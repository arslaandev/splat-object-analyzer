"""Public contracts; source geometry and suggested-camera spaces are explicit."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class ErrorDetail(Model):
    code: str
    message: str
    details: list[dict[str, str]] = Field(default_factory=list)


class ErrorResponse(Model):
    error: ErrorDetail


class Progress(Model):
    stage: str
    completed: int | None = None
    total: int | None = None


class Job(Model):
    id: str
    kind: Literal['analysis', 'query']
    analysis_id: str
    status: Literal['queued', 'running', 'succeeded', 'failed']
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: Progress
    error: ErrorDetail | None = None
    status_url: str
    result_url: str


class Vector3(Model):
    x: float
    y: float
    z: float


class Quaternion(Model):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1


class AxisAlignedBox(Model):
    type: str
    min: Vector3
    max: Vector3


class OrientedBox(Model):
    type: str
    position: Vector3
    scale: Vector3
    rotation: Quaternion


class Evidence(Model):
    frame_index: int
    image_url: str
    detector_score: float | None = None
    mask_quality: float | None = None
    geometry_consistency: float | None = None
    evidence_type: str


class ObjectRecord(Model):
    id: str
    label: str
    category: str
    display_name: str
    status: Literal['supported', 'uncertain', 'rejected', 'unreviewed']
    attributes: list[str] = Field(default_factory=list)
    confidence: float
    confidence_type: str = 'peak_detector_score_not_calibrated_probability'
    supporting_view_count: int
    geometry_supporting_view_count: int
    position: Vector3
    scale: Vector3
    bounding_box: AxisAlignedBox | None = None
    oriented_box: OrientedBox | None = None
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    relationship: str | None = None
    description_source: str
    visual_priority: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class SceneSummary(Model):
    type: str
    description: str
    confidence: str


class ObjectGroup(Model):
    id: str
    name: str
    description: str
    anchor_id: str
    anchor_category: str
    object_ids: list[str]
    confidence: str
    basis: str


class Relation(Model):
    subject: str
    relation: str
    object: str
    basis: str
    gap_source_units: float | None = None


class CoordinateSystem(Model):
    geometry_space: Literal['source'] = 'source'
    units: Literal['source_scene_units'] = 'source_scene_units'
    metric_scale_verified: bool = False
    quaternion_order: Literal['xyzw'] = 'xyzw'
    box_scale: str = 'Full extents along box-local axes; oriented_box.rotation maps box-local to source space.'
    suggested_camera_space: Literal['viewer'] = 'viewer'
    source_to_viewer: list[list[float]] = Field(default_factory=lambda: [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


class Camera(Model):
    space: Literal['viewer'] = 'viewer'
    position: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    fov: float
    fov_axis: Literal['vertical'] = 'vertical'
    fov_unit: Literal['degrees'] = 'degrees'


class SuggestedView(Model):
    id: str
    title: str
    frame_index: int
    image_url: str
    object_ids: list[str]
    subjects: list[str]
    score: float
    camera: Camera
    basis: str


class Selection(Model):
    interpretation: str
    object_ids: list[str]
    group_ids: list[str]
    message: str
    match: dict[str, Any] | None = None


class Source(Model):
    filename: str
    format: Literal['ply', 'spz']
    bytes: int
    sha256: str
    download_url: str


class AnalysisResult(Model):
    schema_version: Literal['1.0'] = '1.0'
    analysis_id: str
    quality: Literal['low', 'medium', 'high']
    source: Source
    coordinates: CoordinateSystem = Field(default_factory=CoordinateSystem)
    rendered_view_count: int
    understanding_included: bool
    scene: SceneSummary | None = None
    objects: list[ObjectRecord]
    groups: list[ObjectGroup] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    hero_object_ids: list[str] = Field(default_factory=list)
    default_selection: Selection | None = None
    suggested_views: list[SuggestedView] = Field(default_factory=list)
    limitations: list[str]


class QueryRequest(Model):
    prompt: str = Field(min_length=1, max_length=1000)


class QueryResult(Model):
    schema_version: Literal['1.0'] = '1.0'
    query_id: str
    analysis_id: str
    prompt: str
    coordinates: CoordinateSystem = Field(default_factory=CoordinateSystem)
    selection: Selection
    suggested_views: list[SuggestedView]
    unshown_object_ids: list[str]
