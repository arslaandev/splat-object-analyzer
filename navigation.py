"""Conservative, geometry-only camera-space estimates for standalone PLY/SPZ files.

Unknown cells are never traversable. An enclosed component is a hypothesis about
an interior, not evidence from an original depth sensor. No holes are repaired.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import time
import tempfile

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

VERSION = 'camera-enclosure-v1'
UNKNOWN, OCCUPIED, FREE, CLEARANCE = 0, 1, 2, 3
FACE_CONNECTED = ndimage.generate_binary_structure(3, 1)
LIMITATIONS = [
    'Candidate space is inferred from modeled geometry, not measured or certified free space.',
    'Missing walls or open boundaries can leave all interior space unknown; no holes are automatically sealed.',
    'Enclosed furniture cavities can resemble interiors. Region size checks do not establish room semantics.',
    'Gaussian opacity and the ellipsoid cutoff are appearance-based heuristics, not calibrated occupancy probabilities.',
    'Coordinates and clearance use source scene units; metric scale and gravity are not assumed.',
]


@dataclass(frozen=True)
class NavigationConfig:
    resolution: int = 128  # approximate cells along the longest robust scene side
    max_cells: int = 2_000_000
    sigma_cutoff: float = 2.0
    opacity_threshold: float = .05
    camera_radius_voxels: float = 1.0
    min_region_cells: int = 128

    def __post_init__(self):
        if not 16 <= self.resolution <= 256 or not 4096 <= self.max_cells <= 8_000_000:
            raise ValueError('Navigation resolution must be 16..256 and max_cells 4096..8000000')
        if not math.isfinite(self.sigma_cutoff) or self.sigma_cutoff <= 0:
            raise ValueError('sigma_cutoff must be positive and finite')
        if not 0 <= self.opacity_threshold < 1 or not math.isfinite(self.camera_radius_voxels) or self.camera_radius_voxels <= 0:
            raise ValueError('Invalid opacity threshold or camera clearance')
        if self.min_region_cells < 1:
            raise ValueError('min_region_cells must be positive')


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def _geometry(arrays, cfg):
    means = np.asarray(arrays['means'], dtype=np.float64)
    scales = np.asarray(arrays['scales'], dtype=np.float64)
    quats = np.asarray(arrays['quats'], dtype=np.float64)
    logits = np.asarray(arrays['opacities'], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3 or scales.shape != means.shape or quats.shape != (len(means), 4) or logits.shape != (len(means),):
        raise ValueError('Invalid Gaussian geometry array shapes')
    finite = np.isfinite(means).all(1) & np.isfinite(scales).all(1) & np.isfinite(quats).all(1) & np.isfinite(logits)
    invalid = int((~finite).sum())
    invalid += int((finite & (scales > 60).any(axis=1)).sum())
    norms = np.linalg.norm(quats, axis=1)
    invalid += int((finite & (norms < 1e-10)).sum())
    alpha = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    keep = finite & (norms >= 1e-10) & (alpha >= cfg.opacity_threshold)
    if not keep.any():
        raise ValueError('No usable Gaussian geometry remains for navigation')
    means, scales, quats = means[keep], scales[keep], quats[keep]
    # Work in float64; clamping only avoids overflow. Extreme extents cover the
    # grid conservatively instead of being dropped as inconvenient outliers.
    axes = np.exp(np.clip(scales, -60, 60)) * cfg.sigma_cutoff
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    return means, axes, rotations, invalid, int((finite & (alpha < cfg.opacity_threshold)).sum())


def _grid_bounds(means, cfg):
    lo, hi = np.percentile(means, [.5, 99.5], axis=0)
    longest = float(np.max(hi - lo))
    if longest <= 1e-12:
        raise ValueError('Navigation requires a scene with nonzero spatial extent')
    voxel = longest / cfg.resolution
    # Four empty boundary cells prevent a cropped surface from masquerading as
    # a closed room boundary at the edge of the region of interest.
    while True:
        shape = np.maximum(np.ceil((hi-lo)/voxel).astype(np.int64), 1) + 8
        if int(np.prod(shape)) <= cfg.max_cells:
            break
        voxel *= max(1.02, (np.prod(shape)/cfg.max_cells)**(1/3))
    return lo - 4*voxel, shape, float(voxel)


def voxelize(means, axes, rotations, origin, shape, voxel, diagnostics=None):
    """Rasterize a conservative superset of cutoff ellipsoid/voxel intersections.

    Transform each voxel's bounding box into Gaussian-local coordinates. Its
    axis intervals bound the true transformed voxel; a lower bound on squared
    Mahalanobis distance can reject it, but cannot miss an intersection. This
    includes thin surfaces that contain no voxel center. All Gaussians are used.
    """
    occupied = np.zeros(tuple(shape), dtype=bool)
    half_local = .5*voxel*np.abs(rotations).sum(axis=1)
    extents = np.sqrt(np.sum((rotations*axes[:, None, :])**2, axis=2))
    lower = np.maximum(np.floor(np.clip((means-extents-origin)/voxel, -1, shape)).astype(np.int64), 0)
    upper = np.minimum(np.floor(np.clip((means+extents-origin)/voxel, -1, shape)).astype(np.int64), shape-1)
    sizes = np.maximum(upper-lower+1, 0)
    counts = np.prod(sizes, axis=1)
    # Huge ellipsoids can span the entire grid. Bound total pair work, filling
    # their world AABBs conservatively rather than skipping collision geometry.
    coarse = counts > 250_000
    remaining = int(counts[~coarse].sum())
    if remaining > 80_000_000:
        largest = np.argsort(counts[~coarse])[::-1]
        indices = np.flatnonzero(~coarse)[largest]
        remove = np.searchsorted(np.cumsum(counts[indices]), remaining-80_000_000)+1
        coarse[indices[:remove]] = True
    if coarse.any():
        difference = np.zeros(tuple(shape+1), dtype=np.int32)
        for bits in itertools.product((0, 1), repeat=3):
            corners = np.where(np.array(bits), upper[coarse]+1, lower[coarse])
            np.add.at(difference, tuple(corners.T), (-1)**sum(bits))
        for axis in range(3):
            np.cumsum(difference, axis=axis, dtype=np.int32, out=difference)
        occupied |= difference[:-1, :-1, :-1] > 0
        counts[coarse] = 0
    if diagnostics is not None:
        diagnostics['conservative_box_fallback_gaussians'] = int(coarse.sum())
    # Bound temporary arrays regardless of splat count or Gaussian extent.
    budget = 250_000
    starts = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    for first in range(0, len(means), 8192):
        stop = min(first+8192, len(means))
        for offset in range(int(starts[first]), int(starts[stop]), budget):
            linear = np.arange(offset, min(offset+budget, int(starts[stop])), dtype=np.int64)
            which = np.searchsorted(starts, linear, side='right')-1
            within = linear-starts[which]
            dims = sizes[which]
            cells = lower[which] + np.column_stack((within//(dims[:, 1]*dims[:, 2]),
                (within//dims[:, 2]) % dims[:, 1], within % dims[:, 2]))
            delta = origin+(cells+.5)*voxel-means[which]
            local = np.einsum('ni,nij->nj', delta, rotations[which])
            distance = np.maximum(np.abs(local)-half_local[which], 0)/axes[which]
            hit = np.sum(distance**2, axis=1) <= 1+1e-10
            occupied[tuple(cells[hit].T)] = True
        if first % 65536 == 0:
            print(f'[navigation] Voxelized {stop}/{len(means)} Gaussians', flush=True)
    return occupied


def classify(occupied, origin, voxel, camera_radius, min_region_cells=128):
    """Find enclosed components in camera configuration space, not point space.

    A crack narrower than the configured camera cannot connect its inside and
    outside. Obstacle clearance handles this without inventing a wall across the
    crack. Broad scan openings remain connected to the outside and stay unknown.
    """
    empty = ~occupied
    if occupied.any():
        clearance = np.maximum(ndimage.distance_transform_edt(empty)*voxel-math.sqrt(3)*voxel/2, 0).astype(np.float32)
    else:
        clearance = np.full(occupied.shape, np.inf, dtype=np.float32)
    camera_space = empty & (clearance > camera_radius+voxel*1e-7)
    boundary = np.zeros_like(occupied)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    # Even a diagonal opening that fits the camera leaks to the exterior.
    outside = ndimage.binary_propagation(boundary & camera_space, mask=camera_space, structure=np.ones((3, 3, 3)))
    enclosed = camera_space & ~outside
    enclosed_labels, count = ndimage.label(enclosed, structure=FACE_CONNECTED)
    populations = np.bincount(enclosed_labels.ravel(), minlength=count+1)
    slices = ndimage.find_objects(enclosed_labels)
    eligible = np.zeros(count+1, dtype=bool)
    min_count = max(min_region_cells, int(occupied.size*.0005))
    for label, section in enumerate(slices, 1):
        spans = np.array([s.stop-s.start for s in section])*voxel
        eligible[label] = populations[label] >= min_count and np.min(spans) >= 2*voxel
    candidates = eligible[enclosed_labels]
    free = candidates
    labels, region_count = ndimage.label(free, structure=FACE_CONNECTED)
    free_counts = np.bincount(labels.ravel(), minlength=region_count+1)
    viable = free_counts >= max(8, min_count//4)
    viable[0] = False
    rejected_pockets = int((~viable[1:]).sum())
    free &= viable[labels]
    labels, region_count = ndimage.label(free, structure=FACE_CONNECTED)
    grid = np.zeros(occupied.shape, dtype=np.uint8)
    grid[occupied] = OCCUPIED
    grid[empty & ~camera_space] = CLEARANCE
    grid[free] = FREE
    regions = []
    for region_id, section in enumerate(ndimage.find_objects(labels), 1):
        local = labels[section] == region_id
        distances = np.where(local, clearance[section], -1)
        cell = np.array(np.unravel_index(np.argmax(distances), distances.shape)) + [s.start for s in section]
        cells = int(local.sum())
        regions.append(dict(id=region_id, cell_count=cells, volume_source_units_cubed=float(cells*voxel**3),
            seed=(origin+(cell+.5)*voxel).tolist(), max_clearance=float(clearance[tuple(cell)]),
            bounds=dict(min=(origin+np.array([s.start for s in section])*voxel).tolist(),
                        max=(origin+np.array([s.stop for s in section])*voxel).tolist())))
    regions.sort(key=lambda r:r['cell_count'], reverse=True)
    diagnostics = dict(enclosed_components=count, accepted_enclosures=int(eligible.sum()),
        rejected_enclosures=count-int(eligible.sum()), connected_camera_regions=region_count,
        exterior_camera_cells=int(outside.sum()),
        rejected_clearance_pockets=rejected_pockets)
    return grid, labels.astype(np.int32), clearance, regions, diagnostics


class NavigationMap:
    def __init__(self, metadata, grid, labels, clearance):
        self.metadata, self.grid, self.labels, self.clearance = metadata, grid, labels, clearance
        self.origin = np.asarray(metadata['origin'], dtype=np.float64)
        self.voxel = metadata['voxel_size']
        self.shape = np.array(grid.shape)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory/'navigation.json').read_text())
        with np.load(directory/'navigation.npz', allow_pickle=False) as archive:
            if str(archive['fingerprint']) != metadata['fingerprint']:
                raise ValueError('Navigation grid and metadata do not match')
            return cls(metadata, archive['grid'], archive['labels'], archive['clearance'])

    def cell(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        relative = (point-self.origin)/self.voxel
        if not np.all((relative >= 0) & (relative < self.shape)):
            return None
        return tuple(np.floor(relative).astype(np.int64))

    def is_valid(self, point):
        index = self.cell(point)
        return index is not None and self.grid[index] == FREE

    def segment_valid(self, start, end):
        """Continuous supercover traversal: also check cells touched at corners.

        A free voxel center has radius+half-diagonal clearance, so its entire
        voxel is valid for the configured camera radius. Grid inflation below
        adds that extra half-diagonal explicitly when building the map.
        """
        if not self.is_valid(start) or not self.is_valid(end):
            return False
        a = (np.asarray(start)-self.origin)/self.voxel
        b = (np.asarray(end)-self.origin)/self.voxel
        cell = np.floor(a).astype(int)
        delta = b-a
        step = np.sign(delta).astype(int)
        nonzero = np.abs(delta) > 1e-14
        crossing = np.full(3, np.inf)
        increment = np.full(3, np.inf)
        increment[nonzero] = 1/np.abs(delta[nonzero])
        crossing[nonzero] = (cell[nonzero]+(step[nonzero] > 0)-a[nonzero])/delta[nonzero]
        parallel_boundary = [i for i in range(3) if not nonzero[i] and abs(a[i]-round(a[i])) < 1e-10]
        def allowed(index):
            for flags in itertools.product((0, 1), repeat=len(parallel_boundary)):
                neighbor = index.copy()
                for axis, flag in zip(parallel_boundary, flags):
                    neighbor[axis] -= flag
                if np.any(neighbor < 0) or np.any(neighbor >= self.shape) or self.grid[tuple(neighbor)] != FREE:
                    return False
            return True
        # The exact start can touch excluded cells on either side of a plane.
        initial_axes = [i for i in range(3) if abs(a[i]-round(a[i])) < 1e-10]
        for flags in itertools.product((0, 1), repeat=len(initial_axes)):
            touched = cell.copy()
            for axis, flag in zip(initial_axes, flags):
                touched[axis] -= flag
            if not allowed(touched):
                return False
        if not allowed(cell):
            return False
        for _ in range(int(self.shape.sum())+4):
            t = float(crossing.min())
            if t > 1+1e-10:
                return True
            axes = np.flatnonzero(np.abs(crossing-t) < 1e-10)
            for flags in itertools.product((0, 1), repeat=len(axes)):
                touched = cell.copy()
                for axis, flag in zip(axes, flags):
                    touched[axis] += step[axis]*flag
                if not allowed(touched):
                    return False
            cell[axes] += step[axes]
            crossing[axes] += increment[axes]
        return False

    def validate(self, points):
        for i, point in enumerate(points):
            if not self.is_valid(point):
                return dict(valid=False, reason='point_is_outside_candidate_space', invalid_point_index=i)
            if i and not self.segment_valid(points[i-1], point):
                return dict(valid=False, reason='segment_crosses_excluded_space', invalid_segment_index=i-1)
        return dict(valid=True, reason='valid_against_candidate_map')

    def find_path(self, start, goal, max_expansions=50_000):
        failed = lambda reason:dict(found=False, reason=reason, points=[], basis='geometry_candidate_map')
        if not self.is_valid(start) or not self.is_valid(goal):
            return failed('endpoint_is_outside_candidate_space')
        a, b = self.cell(start), self.cell(goal)
        if self.labels[a] != self.labels[b]:
            return failed('endpoints_are_in_disconnected_regions')
        if self.segment_valid(start, goal):
            return dict(found=True, reason='path_found', points=[list(start), list(goal)], basis='geometry_candidate_map')
        queue = [(0, a)]
        cost, parent = {a:0}, {}
        closed = set()
        while queue and len(closed) < max_expansions:
            _, current = heapq.heappop(queue)
            if current in closed:
                continue
            closed.add(current)
            if current == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parent[cells[-1]])
                points = [list(start)] + [(self.origin+(np.array(c)+.5)*self.voxel).tolist() for c in reversed(cells)] + [list(goal)]
                # No smoothing or diagonal shortcuts: validate the final segments.
                if not self.validate(points)['valid']:
                    return failed('path_failed_continuous_validation')
                return dict(found=True, reason='path_found', points=points, basis='geometry_candidate_map')
            for axis in range(3):
                for direction in (-1, 1):
                    nxt = list(current); nxt[axis] += direction; nxt = tuple(nxt)
                    if nxt in closed or any(c < 0 or c >= self.shape[i] for i, c in enumerate(nxt)) or self.grid[nxt] != FREE:
                        continue
                    value = cost[current]+1
                    if value < cost.get(nxt, math.inf):
                        cost[nxt], parent[nxt] = value, current
                        heapq.heappush(queue, (value+sum(abs(nxt[i]-b[i]) for i in range(3)), nxt))
        return failed('search_limit_reached' if queue else 'no_path_found')


def build_navigation(arrays, directory, source_hash, cfg=None):
    cfg = cfg or NavigationConfig()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps([VERSION, source_hash, asdict(cfg)], sort_keys=True).encode()).hexdigest()
    if (directory/'navigation.json').exists() and (directory/'navigation-grid.json').exists() and (directory/'navigation.npz').exists():
        try:
            existing = json.loads((directory/'navigation.json').read_text())
            if existing.get('fingerprint') == fingerprint:
                return NavigationMap.load(directory)
        except (OSError, ValueError, KeyError):
            pass  # Rebuild an interrupted or mismatched publication.
    started = time.perf_counter()
    print('[navigation] Building Gaussian collision grid', flush=True)
    means, axes, rotations, invalid, discarded = _geometry(arrays, cfg)
    origin, shape, voxel = _grid_bounds(means, cfg)
    voxel_diagnostics = {}
    occupied = voxelize(means, axes, rotations, origin, shape, voxel, voxel_diagnostics)
    radius = cfg.camera_radius_voxels*voxel
    # The extra half diagonal makes every point in a FREE voxel, not just its
    # center, clear for radius. classify already subtracts the obstacle half.
    grid, labels, clearance, regions, diagnostics = classify(occupied, origin, voxel,
        radius+math.sqrt(3)*voxel/2, cfg.min_region_cells)
    if invalid:
        # Invalid geometry has unknown support, so do not authorize navigation.
        grid[grid != OCCUPIED] = UNKNOWN
        labels.fill(0); regions = []
        diagnostics['connected_camera_regions'] = 0
    counts = np.bincount(grid.ravel(), minlength=4)
    metadata = dict(schema_version='1.0', algorithm=VERSION, fingerprint=fingerprint, source_sha256=source_hash,
        status='candidate_space' if regions else 'uncertain', space='source', units='source_scene_units',
        metric_scale_verified=False, origin=origin.tolist(), shape=shape.tolist(), voxel_size=voxel,
        camera_radius=radius, voxel_clearance_margin=math.sqrt(3)*voxel,
        state_codes=dict(unknown=UNKNOWN, occupied=OCCUPIED, candidate_free=FREE, clearance_excluded=CLEARANCE),
        cell_counts=dict(unknown=int(counts[0]), occupied=int(counts[1]), candidate_free=int(counts[2]), clearance_excluded=int(counts[3])),
        regions=regions, settings=asdict(cfg), diagnostics={**diagnostics, **voxel_diagnostics, 'invalid_gaussians':invalid,
            'below_opacity_threshold':discarded, 'used_gaussians':len(means)},
        seconds=round(time.perf_counter()-started, 3), limitations=LIMITATIONS +
            (['Invalid Gaussian parameters prevent authorizing any camera space.'] if invalid else []))
    archive = directory/'navigation.npz.tmp'
    with archive.open('wb') as out:
        np.savez_compressed(out, grid=grid, labels=labels, clearance=clearance, fingerprint=np.array(fingerprint))
    archive.replace(directory/'navigation.npz')
    flat = grid.ravel()
    changes = np.r_[0, np.flatnonzero(np.diff(flat)) + 1, len(flat)]
    runs = np.column_stack((flat[changes[:-1]], np.diff(changes))).ravel().tolist()
    _json(directory/'navigation-grid.json', dict(**metadata, encoding='rle_xyz_c', runs=runs))
    _json(directory/'navigation.json', metadata)
    print(f'[navigation] {len(regions)} candidate regions; {counts[2]} free cells; {metadata["seconds"]}s', flush=True)
    return NavigationMap(metadata, grid, labels, clearance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splat', required=True, help='Gaussian PLY or SPZ; no camera poses required')
    parser.add_argument('--output', required=True, help='Output directory; can be an existing analysis directory')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--camera-radius-voxels', type=float, default=1)
    args = parser.parse_args()
    source, output = Path(args.splat).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if source.suffix.lower() not in ('.ply', '.spz'):
        parser.error('Provide a .ply or .spz file')
    output.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    from render_cameras import _convert_spz_to_ply, _load_ply_arrays
    with tempfile.TemporaryDirectory(prefix='nav-input-', dir=output) as temporary:
        ply = source
        if source.suffix.lower() == '.spz':
            ply = Path(temporary)/'input.ply'
            _convert_spz_to_ply(str(source), str(ply))
        result = build_navigation(_load_ply_arrays(str(ply)), output, digest,
            NavigationConfig(resolution=args.resolution, camera_radius_voxels=args.camera_radius_voxels))
    print(json.dumps(result.metadata, indent=2))


if __name__ == '__main__':
    main()
