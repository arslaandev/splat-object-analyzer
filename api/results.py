"""Normalize existing pipeline output without changing object IDs or geometry."""
from .schemas import AnalysisResult, QueryResult


def views(response, analysis_id):
    return [dict(id=v['id'], title=v['title'], frame_index=v['frame_idx'],
        image_url=f'/v1/analyses/{analysis_id}/frames/{v["frame_idx"]}', object_ids=v['object_ids'],
        subjects=v['subjects'], score=v['score'], camera=v['camera'], basis=v['basis'])
        for v in response.get('shots', [])]


def analysis_result(record, inventory, transforms, semantic=None):
    if inventory['input']['sha256'] != record['sha256']:
        raise ValueError('Inventory source fingerprint mismatch')
    if semantic and semantic['source_hash'] != record['sha256']:
        raise ValueError('Scene understanding source fingerprint mismatch')
    index = semantic.get('understanding', {}) if semantic else {}
    descriptions = {o['id']:o for o in index.get('objects', [])}
    objects = []
    for o in inventory['objects']:
        description = descriptions.get(o['id'], {})
        evidence = description.get('evidence') or o.get('frames', [])[:4]
        objects.append(dict(id=o['id'], label=o['label'], category=description.get('category', o['label']),
            display_name=description.get('display_name', o['label']), status=description.get('status', 'unreviewed'),
            attributes=description.get('attributes', []), confidence=o['confidence'],
            supporting_view_count=o.get('supporting_view_count', 0),
            geometry_supporting_view_count=o.get('geometry_supporting_view_count', 0),
            position=o['position'], scale=o['scale'], bounding_box=o.get('bounding_box'), oriented_box=o.get('oriented_box'),
            parent_id=o.get('parent_id'), children=o.get('children', []), relationship=o.get('relationship'),
            description_source=description.get('description_source', 'detector_label_only'),
            visual_priority=description.get('visual_priority'),
            evidence=[dict(frame_index=e['frame_idx'], image_url=f'/v1/analyses/{record["id"]}/frames/{e["frame_idx"]}',
                detector_score=e.get('score'), mask_quality=e.get('mask_quality'),
                geometry_consistency=e.get('geometry_consistency'), evidence_type=e.get('evidence_type', 'detector_only'))
                for e in evidence]))
    relations = [dict(subject=r['subject'], relation=r['relation'], object=r['object'], basis=r['basis'],
                      gap_source_units=r.get('gap_source_units')) for r in index.get('relations', [])]
    return AnalysisResult.model_validate(dict(analysis_id=record['id'], quality=record['quality'],
        source=dict(filename=record['filename'], format=record['format'], bytes=record['bytes'], sha256=record['sha256'],
                    download_url=f'/v1/analyses/{record["id"]}/source'),
        rendered_view_count=len(transforms['frames']), understanding_included=semantic is not None,
        scene={k:index['scene'][k] for k in ('type','description','confidence')} if index.get('scene') else None,
        objects=objects, groups=index.get('groups', []), relations=relations, hero_object_ids=index.get('hero_object_ids', []),
        default_selection=semantic.get('selection') if semantic else None, suggested_views=views(semantic or {}, record['id']),
        limitations=list(dict.fromkeys(inventory.get('limitations', []) + index.get('limitations', []) +
            ['Object confidence and semantic similarity scores are not calibrated probabilities.',
             'Geometry uses the source file coordinate space; suggested cameras use the documented viewer transform.']))))


def query_result(record, response):
    if response['source_hash'] != record['sha256']:
        raise ValueError('Query source fingerprint mismatch')
    return QueryResult.model_validate(dict(query_id=record['id'], analysis_id=record['analysis_id'],
        prompt=record['prompt'], selection=response['selection'], suggested_views=views(response, record['analysis_id']),
        unshown_object_ids=response.get('unshown_object_ids', [])))
