"""Resolve candidate identities using SAM masks and their rendered 3D surfaces."""
import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

SAM_MODEL = 'facebook/sam-vit-base'
SAM_REVISION = 'd0d9250bd6a38ebbb901aa29e8d37787e366f7be'

# Compatible names may describe one instance; compatibility alone never merges it.
FAMILIES = [
    {'shelf', 'bookshelf', 'bookcase', 'shelving unit'},
    {'chair', 'armchair', 'office chair', 'swivel chair'},
    {'computer', 'monitor', 'television', 'computer monitor'},
    {'picture', 'painting', 'picture frame', 'poster', 'photo'},
    {'carpet', 'rug', 'mat', 'yoga mat'},
    {'lamp', 'light fixture', 'ceiling fan', 'ceiling light'},
]
PARENTS = {
    'book': {'bookshelf', 'shelf', 'cabinet'},
    'drawer': {'desk', 'cabinet', 'bookshelf', 'shelf'},
    'pillow': {'bed', 'couch', 'chair', 'armchair'},
    'blanket': {'bed', 'couch'},
}


def family(label):
    return next((min(group) for group in FAMILIES if label in group), label)


def voxel_points(points, voxel_size):
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    cells = np.unique(np.floor(np.asarray(points) / voxel_size).astype(np.int64), axis=0)
    return (cells.astype(np.float32) + .5) * voxel_size


def mask_points(mask, depth, transform, metadata):
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > .01))
    if len(xs) > 12000:
        picks = np.linspace(0, len(xs) - 1, 12000, dtype=int)
        ys, xs = ys[picks], xs[picks]
    z = depth[ys, xs]
    camera = np.column_stack(((xs - metadata['cx']) * z / metadata['fl_x'],
                              (ys - metadata['cy']) * z / metadata['fl_y'], z))
    return camera @ transform[:3, :3].T + transform[:3, 3]


def surface_overlap(a, b, tolerance):
    """Directional visible-surface coverage, independent of detector box centres."""
    if not len(a) or not len(b):
        return 0., 0.
    if np.any(a.max(0) + tolerance < b.min(0)) or np.any(b.max(0) + tolerance < a.min(0)):
        return 0., 0.
    da = cKDTree(b).query(a, distance_upper_bound=tolerance)[0]
    db = cKDTree(a).query(b, distance_upper_bound=tolerance)[0]
    return float(np.isfinite(da).mean()), float(np.isfinite(db).mean())


def same_instance(a, b, tolerance):
    if family(a['object']['label']) != family(b['object']['label']):
        return False
    ab, ba = surface_overlap(a['points'], b['points'], tolerance)
    if max(ab, ba) < .68 or (min(ab, ba) < .16 and max(ab, ba) < .85):
        return False
    volumes = [np.maximum(np.ptp(node['points'], axis=0), tolerance).prod() for node in (a, b)]
    contained_fragment = max(ab, ba) >= .9 and max(volumes) >= min(volumes) * 1.8
    shared = a['masks'].keys() & b['masks'].keys()
    # Distinct instances simultaneously visible must not collapse just because
    # their estimated surfaces are close (e.g. neighboring shelves or chairs).
    for frame in shared:
        ma, mb = a['masks'][frame], b['masks'][frame]
        intersection = np.count_nonzero(ma & mb)
        if intersection / max(1, min(ma.sum(), mb.sum())) < .08 and not contained_fragment:
            return False
    return True


def merge_nodes(nodes, voxel_size):
    # ponytail: greedy surface association handles partial observations, but is
    # not a learned object tracker; severe occlusion can still split identities.
    groups = []
    for node in sorted(nodes, key=lambda n: (-len(n['points']), -n['object']['confidence'])):
        for group in groups:
            if same_instance(group, node, voxel_size * 2):
                group['members'].extend(node['members'])
                group['evidence'].extend(node.get('evidence', []))
                group['points'] = voxel_points(np.vstack([group['points'], node['points']]), voxel_size)
                for frame, mask in node['masks'].items():
                    group['masks'][frame] = group['masks'].get(frame, np.zeros_like(mask)) | mask
                break
        else:
            groups.append({'object': node['object'], 'members': list(node['members']),
                           'points': node['points'].copy(), 'masks': dict(node['masks']),
                           'evidence': list(node.get('evidence', []))})
    return groups


def _segment_candidates(objects, root, transforms, device, fingerprint):
    import torch
    from transformers import SamModel, SamProcessor

    cache = root / 'instance_masks'
    cache.mkdir(exist_ok=True)
    requests = defaultdict(list)
    observations = defaultdict(list)
    for obj in objects:
        chosen, positions = [], set()
        for frame in obj['frames']:
            position = transforms['frames'][frame['frame_idx']].get('position_idx', frame['frame_idx'])
            if position not in positions:
                chosen.append(frame)
                positions.add(position)
            if len(chosen) == 4:
                break
        for frame in chosen:
            intrinsics = [transforms[key] for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')]
            key = hashlib.sha256(json.dumps([SAM_REVISION, fingerprint, intrinsics, obj['id'], frame['frame_idx'],
                                            frame['box'], transforms['frames'][frame['frame_idx']]],
                                           sort_keys=True).encode()).hexdigest()[:24]
            requests[frame['frame_idx']].append((obj['id'], frame['box'], cache / f'{key}.npz'))

    pending = sum(not path.exists() for items in requests.values() for _, _, path in items)
    print(f'[identity] SAM: {pending} uncached masks across {len(requests)} views on {device}', flush=True)
    processor = model = None
    if pending:
        processor = SamProcessor.from_pretrained(SAM_MODEL, revision=SAM_REVISION)
        model = SamModel.from_pretrained(SAM_MODEL, revision=SAM_REVISION).eval().to(device)
    try:
        for number, (frame_idx, items) in enumerate(sorted(requests.items())):
            missing = [item for item in items if not item[2].exists()]
            metadata = transforms['frames'][frame_idx]
            if missing:
                with Image.open(root / metadata['file_path']) as source:
                    image = source.convert('RGB')
                prepared = processor(images=image, return_tensors='pt')
                with torch.inference_mode():
                    embeddings = model.get_image_embeddings(prepared['pixel_values'].to(device))
                    for first in range(0, len(missing), 8):
                        batch = missing[first:first + 8]
                        inputs = processor(images=image, input_boxes=[[item[1] for item in batch]],
                                           return_tensors='pt')
                        outputs = model(image_embeddings=embeddings,
                                        input_boxes=inputs['input_boxes'].to(device=device, dtype=torch.float32),
                                        multimask_output=False)
                        masks = processor.image_processor.post_process_masks(
                            outputs.pred_masks.cpu(), inputs['original_sizes'], inputs['reshaped_input_sizes'])[0]
                        qualities = outputs.iou_scores.cpu().numpy()[0, :, 0]
                        for item, mask, quality in zip(batch, masks[:, 0].numpy(), qualities):
                            np.savez_compressed(item[2], mask=mask, quality=float(quality))
                del embeddings, outputs, prepared, inputs
            depth = np.load(root / metadata['depth_path'].replace('.png', '.npy'))
            pose = np.asarray(metadata['transform_matrix'])
            for object_id, box, path in items:
                with np.load(path) as saved:
                    mask, quality = saved['mask'], float(saved['quality'])
                if quality < .65 or mask.sum() < 32:
                    continue
                points = mask_points(mask, depth, pose, transforms)
                if len(points) >= 32:
                    observations[object_id].append({'frame': frame_idx, 'mask': mask, 'points': points,
                                                     'quality': quality, 'path': str(path.relative_to(root))})
            if (number + 1) % 10 == 0 or number + 1 == len(requests):
                print(f'[identity] Segmented {number + 1}/{len(requests)} views', flush=True)
    finally:
        del model
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
        elif device.startswith('cuda'):
            torch.cuda.empty_cache()
    return observations


def _object_from_group(group, index, voxel_size):
    members = group['members']
    labels = Counter()
    frames = {}
    for member in members:
        labels[member['label']] += sum(f['score'] for f in member['frames'])
        for frame in member['frames']:
            if frame['frame_idx'] not in frames or frame['score'] > frames[frame['frame_idx']]['score']:
                frames[frame['frame_idx']] = frame
    label = labels.most_common(1)[0][0]
    if 'bookshelf' in labels:
        label = 'bookshelf'
    if 'chair' in labels:
        label = 'chair'
    points = group['points']
    obj = copy.deepcopy(max(members, key=lambda m: m['confidence']))
    obj.update(id=f'object_{index + 1:04d}', label=label, alternative_labels=sorted(set(labels) - {label}),
               source_candidate_ids=[m['id'] for m in members], parent_id=None, children=[],
               mask_evidence=group.get('evidence', []),
               frames=sorted(frames.values(), key=lambda f: -f['score']), supporting_view_count=len(frames),
               identity_method='segmented_surface_overlap' if len(points) else 'unresolved_candidate')
    if len(points):
        low, high = np.percentile(points, [1, 99], axis=0)
        high = np.maximum(high, low + voxel_size)
        obj['position'] = dict(zip('xyz', map(float, (low + high) / 2)))
        obj['scale'] = dict(zip('xyz', map(float, high - low)))
        obj['bounding_box'] = {'type': 'visible_surface_estimate',
                               'min': dict(zip('xyz', map(float, low))),
                               'max': dict(zip('xyz', map(float, high)))}
    return obj


def attach_contents(objects, groups, voxel_size):
    for child_idx, child in enumerate(objects):
        permitted = PARENTS.get(child['label'])
        if not permitted or not len(groups[child_idx]['points']):
            continue
        best = None
        child_points = groups[child_idx]['points']
        for parent_idx, parent in enumerate(objects):
            if parent['label'] not in permitted or not len(groups[parent_idx]['points']):
                continue
            low = np.array([parent['bounding_box']['min'][a] for a in 'xyz']) - voxel_size * 2
            high = np.array([parent['bounding_box']['max'][a] for a in 'xyz']) + voxel_size * 2
            contained = np.all((child_points >= low) & (child_points <= high), axis=1).mean()
            if contained < .7:
                continue
            coverage, _ = surface_overlap(child_points, groups[parent_idx]['points'], voxel_size * 3)
            score = contained + coverage
            if best is None or score > best[0]:
                best = (score, parent)
        if best:
            parent = best[1]
            child['parent_id'] = parent['id']
            child['relationship'] = 'contents' if child['label'] == 'book' else 'part_of'
            parent['children'].append(child['id'])


def refine_inventory(result, root, device):
    root = Path(root)
    started = time.perf_counter()
    transforms = json.loads((root / 'transforms.json').read_text())
    voxel_size = max(float(transforms['scene_radius']) * .015, 1e-5)
    observations = _segment_candidates(result['objects'], root, transforms, device, result['input']['sha256'])
    nodes = []
    for obj in result['objects']:
        samples = observations[obj['id']]
        points = voxel_points(np.vstack([s['points'] for s in samples]), voxel_size) if samples else np.empty((0, 3))
        nodes.append({'object': obj, 'members': [obj], 'points': points,
                      'masks': {s['frame']: s['mask'] for s in samples},
                      'evidence': [{'frame_idx': s['frame'], 'mask': s['path'],
                                    'mask_quality': s['quality'], 'candidate_id': obj['id']}
                                   for s in samples]})
    groups = merge_nodes(nodes, voxel_size)
    refined = [_object_from_group(group, i, voxel_size) for i, group in enumerate(groups)]
    attach_contents(refined, groups, voxel_size)
    np.savez_compressed(root / 'object_surfaces.npz',
                        **{obj['id']: group['points'] for obj, group in zip(refined, groups)})
    output = copy.deepcopy(result)
    output['objects'] = refined
    output['frame_annotations'] = {}
    for index, obj in enumerate(refined):
        for frame in obj['frames']:
            output['frame_annotations'].setdefault(str(frame['frame_idx']), []).append({
                'label': obj['label'], 'object_idx': index, 'box': frame['box'], 'score': frame['score']})
    output['instance_resolution'] = {
        'model': SAM_MODEL, 'revision': SAM_REVISION, 'voxel_size': voxel_size,
        'input_candidates': len(result['objects']), 'resolved_records': len(refined),
        'top_level_objects': sum(o['parent_id'] is None for o in refined),
        'child_objects': sum(o['parent_id'] is not None for o in refined),
        'seconds': time.perf_counter() - started,
    }
    output['limitations'] = ['Boxes bound observed segmented surfaces; hidden geometry is not measured.',
                             'Mask-based identity and parent/child links are model estimates and can be wrong.']
    print(f'[identity] {len(result["objects"])} candidates → {len(refined)} records; '
          f'{output["instance_resolution"]["top_level_objects"]} top-level objects', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    from pipeline import _select_device
    path = Path(args.input)
    backup = path.with_name('candidates.json')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    result = refine_inventory(json.loads(backup.read_text()), path.parent, args.device or _select_device())
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
