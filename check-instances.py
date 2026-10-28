"""Identity regression checks: one bed, distinct neighbors, nested contents."""
import numpy as np
from collections import Counter
import json
from pathlib import Path
import sys
from refine_inventory import merge_nodes, attach_contents, _object_from_group


grid = np.array(np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))).reshape(2, -1).T
surface = np.column_stack([grid, np.zeros(len(grid))])


def node(label, identity, points=surface, offset=0):
    mask = np.zeros((40, 80), dtype=bool)
    mask[5:35, 5 + offset:35 + offset] = True
    obj = {'id': identity, 'label': label, 'confidence': .9,
           'frames': [{'frame_idx': 0, 'box': [0, 0, 30, 30], 'score': .9}],
           'position': dict.fromkeys('xyz', 0.), 'scale': dict.fromkeys('xyz', 1.),
           'bounding_box': {'min': dict.fromkeys('xyz', 0.), 'max': dict.fromkeys('xyz', 1.)}}
    return {'object': obj, 'members': [obj], 'points': points, 'masks': {0: mask}}


# A whole-bed mask and a partial-bed mask must share one identity even if the
# original detector placed their centres on different parts of the mattress.
bed = merge_nodes([node('bed', 'a'), node('bed', 'b', surface[grid[:, 0] > .4])], .025)
assert len(bed) == 1 and len(bed[0]['members']) == 2
fragment = surface[(grid[:, 0] > .6) & (grid[:, 1] > .6)]
assert len(merge_nodes([node('bookshelf', 'a'), node('shelf', 'b', fragment, offset=40)], .025)) == 1
assert len(merge_nodes([node('bed', 'a'), node('bed', 'b', surface + [3, 0, 0])], .025)) == 2

# Close 3D surfaces cannot override two disjoint instance masks in one image.
neighbors = merge_nodes([node('chair', 'a'), node('chair', 'b', surface + [0, 0, .01], offset=40)], .025)
assert len(neighbors) == 2

# Equivalent labels become alternatives on the same physical storage unit.
shelves = merge_nodes([node('shelf', 'a'), node('bookshelf', 'b')], .025)
assert len(shelves) == 1
storage = _object_from_group(shelves[0], 0, .025)
assert storage['label'] == 'bookshelf' and storage['alternative_labels'] == ['shelf']

# A book stays a separate child; surface containment must not erase it.
groups = merge_nodes([node('bookshelf', 'a'), node('book', 'b', surface[(grid[:, 0] > .2) & (grid[:, 0] < .5)])], .025)
assert len(groups) == 2
objects = [_object_from_group(group, i, .025) for i, group in enumerate(groups)]
attach_contents(objects, groups, .025)
book = next(o for o in objects if o['label'] == 'book')
assert book['parent_id'] == next(o['id'] for o in objects if o['label'] == 'bookshelf')
assert book['relationship'] == 'contents'

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    result = json.loads(path.read_text())
    candidates = json.loads(path.with_name('candidates.json').read_text())
    assert result['input'] == candidates['input']
    records = {o['id']: o for o in result['objects']}
    source_ids = [source for obj in records.values() for source in obj['source_candidate_ids']]
    assert Counter(source_ids) == Counter(o['id'] for o in candidates['objects']), 'Candidates lost or duplicated'
    for obj in records.values():
        if obj['parent_id']:
            assert obj['parent_id'] != obj['id']
            assert obj['id'] in records[obj['parent_id']]['children']
        assert all(records[child]['parent_id'] == obj['id'] for child in obj['children'])
        assert all((path.parent / sample['mask']).exists() for sample in obj['mask_evidence'])
    # This source is the reviewed single-bed bedroom, not a universal bed cap.
    if result['input']['sha256'] == '26a0a0316c3a89d321d89b53f46174f24962392f78af0b25c3d070afce57a782':
        assert sum(o['label'] == 'bed' for o in records.values()) == 1
        assert all(o['parent_id'] for o in records.values() if o['label'] == 'book')
        before = sum(o['label'] in {'shelf', 'bookshelf'} for o in candidates['objects'])
        after = sum(o['label'] in {'shelf', 'bookshelf'} and not o['parent_id'] for o in records.values())
        assert after < before
    print(f'Validated {len(records)} instance records with complete candidate provenance and valid hierarchy.')
print('Instance identity and contents checks passed.')
