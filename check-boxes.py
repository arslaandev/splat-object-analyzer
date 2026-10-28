"""Check rotated furniture boxes and legacy axis-aligned bounds."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from refine_inventory import fit_oriented_box, coherent_samples


def local_points(points, box):
    q = [box['rotation'][a] for a in 'xyzw']
    return (points - [box['position'][a] for a in 'xyz']) @ Rotation.from_quat(q).as_matrix()


grid = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-.4, .4, 11), np.linspace(-.5, .5, 13))
local = np.array(grid).reshape(3, -1).T
points = local @ Rotation.from_euler('y', 37, degrees=True).as_matrix().T + [4, 2, -3]
box = fit_oriented_box(points, .01)
size = np.array([box['scale'][a] for a in 'xyz'])
assert np.prod(size) < np.prod(np.ptp(points, axis=0)) * .6, 'Rotated furniture still has an axis-aligned box'
assert np.mean(np.all(np.abs(local_points(points, box)) <= size / 2 + 1e-6, axis=1)) >= .95
np.testing.assert_allclose(sorted(size), [.8, 1, 4], atol=.06)
aligned = fit_oriented_box(local, .01)
assert abs(aligned['rotation']['y']) < 1e-8, 'Aligned bedroom box rotated unnecessarily'
corners = np.array([[x, y, z] for x in [-1,1] for y in [-.4,.4] for z in [-.2,.2]])
small = fit_oriented_box(corners, .01)
assert np.all(np.abs(local_points(corners, small)) <= np.array(list(small['scale'].values())) / 2 + 1e-6), 'Tiny objects lost their corner points'
observations = [{'frame':0,'quality':.95,'points':local},
                {'frame':1,'quality':.9,'points':local + [8,0,0]},
                {'frame':2,'quality':.9,'points':local[local[:,0] > 0]}]
obj = {'frames':[{'frame_idx':i,'score':.9-i*.1} for i in range(3)]}
assert [s['frame'] for s in coherent_samples(observations, obj, .05)] == [0,2], 'Nearby-object observations polluted one box'

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    result = json.loads(path.read_text())
    count = 0
    surfaces_path = path.with_name('object_surfaces.npz')
    surfaces = np.load(surfaces_path) if surfaces_path.exists() else None
    for obj in result['objects']:
        if not obj.get('oriented_box'):
            continue
        box = obj['oriented_box']
        q = np.array([box['rotation'][a] for a in 'xyzw'])
        size = np.array([box['scale'][a] for a in 'xyz'])
        assert np.isfinite(q).all() and np.isclose(np.linalg.norm(q), 1.)
        assert np.isfinite(size).all() and (size > 0).all()
        assert all(np.isfinite(box['position'][a]) for a in 'xyz')
        if surfaces is not None and obj['id'] in surfaces and len(surfaces[obj['id']]):
            coverage = np.mean(np.all(np.abs(local_points(surfaces[obj['id']], box)) <= size / 2 + 1e-5, axis=1))
            assert coverage >= .9, f'{obj["id"]}: oriented box lost its observed surfaces'
        count += 1
    assert count, 'No oriented surface boxes in output'
    if surfaces is not None:
        surfaces.close()
    print(f'Validated {count} oriented boxes; legacy bounds retained.')
print('Oriented box checks passed.')
