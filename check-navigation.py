"""Geometry regressions: closed/open rooms, nonconvex routes, extents and clearance."""
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from navigation import (UNKNOWN, OCCUPIED, FREE, NavigationMap, NavigationConfig,
    build_navigation, classify, voxelize)


def map_from_obstacles(obstacles, radius=.4, voxel=1, origin=(0, 0, 0)):
    origin = np.array(origin, dtype=float)
    grid, labels, clearance, regions, diagnostics = classify(obstacles, origin, voxel, radius, 64)
    metadata = dict(origin=origin.tolist(), voxel_size=voxel, regions=regions, diagnostics=diagnostics)
    return NavigationMap(metadata, grid, labels, clearance)


inside = np.zeros((40, 42, 30), dtype=bool)
inside[5:33, 5:15, 5:25] = True
inside[23:33, 5:37, 5:25] = True
walls = ndimage.binary_dilation(inside, structure=np.ones((3, 3, 3))) & ~inside
room = map_from_obstacles(walls)
assert len(room.metadata['regions']) == 1
assert room.grid[0, 0, 0] == UNKNOWN
assert room.grid[4, 8, 10] == OCCUPIED
start, end = [9.5, 9.5, 12.5], [27.5, 32.5, 12.5]
assert room.is_valid(start) and room.is_valid(end)
assert not room.segment_valid(start, end), 'L-shaped corner was cut'
route = room.find_path(start, end)
assert route['found'], route
assert room.validate(route['points'])['valid']
assert not room.validate([start, end])['valid']
assert not room.is_valid([-1, 10, 12])
assert not room.is_valid([float('nan'), 10, 12])
assert not room.is_valid([100, 10, 12])
assert room.segment_valid(start, start)

# Neither a scan hole nor a missing roof is silently treated as a doorway.
gap = walls.copy(); gap[4, 7:13, 8:18] = False
assert not np.any(map_from_obstacles(gap).grid == FREE), 'A boundary leak was sealed'
crack = walls.copy(); crack[4, 8, 12] = False
assert np.any(map_from_obstacles(crack).grid == FREE), 'A sub-camera crack incorrectly connected the camera space to the exterior'
assert map_from_obstacles(crack).grid[4,8,12] != OCCUPIED, 'Clearance must not invent obstacle geometry'
roof = walls.copy(); roof[:, :, 25] = False
assert not np.any(map_from_obstacles(roof).grid == FREE)

# Keep a small enclosed furniture cavity unknown while retaining room space.
furniture = walls.copy()
furniture[13:18, 7:12, 10:15] = True
furniture[14:17, 8:11, 11:14] = False
furnished = map_from_obstacles(furniture)
assert furnished.grid[15, 9, 12] == UNKNOWN
assert furnished.grid[13, 9, 12] == OCCUPIED
assert furnished.is_valid([9.5, 9.5, 12.5])

# A narrow interior portal connects rooms only when camera clearance permits it.
box = np.zeros((40, 32, 30), dtype=bool); box[3:37, 3:29, 3:27] = True
divider = ndimage.binary_dilation(box, structure=np.ones((3, 3, 3))) & ~box
divider[20, 3:29, 3:27] = True
divider[20, 14:17, 13:16] = False
small, large = map_from_obstacles(divider, radius=.01), map_from_obstacles(divider, radius=2)
assert len(small.metadata['regions']) == 1
assert len(large.metadata['regions']) == 2
assert not large.find_path([10.5, 15.5, 14.5], [30.5, 15.5, 14.5])['found']

# Continuous validation includes touching diagonal cells and paths on grid planes.
grid = np.full((5, 5, 5), FREE, dtype=np.uint8)
grid[2, 1, 2] = OCCUPIED
toy = NavigationMap(dict(origin=[0,0,0],voxel_size=1), grid, np.ones_like(grid), np.ones_like(grid))
assert not toy.segment_valid([1.5, 1.5, 2.5], [2.5, 2.5, 2.5])
assert not toy.segment_valid([1.5, 2, 2.5], [3.5, 2, 2.5])

# Source translation/scale is retained, with no hard-coded gravity or meters.
origin, voxel = np.array([-17., 31., 7.]), .3
transformed = map_from_obstacles(walls, radius=.12, origin=origin, voxel=voxel)
world_start, world_end = origin+np.array(start)*voxel, origin+np.array(end)*voxel
assert transformed.is_valid(world_start)
assert transformed.find_path(world_start, world_end)['found']

# Rasterization must include a thin rotated Gaussian even if no cell center is
# inside it. Every sampled ellipsoid point must land in an occupied voxel.
rng = np.random.default_rng(12)
rot = Rotation.from_euler('xyz', [20, 35, 48], degrees=True).as_matrix()
mean, axes = np.array([.03, .02, -.01]), np.array([1.2, .06, .003])
origin = np.array([-2., -2., -2.]); shape = np.array([40,40,40]); voxel = .1
raster = voxelize(mean[None], axes[None], rot[None], origin, shape, voxel)
samples = rng.normal(size=(4000,3)); samples /= np.linalg.norm(samples,axis=1)[:,None]
samples = (samples*axes) @ rot.T+mean
cells = np.floor((samples-origin)/voxel).astype(int)
assert np.all(raster[tuple(cells.T)]), 'Thin/rotated support was omitted'
assert not raster[0,0,0]
huge = voxelize(mean[None], np.full((1,3),1e26), rot[None], origin, shape, voxel)
assert huge.all(), 'Extreme scales overflowed voxel index calculations'

# End-to-end Gaussian input, bounded grid, source identity and published assets.
with tempfile.TemporaryDirectory() as temporary:
    shell = np.array(np.nonzero(walls)).T.astype(float)+.5
    arrays = dict(means=shell, scales=np.full_like(shell, np.log(.55)),
                  quats=np.tile([1.,0,0,0], (len(shell),1)), opacities=np.full(len(shell),5.))
    cfg = NavigationConfig(resolution=64, max_cells=250_000)
    generated = build_navigation(arrays, temporary, 'a'*64, cfg)
    assert generated.metadata['status'] == 'candidate_space'
    assert generated.grid.size <= cfg.max_cells
    assert generated.metadata['space'] == 'source'
    assert not generated.metadata['metric_scale_verified']
    loaded = NavigationMap.load(temporary)
    assert np.array_equal(loaded.grid, generated.grid)
    exported = json.loads((Path(temporary)/'navigation-grid.json').read_text())
    runs = np.asarray(exported['runs']).reshape(-1,2)
    assert np.array_equal(np.repeat(runs[:,0],runs[:,1]), generated.grid.ravel())
    cached = build_navigation(arrays, temporary, 'a'*64, cfg)
    assert cached.metadata['fingerprint'] == generated.metadata['fingerprint']
    # A changed input fingerprint prevents stale navigation from being reused.
    changed = build_navigation(arrays, temporary, 'b'*64, cfg)
    assert changed.metadata['fingerprint'] != cached.metadata['fingerprint']
    arrays['quats'][0] = 0
    invalid = build_navigation(arrays, temporary, 'c'*64, cfg)
    assert invalid.metadata['status'] == 'uncertain' and not np.any(invalid.grid == FREE)

print('Navigation: closed/open rooms, L-shaped paths, clearance, cavities, source coordinates, rotated Gaussian extents, bounded export and cache checks passed.')
