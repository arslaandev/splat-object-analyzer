"""Camera regression checks: reject obstruction/background and preserve coverage."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter
import torch

from config import PipelineConfig
from render_cameras import (_generate_camera_positions, _view_quality,
                            _select_preview_positions, _preview_camera_positions)

# A hollow room must yield clear interior candidates without a density fallback.
grid = np.linspace(-1., 1., 31)
volume = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
shell = volume[np.any(np.abs(volume) == 1, axis=1)]
cameras, _ = _generate_camera_positions(shell, 3, PipelineConfig())
clearance = np.min(np.linalg.norm(cameras[:, None] - shell[None], axis=2), axis=1)
assert len(cameras) == 3 and (clearance > .5).all()
try:
    _generate_camera_positions(volume, 3, PipelineConfig())
except ValueError as error:
    assert 'No clear camera candidates' in str(error)
else:
    raise AssertionError('Dense solid was filled with unchecked camera positions')

image = np.full((32, 32, 3), 150, dtype=np.uint8)
depth = np.full((32, 32), 2.)
assert _view_quality(image, depth, .3, 10)['accepted'], 'A clear blank wall is still a valid view'
assert not _view_quality(image, depth * .01, .3, 10)['accepted'], 'Camera inside a surface accepted'
assert not _view_quality(image * 0, depth * 0, .3, 10)['accepted'], 'Empty exterior view accepted'
depth[:16] = 0
assert not _view_quality(image, depth, .3, 10)['accepted'], 'Mostly empty view accepted'

# Detail should distinguish a sharp preview from a smeared surface with valid depth.
checker = np.indices((128, 128)).sum(axis=0) // 8 % 2 * 200 + 30
sharp = np.repeat(checker[:, :, None], 3, axis=2).astype(np.uint8)
blurred = np.asarray(Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(4)))
flat_depth = np.full((128, 128), 2.)
assert _view_quality(sharp, flat_depth, .3, 10)['score'] > 3 * _view_quality(blurred, flat_depth, .3, 10)['score']

# A different surface should win over an equally good duplicate view.
positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
selection = _select_preview_positions(positions, [{1, 2}, {1, 2}, {3, 4}], [1, 1, 1], 2, .1)
assert selection == [0, 2], 'Camera selection did not favor new visible surfaces'
assert _select_preview_positions(positions, [set()] * 3, [0, 0, 0], 3, .1) == [], 'Invalid candidates used as filler'

class EmptyRenderer:
    device = torch.device('cpu')

    def render_rgb(self, g, poses, intrinsics, width, height):
        return np.zeros((len(poses), height, width, 3), dtype=np.uint8)

    def render_depth(self, g, poses, intrinsics, width, height):
        return np.zeros((len(poses), height, width), dtype=np.float32)


# Failed preview validation must preserve diagnostics and stop before detection.
with tempfile.TemporaryDirectory() as temporary:
    diagnostics = Path(temporary) / 'camera_diagnostics.json'
    try:
        _preview_camera_positions(shell, EmptyRenderer(), {},
            PipelineConfig(n_positions=3, camera_candidates=4, preview_size=32), diagnostics)
    except ValueError as error:
        assert 'No camera candidates passed' in str(error)
        assert json.loads(diagnostics.read_text())['selected_positions'] == 0
    else:
        raise AssertionError('Empty previews did not stop analysis')

if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    transforms = json.loads((root / 'transforms.json').read_text())
    assert transforms['camera_placement'] == 'preview_validated_surface_coverage'
    validation = transforms['camera_validation']
    assert validation['selected_positions'] <= validation['requested_positions']
    assert validation['accepted_views'] == len(transforms['frames'])
    assert validation['rendered_views'] == len(transforms['frames']) + len(transforms['rejected_views'])
    for index in validation['selected_indices']:
        candidate = validation['candidates'][index]
        assert candidate['score'] > 0 and sum(v['accepted'] for v in candidate['views']) >= 3
    for frame in transforms['frames']:
        assert frame['quality']['accepted']
        assert (root / frame['file_path']).is_file()
    print(f'Validated {len(transforms["camera_positions"])} cameras and {len(transforms["frames"])} accepted views.')

print('Camera placement regression checks passed.')
