#!/usr/bin/env python3
"""Serve the viewer plus one explicit scene/result pair on localhost."""
import argparse
from functools import partial
import http.server
import io
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


def make_config(result, result_path, scene):
    config = {'splatUrl': '/scene' + scene.suffix.lower(), 'annotationsUrl': '/interactions.json'}
    transforms_path = result_path.parent / 'transforms.json'
    if transforms_path.exists():
        transforms = json.loads(transforms_path.read_text())
        root_indices = {i for i, obj in enumerate(result['objects']) if not obj.get('parent_id')}
        annotations = result.get('frame_annotations', {})
        best = max(range(len(transforms['frames'])), key=lambda i: sum(
            a['score'] for a in annotations.get(str(i), []) if a['object_idx'] in root_indices))
        pose = transforms['frames'][best]['transform_matrix']
        position = [pose[0][3], -pose[1][3], -pose[2][3]]
        direction = [pose[0][2], -pose[1][2], -pose[2][2]]
        if all(math.isfinite(n) for n in position + direction):
            config['camera'] = {'position': position, 'target': [p + d for p, d in zip(position, direction)]}
    return {'scene': config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', help='Path to interactions.json')
    parser.add_argument('--scene', help='Override the original input scene path, e.g. after copying results')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()
    result_path = Path(args.results).expanduser().resolve()
    result = json.loads(result_path.read_text())
    scene = Path(args.scene or result['input']['path']).expanduser().resolve()
    if not scene.is_file():
        parser.error('Scene file not found; use --scene /path/to/scene.spz')
    if scene.suffix.lower() not in {'.ply', '.spz'}:
        parser.error('Use a .ply or .spz Gaussian splat scene')
    config_bytes = json.dumps(make_config(result, result_path, scene)).encode()
    viewer = Path(__file__).resolve().parent / 'viewer'

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def send_head(self):
            route = urlsplit(self.path).path
            if route == '/viewer-config.json':
                content, content_type, length = io.BytesIO(config_bytes), 'application/json', len(config_bytes)
            elif route in {'/scene' + scene.suffix.lower(), '/interactions.json'}:
                path = scene if route.startswith('/scene') else result_path
                content = path.open('rb')
                content_type = 'application/json' if route == '/interactions.json' else 'application/octet-stream'
                length = path.stat().st_size
            else:
                return super().send_head()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.end_headers()
            return content

    with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(viewer))) as server:
        print(f'Viewer: http://127.0.0.1:{args.port}/', flush=True)
        print(f'Results: {result_path}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
