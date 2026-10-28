"""Local viewer and one-at-a-time Gaussian splat analysis jobs (stdlib only)."""
import argparse
from functools import partial
import http.server
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
from urllib.parse import parse_qs, urlsplit
import uuid


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 2 * 1024**3


def frame_camera(pose):
    position = [pose[0][3], -pose[1][3], -pose[2][3]]
    direction = [pose[0][2], -pose[1][2], -pose[2][2]]
    if all(math.isfinite(n) for n in position + direction):
        return {'position': position, 'target': [p + d for p, d in zip(position, direction)]}
    return None


def make_config(result, result_path, scene, prefix=''):
    config = {'splatUrl': prefix + '/scene' + scene.suffix.lower(),
              'annotationsUrl': prefix + '/interactions.json', 'name': scene.name,
              'sourceHash': result['input']['sha256']}
    transforms_path = result_path.parent / 'transforms.json'
    if transforms_path.exists():
        transforms = json.loads(transforms_path.read_text())
        frames = transforms.get('frames', [])
        roots = {i for i, obj in enumerate(result['objects']) if not obj.get('parent_id')}
        annotations = result.get('frame_annotations', {})
        if frames:
            best = max(range(len(frames)), key=lambda i: sum(
                a['score'] for a in annotations.get(str(i), []) if a['object_idx'] in roots))
            camera = frame_camera(frames[best]['transform_matrix'])
            if camera:
                config['camera'] = camera
    return {'scene': config}


def progress_from_line(line):
    """Translate real pipeline messages into stages; no invented overall percent."""
    if '[render]' in line or '[pipeline] Step 1' in line or '[spz]' in line:
        stage = 'Selecting and rendering views'
    elif '[discover]' in line:
        stage = 'Discovering object names'
    elif '[detect] Processed' in line or 'Loading OWLv2' in line or '[pipeline] Step 2' in line:
        stage = 'Detecting objects'
    elif '[pipeline] Step 3' in line or '[cluster]' in line:
        stage = 'Associating detections'
    elif '[identity]' in line:
        stage = 'Segmenting objects and fitting boxes'
    else:
        return None
    match = re.search(r'(\d+)\s*/\s*(\d+)', line)
    update = {'stage': stage, 'completed': None, 'total': None}
    if match:
        update.update(completed=int(match[1]), total=int(match[2]))
    return update


class AnalysisJobs:
    def __init__(self, directory, initial=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.initial = initial
        self.jobs = {}
        self.active = None
        self.latest = None
        self.lock = threading.Lock()

    def reserve(self, name, quality):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Another scene is being processed. Wait for it to finish.')
            job_id = uuid.uuid4().hex
            path = self.directory / job_id
            path.mkdir()
            job = {'id': job_id, 'name': name, 'quality': quality,
                   'status': 'uploading', 'stage': 'Uploading scene',
                   'completed': None, 'total': None,
                   'source': path / ('input' + Path(name).suffix.lower()),
                   'output': path / 'analysis'}
            self.jobs[job_id] = job
            self.active = job_id
            return job

    def fail(self, job_id, message):
        with self.lock:
            self.jobs[job_id].update(status='failed', error=message, stage='Analysis failed')
            self.active = None

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job['status'] == 'running' and not job.get('previewCamera'):
                try:
                    frames = json.loads((job['output'] / 'transforms.json').read_text())['frames']
                    best = max(frames, key=lambda f: f.get('quality', {}).get('score', 0) *
                               (1 - abs(f['transform_matrix'][1][2])))
                    job['previewCamera'] = frame_camera(best['transform_matrix'])
                except (OSError, ValueError, KeyError):
                    pass  # Render metadata may not have been written completely yet.
            public = {k: v for k, v in job.items() if k not in {'source', 'output'}}
            if job['status'] != 'uploading':
                public['splatUrl'] = f'/api/jobs/{job_id}/scene{job["source"].suffix}'
            return public

    def configuration(self):
        with self.lock:
            latest = self.latest
            active = self.active
        pair = latest or self.initial
        config = make_config(*pair) if pair else {'scene': None}
        if latest:
            config = make_config(*latest, prefix='/api/jobs/' + latest[1].parent.parent.name)
            config['scene']['name'] = self.jobs[latest[1].parent.parent.name]['name']
        with self.lock:
            scenes = [{'id': job['id'], 'label': job['name'], **job['config']['scene']}
                      for job in reversed(list(self.jobs.values())) if job['status'] == 'complete']
        if self.initial:
            pair = make_config(*self.initial)['scene']
            scenes.append({'id': 'initial', 'label': pair['name'], **pair})
        return {**config, 'scenes': scenes, 'analysisAvailable': True, 'defaultQuality': 'high', 'activeJob': active}

    def run(self, job_id):
        job = self.jobs[job_id]
        with self.lock:
            job.update(status='running', stage='Starting analysis')
        log_path = job['source'].parent / 'run.log'
        last_line = ''
        try:
            # run.sh configures the venv tools and model caches consistently with CLI runs.
            command = ['/bin/sh', str(ROOT / 'run.sh'), '--ply', str(job['source']),
                       '--quality', job['quality'], '--job_dir', str(job['output'])]
            with log_path.open('w') as log:
                with subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.strip():
                            last_line = line.strip()
                        update = progress_from_line(line)
                        if update:
                            with self.lock:
                                job.update(update)
                    code = process.wait()
            if code:
                raise RuntimeError(last_line[:400] or f'Analysis exited with code {code}')
            path = job['output'] / 'interactions.json'
            result = json.loads(path.read_text())
            config = make_config(result, path, job['source'], prefix='/api/jobs/' + job_id)
            config['scene']['name'] = job['name']
            with self.lock:
                job.update(status='complete', stage='Analysis complete', config=config,
                           recordCount=len(result['objects']),
                           rootCount=sum(not o.get('parent_id') for o in result['objects']),
                           completed=None, total=None)
                self.latest = (result, path, job['source'])
                self.active = None
        except Exception as error:
            self.fail(job_id, str(error))


def handler_for(jobs, port):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def json_response(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            # Custom header plus same-origin validation prevent cross-site uploads to localhost.
            host = self.headers.get('Host')
            origin = self.headers.get('Origin')
            if not self.valid_host() or (origin and origin != 'http://' + host) or self.headers.get('X-Splat-Viewer') != '1':
                self.json_response({'error': 'Only same-origin viewer requests are allowed'}, 403)
                return
            url = urlsplit(self.path)
            if url.path != '/api/jobs':
                self.json_response({'error': 'Unknown endpoint'}, 404)
                return
            query = parse_qs(url.query)
            name = Path(query.get('filename', ['scene'])[0].replace('\\', '/')).name[:200]
            quality = query.get('quality', ['high'])[0]
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                size = 0
            if Path(name).suffix.lower() not in {'.ply', '.spz'} or quality not in {'low', 'medium', 'high'}:
                self.json_response({'error': 'Choose a .ply or .spz file and a valid quality setting'}, 400)
                return
            if size <= 0 or size > MAX_UPLOAD:
                self.json_response({'error': 'Choose a nonempty file smaller than 2 GiB'}, 413)
                return
            try:
                job = jobs.reserve(name, quality)
            except RuntimeError as error:
                self.json_response({'error': str(error)}, 409)
                return
            try:
                remaining = size
                with job['source'].open('wb') as destination:
                    while remaining:
                        block = self.rfile.read(min(1024**2, remaining))
                        if not block:
                            raise ValueError('The upload was interrupted')
                        destination.write(block)
                        remaining -= len(block)
                with job['source'].open('rb') as source:
                    magic = source.read(4)
                if ((job['source'].suffix == '.ply' and not magic.startswith(b'ply')) or
                    (job['source'].suffix == '.spz' and not magic.startswith(b'\x1f\x8b'))):
                    raise ValueError('The file contents do not match a supported Gaussian splat format')
            except Exception as error:
                jobs.fail(job['id'], str(error))
                self.json_response({'error': str(error)}, 400)
                return
            threading.Thread(target=jobs.run, args=(job['id'],), daemon=True).start()
            self.json_response({'id': job['id']}, 202)

        def do_GET(self):
            if not self.valid_host():
                self.json_response({'error': 'Unexpected host'}, 403)
                return
            route = urlsplit(self.path).path
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})', route)
            if match:
                try:
                    self.json_response(jobs.snapshot(match[1]))
                except KeyError:
                    self.json_response({'error': 'Analysis job not found'}, 404)
                return
            super().do_GET()

        def do_HEAD(self):
            if not self.valid_host():
                self.send_error(403)
                return
            super().do_HEAD()

        def send_head(self):
            route = urlsplit(self.path).path
            if route == '/viewer-config.json':
                data = json.dumps(jobs.configuration()).encode()
                content, content_type, length = io.BytesIO(data), 'application/json', len(data)
            else:
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/(scene\.(?:ply|spz)|interactions\.json)', route)
                pair = jobs.initial
                if match:
                    job = jobs.jobs.get(match[1])
                    if not job:
                        self.send_error(404)
                        return None
                    path = job['source'] if match[2].startswith('scene.') else job['output'] / 'interactions.json'
                    if match[2].startswith('scene.') and match[2] != 'scene' + path.suffix:
                        self.send_error(404)
                        return None
                elif pair and route in {'/scene' + pair[2].suffix.lower(), '/interactions.json'}:
                    path = pair[2] if route.startswith('/scene') else pair[1]
                else:
                    return super().send_head()
                if not path.is_file():
                    self.send_error(404)
                    return None
                content = path.open('rb')
                content_type = 'application/json' if path.suffix == '.json' else 'application/octet-stream'
                length = path.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.end_headers()
            return content

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', help='Optional initial interactions.json')
    parser.add_argument('--scene', help='Override the initial result scene path')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--jobs-dir', default=str(ROOT / 'outputs' / 'viewer'))
    args = parser.parse_args()
    initial = None
    if args.results:
        path = Path(args.results).expanduser().resolve()
        result = json.loads(path.read_text())
        scene = Path(args.scene or result['input']['path']).expanduser().resolve()
        if not scene.is_file() or scene.suffix.lower() not in {'.ply', '.spz'}:
            parser.error('Initial scene not found; provide --scene /path/to/scene.ply or scene.spz')
        initial = (result, path, scene)
    jobs = AnalysisJobs(Path(args.jobs_dir).expanduser(), initial)
    handler = partial(handler_for(jobs, args.port), directory=str(ROOT / 'viewer'))
    with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Viewer: http://127.0.0.1:{args.port}/', flush=True)
        print(f'Analysis uploads and results: {jobs.directory}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
