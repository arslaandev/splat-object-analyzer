"""Exercise upload, progress, result isolation, and failure handling without models."""
from functools import partial
import hashlib
import http.server
import json
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import analysis_server


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    script = root / 'fake_pipeline.py'
    script.write_text('''import argparse, hashlib, json, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--ply');p.add_argument('--quality');p.add_argument('--job_dir')
a=p.parse_args();source=Path(a.ply)
out=Path(a.job_dir);out.mkdir()
(out/'transforms.json').write_text(json.dumps({'frames':[{'quality':{'score':1},'transform_matrix':[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]}]}))
print('[discover] 1/4 views; 2 labels discovered', flush=True)
time.sleep(.35)
if b'fail' in source.read_bytes():
 print('ValueError: unsupported test scene', flush=True);raise SystemExit(3)
(out/'interactions.json').write_text(json.dumps({'input':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'objects':[]}))
''')
    (root / 'run.sh').write_text('exec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(script)) + ' "$@"\n')
    analysis_server.ROOT = root
    jobs = analysis_server.AnalysisJobs(root / 'jobs')
    handler = partial(analysis_server.handler_for(jobs, 0), directory=str(root))
    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'

        def request(path, data=None, headers=None):
            req = Request(base + path, data=data, headers=headers or {})
            try:
                with urlopen(req, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except HTTPError as error:
                return error.code, json.loads(error.read())

        def wait(job):
            deadline = time.monotonic() + 10
            seen = set()
            preview_seen = False
            while time.monotonic() < deadline:
                code, state = request('/api/jobs/' + job)
                seen.add(state['status'])
                if state.get('previewCamera'):
                    assert state['previewCamera']['position'] == [1, -2, -3]
                    preview_seen = True
                if state['status'] in {'complete', 'failed'}:
                    return state, seen, preview_seen
                time.sleep(.03)
            raise AssertionError('Job never finished')

        headers = {'Content-Type': 'application/octet-stream', 'X-Splat-Viewer': '1'}
        assert request('/api/jobs?filename=scene.ply', b'ply\n', {})[0] == 403
        assert request('/api/jobs?filename=scene.exe', b'ply\n', headers)[0] == 400
        assert request('/api/jobs?filename=bad.ply', b'not a ply', headers)[0] == 400
        assert request('/api/jobs?filename=scene.ply', b'ply\n', {**headers, 'Origin':'http://outside.example'})[0] == 403
        code, result = request('/api/jobs?filename=../../scene.ply&quality=high', b'ply\nvalid', headers)
        assert code == 202
        job_id = result['id']
        assert request('/api/jobs?filename=second.ply', b'ply\n', headers)[0] == 409
        state, seen, preview_seen = wait(job_id)
        assert state['status'] == 'complete' and 'running' in seen
        assert preview_seen, 'A usable camera was not published during processing'
        assert state['name'] == 'scene.ply'
        assert jobs.jobs[job_id]['source'].is_relative_to((root / 'jobs').resolve())
        code, data = request(state['config']['scene']['annotationsUrl'])
        assert code == 200 and data['input']['sha256'] == hashlib.sha256(b'ply\nvalid').hexdigest()
        assert request('/viewer-config.json')[1]['scene']['sourceHash'] == data['input']['sha256']
        assert request('/viewer-config.json', headers={'Host': 'outside.example'})[0] == 403
        assert request('/api/jobs/' + '0'*32)[0] == 404
        code, failed = request('/api/jobs?filename=fail.ply', b'ply\nfail', headers)
        assert code == 202
        state, _, _ = wait(failed['id'])
        assert state['status'] == 'failed' and 'unsupported test scene' in state['error']
        assert jobs.active is None
        assert analysis_server.progress_from_line('[identity] Segmented 3/10 views')['completed'] == 3
        assert analysis_server.progress_from_line('[detect] Processed 7/90 views')['total'] == 90
        assert analysis_server.progress_from_line('arbitrary warning') is None
        server.shutdown()
print('Viewer upload, progress, result isolation, and failure checks passed.')
