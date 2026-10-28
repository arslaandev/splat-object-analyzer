"""Background local-model requests sharing the analyzer's single GPU slot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid


class UnderstandingJobs:
    def __init__(self, owner, root):
        self.owner = owner
        self.root = Path(root)
        self.requests = {}
        self.process = None

    def scene(self, scene_id):
        if scene_id == 'initial' and self.owner.initial:
            return self.owner.initial
        job = self.owner.jobs.get(scene_id)
        if not job or job['status'] != 'complete':
            raise ValueError('This scene has no completed analysis. Analyze it first.')
        return (json.loads((job['output'] / 'interactions.json').read_text()),
                job['output'] / 'interactions.json', job['source'])

    def submit(self, scene_id, source_hash, prompt):
        with self.owner.lock:
            result, path, _ = self.scene(scene_id)
            if result['input']['sha256'] != source_hash:
                raise ValueError('The request belongs to a different scene. Reload this scene.')
            active = self.owner.understanding_active
            if active:
                request = self.requests[active]
                if request['sceneId'] == scene_id and (request['prompt'] == prompt or not prompt):
                    return request['id']
                raise RuntimeError('Scene understanding is busy. Wait for the current request to finish.')
            if self.owner.active:
                raise RuntimeError('Object analysis is busy. Wait for it to finish.')
            request_id = uuid.uuid4().hex
            output = path.parent / 'understanding' / ('request-'+request_id+'.json')
            self.requests[request_id] = dict(id=request_id, sceneId=scene_id, sourceHash=source_hash,
                prompt=prompt, status='running', stage='Preparing scene understanding', completed=None, total=None,
                output=output, result_path=path)
            self.owner.understanding_active = request_id
        threading.Thread(target=self.run, args=(request_id,), daemon=True).start()
        return request_id

    def snapshot(self, request_id):
        with self.owner.lock:
            request = self.requests[request_id]
            public = {key: value for key, value in request.items() if key not in {'output', 'result_path'}}
            if request['status'] == 'complete':
                public['result'] = json.loads(request['output'].read_text())
            return public

    def run(self, request_id):
        request = self.requests[request_id]
        try:
            request['output'].parent.mkdir(exist_ok=True)
            command = [sys.executable, str(self.root / 'scene_understanding.py'), '--result', str(request['result_path']),
                       '--prompt', request['prompt'], '--output', str(request['output'])]
            env = os.environ.copy()
            env.setdefault('HF_HOME', str(self.root / '.cache' / 'huggingface'))
            env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            last = ''
            with request['output'].with_suffix('.log').open('w') as log:
                with subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    with self.owner.lock:
                        self.process = process
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.strip():
                            last = line.strip()
                        if line.startswith('[understand] '):
                            update = json.loads(line[len('[understand] '):])
                            with self.owner.lock:
                                request.update({k: update[k] for k in ('stage','completed','total')})
                    code = process.wait()
            if code:
                raise RuntimeError(last[:500] or f'Scene understanding exited with code {code}')
            result = json.loads(request['output'].read_text())
            if result['source_hash'] != request['sourceHash']:
                raise ValueError('Scene-understanding output belongs to a different scene.')
            for shot in result['shots']:
                shot['imageUrl'] = '/api/scene-assets/' + request['sceneId'] + '/' + shot['image']
            # Publish a complete response only after its scene and assets are validated.
            from scene_understanding import atomic_json
            atomic_json(request['output'], result)
            with self.owner.lock:
                request.update(status='complete', stage='Suggested views ready', completed=None, total=None)
        except Exception as error:
            with self.owner.lock:
                request.update(status='failed', stage='Scene understanding failed', error=str(error))
        finally:
            with self.owner.lock:
                self.process = None
                if self.owner.understanding_active == request_id:
                    self.owner.understanding_active = None

    def close(self):
        with self.owner.lock:
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
