"""Exercise the public API and real queue/process lifecycle without model downloads."""
import json
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
from api_server import create_app, Settings
from api.worker import Worker

KEY = 'test-api-key-' + 'x'*32
AUTH = {'Authorization':'Bearer '+KEY}
HEADERS = {**AUTH, 'Content-Type':'application/octet-stream'}


def wait(client, job_id, desired=None):
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        response = client.get('/v1/jobs/'+job_id, headers=HEADERS)
        assert response.status_code == 200, response.text
        job = response.json()
        if desired and desired(job):
            return job
        if not desired and job['status'] in ('succeeded','failed'):
            return job
        time.sleep(.02)
    raise AssertionError('Job did not reach the expected state')


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    program = root/'program'; program.mkdir()
    (program/'run_local.py').write_text('''import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ply');p.add_argument('--quality');p.add_argument('--job_dir');a=p.parse_args()
out=Path(a.job_dir);out.mkdir(exist_ok=True);source=Path(a.ply);content=source.read_bytes()
with (out/'pipeline-runs.txt').open('a') as log:log.write('run\\n')
print('[detect] Processed 1/2 views',flush=True)
time.sleep(.15)
if content.endswith(b'fail'): raise RuntimeError('fixture processing failure')
if content.endswith(b'timeout'): time.sleep(10)
digest=hashlib.sha256(content).hexdigest()
if content.endswith(b'bad-hash'):digest='wrong'
obj=dict(id='object_0001',label='desk',confidence=.8,supporting_view_count=2,geometry_supporting_view_count=2,
 position=dict(x=1,y=2,z=3),scale=dict(x=2,y=1,z=1),parent_id=None,children=[],
 oriented_box=dict(type='upright_visible_surface_estimate',position=dict(x=1,y=2,z=3),scale=dict(x=2,y=1,z=1),rotation=dict(x=0,y=.2,z=0,w=.9797958971)),
 bounding_box=dict(type='visible_surface_estimate',min=dict(x=0,y=1.5,z=2.5),max=dict(x=2,y=2.5,z=3.5)),
 frames=[dict(frame_idx=0,score=.8,box=[0,0,10,10])])
(out/'interactions.json').write_text(json.dumps(dict(input=dict(path=str(source),sha256=digest),objects=[obj],limitations=['Test estimate'])))
(out/'transforms.json').write_text(json.dumps(dict(frames=[dict(file_path='frames/frame_0000.png')])) )
(out/'frames').mkdir(exist_ok=True);(out/'frames/frame_0000.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')
''')
    (program/'scene_understanding.py').write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result');p.add_argument('--prompt');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.result).read_text());out=Path(a.output)
print('[understand] '+json.dumps(dict(stage='Describing objects from multiple views',completed=1,total=1)),flush=True)
if next(Path(a.result).parent.parent.glob('source.*')).read_bytes().endswith(b'restart') and not (out.parent/'pause-marker').exists():
 (out.parent/'pause-marker').write_text('once');time.sleep(10)
if a.prompt=='fail-query':raise RuntimeError('fixture query failure')
objects=[dict(**o,category=o['label'],display_name='wooden desk',status='supported',description_source='vision_language_model',visual_priority='medium',attributes=[]) for o in r['objects']]
selection=dict(interpretation=a.prompt or 'Tour',object_ids=[] if 'piano' in a.prompt else ['object_0001'],group_ids=[],message='No matching subjects' if 'piano' in a.prompt else '')
shots=[dict(id='view_1',title='Subject overview',frame_idx=0,object_ids=['object_0001'],subjects=['wooden desk'],score=.9,camera=dict(position=[1,-2,-3],target=[1,-2,-2],fov=90),basis='existing_render_with_depth_visibility_check')] if selection['object_ids'] else []
out.write_text(json.dumps(dict(source_hash=r['input']['sha256'],selection=selection,shots=shots,unshown_object_ids=[],understanding=dict(objects=objects,groups=[],relations=[],hero_object_ids=['object_0001'],scene=dict(type='study',description='A workspace.',confidence='medium'),limitations=['Inferred descriptions']))))
''')
    settings = Settings(root/'data', KEY, max_upload_bytes=512, cors_origins=('https://app.example.com',))
    app = create_app(settings, program)
    with TestClient(app) as client:
        assert client.get('/health/ready').status_code == 200
        assert client.get('/v1/jobs/'+'0'*32).status_code == 401
        spec = client.get('/openapi.json').json()
        assert spec['paths']['/v1/analyses']['post']['requestBody']['content']['application/octet-stream']
        assert 'AnalysisResult' in spec['components']['schemas'] and 'QueryResult' in spec['components']['schemas']
        assert spec['paths']['/v1/analyses']['post']['security']
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n',headers={}).status_code==401
        assert client.post('/v1/analyses?filename=../scene.ply',content=b'ply\n',headers=HEADERS).status_code==400
        assert client.post('/v1/analyses?filename=scene.ply',content=b'wrong',headers=HEADERS).status_code==400
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n',headers={**HEADERS,'Content-Type':'application/json'}).status_code==415
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n'+b'x'*513,headers=HEADERS).status_code==413
        assert client.post('/v1/analyses?filename=scene.ply',content=iter([b'ply\n',b'x'*513]),headers=HEADERS).status_code==413
        assert not list((settings.data_dir/'jobs').iterdir()), 'Rejected upload left data behind'
        response = client.post('/v1/analyses?filename=room.ply',content=b'ply\nvalid',headers=HEADERS)
        assert response.status_code==202, response.text
        job = response.json();analysis_id=job['id']
        assert response.headers['Location']==job['status_url']
        assert client.get(job['result_url'],headers=HEADERS).status_code==409
        assert wait(client,analysis_id)['status']=='succeeded'
        result=client.get(job['result_url'],headers=HEADERS)
        assert result.status_code==200, result.text
        data=result.json()
        assert data['objects'][0]['position']==dict(x=1,y=2,z=3), 'Source geometry was transformed'
        assert data['objects'][0]['oriented_box']['rotation']==dict(x=0,y=.2,z=0,w=.9797958971), 'Box rotation was changed'
        assert data['suggested_views'][0]['camera']['space']=='viewer'
        assert data['suggested_views'][0]['camera']['position']==[1,-2,-3]
        assert data['coordinates']['metric_scale_verified'] is False
        assert str(root) not in result.text, 'Internal filesystem paths leaked into public results'
        image=client.get(data['suggested_views'][0]['image_url'],headers=HEADERS)
        assert image.status_code==200 and image.headers['Content-Type']=='image/png'
        assert client.get(data['source']['download_url'],headers=HEADERS).content==b'ply\nvalid'
        assert client.get('/v1/analyses/'+analysis_id+'/frames/99',headers=HEADERS).status_code==404
        metadata_path=settings.data_dir/'jobs'/analysis_id/'analysis/transforms.json'
        saved_metadata=metadata_path.read_text()
        metadata_path.write_text(json.dumps(dict(frames=[dict(file_path='../private.png')])) )
        assert client.get('/v1/analyses/'+analysis_id+'/frames/0',headers=HEADERS).status_code==404
        metadata_path.write_text(saved_metadata)
        assert client.get('/v1/jobs/not-a-job',headers=HEADERS).status_code==422
        assert client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':' '},headers=AUTH).status_code==400
        assert client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'x'*1001},headers=AUTH).status_code==422
        query=client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'show desk'},headers=AUTH)
        assert query.status_code==202
        assert wait(client,query.json()['id'])['status']=='succeeded'
        assert client.get(query.json()['result_url'],headers=HEADERS).json()['selection']['object_ids']==['object_0001']
        absent=client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'grand piano'},headers=AUTH).json()
        assert wait(client,absent['id'])['status']=='succeeded'
        absent_result=client.get(absent['result_url'],headers=HEADERS).json()
        assert not absent_result['selection']['object_ids'] and not absent_result['suggested_views']
        failure=client.post('/v1/analyses?filename=bad.ply',content=b'ply\nfail',headers=HEADERS).json()
        assert wait(client,failure['id'])['status']=='failed'
        assert str(root) not in json.dumps(wait(client,failure['id'])['error'])
        mismatch=client.post('/v1/analyses?filename=mismatch.ply&understanding=false',content=b'ply\nbad-hash',headers=HEADERS).json()
        assert wait(client,mismatch['id'])['status']=='failed', 'Mismatched source fingerprint was published'
        spz=client.post('/v1/analyses?filename=room.spz',content=b'\x1f\x8bfixture',headers=HEADERS).json()
        assert wait(client,spz['id'])['status']=='succeeded'
        assert client.get(spz['result_url'],headers=HEADERS).json()['source']['format']=='spz'
        fast=client.post('/v1/analyses?filename=fast.ply&understanding=false',content=b'ply\nvalid',headers=HEADERS).json()
        assert wait(client,fast['id'])['status']=='succeeded'
        assert client.get(fast['result_url'],headers=HEADERS).json()['understanding_included'] is False
        assert client.post('/v1/analyses/'+fast['id']+'/queries',json={'prompt':'desk'},headers=AUTH).status_code==409
        cors=client.options('/v1/analyses',headers={'Origin':'https://app.example.com','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'Authorization,Content-Type'})
        assert cors.headers['Access-Control-Allow-Origin']=='https://app.example.com'
        second=Worker(app.state.store,program)
        try:second.start()
        except RuntimeError:pass
        else:raise AssertionError('A second GPU worker acquired the same queue')
        restart=client.post('/v1/analyses?filename=restart.ply',content=b'ply\nrestart',headers=HEADERS).json()
        wait(client,restart['id'],lambda j:j['progress']['stage']=='Describing objects from multiple views')
    with TestClient(create_app(settings,program)) as client:
        assert wait(client,restart['id'])['status']=='succeeded', 'Interrupted job was not resumed'
        assert (settings.data_dir/'jobs'/restart['id']/'analysis/pipeline-runs.txt').read_text().splitlines()==['run'], 'Completed geometry checkpoint was rerun'
        assert client.get(job['result_url'],headers=HEADERS).status_code==200, 'Completed result did not survive restart'

    limited=Settings(root/'limited', KEY, max_pending_jobs=1, stage_timeout_seconds=1)
    with TestClient(create_app(limited,program)) as client:
        slow=client.post('/v1/analyses?filename=slow.ply',content=b'ply\ntimeout',headers=HEADERS).json()
        rejected=client.post('/v1/analyses?filename=extra.ply',content=b'ply\nvalid',headers=HEADERS)
        assert rejected.status_code==429 and len(list((limited.data_dir/'jobs').iterdir()))==1
        assert wait(client,slow['id'])['status']=='failed', 'Stage timeout did not release GPU worker'
        next_job=client.post('/v1/analyses?filename=next.ply',content=b'ply\nvalid',headers=HEADERS).json()
        assert wait(client,next_job['id'])['status']=='succeeded'

    try:
        with TestClient(create_app(Settings(root/'bad-key','short'),program)):pass
    except RuntimeError:pass
    else:raise AssertionError('API started with an unsafe default key')
print('API authentication, upload limits, queue, results, queries, coordinates, assets, restart recovery, failures, and timeout checks passed.')
