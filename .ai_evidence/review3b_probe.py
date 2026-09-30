import ast, importlib.util, logging, types, tempfile, subprocess, json
from pathlib import Path
from unittest.mock import patch
from flask import Flask, request, jsonify
from datetime import datetime, timezone

def fn(path,name,ns):
 tree=ast.parse(Path(path).read_text(encoding='utf-8'))
 n=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 n.decorator_list=[]
 exec(compile(ast.Module(body=[n],type_ignores=[]),path,'exec'),ns)
 return ns[name]
spec=importlib.util.spec_from_file_location('media','app/services/media_token.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
clock=[0];closed=[]
def late():
 try:
  clock[0]=601
  yield b'private'
 finally:closed.append(True)
with patch.object(m.time,'monotonic',lambda:clock[0]):assert list(m.bounded(late(),600))==[] and closed
print('PASS late frame suppressed and generator closed')
clock[0]=0
fake_time=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
stream=types.SimpleNamespace(is_active=lambda:True,wait_for_new_frame=lambda since:clock.__setitem__(0,clock[0]+1),get_mjpeg_frame=lambda **kw:None)
gen=fn('app/services/stream_service.py','generate_mjpeg_stream',{'time':fake_time,'stream_manager':types.SimpleNamespace(get_stream=lambda *a:stream)})
assert list(gen(1,'fake','fake',stop_at=600))==[] and 600<=clock[0]<602
print('PASS no-frame producer exits at synthetic t=',clock[0])
for key in ['t=','%74=','t%3D','%2574=','token=']:
 r=logging.LogRecord('werkzeug',20,'',0,'GET /?'+key+'PRIVATE',(),None);m.RedactMediaTokens().filter(r);assert 'PRIVATE' not in r.getMessage()
print('PASS 5 raw/encoded log token cases')
class Broken:
 def filter_by(self,**kw):raise RuntimeError('fixture read fault')
f=fn('app/models/token_blocklist.py','is_jti_blacklisted',{})
assert f(types.SimpleNamespace(query=Broken()),'j') is True
assert f(types.SimpleNamespace(query=types.SimpleNamespace(filter_by=lambda **kw:types.SimpleNamespace(first=lambda:None))),'j') is False
print('PASS blocklist read fault rejects; unknown JTI not revoked')
app=Flask(__name__)
for name in ['logout','logout_all']:
 ns={'get_jwt':lambda:{'jti':'j','type':'access','exp':2000000000},'get_jwt_identity':lambda:'7','datetime':datetime,'tz':timezone.utc,'TokenBlocklist':types.SimpleNamespace(add_token_to_blacklist=lambda *a:False),'jsonify':jsonify}
 f=fn('app/routes/auth.py',name,ns)
 with app.app_context():assert f()[1]==500
print('PASS both actual logout bodies return 500 on failed revocation write (decorators omitted)')
f=fn('app/__init__.py','_no_cors_on_live_view',{'request':request})
app.after_request(f)
@app.after_request
def simulate_cors(response):response.headers['Access-Control-Allow-Origin']='https://unrelated.example';return response
app.add_url_rule('/stream',endpoint='stream.stream_camera',view_func=lambda:'frame')
app.add_url_rule('/other',endpoint='other',view_func=lambda:'ok')
client=app.test_client()
assert 'Access-Control-Allow-Origin' not in client.get('/stream',headers={'Origin':'https://unrelated.example'}).headers
assert 'Access-Control-Allow-Origin' in client.get('/other').headers
print('PASS actual CORS stripping hook/order with simulated CORS injector; flask_cors unavailable')
print('Synthetic fixtures; 0 clips; no live endpoints/cameras')
