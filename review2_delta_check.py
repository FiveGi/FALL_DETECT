"""REVIEW-2 isolated checks; synthetic fixtures, zero video clips."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent

def extract(path,names,ns):
    nodes=[n for n in ast.parse((ROOT/path).read_text(encoding='utf-8')).body if getattr(n,'name',None) in names]
    for n in nodes:
        if isinstance(n,ast.FunctionDef):
            n.decorator_list=[]
            n.body=[s for s in n.body if not isinstance(s,ast.ImportFrom)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),path,'exec'),ns)

def load(path):
    spec=importlib.util.spec_from_file_location(Path(path).stem,ROOT/path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

c=load('tools/check_config_coherence.py'); dp=load('app/services/detector_profiles.py')
r={'_coherence_module':lambda:c,'_measured_table':lambda:c.MEASURED,'jsonify':lambda x:x,'dp':dp}
extract('app/routes/detector_info.py',{'detector_info','list_profiles'},r)
for name,p in dp.PROFILES.items():
    e=p['env']
    config=dict(input_size=int(e['V3_IMGSZ']),window_frames=15,target_fps=float(e['V3_TARGET_FPS']),partial_window_from=int(e['V3_PARTIAL_MIN']),threshold=float(e['V3_THRESHOLD']),preprocess=e['V3_PREPROCESS'].split(','),roi_input_size=int(e['V3_ROI_IMGSZ']),roi_full_every=int(e.get('V3_ROI_FULL_EVERY',0)),preprocess_dark_below=float(e.get('V3_PREPROCESS_DARK_BELOW',70)))
    r['read_detector_config']=lambda:config
    assert r['detector_info']()[0]['data']['measured_known']
    assert r['list_profiles']()[0]['data']['running']==name
    print('PASS isolated route bodies:',name)
    if name=='cpu_far_people':
        config['roi_full_every']=1
        assert not r['detector_info']()[0]['data']['measured_known']
        assert r['list_profiles']()[0]['data']['running'] is None
        print('PASS cadence 1 rejected by both route bodies')
ns={'STILL_DOWN_FRACTION':.8,'STILL_DOWN_SEEN_FRACTION':.3,'MAX_MISSED_FRAMES':15,'is_upright':lambda k:k}
extract('app/services/notification_service.py',{'still_down_confirmed'},ns)
extract('app/detection/v3_fall_detection.py',{'PersonTracker'},ns)
t=ast.parse((ROOT/'app/detection/v3_fall_detection.py').read_text(encoding='utf-8'))
step=next(n for n in t.body if getattr(n,'name','')=='_step_person')
branch=next(n for n in step.body if isinstance(n,ast.If))
branch.orelse=branch.orelse[:1]
prefix=ast.parse('def counter_step(kpts,person_found,state):\n pass').body[0]
prefix.body=[branch]
exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[])),'counter-prefix','exec'),ns)
def lifecycle(prior):
    s=SimpleNamespace(frames_since_upright=0,frames_seen_down=0,raw_buffer=[])
    for _ in range(prior): ns['counter_step'](False,True,s)
    tracker=ns['PersonTracker'](); tracker.tracks[0]={'centroid':None,'missed':0}
    for _ in range(8):
        assert tracker.update([])==[(0,None,False)]
        ns['counter_step'](None,False,s)
    return s,ns['still_down_confirmed'](s.frames_since_upright,s.frames_seen_down,8)
assert lifecycle(0)[1] is False
state,result=lifecycle(3)
assert result is True
print('REPRO P1: 3 pre-alert down sightings + 8 absent frames at 0.8 fps:',state.frames_since_upright,state.frames_seen_down,'elapsed=8 confirmed=',result,'post-alert sightings=0')
v3=SimpleNamespace(V3MultiPersonFallState=lambda:SimpleNamespace(person_states={0:state}),detect_v3_fall_multi=lambda *a,**k:[(0,True,.9)],alert_result=lambda r:r[0])
ns2={'v3':v3,'STILL_DOWN_SECONDS':10,'v3_still_down':ns['still_down_confirmed']}
extract('training/measure/tier_accuracy.py',{'tier_for_clip'},ns2)
answer=ns2['tier_for_clip'](SimpleNamespace(),[[] for _ in range(9)],.8)
assert answer[1]=='confirmed'
print('REPRO evaluator with injected alert and terminal state:',answer)
print('Dataset: synthetic states/configs; clips=0; no live HTTP/model/accuracy replay.')
