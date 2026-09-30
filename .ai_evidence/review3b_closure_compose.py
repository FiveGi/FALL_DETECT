import importlib.util,tempfile,json,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('c','tools/check_config_coherence.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
with tempfile.TemporaryDirectory() as d:
 p=Path(d);c.ROOT=d
 (p/'docker-compose.yml').write_text('services:\n  celery_worker:\n    image: fixture\n    env_file: [base.env]\n    environment:\n      V3_IMGSZ: "960"\n      V3_TARGET_FPS: "20"\n')
 (p/'docker-compose.gpu.yml').write_text('services:\n  celery_worker:\n    env_file: [gpu.env]\n')
 (p/'base.env').write_text('V3_ROI_IMGSZ=300\nV3_ROI_FULL_EVERY=1\n')
 (p/'gpu.env').write_text('UNRELATED=1\n')
 got=c.compose_profiles()['gpu'];key=c.measured_key(got['imgsz'],15,got['fps'],4,['auto'],.65,got['roi'],got['roi_every'])
 print('Checker GPU:',got,'measured=',key in c.MEASURED)
 r=subprocess.run(['docker','compose','-f',str(p/'docker-compose.yml'),'-f',str(p/'docker-compose.gpu.yml'),'config','--format','json'],capture_output=True,text=True)
 if r.returncode:print('Compose failed:',r.stderr[:600]);raise SystemExit(1)
 env=json.loads(r.stdout)['services']['celery_worker']['environment']
 print('Compose GPU:',{k:v for k,v in env.items() if k.startswith('V3_')})
 assert got['roi']==300 and got['roi_every']==1 and key not in c.MEASURED and str(env['V3_ROI_IMGSZ'])=='300' and str(env['V3_ROI_FULL_EVERY'])=='1'
 print('PASS: checker and Compose retain 300/1; checker reports unmeasured. Synthetic, 0 clips.')
