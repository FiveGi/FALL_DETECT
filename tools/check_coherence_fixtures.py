"""Negative fixtures for compose_profiles()/measured_key(), in a temp dir (real files untouched)."""
import importlib.util, os, shutil, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('coh', os.path.join(REPO, 'tools', 'check_config_coherence.py'))
coh = importlib.util.module_from_spec(spec); spec.loader.exec_module(coh)
base = open(os.path.join(REPO, 'docker-compose.yml'), encoding='utf-8').read()
gpu = open(os.path.join(REPO, 'docker-compose.gpu.yml'), encoding='utf-8').read()
fails = []


def verdict(base_text, gpu_text, env_text=''):
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, 'docker-compose.yml'), 'w', encoding='utf-8').write(base_text)
        open(os.path.join(d, 'docker-compose.gpu.yml'), 'w', encoding='utf-8').write(gpu_text)
        open(os.path.join(d, '.env'), 'w', encoding='utf-8').write(env_text)
        coh.ROOT = d
        p = coh.compose_profiles()['cpu']
        key = coh.measured_key(p['imgsz'], 15, p['fps'], 4, ['auto'], 0.65, p['roi'], p['roi_every'])
        return p, key in coh.MEASURED
    finally:
        shutil.rmtree(d)


def check(name, got, want):
    print('%-4s %s' % ('PASS' if got == want else 'FAIL', name))
    if got != want:
        fails.append(name)


_, m = verdict(base, gpu)
check('real files: cpu crop 256/8 is measured', m, True)
_, m = verdict(base.replace('V3_ROI_FULL_EVERY=8', 'V3_ROI_FULL_EVERY=0'), gpu)
check('explicit cadence 0 (= every frame at runtime) is NOT measured', m, False)
_, m = verdict(base.replace('V3_ROI_FULL_EVERY=8', 'V3_ROI_FULL_EVERY=1'), gpu)
check('cadence 1 is NOT measured', m, False)
no_roi = base.replace('      - V3_ROI_IMGSZ=256\n', '').replace('      - V3_ROI_FULL_EVERY=8\n', '')
p, m = verdict(no_roi, gpu, 'V3_ROI_IMGSZ=300\nV3_ROI_FULL_EVERY=1\n')
check('crop 300/1 coming from .env is seen (roi=%s) and NOT measured' % p['roi'], (p['roi'], m), (300, False))
p, m = verdict(no_roi, gpu, '')
check('no crop anywhere -> full-frame row, measured', (p['roi'], m), (0, True))
# GPU overlay with its OWN env_file: Compose concatenates, so base .env still applies to GPU.
def gpu_verdict(base_text, gpu_text, env_text, extra_env_text):
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, 'docker-compose.yml'), 'w', encoding='utf-8').write(base_text)
        open(os.path.join(d, 'docker-compose.gpu.yml'), 'w', encoding='utf-8').write(gpu_text)
        open(os.path.join(d, '.env'), 'w', encoding='utf-8').write(env_text)
        open(os.path.join(d, 'gpu.env'), 'w', encoding='utf-8').write(extra_env_text)
        coh.ROOT = d
        return coh.compose_profiles()['gpu']
    finally:
        shutil.rmtree(d)


NL = chr(10)
gpu_no_roi = gpu.replace('      - V3_ROI_IMGSZ=0' + NL, '').replace(
    '  celery_worker:' + NL,
    '  celery_worker:' + NL + '    env_file:' + NL + '      - gpu.env' + NL, 1)
g = gpu_verdict(no_roi, gpu_no_roi, 'V3_ROI_IMGSZ=300' + NL + 'V3_ROI_FULL_EVERY=1' + NL,
                'X=1' + NL)
check('GPU overlay with its own env_file still sees the base .env crop (roi=%s)' % g['roi'],
      g['roi'], 300)
os.environ['V3_ROI_IMGSZ'] = '300'
_, m = verdict(base, gpu)
check('shell env V3_ROI_IMGSZ=300 cannot change the verdict', m, True)
print('FAILED %s' % fails if fails else 'all coherence fixtures passed')
sys.exit(1 if fails else 0)
