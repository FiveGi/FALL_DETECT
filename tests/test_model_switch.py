"""F4 (9 Oct plan): the model is chosen from the environment alone and the running identity is provable.
Run: python tests/test_model_switch.py   (CPU, ~1 min)"""
import hashlib, importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
fails = []
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]
def load(env):
    for k in ('V3_CLASSIFIER', 'V3_POSE_MODEL', 'V3_THRESHOLD'):
        os.environ.pop(k, None)
    os.environ.update(env, V3_DEVICE='cpu')
    spec = importlib.util.spec_from_file_location('v3_%d' % len(sys.modules), 'app/detection/v3_fall_detection.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.V3PoseFallDetector(model_dir='models')
    return m.LOADED_IDENTITY
def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (' -- %s' % detail if detail else ''), flush=True)
    if not cond: fails.append(name)
prod = load({})
check('default = production', prod['classifier'] == 'fall_classifier_v3.onnx' and prod['pose'] == 'yolo26s-pose.pt'
      and prod['classifier_sha'] == sha('models/fall_classifier_v3.onnx') and abs(prod['threshold'] - 0.65) < 1e-9, prod)
a = load({'V3_CLASSIFIER': 'fall_classifier_t2full_s45.onnx', 'V3_POSE_MODEL': 'pose_nightaug_s44.pt', 'V3_THRESHOLD': '0.70'})
check('A via env', a['classifier_sha'] == sha('training/data/stage1/t2_truncfull_s45/fall_classifier_v3.onnx')
      and a['pose_sha'] == sha('training/data/pose_ir/nightaug_s44_p2/weights/best.pt') and abs(a['threshold'] - 0.70) < 1e-9, a)
back = load({})
check('rollback = remove the env lines', back == prod, back)
print('RESULT', 'FAIL' if fails else 'PASS', fails); sys.exit(1 if fails else 0)
