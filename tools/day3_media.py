"""Day-3 report illustrations; CPU only, one thread, no training or HTML.

Run from repo root: python tools/day3_media.py chart|grid|clips|verify
Outputs are staged inside this writable workspace's report/ directory.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CPU_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import json
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'report'
EVID = ROOT / '.ai_evidence/day3_media'
for directory in (OUT / 'img', OUT / 'videos', EVID):
    directory.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(1)
FONT = 'C:/Windows/Fonts/tahoma.ttf'
NIGHT = ROOT / 'training/data/pose_ir/nightaug_s44_p2/weights/best.pt'
POSNEG = ROOT / 'training/data/pose_ir/posneg_s44_p2/weights/best.pt'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=FONT)
    path = ROOT / 'training/data/stage1/results_pinned.jsonl'
    rows = {r['name']: (i, r) for i, line in enumerate(path.read_text().splitlines(), 1)
            for r in [json.loads(line)]}
    tracks = {}
    for name in ('track_metric_t2.txt', 'track_metric_step1b.txt'):
        p = ROOT / 'training/data/multi_diag_v2' / name
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.startswith('{'):
                r = json.loads(line)
                tracks[(Path(r['model_dir']).name, r['threshold'])] = (str(p.relative_to(ROOT)), i, r)
    buckets = read(ROOT / 'training/data/day_first/owner_size_buckets_v4.json')
    near = [k for k, v in buckets.items() if v['bucket'] == 'near']
    assert len(near) == 25
    evidence = {'command': 'python tools/day3_media.py chart', 'threshold': 0.65,
                'pose': 'nightaug_s44', 'phases': 8, 'counts': [75,25,25,14],
                'near_keys': near, 'pairs': []}
    vals = []
    for seed in (45,46,47):
        pair = []
        refs = []
        for prefix in ('T1_ctrl', 'T2_truncfull'):
            name = f'{prefix}_s{seed}_065'
            line, r = rows[name]
            assert r['threshold'] == .65 and r['n_owner'] == 8
            near_counts = []
            for phase in range(8):
                p = ROOT / f'training/data/stage1/{name}_pinned_owner{phase}.json'
                segs = read(p)['segments']
                near_counts.append(sum(bool(segs[k]['alerts_at']) for k in near))
            tp, tl, tr = tracks[(Path(r['model_dir']).name, .65)]
            assert tr['n_scored'] == 14 and len(tr['correct']) == 8
            pair.append([float(np.mean(r['owner_caught'])), float(np.mean(r['owner_multi'])),
                         float(np.mean(near_counts)), float(np.mean(tr['correct']))])
            refs.append({'row':f'training/data/stage1/results_pinned.jsonl:{line}',
                         'track':f'{tp}:{tl}', 'owner_files':f'training/data/stage1/{name}_pinned_owner[0-7].json',
                         'near_counts':near_counts})
        delta = (np.array(pair[1]) - pair[0]).tolist()
        vals.append(delta)
        evidence['pairs'].append({'seed':seed, 'control':pair[0], 'full':pair[1], 'delta':delta, 'sources':refs})
    fig, ax = plt.subplots(figsize=(12,6.6), dpi=150)
    fig.patch.set_facecolor('#f6f8fc'); ax.set_facecolor('#f6f8fc')
    x = np.arange(4)
    for j, (seed, color) in enumerate(zip((45,46,47), ('#176baf','#20a58a','#de9326'))):
        bars = ax.bar(x + (j-1)*.25, vals[j], .23, color=color, label=f'seed {seed}')
        ax.bar_label(bars, labels=[f'+{v:.2f}' for v in vals[j]], padding=5, fontsize=10)
    ax.set_xticks(x, ['จับล้มเจ้าของ\n75 คลิป', 'หลายคน\n25 คลิป', 'ระยะใกล้\n25 คลิป', 'D3 เตือนถูกคน\n14 คลิป'], fontproperties=font, fontsize=12)
    ax.set_ylabel('จำนวนคลิปที่เพิ่มขึ้น (เฉลี่ย 8 เฟส)', fontproperties=font, fontsize=12)
    ax.set_title('T2-full เทียบคู่กับ T1 control: ดีขึ้นเท่าไรในแต่ละ seed', fontproperties=font, fontsize=18, pad=24)
    ax.set_ylim(0,9); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False, ncol=3)
    fig.text(.08,.055,'ทั้งสองฝั่งใช้ nightaug_s44 และเกณฑ์ 0.65 | คลิปเปรียบเทียบใช้ตัวเลือก s45 ที่ 0.70',fontproperties=font,fontsize=11)
    fig.text(.08,.02,'ผลจาก cache เดิม • ใกล้ = กลุ่มจาก stock-pose • ผลบวกไม่ใช่การผ่านเกณฑ์ทั้งหมด / ไม่ใช่ผล Le2i',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(OUT / 'img/day3_t2_paired_deltas.png'); plt.close(fig)
    save(EVID / 'chart.json',evidence)
    print(json.dumps(evidence['pairs'],ensure_ascii=False), flush=True)

def grid():
    import torch
    import ultralytics.utils.torch_utils as tu
    tu.NUM_THREADS = 1
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    obj = module('day3_obj', 'training/measure/object_false_person.py')
    paths = obj.test_set()
    models = [obj.YOLO(str(p)) for p in (NIGHT, POSNEG)]
    selected = []
    examined = 0
    for path in paths:
        frame = cv2.imread(path)
        a = models[0].predict(frame, verbose=False, conf=.30, classes=[0], imgsz=320, device='cpu')[0]
        torch.set_num_threads(1)
        examined += 1
        if a.boxes is None or not len(a.boxes):
            continue
        b = models[1].predict(frame, verbose=False, conf=.30, classes=[0], imgsz=320, device='cpu')[0]
        torch.set_num_threads(1)
        if b.boxes is not None and len(b.boxes):
            continue
        selected.append({'path':path, 'nightaug_boxes':a.boxes.xyxy.cpu().tolist(),
                         'nightaug_conf':a.boxes.conf.cpu().tolist(), 'posneg_count':0})
        print('selected',len(selected),Path(path).name,'examined',examined,flush=True)
        if len(selected) == 6:
            break
    assert len(selected) == 6, 'Insufficient qualifying examples'
    canvas = Image.new('RGB',(1500,1000),'#f6f8fc')
    draw = ImageDraw.Draw(canvas)
    f = lambda n: ImageFont.truetype(FONT,n)
    draw.text((30,16),'ภาพ COCO val ที่ไม่มีป้ายกำกับคน: nightaug เห็นคน แต่ posneg ไม่เห็น',font=f(30),fill='#14243b')
    draw.text((30,65),'s44 ทั้งคู่ • CPU 320 px • confidence ≥ 0.30 • กรอบแดง = nightaug • posneg = 0 ทุกภาพ',font=f(22),fill='#42546b')
    for idx, r in enumerate(selected):
        im = Image.open(r['path']).convert('RGB')
        iw,ih = im.size; scale=min(470/iw,340/ih)
        im=im.resize((round(iw*scale),round(ih*scale)))
        x=20+(idx%3)*495+(470-im.width)//2; y=125+(idx//3)*405
        canvas.paste(im,(x,y))
        for box, conf in zip(r['nightaug_boxes'],r['nightaug_conf']):
            xy=[x+box[0]*scale,y+box[1]*scale,x+box[2]*scale,y+box[3]*scale]
            draw.rectangle(xy,outline='#ff3535',width=4)
            draw.text((xy[0]+3,xy[1]+2),f'{conf:.2f}',font=f(18),fill='#ff3535',stroke_width=1,stroke_fill='white')
        draw.text((25+(idx%3)*495,y+348),f"COCO {Path(r['path']).stem} | {len(r['nightaug_boxes'])} → 0",font=f(20),fill='#14243b')
    draw.text((30,949),f'6 ตัวอย่างแรกที่เข้าเงื่อนไขจาก {examined} ภาพที่ตรวจ (ชุดมี {len(paths):,} ภาพ) • ภาพประกอบ ไม่ใช่การวัดอัตรารวม',font=f(21),fill='#42546b')
    canvas.save(OUT / 'img/day3_coco_false_person_grid.jpg',quality=94)
    save(EVID / 'grid.json',{'command':'python tools/day3_media.py grid', 'logic':'training/measure/object_false_person.py:26-36,48',
                           'dataset_count':len(paths),'examined':examined,'selection':'first 6 qualifying in sorted COCO image ID order',
                           'models':{str(p.relative_to(ROOT)):sha(p) for p in (NIGHT,POSNEG)},'examples':selected,
                           'torch_threads':torch.get_num_threads(), 'device':'cpu'})

def clips():
    import torch
    import ultralytics.utils.torch_utils as tu
    tu.NUM_THREADS = 1
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    # Explicitly reset only this renderer's environment, never the application's files.
    for key in list(os.environ):
        if key.startswith('V3_'):
            del os.environ[key]
    env = {'V3_DEVICE':'cpu','V3_IMGSZ':'320','V3_ROI_IMGSZ':'256',
           'V3_ROI_FULL_EVERY':'8','V3_POSE_CONF':'0.3','V3_PREPROCESS':'auto',
           'V3_PREPROCESS_DARK_BELOW':'70','V3_PARTIAL_MIN':'4','V3_TARGET_FPS':'8'}
    os.environ.update(env)
    sides = []
    for name, threshold, pose, classifier in (
            ('production',.65,ROOT/'models/yolo26s-pose.pt',ROOT/'models'),
            ('candidate',.70,NIGHT,ROOT/'training/data/stage1/t2_truncfull_s45')):
        os.environ.update(V3_THRESHOLD=str(threshold),V3_POSE_MODEL=str(pose))
        v3=module('day3_'+name,'app/detection/v3_fall_detection.py')
        det=v3.V3PoseFallDetector(model_dir=str(classifier),device='cpu')
        side={'name':name,'v3':v3,'det':det,'people':[], 'threshold':threshold,
              'pose':str(pose.relative_to(ROOT)), 'pose_sha256':sha(pose),
              'classifier':str(classifier.relative_to(ROOT)),
              'classifier_sha256':sha(classifier/'fall_classifier_v3.onnx')}
        real=det.extract_all_keypoints
        def spy(frame, side=side, real=real):
            side['people']=real(frame)
            return side['people']
        det.extract_all_keypoints=spy
        sides.append(side)
    frames_module=module('day3_frames','training/measure/cache_owner_segments.py')
    edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    incidents=read(ROOT/'test_result/incidents/incidents.json')
    evidence={'command':'python tools/day3_media.py clips','env':env,'device':'cpu','phases':[0],
              'selection':'selected phase-0 gains for illustration; not representative sampling',
              'pipeline_sha256':sha(ROOT/'app/detection/v3_fall_detection.py'),
              'models':[{k:v for k,v in s.items() if k not in ('v3','det','people')} for s in sides], 'clips':[]}
    for kind,key in (('near','7.mp4#2'),('multi','1.mp4#15')):
        clip,seg=key.split('#')
        row=next(r for r in incidents['clips'][clip] if r['segment']==int(seg))
        start,end=row['start_s'],row['end_s']
        for side in sides:
            side['det'].reset_roi_state(0)
            side['state']=side['v3'].V3MultiPersonFallState()
        path=OUT/f'videos/day3_compare_{kind}.mp4'
        writer=None; timeline=[]
        for i,source_fps,frame in frames_module.segment_frames(str(ROOT/'Test'/clip),start,end):
            t=round(i/source_fps-start,2)
            panels=[]; entry={'t':t,'source_frame':i-1,'sides':{}}
            for side in sides:
                v3,det=side['v3'],side['det']
                results=v3.detect_v3_fall_multi(frame,side['state'],det,config=None)
                h,w=frame.shape[:2]; pw=480; ph=round(h*pw/w)//2*2
                panel=cv2.resize(frame,(pw,ph))
                # Draw actual track centroids from detector output. Pose skeletons stay yellow;
                # red circles mark alert tracks directly, avoiding guessed pose-to-track matches.
                for kp,hip in side['people']:
                    pts=[(int(x*pw),int(y*ph)) if c>.3 else None for x,y,c in kp]
                    for a,b in edges:
                        if pts[a] and pts[b]: cv2.line(panel,pts[a],pts[b],(0,220,250),2,cv2.LINE_AA)
                tracks=[]
                for tid,alert,score,label,centroid in results:
                    x,y=int(centroid[0]*pw),int(centroid[1]*ph)
                    color=(30,40,240) if alert else (255,210,40)
                    cv2.circle(panel,(x,y),12,color,3,cv2.LINE_AA)
                    cv2.putText(panel,f'ID{tid} {score:.2f}',(max(0,min(x-30,pw-130)),max(22,y-18)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
                    tracks.append({'id':int(tid),'alert':bool(alert),'score':float(score),'label':str(label),'centroid':np.asarray(centroid).tolist()})
                alert=any(r[1] for r in results)
                entry['sides'][side['name']]={'alert':alert,'people':len(side['people']),'tracks':tracks}
                pil=Image.new('RGB',(pw,ph+104),'#18263b')
                pil.paste(Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)),(0,104))
                d=ImageDraw.Draw(pil); f=lambda n:ImageFont.truetype(FONT,n)
                title='ระบบปัจจุบัน' if side['name']=='production' else 'ตัวเลือกใหม่ (ยังไม่ใช้งานจริง)'
                model='stock pose + models/ | 0.65' if side['name']=='production' else 'nightaug_s44 + T2-full s45 | 0.70'
                d.text((12,6),title,font=f(24),fill='white'); d.text((12,39),model,font=f(19),fill='#d1e0f0')
                d.rectangle((0,70,pw,103),fill='#c93038' if alert else '#34465c')
                d.text((12,73),f'{t:4.2f} s | '+('แจ้งเตือน' if alert else 'กำลังเฝ้าดู'),font=f(20),fill='white')
                panels.append(pil)
            canvas=Image.new('RGB',(960,panels[0].height+66),'#eef3f9')
            for j,panel in enumerate(panels):canvas.paste(panel,(j*480,0))
            d=ImageDraw.Draw(canvas); f=ImageFont.truetype(FONT,18)
            d.text((12,panels[0].height+6),f'Test/{key} | {start:.2f}–{end:.2f} s | CPU 320 / crop 256 / 8 fps / phase 0',font=f,fill='#18304a')
            d.text((12,panels[0].height+34),'ตัวอย่างที่เลือก • ไม่ใช่ผลรวม • รีเซ็ตตอนเริ่มช่วง • แดง = track ที่แจ้งเตือน • ไม่มีเสียง',font=f,fill='#18304a')
            outframe=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'avc1'),8,(canvas.width,canvas.height))
                assert writer.isOpened(),'H264 encoder unavailable'
            writer.write(outframe); timeline.append(entry)
        assert writer is not None
        writer.release()
        # Compare the complete alert timeline with the pinned phase-0 evaluation.
        comparison={}
        for side,prefix in zip(sides,('parity_deployed','T2_truncfull_s45_rule')):
            ref=f'training/data/stage1/{prefix}_pinned_owner0.json'
            expected=read(ROOT/ref)['segments'][key]['alerts_at']
            observed=[r['t'] for r in timeline if r['sides'][side['name']]['alert']]
            comparison[side['name']]={'source':ref,'expected':expected,'observed':observed,'exact_match':expected==observed}
        evidence['clips'].append({'key':key,'start_s':start,'end_s':end,'frames':len(timeline),
                                 'source_sha256':sha(ROOT/'Test'/clip),'output':str(path.relative_to(ROOT)),
                                 'comparison':comparison,'timeline':timeline})
        save(EVID/'clips.json',evidence)
        print(key,'frames',len(timeline),'parity',comparison,flush=True)
    print('torch threads',torch.get_num_threads(),flush=True)

def verify():
    evidence=read(EVID/'clips.json')
    checks=[]
    for clip in evidence['clips']:
        path=ROOT/clip['output']
        cap=cv2.VideoCapture(str(path))
        assert cap.isOpened(),str(path)
        fps=cap.get(cv2.CAP_PROP_FPS)
        fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
        codec=''.join(chr((fourcc >> 8*i)&255) for i in range(4))
        count=0; dims=None
        # Decode EVERY frame; retain the first alert frame and final frame for visual review.
        alert_idx=next(i for i,r in enumerate(clip['timeline']) if r['sides']['candidate']['alert'])
        while True:
            ok,frame=cap.read()
            if not ok:break
            dims=list(frame.shape[:2][::-1])
            if count in (0,alert_idx,clip['frames']-1):
                cv2.imwrite(str(EVID/f"{path.stem}_frame{count}.jpg"),frame)
            count+=1
        cap.release()
        assert count==clip['frames'],(count,clip['frames'])
        assert fps==8,(fps,path)
        assert codec.lower() in ('h264','avc1'),codec
        assert all(v['exact_match'] for v in clip['comparison'].values())
        checks.append({'output':clip['output'],'sha256':sha(path),'frames_decoded':count,'fps':fps,
                       'seconds':count/fps,'dimensions':dims,'codec':codec,'alert_timeline_parity':True})
    for p in sorted((OUT/'img').glob('day3_*')):
        with Image.open(p) as im:
            im.load(); checks.append({'output':str(p.relative_to(ROOT)),'dimensions':list(im.size),'sha256':sha(p)})
    save(EVID/'verification.json',{'command':'python tools/day3_media.py verify','checks':checks})
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    globals()[sys.argv[1]]()
