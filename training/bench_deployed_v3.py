"""Time the currently deployed v3 pipeline (YOLO26s-pose + ONNX temporal classifier).

Written to answer SKILL.md SS7.1 for the model that is actually live now: every speed number
in that section was measured for RF-DETR, MediaPipe or yolov10x, none of which is what
production runs today. The pose extractor runs on EVERY frame with no stride, so its
per-frame latency -- not the classifier's -- is what gates the real-time loop.

Two knobs matter for reproducing the production server (SKILL.md SS9): its QEMU virtual CPU
exposes only SSE4.1/4.2, no AVX/AVX2/FMA, and celery_worker is pinned to OMP_NUM_THREADS=1
under --cpus=2.0.

  --no-avx  sets ATEN_CPU_CAPABILITY=default before torch is imported, which forces
            PyTorch's non-vectorised CPU kernels. This is a PROXY for the server, not a
            substitute for running there: it constrains ATen's dispatch only, while the
            ONNX Runtime session and any oneDNN path keep using whatever this machine has.
            Treat the result as a lower bound on the server's latency, and say so.

Usage:
    python training/bench_deployed_v3.py --frames 40
    python training/bench_deployed_v3.py --frames 40 --no-avx
"""
import argparse
import glob
import os
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--frames', type=int, default=40)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--no-avx', action='store_true',
                    help='force PyTorch non-vectorised CPU kernels (proxy for the AVX-less server)')
parser.add_argument('--gpu', action='store_true', help='run the pose model on CUDA')
parser.add_argument('--video', default=None, help='video file to read frames from')
args = parser.parse_args()

# Must be set before torch/ultralytics import to have any effect.
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ['MKL_NUM_THREADS'] = str(args.threads)
# --gpu leaves the GPU visible; otherwise this pins CPU-only to match the container.
if not args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # '' makes torch raise "Invalid device id" on a GPU host
if args.no_avx:
    os.environ['ATEN_CPU_CAPABILITY'] = 'default'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import torch  # noqa: E402

# Loaded by path, not as app.detection.*: importing the package pulls in Flask/SQLAlchemy,
# which this benchmark does not need and a bare training environment may not have.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'app', 'detection', 'v3_fall_detection.py'))
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
V3PoseFallDetector = _v3.V3PoseFallDetector
V3MultiPersonFallState = _v3.V3MultiPersonFallState
detect_v3_fall_multi = _v3.detect_v3_fall_multi


def load_frames(n):
    path = args.video
    if path is None:
        candidates = sorted(glob.glob(os.path.join('Test', '*.mp4')))
        if not candidates:
            raise SystemExit('no clips found in Test/ -- pass --video')
        path = candidates[0]
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit('could not read frames from ' + path)
    return path, frames


def main():
    torch.set_num_threads(args.threads)
    clip, frames = load_frames(args.frames)

    detector = V3PoseFallDetector(model_dir='models', device='cuda' if args.gpu else 'cpu')
    state = V3MultiPersonFallState()

    # One untimed pass: the first call pays for lazy model load and allocator warm-up, which
    # would otherwise dominate a 40-frame average and overstate steady-state latency.
    detect_v3_fall_multi(frames[0], state, detector, {})

    state = V3MultiPersonFallState()
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        detect_v3_fall_multi(frame, state, detector, {})
        times.append((time.perf_counter() - t0) * 1000.0)

    times_sorted = sorted(times)
    p = lambda q: times_sorted[min(len(times_sorted) - 1, int(len(times_sorted) * q))]
    mean = statistics.mean(times)

    print(f'clip           : {clip}  ({len(frames)} frames)')
    print(f'threads        : {args.threads}   ATEN_CPU_CAPABILITY={os.environ.get("ATEN_CPU_CAPABILITY", "(default vectorised)")}')
    print(f'mean           : {mean:.1f} ms/frame  ({1000.0 / mean:.2f} fps)')
    print(f'p50 / p95 / max: {p(0.5):.1f} / {p(0.95):.1f} / {max(times):.1f} ms')
    # A camera loop must finish a frame faster than the camera produces one; below ~10 fps
    # the pipeline is dropping most of what it sees.
    print(f'keeps up with  : {1000.0 / mean:.1f} fps camera at best')


if __name__ == '__main__':
    main()
