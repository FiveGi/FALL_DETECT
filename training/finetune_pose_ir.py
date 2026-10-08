# -*- coding: utf-8 -*-
"""POSE-IR: fine-tune yolo26s-pose so it keeps finding people in night-like pictures.

Owner, 2026-10-01: "make greyscale or unclear images and train with them". The night loss is in
the POSE stage (SKILL.md section 74: colour 45/60 URFD falls, plain greyscale 35/60, greyscale +
IR vignette + noise 15-20/60), so the pose model is what is trained -- the fall classifier is not
touched. Design: Codex, AI_HANDOFF.md 2026-10-01 "POSE-IR design"; decisions in the Claude entry
"POSE-IR adopted as designed".

Each training image, after ultralytics' own augmentation and before formatting, is left in colour
(70%), turned plain greyscale (15%) or into simulated IR (15%): greyscale, a radial vignette of
random strength 0-0.6 and Gaussian sensor noise sigma 0-8 levels (the cache simulator's defaults,
0.45 / 5.0, sit inside these ranges). Half of the degraded images are also blurred -- defocus
(Gaussian sigma 0.5-2 px) or motion (line kernel 3-7 px, random angle) -- for long night
exposures (owner's "unclear"). Labels are untouched: these change pixels, not geometry.
CONTROL=1 runs the identical schedule with no degradation, so any change is attributable.

Schedule: phase 1, 2 epochs, backbone (11 layers) frozen, AdamW lr 1e-4; phase 2, 8 epochs,
everything trainable, AdamW lr 5e-5 cosine to 10%. Codex proposed backbone 1e-5 / head 5e-5 in
phase 2; ultralytics has one learning rate for all layers, so 5e-5 everywhere is the deviation.
Input 320 (the CPU profile). Checkpoint chosen on COCO val (colour) by ultralytics, never on any
evaluation set used by the gates.

Usage: SEED=42 [CONTROL=1] python training/finetune_pose_ir.py
"""
import math
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('COCO_POSE_YAML', 'D:/project/PROJECT/datasets/coco-pose.yaml')
BASE = os.path.join(ROOT, 'models', 'yolo26s-pose.pt')
SEED = int(os.environ.get('SEED', 42))
CONTROL = os.environ.get('CONTROL', '0') == '1'
P_GREY, P_IR, P_BLUR = (0.0, 0.0, 0.0) if CONTROL else (0.15, 0.15, 0.5)
SMOKE = os.environ.get('SMOKE', '0') == '1'   # 2% of COCO, 1+1 epochs: checks the pipeline only
NAME = os.environ.get('RUN_NAME') or '%s_s%d%s' % ('colour' if CONTROL else 'nightaug', SEED, '_smoke' if SMOKE else '')
PROJECT = os.path.join(ROOT, 'training', 'data', 'pose_ir')


class NightDegrade:
    """labels['img'] (BGR uint8, already augmented) -> colour / greyscale / simulated IR (+blur)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.counts = {'colour': 0, 'grey': 0, 'ir': 0, 'blur': 0}

    def __call__(self, labels):
        u = self.rng.random()
        if u >= P_GREY + P_IR:
            self.counts['colour'] += 1
            return labels
        img = labels['img']
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if u < P_IR:
            h, w = g.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cy, cx = h * self.rng.uniform(0.35, 0.65), w * self.rng.uniform(0.35, 0.65)
            r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2) / math.sqrt(0.5)
            g *= 1.0 - self.rng.uniform(0.0, 0.6) * np.clip(r, 0, 1) ** 2
            g += np.random.default_rng(self.rng.getrandbits(32)).normal(
                0, self.rng.uniform(0.0, 8.0), g.shape).astype(np.float32)
            self.counts['ir'] += 1
        else:
            self.counts['grey'] += 1
        if self.rng.random() < P_BLUR:
            if self.rng.random() < 0.5:
                g = cv2.GaussianBlur(g, (0, 0), self.rng.uniform(0.5, 2.0))
            else:
                k = self.rng.choice([3, 5, 7])
                kern = np.zeros((k, k), np.float32)
                kern[k // 2, :] = 1.0 / k
                rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), self.rng.uniform(0, 180), 1.0)
                kern = cv2.warpAffine(kern, rot, (k, k))
                g = cv2.filter2D(g, -1, kern / max(kern.sum(), 1e-6))
            self.counts['blur'] += 1
        g = np.clip(g, 0, 255).astype(np.uint8)
        labels['img'] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return labels


def trainer_class(seed):
    from ultralytics.models.yolo.pose import PoseTrainer

    class NightPoseTrainer(PoseTrainer):
        def build_dataset(self, img_path, mode='train', batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == 'train' and not CONTROL:
                # Before Format (the last transform), after every geometric/colour augmentation.
                ds.transforms.transforms.insert(len(ds.transforms.transforms) - 1, NightDegrade(seed))
            return ds
    return NightPoseTrainer


def main():
    from ultralytics import YOLO
    random.seed(SEED)
    np.random.seed(SEED)
    common = dict(data=DATA, imgsz=320, batch=64, workers=int(os.environ.get('WORKERS', 4)), seed=SEED, deterministic=False,
                  optimizer='AdamW', project=PROJECT, exist_ok=True, plots=False, verbose=False,
                  fraction=0.02 if SMOKE else 1.0)
    p1 = YOLO(BASE).train(trainer=trainer_class(SEED), epochs=1 if SMOKE else 2, freeze=11, lr0=1e-4, lrf=1.0,
                          warmup_epochs=0, name=NAME + '_p1', **common)
    w1 = os.path.join(PROJECT, NAME + '_p1', 'weights', 'last.pt')
    YOLO(w1).train(trainer=trainer_class(SEED + 1000), epochs=1 if SMOKE else 8, lr0=5e-5, lrf=0.1, cos_lr=True,
                   warmup_epochs=0, name=NAME + '_p2', **common)
    print('done', NAME, os.path.join(PROJECT, NAME + '_p2', 'weights', 'best.pt'))


if __name__ == '__main__':
    sys.exit(main())
