"""Independent CPU-only replay of existing pose caches; never loads/runs the pose model.

Uses the production classifier feature function, ONNX file, tracking and alert state
machine. Constructor is deliberately bypassed because it loads an unused pose model.
This validates cached classifier outcomes, NOT camera speed or simulation realism.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def main():
    # Process-local environment only. Do not read or change the deployment .env.
    for name in list(os.environ):
        if name.startswith("V3_"):
            del os.environ[name]
    os.environ.update(V3_IMGSZ="320", V3_TARGET_FPS="8", V3_THRESHOLD="0.65",
                      V3_PARTIAL_MIN="4", V3_PREPROCESS="auto", V3_PREPROCESS_DARK_BELOW="70")
    path = ROOT / "app/detection/v3_fall_detection.py"
    spec = importlib.util.spec_from_file_location("audit_v3", path)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    model = ROOT / "models/fall_classifier_v3.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    det = v3.V3PoseFallDetector.__new__(v3.V3PoseFallDetector)
    det.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    det.extra_sessions = []
    det.extract_all_keypoints = lambda frame: det._replay
    report = {
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": {k: getattr(v3, k) for k in ["WINDOW_SIZE", "THRESHOLD", "PARTIAL_MIN",
                    "SMOOTH_NEED", "SMOOTH_OF", "STRIDE", "TRACKER", "COLLAPSE_ENABLED"]},
        "caches": {}}
    reference_names = None
    for label, digest in [("daylight", "58bd55e55f1d"), ("greyscale", "932fbee53d48"),
                          ("simulated_ir", "cb5439b03087"), ("simulated_ir_forced_cleanup", "9c2963624342")]:
        folder = ROOT / "training/data/pose_cache" / digest
        names = sorted(p.name for p in folder.glob("urfd_*.npz"))
        if reference_names is None:
            reference_names = names
        assert names == reference_names, "Caches do not cover identical clips"
        assert len(names) == 100, (label, len(names))
        rows = []
        for name in names:
            state = v3.V3MultiPersonFallState()
            with np.load(folder / name) as data:
                counts, kpts = data["counts"], data["kpts"]
            assert int(counts.sum()) == len(kpts)
            at, hit = 0, False
            for count in counts:
                people = kpts[at:at + int(count)]
                at += int(count)
                det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2) for kp in people]
                current = v3.detect_v3_fall_multi(None, state, det, config=None)
                hit = hit or any(r[1] for r in current)
            rows.append({"clip": name, "alerted": hit})
        summary = {}
        for group in ["urfd_fall", "urfd_adl"]:
            subset = [r for r in rows if r["clip"].startswith(group + "__")]
            summary[group] = {"clips": len(subset), "alerted": sum(r["alerted"] for r in subset)}
        report["caches"][label] = {"directory": digest,
            "key": json.loads((folder / "key.json").read_text()), "summary": summary, "rows": rows}
        print(label, json.dumps(summary), flush=True)
    target = Path(__file__).with_name("2026-09-29-cache-replay.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", target)


if __name__ == "__main__":
    main()
