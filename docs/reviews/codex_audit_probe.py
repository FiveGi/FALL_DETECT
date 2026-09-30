"""Offline review probes; no app startup, real credentials, network, or production writes.

Run from repository root: python docs/reviews/codex_audit_probe.py
AST extraction executes the checked-in function bodies with isolated dependencies.
These are diagnostic reproductions of current defects, not a passing regression suite.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from flask import Flask

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    from flask import jsonify, request
    results = {}
    app = Flask("isolated_audit")
    writes = []
    services = types.ModuleType("app.services")
    services.detector_profiles = types.SimpleNamespace(
        apply_profile=lambda key: writes.append(key) or {"V3_THRESHOLD": "0.70"})
    sys.modules["app.services"] = services
    claims = {"sub": "regular-user", "type": "access"}
    ns = functions("app/routes/detector_info.py", {"apply_profile"},
                   {"get_jwt": lambda: claims, "jsonify": jsonify, "request": request})
    with app.test_request_context(json={"profile": "cpu_fewer_false_alarms"}):
        response, status = ns["apply_profile"]()
    results["profile_guard_missing_role"] = {"http_status": status, "write_calls": writes}

    class Query:
        def get(self, camera_id):
            return {"camera_id": camera_id, "owner": "another-user"}
    def invalid_token(**kwargs):
        raise ValueError("invalid token")
    ns = functions("app/routes/stream.py", {"get_current_user_id", "get_user_camera_or_404"},
                   {"verify_jwt_in_request": invalid_token,
                    "Camera": types.SimpleNamespace(query=Query())})
    uid = ns["get_current_user_id"]()
    results["invalid_stream_token"] = {
        "user_id": uid, "camera_returned": ns["get_user_camera_or_404"](17, uid)}

    spec = importlib.util.spec_from_file_location("coherence_audit", ROOT / "tools/check_config_coherence.py")
    measured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measured)
    config = {"input_size": 320, "window_frames": 15, "target_fps": 8.0,
              "partial_window_from": 4, "preprocess": ["auto"], "threshold": 0.65}
    dispatch = types.ModuleType("app.services.detection_dispatch")
    dispatch.read_detector_config = lambda: config
    sys.modules["app.services.detection_dispatch"] = dispatch
    ns = functions("app/routes/detector_info.py", {"detector_info"},
                   {"jsonify": jsonify, "_measured_table": lambda: measured.MEASURED})
    results["measurement_key_omits_threshold"] = {}
    with app.app_context():
        for threshold in [0.65, 0.99]:
            config["threshold"] = threshold
            response, status = ns["detector_info"]()
            data = response.get_json()["data"]
            results["measurement_key_omits_threshold"][str(threshold)] = {
                "measured_known": data["measured_known"], "measurement": data["measured"]}

    spec = importlib.util.spec_from_file_location("audit_profiles", ROOT / "app/services/detector_profiles.py")
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    env = dict(profiles.PROFILES["cpu_fewer_false_alarms"]["env"])
    # list_profiles converts the worker's numeric threshold to a string with %g.
    env["V3_THRESHOLD"] = "%g" % float(env["V3_THRESHOLD"])
    results["profile_numeric_format"] = {
        "threshold_from_worker": env["V3_THRESHOLD"],
        "matched_profile": profiles.current_profile(env)}

    # Execute the worker's actual top assignment, with a valid N-of-M scenario:
    # track 1 remains detected from its recent history; track 2 has a higher current
    # score below the decision threshold, but no positive smoothing history.
    tree = ast.parse((ROOT / "app/services/camera_manager.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_v2_fall_detection")
    top = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "top" for t in n.targets))
    ns = {"results": [(1, True, 0.2, "fall", None), (2, False, 0.6, "no_fall", None)]}
    exec(compile(ast.Module(body=[top], type_ignores=[]), "worker_top", "exec"), ns)
    results["alert_track_selection"] = {"alerting_track": 1, "selected_track": ns["top"][0]}

    # Cache generation skips existing clips before drawing random noise. Show why
    # one global seed cannot preserve a later clip when the previous one is skipped.
    rng = np.random.RandomState(0)
    rng.normal(size=(4, 4, 3))  # already-cached clip A
    uninterrupted_b = rng.normal(size=(4, 4, 3))
    resumed_b = np.random.RandomState(0).normal(size=(4, 4, 3))
    results["resumed_global_noise_stream"] = {
        "same_clip_same_seed_identical": bool(np.array_equal(uninterrupted_b, resumed_b))}

    # Worker reads every consecutive file frame; the offline sampler drops source
    # frames according to timestamps. No real camera is opened for this probe.
    source_fps, target_fps, frame_count = 30, 8, 300
    slots = [int(i * target_fps / source_fps) for i in range(frame_count)]
    sampled = [i for i, slot in enumerate(slots) if i == 0 or slot != slots[i - 1]]
    results["file_sampling"] = {
        "source_seconds": frame_count / source_fps,
        "offline_classifier_frames": len(sampled),
        "worker_classifier_frames_per_loop": frame_count,
        "worker_wall_seconds_at_target": frame_count / target_fps}

    # Arithmetic of the worker's still-down branch, for a loaded CPU worker.
    elapsed, observed_down_frames, configured_fps = 10.0, 40, 8
    seconds = observed_down_frames / configured_fps
    results["still_down_below_target"] = {
        "scenario": "person down continuously for 10 s; actual 4 fps, target 8 fps",
        "calculated_down_seconds": seconds,
        "discards_followup_as_upright": seconds < 0.8 * elapsed}

    output = json.dumps(results, indent=2)
    Path(__file__).with_name("2026-09-29-audit-probes.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
