import collections
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos, split_by_group, group_key, FallWindowDataset
from model import FallClassifier

# GMDCSA24_DIR_NAME lets relabel_gmdcsa24_gemini.py's downstream comparison point this
# at "poses" (original motion-peak heuristic) or "poses_gmdcsa24_v2" (Gemini-verified
# fall onset) without editing this file, so both runs use identical code.
# Defaults reproduce the DEPLOYED model. They were once the MediaPipe-era directories, which
# meant `python train.py` with no environment produced a model three generations old whose
# ONNX export the runtime cannot even load (it expects 15x85, not 30x85). Anyone cloning this
# repo got that silently. Every old configuration is still reachable through the variables.
GMDCSA24_DIR_NAME = os.environ.get("GMDCSA24_DIR_NAME", "poses_yolopose")
# CAUCAFALL_DIR_NAME/OFITW_DIR_NAME: same pattern as GMDCSA24_DIR_NAME, added so the
# YOLO-pose-vs-MediaPipe pose-extractor comparison can point all three video-derived
# datasets at their yolopose_extractor.py-produced siblings (poses_yolopose,
# poses_caucafall_yolopose, poses_ofitw_yolopose_matched) without editing this file.
# poses_fallvision is untouched either way -- it's pre-extracted external keypoints,
# not derived from either pose backend, so it stays a shared constant across both runs.
CAUCAFALL_DIR_NAME = os.environ.get("CAUCAFALL_DIR_NAME", "poses_caucafall_yolopose")
OFITW_DIR_NAME = os.environ.get("OFITW_DIR_NAME", "poses_ofitw_yolopose_matched")
POSE_DIRS = [
    os.path.join(os.path.dirname(__file__), "data", GMDCSA24_DIR_NAME),    # GMDCSA24
    os.path.join(os.path.dirname(__file__), "data", "poses_fallvision"),   # FallVision (COCO-17, heuristic labels)
    os.path.join(os.path.dirname(__file__), "data", CAUCAFALL_DIR_NAME),   # CAUCAFall (MediaPipe-33, REAL per-frame labels)
    os.path.join(os.path.dirname(__file__), "data", OFITW_DIR_NAME),       # OmniFall OF-ItW / OOPS (MediaPipe-33, REAL segment labels, real-world not staged)
]
# USE_REALTEST_V1: 67 Gemini+Claude-verified segments from Test/4.mp4-11.mp4 (SS21/SS22)
# -- 45 real falls, 22 explicit hard negatives (bed-lying, dancing, standing-near-objects
# that the deployed model currently misfires on). Off by default so SS20/SS21's
# baseline numbers stay reproducible; set to 1 for the SS22 before/after comparison.
if os.environ.get("USE_REALTEST_V1") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_realtest_v1"))
# USE_OMNIFALL_ADL: 117 real OmniFall segments labeled lying/lie_down/sitting/sit_down/
# kneeling/squatting (SS28) -- unlike SS22, these directly target the SPECIFIC pattern
# (bed/floor-lying, confirmed across SS17/SS20/SS21/SS23/SS27) the deployed model keeps
# misfiring on, sourced from ~100 different OOPS subjects/rooms instead of GMDCSA24's 4.
if os.environ.get("USE_OMNIFALL_ADL") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_omnifall_adl"))
# USE_URFD_ADL: 20 of URFD's 40 ADL clips (the even-numbered half; the odd half is never
# trained on and stays the held-out measure). Indoor, fixed camera, ordinary activity -- aimed
# at the deep-bend false alarm that survived every decision-rule attempt (SS33, SS40), and the
# same targeted-hard-negative approach that worked in SS28. Fall clips are never included, so
# URFD recall remains a clean measurement.
if os.environ.get("USE_URFD_ADL") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_urfd_adl_train"))
# USE_LE2I: Le2i / ImViA (190 videos, single fixed camera, OmniFall labels), extracted 2026-10-01 by
# extract_omnifall_staged.py with the deployed CPU pipeline at 8 fps (each file carries fps=8).
if os.environ.get("USE_LE2I") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_le2i"))
# SKIP_POSE_DIRS: comma-separated directory names to drop from POSE_DIRS. It exists for one
# control in particular: USE_FRAME_POSITION reads raw frame coordinates, and FallVision -- 58%
# of the videos -- stores them in pixels, so dataset.COORD_SCALE has to put it in the same
# space as everything else. That scale is measured rather than documented, so the arm that
# checks it is one trained with `SKIP_POSE_DIRS=poses_fallvision`, where no scale is applied
# because the data is not there. If the two arms disagree, the scale is wrong.
_skip = {n.strip() for n in os.environ.get("SKIP_POSE_DIRS", "").split(",") if n.strip()}
if _skip:
    POSE_DIRS = [d for d in POSE_DIRS if os.path.basename(d) not in _skip]
    print(f"SKIP_POSE_DIRS dropped {sorted(_skip)}; {len(POSE_DIRS)} pose directories left")

CKPT_PATH = os.environ.get(
    "CKPT_PATH", os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
)
EPOCHS = int(os.environ.get("EPOCHS", 60))
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("TRAIN_SEED", 42))  # fixes torch's init/shuffle RNG so before/after label-quality comparisons aren't noise

# Stage 1 controls (Codex DATA-DESIGN, AI_HANDOFF.md 2026-10-01). All default to the old
# behaviour so earlier runs stay reproducible.
# SPLIT=group: deterministic split by group_key (person / recording / archive) instead of the
#   random per-label shuffle, which leaked OF-ItW segments of one video across train and val and
#   reshuffled every existing video whenever a source was added.
# SYN_DIR / SYN_FRACTION: synthetic pose files (OF-Syn), TRAINING ONLY, drawn as this share of
#   each epoch's windows. SYN_AGES filters by the file's age_group (the audit rejected children).
# BALANCED_SAMPLING=1: every epoch draws the same number of windows (= the real training
#   windows), with replacement, each CLIP weighted equally within its pool (real / synthetic), so
#   long clips cannot dominate and every arm takes the same optimizer steps. Implied by
#   SYN_FRACTION > 0; set it for the 0% arm too so the arms differ only in data.
SPLIT = os.environ.get("SPLIT", "random")
SYN_DIR = os.environ.get("SYN_DIR", "")
SYN_FRACTION = float(os.environ.get("SYN_FRACTION", 0))
SYN_AGES = {a for a in os.environ.get(
    "SYN_AGES", "elderly_65_plus,middle_aged_35_64,young_adults_18_34").split(",") if a}
BALANCED_SAMPLING = os.environ.get("BALANCED_SAMPLING", "0") == "1" or SYN_FRACTION > 0


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    videos = load_all_videos(POSE_DIRS)
    print(f"Loaded {len(videos)} videos")
    labels = [v["label"] for v in videos]
    print(f"  Fall: {sum(labels)}, ADL: {len(labels) - sum(labels)}")

    if SPLIT == "group":
        # Sources that ARE evaluation sets may not be trained on under the Stage 1 recipe at all
        # (Codex review P1): URFD ADL is the night/day gate's negative set, realtest_v1 is cut
        # from the owner's Test/ clips. omnifall_adl is refused until its files carry canonical
        # OOPS recording ids: today all 117 collapse into one group and 21 share recordings with
        # OF-ItW validation segments.
        HELDOUT_SOURCES = {"poses_urfd_adl_train", "poses_realtest_v1", "poses_omnifall_adl"}
        bad = sorted({v["source"] for v in videos} & HELDOUT_SOURCES)
        if bad:
            raise SystemExit(f"SPLIT=group refuses held-out or ungroupable sources: {bad}")
        # Held-out evaluation files are removed before the split, so they can land on neither
        # side: not trained on, and not used to pick the checkpoint either.
        excl_path = os.environ.get("HELDOUT_EXCLUDE", os.path.join(os.path.dirname(__file__), "data", "heldout_exclude.txt"))
        excl = {l.strip() for l in open(excl_path) if l.strip() and not l.startswith("#")}
        before = len(videos)
        videos = [v for v in videos if v["source"] + "/" + v["name"] not in excl]
        print(f"Held-out exclusions ({os.path.basename(excl_path)}): removed {before - len(videos)} files")
        train_videos, val_videos = split_by_group(videos)
    else:
        train_videos, val_videos = split_videos(videos, val_ratio=0.2)
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)} (split={SPLIT})")
    # Record exactly which video went where, so a run can be audited and repeated.
    import json, collections
    manifest = {"split": SPLIT, "seed": SEED, "train": sorted(v["source"] + "/" + v["name"] for v in train_videos),
                "val": sorted(v["source"] + "/" + v["name"] for v in val_videos)}
    by_src = collections.defaultdict(lambda: [0, 0, 0, 0])
    for side, vs in ((0, train_videos), (2, val_videos)):
        for v in vs:
            by_src[v["source"]][side + v["label"]] += 1
    for src, (tn, tf, vn, vf) in sorted(by_src.items()):
        print(f"  {src:32s} train {tn:5d} adl {tf:5d} fall | val {vn:5d} adl {vf:5d} fall")

    train_samples = make_windows(train_videos)
    val_samples = make_windows(val_videos)
    print(f"Train windows: {len(train_samples)}, Val windows: {len(val_samples)} (val is real data only)")

    syn_samples = []
    if SYN_DIR:
        syn_videos = [v for v in load_all_videos([SYN_DIR])]
        meta_ok = []
        for v in syn_videos:
            z = np.load(os.path.join(SYN_DIR, v["name"]), allow_pickle=True)
            if str(z["age_group"]) in SYN_AGES:
                meta_ok.append(v)
        if SYN_FRACTION > 0 and not meta_ok:
            raise SystemExit(f"SYN_FRACTION={SYN_FRACTION} but no synthetic file in {SYN_DIR} "
                             f"matches SYN_AGES={sorted(SYN_AGES)}")
        syn_samples = make_windows(meta_ok)
        manifest["synthetic"] = sorted(v["name"] for v in meta_ok)
        print(f"Synthetic: {len(meta_ok)} files of {len(syn_videos)} (ages {sorted(SYN_AGES)}), "
              f"{len(syn_samples)} windows, {sum(l for _, l, _ in syn_samples)} fall; "
              f"share of each epoch {SYN_FRACTION:.0%}")
    json.dump(manifest, open(os.path.splitext(CKPT_PATH)[0] + "_split.json", "w"), indent=0)

    # AUGMENT (SS35): horizontal-flip + synthetic per-joint occlusion, applied only to the
    # training split (never val, which must stay a clean measure of real generalization).
    # Off by default so every prior run in this file stays exactly reproducible.
    AUGMENT = os.environ.get("AUGMENT", "1") == "1"
    if SYN_FRACTION > 0 and not syn_samples:
        raise SystemExit("SYN_FRACTION > 0 but SYN_DIR is unset or produced no windows")
    pool = train_samples + (syn_samples if SYN_FRACTION > 0 else [])
    train_ds = FallWindowDataset(pool, augment=AUGMENT)
    val_ds = FallWindowDataset(val_samples, augment=False)
    if BALANCED_SAMPLING:
        from torch.utils.data import WeightedRandomSampler
        def clip_weights(samples, share):
            per_clip = collections.Counter(n for _, _, n in samples)
            return [share / (len(per_clip) * per_clip[n]) for _, _, n in samples]
        w = clip_weights(train_samples, 1.0 - (SYN_FRACTION if syn_samples else 0.0))
        if SYN_FRACTION > 0 and syn_samples:
            w += clip_weights(syn_samples, SYN_FRACTION)
        weights = torch.tensor(w, dtype=torch.double)
        gen = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True, generator=gen)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        expected_pos = float((weights * torch.tensor([l for _, l, _ in pool], dtype=torch.double)).sum() / weights.sum())
        print(f"Balanced sampling: {len(train_samples)} windows/epoch, expected fall share {expected_pos:.3f}")
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        expected_pos = None
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Missing a real fall is far worse than a false alarm, so weight the loss to punish
    # false negatives harder than the raw class ratio would -- pushes the model away from
    # the "always predict no-fall" local optimum it settled into last run.
    # POS_WEIGHT_MULT (SS29): SS28 found adding real targeted lying/sitting hard negatives
    # pushed recall to 100% but didn't fix a single one of the ADL false positives it was
    # meant to fix -- testing whether this 1.5x recall bias is why, by trying it at 1.0x
    # (no artificial bias beyond the natural class ratio) alongside the same SS28 data.
    POS_WEIGHT_MULT = float(os.environ.get("POS_WEIGHT_MULT", "1.5"))
    train_labels = np.array([lbl for _, lbl, _ in train_samples])
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    if expected_pos is not None:
        # The class ratio the model actually sees is the sampler's, not the pool's.
        n_pos, n_neg = expected_pos, 1.0 - expected_pos
    pos_weight = torch.tensor([(n_neg / max(n_pos, 1e-6)) * POS_WEIGHT_MULT], device=DEVICE)
    print(f"Train windows: {n_pos} fall, {n_neg} no-fall -> pos_weight={pos_weight.item():.2f} (mult={POS_WEIGHT_MULT})")

    # HIDDEN_SIZE (SS31): model.py's docstring notes a much bigger architecture (ST-GCN,
    # ~17x more params) underperformed this one on the old data (0.582 vs 0.615 F1) --
    # testing a modest capacity increase (not 17x) on top of SS29's winning data/pos_weight
    # config specifically, since that comparison predates this session's data additions.
    HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", "128"))
    model = FallClassifier(hidden=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # EMA=1 (proposal 2026-10-05, off by default): keep an exponential moving average of the weights (BatchNorm
    # BN statistics recomputed each epoch) and evaluate/save THAT model. Six T2-full seeds passed the frozen gates only 2/6 times and the
    # rule threshold swung 0.45-0.70; picking one noisy epoch by val F1 is a likely source. Still one model at runtime.
    EMA = os.environ.get("EMA") == "1"
    EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
    if EMA:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        from torch.optim.swa_utils import update_bn
        # Parameters only; BatchNorm statistics are RECOMPUTED on the training data before every evaluation.
        # Gemini's test 2026-10-05: averaged BN buffers gave val F1 0.02 vs 0.51 recomputed (decay 0.9999, epoch 1).
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=False)
        print(f"EMA on: decay {EMA_DECAY} per step, BN stats recomputed (update_bn); eval + checkpoint use the averaged model")
    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if EMA:
                ema.update_parameters(model)
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / (len(train_samples) if BALANCED_SAMPLING else len(train_ds))

        if EMA:
            update_bn(((x.to(DEVICE),) for x, _ in train_loader), ema.module, device=DEVICE)
        eval_model = ema.module if EMA else model
        acc, prec, rec, f1 = evaluate(eval_model, val_loader)
        scheduler.step(f1)
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val_acc {acc:.3f} val_prec {prec:.3f} val_rec {rec:.3f} val_f1 {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state": eval_model.state_dict(),
                "val_f1": f1,
                "val_acc": acc,
                "epoch": epoch,
            }, CKPT_PATH)
            print(f"  -> saved new best (f1={f1:.3f})")

    print(f"\nBest val F1: {best_f1:.3f}")
    print(f"Checkpoint saved to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
