#!/bin/sh
# Owner's PC in use (game open, RAM < 12 GB free): night screening paused 21:05. Resume when the
# game is closed and >= 13.5 GB is free. A build killed mid-write can leave a truncated .npz that
# the per-clip resume would skip, so unreadable files are removed first.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir_eval/queue.log
until [ -z "$(tasklist //FI "IMAGENAME eq RobloxPlayerBeta.exe" //NH | grep -i roblox)" ] && \
      [ "$(powershell -NoProfile -c "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB*10)" | tr -d '\r')" -ge 135 ]; do
  sleep 60
done
python - <<'PY' >> $L 2>&1
import glob, os, numpy as np
bad = 0
for f in glob.glob('training/data/pose_ir_eval/*/*/*.npz'):
    try:
        np.load(f)['counts']
    except Exception:
        os.remove(f); bad += 1
print('resume: removed %d unreadable npz' % bad)
PY
echo "$(date +%H:%M) resumed after pause" >> $L
sh training/pose_ir_night_eval.sh
