#!/bin/sh
# Owner's PC in use (game open): E2 seed 47 stopped mid-training 21:10. Resume (seed 47 from the
# start, then the frozen E2 ensemble eval) when the game is closed and >= 13.5 GB is free.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until [ -z "$(tasklist //FI "IMAGENAME eq RobloxPlayerBeta.exe" //NH | grep -i roblox)" ] && \
      [ "$(powershell -NoProfile -c "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB*10)" | tr -d '\r')" -ge 135 ]; do
  sleep 60
done
echo "$(date +%H:%M) E2 resumed after pause (seed 47 restarted)" >> training/data/stage1/ablate.log
E2_SEEDS=47 sh training/e2_resume.sh
