#!/bin/sh
# When FALLNIGHT s42 has trained: its screening caches (day 0/4, IR 0, NIGHT_ALT 0, owner 0/4), 2 at a time.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until grep -q "fallnight s42 \(done\|FAILED\)" training/data/fallpose_queue.log; do sleep 30; done
grep -q "fallnight s42 done" training/data/fallpose_queue.log && sh training/gate_jobs_runner.sh training/data/gate_jobs_fallnight.txt
