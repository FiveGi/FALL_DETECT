#!/bin/sh
# The running t3_posneg.sh was started before the marker line was added; create the marker when T3 truly ends.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until [ -f training/data/multi_diag_v2/presence_t3.txt ] && [ -s training/data/multi_diag_v2/presence_t3.txt ]; do sleep 60; done
echo ok > training/data/holiday_queue.log.t3marker
