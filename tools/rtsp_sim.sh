#!/usr/bin/env bash
# Simulated IP cameras for the 9 Oct system test (no real camera is available before then).
#
# Starts an RTSP server (mediamtx) and N publishers that loop owner Test/ clips the way a CCTV camera
# sends them: constant frame rate, fixed GOP, a typical bitrate. Streams: rtsp://<host>:8554/camN
#
# Usage:
#   tools/rtsp_sim.sh up   N [codec] [res] [kbps]   codec h264|h265 (default h264), res 1080|720|360, kbps default by res
#   tools/rtsp_sim.sh stop K       stop publisher camK only (simulates a camera dropping)
#   tools/rtsp_sim.sh start K      restart publisher camK (camera comes back)
#   tools/rtsp_sim.sh down         remove everything
# Clips used, in order: Test/1.mp4 Test/4.mp4 Test/5.mp4 Test/9.mp4 (camK uses the K-th; all contain falls).
set -euo pipefail
cd "$(dirname "$0")/.."
NET=rtsp-sim
# SIM_CLIPS overrides the clip list (paths relative to the repo), e.g. SIM_CLIPS="Test/1.mp4 training/data/system_test/empty_room.mp4"
read -r -a CLIPS <<< "${SIM_CLIPS:-Test/1.mp4 Test/4.mp4 Test/5.mp4 Test/9.mp4}"
cmd=${1:-}; shift || true

publisher() {   # $1 index  $2 codec  $3 res  $4 kbps
  local k=$1 codec=$2 res=$3 kbps=$4 clip=${CLIPS[$(( ($1 - 1) % ${#CLIPS[@]} ))]}
  local enc=libx264; [ "$codec" = h265 ] && enc=libx265
  local h=$res
  docker rm -f "rtsp-pub$k" >/dev/null 2>&1 || true
  MSYS_NO_PATHCONV=1 docker run -d --restart unless-stopped --name "rtsp-pub$k" --network $NET -v "$(pwd -W):/repo:ro" linuxserver/ffmpeg:latest \
    -hide_banner -loglevel error -re -stream_loop -1 -i "/repo/$clip" -an \
    -vf "scale=-2:$h,pad=$(( h*16/9 )):$h:(ow-iw)/2:0,fps=25" -c:v $enc -preset veryfast -tune zerolatency -g 50 -b:v ${kbps}k -maxrate ${kbps}k -bufsize $((kbps*2))k \
    -f rtsp -rtsp_transport tcp "rtsp://rtsp-server:8554/cam$k" >/dev/null
}

case "$cmd" in
  up)
    n=${1:-1}; codec=${2:-h264}; res=${3:-1080}
    case $res in 1080) def=4000;; 720) def=2000;; *) def=512;; esac
    kbps=${4:-$def}
    docker network inspect $NET >/dev/null 2>&1 || docker network create $NET >/dev/null
    docker rm -f rtsp-server >/dev/null 2>&1 || true
    docker run -d --name rtsp-server --network $NET -p 8554:8554 bluenviron/mediamtx:latest >/dev/null
    sleep 2
    for k in $(seq 1 "$n"); do publisher "$k" "$codec" "$res" "$kbps"; done
    echo "$n camera(s) $codec ${res}p ${kbps}kbps: rtsp://localhost:8554/cam1..cam$n (inside docker: rtsp://rtsp-server:8554/camK on network $NET)"
    ;;
  stop)  docker stop "rtsp-pub$1" >/dev/null && echo "cam$1 stopped" ;;
  start) docker start "rtsp-pub$1" >/dev/null && echo "cam$1 started" ;;
  down)  docker rm -f rtsp-server $(docker ps -aq --filter name=rtsp-pub) >/dev/null 2>&1 || true; echo down ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
