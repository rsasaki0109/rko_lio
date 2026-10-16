#!/usr/bin/env bash
# Usage: pipe_once.sh <csv> <label> <ws>  -- one full-pipeline run on exp04 (same as one pipe_abab.sh run).
csv=$1; l=$2; w=$3
SLAM=/home/user/rsasaki0109/lidar_slam_ros2
out="/home/user/bench/pipe/$(basename "${csv}" .csv)_${l}"
rm -rf "${out}"
read -r l1 l5 l15 _ < /proc/loadavg
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
(cd "${SLAM}" && LIDARSLAM_WS_ROOT="${w}" ROS_DOMAIN_ID=32 timeout 3000 bash scripts/run_rko_lio_graph_benchmark.sh --gt-blind \
  --bag /home/user/data/hilti2022/exp04_ros2 --lidar-topic /hesai/pandar --imu-topic /alphasense/imu \
  --lidarslam-param configs/hilti2022/lidarslam_competitive_v2.yaml \
  --rko-param configs/hilti2022/rko_lio_hilti2022_pandar_competitive_v2.yaml \
  --output-dir "${out}" --run-name front --save-timeout-secs 420 --skip-map-save) > "${out}.log" 2>&1
rc=$?
wall=$(python3 -c "import json;print(json.load(open('${out}/gt_blind_run.json'))['wall_seconds'])" 2>/dev/null || echo NA)
rtf=$(python3 -c "print(f'{${wall}/125.814:.4f}')" 2>/dev/null || echo NA)
tum="${out}/front_0/front_tum_0.txt"
if [[ -f "${tum}" ]]; then md5=$(md5sum "${tum}" | cut -d' ' -f1); n=$(wc -l < "${tum}"); else md5=missing; n=0; fi
took=$(grep -ohP 'RKO LIO Offline Node took \K[0-9.]+' "${out}"/*.log 2>/dev/null | tail -1)
echo "${l},1,${rc},${wall},${rtf},${l1},${l5},${l15},${start_utc},${md5},${n},${took:-NA}" | tee -a "${csv}"
