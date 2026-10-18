#!/usr/bin/env bash
# Usage: pipe_abab.sh <csv> <pairs> <labelA> <wsA> <labelB> <wsB>
# Full pipeline (RKO-LIO + graph_based_slam) on exp04 with run_rko_lio_graph_benchmark.sh --gt-blind,
# A and B alternately. Appends label,pair,rc,wall_seconds,rtf,load1,load5,load15,start_utc,front_md5,lines,node_took
csv=$1; pairs=$2; la=$3; wa=$4; lb=$5; wb=$6
SLAM=/home/user/rsasaki0109/lidar_slam_ros2
for ((i = 1; i <= pairs; i++)); do
  for side in A B; do
    if [[ ${side} == A ]]; then l=${la}; w=${wa}; else l=${lb}; w=${wb}; fi
    sleep "${PAUSE:-20}"
    out="/home/user/bench/pipe/$(basename "${csv}" .csv)_${l}_${i}"
    rm -rf "${out}"
    read -r l1 l5 l15 _ < /proc/loadavg
    start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    (cd "${SLAM}" && LIDARSLAM_WS_ROOT="${w}" ROS_DOMAIN_ID=31 timeout 3000 bash scripts/run_rko_lio_graph_benchmark.sh --gt-blind \
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
    echo "${l},${i},${rc},${wall},${rtf},${l1},${l5},${l15},${start_utc},${md5},${n},${took:-NA}" | tee -a "${csv}"
  done
done
