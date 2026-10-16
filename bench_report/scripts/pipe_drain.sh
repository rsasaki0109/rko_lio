#!/usr/bin/env bash
# Usage: pipe_drain.sh <csv> <label> <ws> <lidarslam_yaml> <out_dir>
# Launches the same lidarslam rko_lio_slam.launch.py as run_rko_lio_graph_benchmark.sh (--skip-map-save
# mode), but keeps graph_based_slam alive after the offline node exits, and records per second the CPU of
# both processes. Reports when the frontend ended and when the backend went idle (5 s under 0.05 cores).
csv=$1; label=$2; w=$3; yaml=$4; out=$5
rm -rf "${out}"; mkdir -p "${out}"
set +u; source "${w}/install/setup.bash"; set -u
export ROS_DOMAIN_ID=33 ROS_LOG_DIR="${out}/roslog" M6A10_BENCHMARK_NO_MAP_ARTIFACTS=1
cp /home/user/bench/params/rko_params.ros.yaml "${out}/rko_params.ros.yaml"
cd /home/user/rsasaki0109/lidar_slam_ros2
setsid ros2 launch lidarslam rko_lio_slam.launch.py "main_param_dir:=${yaml}" "rko_param_file:=${out}/rko_params.ros.yaml" \
  bag_path:=/home/user/data/hilti2022/exp04_ros2 lidar_topic:=/hesai/pandar imu_topic:=/alphasense/imu \
  base_frame:=base_link publish_static_tf:=true "save_dir:=${out}" "results_dir:=${out}" run_name:=front \
  dump_results:=true use_rviz:=false > "${out}/slam.launch.log" 2>&1 &
lpid=$!
t0=$(date +%s.%N)
echo "t,offline_ticks,graph_ticks" > "${out}/cpu.csv"
seen_front=0; front_end=""; idle_run=0; idle_start=""; prev_graph=""
loops_at_front_end=""
for ((s = 1; s <= 1800; s++)); do
  sleep 1
  opid=$(pgrep -f "lib/rko_lio/offline_node( |$)" | head -1)
  gpid=$(pgrep -f "lib/graph_based_slam/graph_based_slam_node( |$)" | head -1)
  ot=""; gt=""
  if [[ -n "${opid}" && -r /proc/${opid}/stat ]]; then read -r -a f < /proc/${opid}/stat; ot=$(( f[13] + f[14] )); seen_front=1; fi
  if [[ -n "${gpid}" && -r /proc/${gpid}/stat ]]; then read -r -a g < /proc/${gpid}/stat; gt=$(( g[13] + g[14] )); fi
  echo "${s},${ot},${gt}" >> "${out}/cpu.csv"
  if [[ ${seen_front} == 1 && -z "${opid}" && -z "${front_end}" ]]; then
    front_end=${s}
    loops_at_front_end=$(grep -c "best_loop_candidate" "${out}/slam.launch.log")
  fi
  if [[ -n "${front_end}" && -n "${gt}" && -n "${prev_graph}" ]]; then
    if (( gt - prev_graph < 5 )); then idle_run=$((idle_run + 1)); else idle_run=0; fi
    if (( idle_run >= 5 )); then idle_start=$((s - 5)); break; fi
  fi
  prev_graph=${gt}
done
kill -INT -- -"${lpid}" 2>/dev/null; sleep 5; kill -KILL -- -"${lpid}" 2>/dev/null
took=$(grep -ohP 'RKO LIO Offline Node took \K[0-9.]+' "${out}/slam.launch.log" | tail -1)
loops_total=$(grep -c "best_loop_candidate" "${out}/slam.launch.log")
tum="${out}/front_0/front_tum_0.txt"
md5=$( [[ -f "${tum}" ]] && md5sum "${tum}" | cut -d' ' -f1 || echo missing)
drain=$( [[ -n "${front_end}" && -n "${idle_start}" ]] && echo $((idle_start - front_end)) || echo NA)
echo "${label},${took:-NA},${front_end:-NA},${idle_start:-NA},${drain},${loops_at_front_end:-NA},${loops_total},${md5}" | tee -a "${csv}"
