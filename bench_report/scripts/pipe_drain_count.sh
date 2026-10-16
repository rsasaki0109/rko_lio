#!/usr/bin/env bash
# Usage: pipe_drain_count.sh <csv> <label> <ws> <lidarslam_yaml> <out_dir>
# Like pipe_drain.sh (same launch, backend kept alive until idle), then calls /map_save once and counts
# the submaps in the saved trajectory_optimized.tum. Map artifacts are written (no M6A10 no-map marker).
csv=$1; label=$2; w=$3; yaml=$4; out=$5
rm -rf "${out}"; mkdir -p "${out}"
set +u; source "${w}/install/setup.bash"; set -u
export ROS_DOMAIN_ID=34 ROS_LOG_DIR="${out}/roslog"
cp /home/user/bench/params/rko_params.ros.yaml "${out}/rko_params.ros.yaml"
cd /home/user/rsasaki0109/lidar_slam_ros2
setsid ros2 launch lidarslam rko_lio_slam.launch.py "main_param_dir:=${yaml}" "rko_param_file:=${out}/rko_params.ros.yaml" \
  bag_path:=/home/user/data/hilti2022/exp04_ros2 lidar_topic:=/hesai/pandar imu_topic:=/alphasense/imu \
  base_frame:=base_link publish_static_tf:=true "save_dir:=${out}" "results_dir:=${out}" run_name:=front \
  dump_results:=true use_rviz:=false > "${out}/slam.launch.log" 2>&1 &
lpid=$!
seen_front=0; front_end=""; idle_run=0; idle_start=""; prev_graph=""
for ((s = 1; s <= 1800; s++)); do
  sleep 1
  opid=$(pgrep -f "lib/rko_lio/offline_node( |$)" | head -1)
  gpid=$(pgrep -f "lib/graph_based_slam/graph_based_slam_node( |$)" | head -1)
  gt=""
  [[ -n "${opid}" ]] && seen_front=1
  if [[ -n "${gpid}" && -r /proc/${gpid}/stat ]]; then read -r -a g < /proc/${gpid}/stat; gt=$(( g[13] + g[14] )); fi
  if [[ ${seen_front} == 1 && -z "${opid}" && -z "${front_end}" ]]; then front_end=${s}; fi
  if [[ -n "${front_end}" && -n "${gt}" && -n "${prev_graph}" ]]; then
    if (( gt - prev_graph < 5 )); then idle_run=$((idle_run + 1)); else idle_run=0; fi
    if (( idle_run >= 5 )); then idle_start=$((s - 5)); break; fi
  fi
  prev_graph=${gt}
done
timeout 120 ros2 service call /map_save std_srvs/srv/Empty > "${out}/map_save_call.log" 2>&1
for ((s = 0; s < 600; s++)); do [[ -s "${out}/trajectory_optimized.tum" ]] && break; sleep 1; done
sleep 5
kill -INT -- -"${lpid}" 2>/dev/null; sleep 5; kill -KILL -- -"${lpid}" 2>/dev/null
submaps=$( [[ -f "${out}/trajectory_optimized.tum" ]] && grep -vc '^#' "${out}/trajectory_optimized.tum" || echo NA)
loops=$(grep -c "best_loop_candidate" "${out}/slam.launch.log")
echo "${label},${front_end:-NA},${idle_start:-NA},${submaps},${loops}" | tee -a "${csv}"
du -sh "${out}" | cut -f1
