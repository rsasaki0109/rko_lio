#!/usr/bin/env bash
# Usage: sample_proc_cpu.sh <out.csv> <seconds>
# Every second, records utime+stime (clock ticks) of offline_node and graph_based_slam_node.
out=$1; dur=$2
echo "t,proc,pid,cpu_ticks,threads" > "${out}"
for ((s = 0; s < dur; s++)); do
  for name in offline_node graph_based_slam_node; do
    for pid in $(pgrep -f "lib/[a-z_]*/${name}( |$)" 2>/dev/null); do
      if [[ -r /proc/${pid}/stat ]]; then
        read -r -a f < /proc/${pid}/stat
        echo "${s},${name},${pid},$(( f[13] + f[14] )),${f[19]}" >> "${out}"
      fi
    done
  done
  sleep 1
done
