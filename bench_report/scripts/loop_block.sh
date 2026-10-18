#!/usr/bin/env bash
# Loop closure enabled: full-pipeline ABAB (benchmark script, --skip-map-save) and backend drain checks.
W=/home/user/ws
P=/home/user/bench/pipe_loop
LOOP=/home/user/bench/params/lidarslam_competitive_v2_loop.yaml
OFF=/home/user/rsasaki0109/lidar_slam_ros2/configs/hilti2022/lidarslam_competitive_v2.yaml
echo "start $(date -u +%H:%M:%S)"
LIDARSLAM_PARAM=${LOOP} PIPE_OUT=${P} /home/user/scripts/pipe_abab_cfg.sh ${P}/loop1_base_f6b0425.csv 3 80f3c1c ${W}/pipe_base f6b0425 ${W}/pipe_f6b0425
LIDARSLAM_PARAM=${LOOP} PIPE_OUT=${P} /home/user/scripts/pipe_abab_cfg.sh ${P}/loop2_f6b0425_734d54f.csv 3 f6b0425 ${W}/pipe_f6b0425 734d54f ${W}/pipe_734d54f
echo "abab done $(date -u +%H:%M:%S)"
for spec in 80f3c1c_loop:pipe_base:${LOOP} 734d54f_loop:pipe_734d54f:${LOOP} f6b0425_off:pipe_f6b0425:${OFF} 734d54f_off:pipe_734d54f:${OFF}; do
  IFS=: read -r label ws yaml <<< "${spec}"
  sleep 20
  /home/user/scripts/pipe_drain.sh ${P}/drain.csv ${label} ${W}/${ws} ${yaml} ${P}/drain_${label}
done
echo "drain done $(date -u +%H:%M:%S)"
