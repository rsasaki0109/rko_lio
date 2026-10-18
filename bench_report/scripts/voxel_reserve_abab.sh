#!/usr/bin/env bash
# Frontend only, ABAB: rko_lio 40b3a06 (A) against 77072e4 (B), one run at a time, warm bags.
# exp04 3 pairs, then Oxford Keble 05 2 pairs, frontend-only parameters with the moving-start guard (14 章).
set -uo pipefail
P=/home/user/bench/params; O=/home/user/bench/voxel_abab; mkdir -p ${O}
dd if=/home/user/data/hilti2022/exp04_ros2/exp04_ros2.db3 of=/dev/null bs=16M 2>&1 | tail -1
PARAMS=${P}/exp04_gyro002.yaml bash /home/user/scripts/abab.sh ${O}/exp04.csv 3 \
  a40b3a06 /home/user/ws/cand_40b3a06 b77072e4 /home/user/ws/cand_77072e4
dd if=/home/user/data/oxford_keble05/ros2/$(ls /home/user/data/oxford_keble05/ros2 | grep db3) of=/dev/null bs=16M 2>&1 | tail -1
PARAMS=${P}/oxford_keble05_gyro002.yaml BAG=/home/user/data/oxford_keble05/ros2 LIDAR=/hesai/pandar \
  IMU=/alphasense_driver_ros/imu bash /home/user/scripts/abab.sh ${O}/oxford.csv 2 \
  a40b3a06 /home/user/ws/cand_40b3a06 b77072e4 /home/user/ws/cand_77072e4
python3 /home/user/scripts/summarize_abab.py ${O}/exp04.csv ${O}/oxford.csv
echo "voxel abab done"
