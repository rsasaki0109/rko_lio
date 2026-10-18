#!/usr/bin/env bash
# rko_lio 40b3a06 (reader keeps reading while registration waits for IMU): the MapsHD underground-garage trial-1
# bag (IMU ends 9.9 s before the LiDAR) must now finish, and exp04, MID-360 and Oxford must keep their md5.
set -uo pipefail
P=/home/user/bench/params; O=/home/user/bench/deadlock; WS=/home/user/ws/cand_40b3a06
M=/home/user/data/mid360_public/driving_slam_mid360/extracted/rosbag2_2024_04_16-14_17_01/rosbag2_2024_04_16-14_17_01
mkdir -p ${O}
export ROS_DOMAIN_ID=27 RUN_TIMEOUT=3000
B=/home/user/data/mapshd/bags/underground-garage_trial-1
dd if=$(ls ${B}/*.db3) of=/dev/null bs=16M 2>&1 | tail -1
bash /home/user/scripts/run_front.sh garage1_40b3a06 ${WS} ${P}/mapshd_gyro002.yaml ${B} /livox/pointcloud /livox/imu ${O}/garage1_40b3a06 ${O}/runs.csv | tail -1
grep -E "Bag exhausted|took" ${O}/garage1_40b3a06/stdout.log
dd if=/home/user/data/hilti2022/exp04_ros2/exp04_ros2.db3 of=/dev/null bs=16M 2>&1 | tail -1
bash /home/user/scripts/run_front.sh exp04_40b3a06 ${WS} ${P}/exp04_gyro002.yaml /home/user/data/hilti2022/exp04_ros2 /hesai/pandar /alphasense/imu ${O}/exp04_40b3a06 ${O}/runs.csv | tail -1
dd if=${M}/rosbag2_2024_04_16-14_17_01_0.db3 of=/dev/null bs=16M 2>&1 | tail -1
bash /home/user/scripts/run_front.sh mid360_40b3a06 ${WS} ${P}/mid360_gyro002.yaml ${M} /livox/lidar /livox/imu ${O}/mid360_40b3a06 ${O}/runs.csv | tail -1
dd if=/home/user/data/oxford_keble05/ros2/ros2_0.db3 of=/dev/null bs=16M 2>&1 | tail -1
bash /home/user/scripts/run_front.sh oxford_40b3a06 ${WS} ${P}/oxford_keble05_gyro002.yaml /home/user/data/oxford_keble05/ros2 /hesai/pandar /alphasense_driver_ros/imu ${O}/oxford_40b3a06 ${O}/runs.csv | tail -1
echo "deadlock fix check done $(date -u +%T)"
