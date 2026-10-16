#!/usr/bin/env bash
# Usage: md5_pair.sh <label> <ws> [extra ros args...]  -- runs exp04 and MID-360 once each for the md5 table.
label=$1; ws=$2; shift 2
M=/home/user/data/mid360_public/driving_slam_mid360/extracted/rosbag2_2024_04_16-14_17_01/rosbag2_2024_04_16-14_17_01
R=/home/user/bench/runs/md5
/home/user/scripts/run_front.sh "${label}_exp04" "${ws}" /home/user/bench/params/exp04.yaml /home/user/data/hilti2022/exp04_ros2 /hesai/pandar /alphasense/imu "${R}/${label}_exp04" "${R}/results_md5.csv" "$@"
/home/user/scripts/run_front.sh "${label}_mid360" "${ws}" /home/user/bench/params/mid360.yaml "${M}" /livox/lidar /livox/imu "${R}/${label}_mid360" "${R}/results_md5.csv" "$@"
