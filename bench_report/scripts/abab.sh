#!/usr/bin/env bash
# Usage: [PARAMS=.. BAG=.. LIDAR=.. IMU=..] abab.sh <csv> <pairs> <labelA> <wsA> <labelB> <wsB>
# Runs A,B alternately <pairs> times on exp04 by default (frontend only), one run at a time.
csv=$1; pairs=$2; la=$3; wa=$4; lb=$5; wb=$6
R=/home/user/bench/runs/abab
mkdir -p "${R}"
for ((i = 1; i <= pairs; i++)); do
  for side in A B; do
    if [[ ${side} == A ]]; then l=${la}; w=${wa}; else l=${lb}; w=${wb}; fi
    sleep "${PAUSE:-15}"
    out="${R}/$(basename "${csv}" .csv)_${l}_${i}"
    ROS_DOMAIN_ID=21 /home/user/scripts/run_front.sh "${l},${i}" "${w}" "${PARAMS:-/home/user/bench/params/exp04.yaml}" \
      "${BAG:-/home/user/data/hilti2022/exp04_ros2}" "${LIDAR:-/hesai/pandar}" "${IMU:-/alphasense/imu}" "${out}" "${csv}" > /dev/null
    icp=$(grep -A2 "^	ICP:" "${out}/stdout.log" | grep -oP 'Average time: \K[0-9.]+')
    reg=$(grep -A2 "ROS Registration Loop:" "${out}/stdout.log" | grep -oP 'Average time: \K[0-9.]+')
    sed -i '$ s/$/,'"${icp:-NA},${reg:-NA}"'/' "${csv}"
    tail -1 "${csv}"
    rm -rf "${out}/roslog"
  done
done
