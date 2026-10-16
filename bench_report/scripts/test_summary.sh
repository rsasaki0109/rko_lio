#!/usr/bin/env bash
# Per build: ctest totals and the gtest cases of test_voxel_hash_map.
set +u; source /opt/ros/jazzy/setup.bash
for n in 80f3c1c f6b0425 f089726 5a2e24c 3d520b1 634393a 0df11e8 f4a6547 734d54f; do
  log=/home/user/bench/tests/ctest_$n.log
  total=$(grep -oP 'tests passed, \K[0-9]+ tests failed out of [0-9]+' "$log")
  vhm=$(/home/user/ws/test_$n/build/rko_lio/test_voxel_hash_map --gtest_brief=1 2>&1 | grep -oP '\[==========\] \K[0-9]+ tests? from')
  echo "$n | $(grep -oP '[0-9]+% tests passed' "$log") | $total | test_voxel_hash_map: $vhm"
done
