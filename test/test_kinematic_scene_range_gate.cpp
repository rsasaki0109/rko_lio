#include <gtest/gtest.h>

#include "rko_lio/core/kinematic_scene_range_gate.hpp"

#include <Eigen/Core>

#include <limits>
#include <vector>

namespace {

using rko_lio::core::evaluate_kinematic_scene_range_support;
using rko_lio::core::update_kinematic_scene_reenable_gate;
using rko_lio::core::update_kinematic_persistent_speed_gate;
using rko_lio::core::update_kinematic_speed_reenable_gate;

std::vector<Eigen::Vector3d> scan_with_ranges(
    const std::size_t near_count,
    const std::size_t middle_count,
    const std::size_t far_count) {
  std::vector<Eigen::Vector3d> points;
  points.reserve(near_count + middle_count + far_count);
  points.insert(points.end(), near_count, Eigen::Vector3d(2.0, 0.0, 0.0));
  points.insert(points.end(), middle_count, Eigen::Vector3d(5.0, 0.0, 0.0));
  points.insert(points.end(), far_count, Eigen::Vector3d(15.0, 0.0, 0.0));
  return points;
}

TEST(KinematicSceneRangeGate, StructuralTunnelSupportPasses) {
  const auto result = evaluate_kinematic_scene_range_support(
      scan_with_ranges(20, 70, 10), 0.7, 100.0, 3.0, 0.5, 10.0, 0.05, 100);
  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(result.trusted);
  EXPECT_DOUBLE_EQ(result.near_fraction, 0.2);
  EXPECT_DOUBLE_EQ(result.far_fraction, 0.1);
}

TEST(KinematicSceneRangeGate, NearFieldClutterIsRejected) {
  const auto result = evaluate_kinematic_scene_range_support(
      scan_with_ranges(90, 10, 0), 0.7, 100.0, 3.0, 0.5, 10.0, 0.05, 100);
  ASSERT_TRUE(result.valid);
  EXPECT_FALSE(result.trusted);
  EXPECT_DOUBLE_EQ(result.near_fraction, 0.9);
  EXPECT_DOUBLE_EQ(result.far_fraction, 0.0);
}

TEST(KinematicSceneRangeGate, MissingLongRangeStructureIsRejected) {
  const auto result = evaluate_kinematic_scene_range_support(
      scan_with_ranges(20, 80, 0), 0.7, 100.0, 3.0, 0.5, 10.0, 0.05, 100);
  ASSERT_TRUE(result.valid);
  EXPECT_FALSE(result.trusted);
}

TEST(KinematicSceneRangeGate, TooFewValidPointsFailClosed) {
  auto points = scan_with_ranges(1, 1, 1);
  points.emplace_back(Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()));
  const auto result = evaluate_kinematic_scene_range_support(
      points, 0.7, 100.0, 3.0, 0.5, 10.0, 0.05, 100);
  EXPECT_FALSE(result.valid);
  EXPECT_FALSE(result.trusted);
  EXPECT_EQ(result.valid_point_count, 3U);
}

TEST(KinematicSceneRangeGate, InvalidThresholdOrderingFailsClosed) {
  const auto result = evaluate_kinematic_scene_range_support(
      scan_with_ranges(20, 70, 10), 0.7, 100.0, 10.0, 0.5, 3.0, 0.05, 100);
  EXPECT_FALSE(result.valid);
  EXPECT_FALSE(result.trusted);
}

TEST(KinematicSceneRangeGate, SpeedGateRequiresPersistentEnvelopeViolation) {
  const auto first =
      update_kinematic_persistent_speed_gate(4.0, 2.5, 3, 0);
  const auto second =
      update_kinematic_persistent_speed_gate(4.0, 2.5, 3, first.streak);
  const auto third =
      update_kinematic_persistent_speed_gate(4.0, 2.5, 3, second.streak);
  EXPECT_FALSE(first.rejected);
  EXPECT_FALSE(second.rejected);
  EXPECT_TRUE(third.rejected);
}

TEST(KinematicSceneRangeGate, SpeedGateResetsAfterInEnvelopeScan) {
  const auto result =
      update_kinematic_persistent_speed_gate(1.7, 2.5, 3, 2);
  EXPECT_EQ(result.streak, 0U);
  EXPECT_FALSE(result.rejected);
}

TEST(KinematicSceneRangeGate, DisabledSpeedGateIsNeutral) {
  const auto result =
      update_kinematic_persistent_speed_gate(10.0, 0.0, 3, 2);
  EXPECT_EQ(result.streak, 0U);
  EXPECT_FALSE(result.rejected);
}

TEST(KinematicSceneRangeGate, SceneRejectionStartsReenableCooldown) {
  const auto rejected =
      update_kinematic_scene_reenable_gate(false, 10.0, 60.0, -1.0);
  EXPECT_TRUE(rejected.rejected);
  EXPECT_FALSE(rejected.cooldown_active);
  EXPECT_DOUBLE_EQ(rejected.last_rejected_time_sec, 10.0);

  const auto early =
      update_kinematic_scene_reenable_gate(
          true, 69.9, 60.0, rejected.last_rejected_time_sec);
  EXPECT_TRUE(early.rejected);
  EXPECT_TRUE(early.cooldown_active);

  const auto ready =
      update_kinematic_scene_reenable_gate(
          true, 70.0, 60.0, rejected.last_rejected_time_sec);
  EXPECT_FALSE(ready.rejected);
  EXPECT_FALSE(ready.cooldown_active);
}

TEST(KinematicSceneRangeGate, TrustedSceneWithoutPriorRejectionIsImmediate) {
  const auto result =
      update_kinematic_scene_reenable_gate(true, 1.0, 60.0, -1.0);
  EXPECT_FALSE(result.rejected);
  EXPECT_FALSE(result.cooldown_active);
}

TEST(KinematicSceneRangeGate, SpeedReenableCooldownRejectsShortSlowdown) {
  const auto high =
      update_kinematic_speed_reenable_gate(false, 70.0, 60.0, -1.0);
  const auto slowdown =
      update_kinematic_speed_reenable_gate(
          true, 74.7, 60.0, high.last_rejected_time_sec);
  EXPECT_TRUE(slowdown.rejected);
  EXPECT_TRUE(slowdown.cooldown_active);
}

} // namespace
