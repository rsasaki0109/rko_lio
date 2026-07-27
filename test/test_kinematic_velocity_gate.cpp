// Unit tests for the accelerometer-consistency velocity gate
// (kinematic_velocity_gate.hpp).
#include <gtest/gtest.h>

#include "rko_lio/core/kinematic_velocity_gate.hpp"

#include <Eigen/Core>

namespace {

using rko_lio::core::clamp_velocity_change_to_accel;

constexpr double kDt = 0.1;

TEST(KinematicVelocityGate, FreezeStyleDecelerationIsClamped) {
  // Walking at 1.7 m/s; ICP claims a full stop in one 0.1 s scan (17 m/s^2)
  // while the accelerometer measures only perpendicular (vertical) vibration:
  // the axis projection is zero, so only the margin opens the bound.
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(0.0, 0.0, 0.0);
  const Eigen::Vector3d vibration(0.0, 0.0, 0.5);
  const auto clamp = clamp_velocity_change_to_accel(v_prev, icp_step, kDt, vibration, 0.3, 0.3, 1.0);
  ASSERT_TRUE(clamp.corrected);
  EXPECT_NEAR(clamp.implied_accel, 17.0, 1.0e-9);
  EXPECT_NEAR(clamp.allowed_accel, 0.3, 1.0e-9);
  // Corrected step must decelerate at exactly the bound: new speed 1.7 - 0.03.
  const Eigen::Vector3d corrected_step = icp_step + clamp.correction;
  EXPECT_NEAR(corrected_step.x() / kDt, 1.7 - 0.3 * kDt, 1.0e-9);
  EXPECT_NEAR(corrected_step.y(), 0.0, 1.0e-12);
  EXPECT_NEAR(corrected_step.z(), 0.0, 1.0e-12);
}

TEST(KinematicVelocityGate, GenuineBrakingPassesBecauseAccelIsMeasured) {
  // Same full-stop claim, but the accelerometer actually measured the braking.
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(0.0, 0.0, 0.0);
  const auto clamp =
      clamp_velocity_change_to_accel(v_prev, icp_step, kDt, Eigen::Vector3d(-17.0, 0.0, 0.5), 1.0, 0.3, 1.0);
  EXPECT_FALSE(clamp.corrected);
}

TEST(KinematicVelocityGate, ConsistentMotionIsUntouched) {
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(1.68 * kDt, 0.01, 0.0); // 0.2 m/s^2 change
  const auto clamp =
      clamp_velocity_change_to_accel(v_prev, icp_step, kDt, Eigen::Vector3d(0.0, 0.0, 0.5), 0.3, 0.3, 1.0);
  EXPECT_FALSE(clamp.corrected);
}

TEST(KinematicVelocityGate, RunawayAccelerationIsClampedSymmetrically) {
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(3.4 * kDt, 0.0, 0.0); // +17 m/s^2 jump
  const auto clamp =
      clamp_velocity_change_to_accel(v_prev, icp_step, kDt, Eigen::Vector3d(0.0, 0.0, 0.5), 0.3, 0.3, 1.0);
  ASSERT_TRUE(clamp.corrected);
  const Eigen::Vector3d corrected_step = icp_step + clamp.correction;
  EXPECT_NEAR(corrected_step.x() / kDt, 1.7 + 0.3 * kDt, 1.0e-9);
}

TEST(KinematicVelocityGate, LateralMotionIsNeverTouched) {
  // The clamp acts only along the previous motion direction: a lateral ICP
  // component is preserved exactly even when the along-axis part is clamped.
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(0.0, 0.05, -0.02);
  const auto clamp =
      clamp_velocity_change_to_accel(v_prev, icp_step, kDt, Eigen::Vector3d(0.0, 0.0, 0.5), 0.3, 0.3, 1.0);
  ASSERT_TRUE(clamp.corrected);
  const Eigen::Vector3d corrected_step = icp_step + clamp.correction;
  EXPECT_NEAR(corrected_step.y(), 0.05, 1.0e-12);
  EXPECT_NEAR(corrected_step.z(), -0.02, 1.0e-12);
}

TEST(KinematicVelocityGate, BelowMinSpeedOrBadInputsAreNoOps) {
  const Eigen::Vector3d icp_step(0.0, 0.0, 0.0);
  const Eigen::Vector3d accel(0.0, 0.0, 0.5);
  EXPECT_FALSE(
      clamp_velocity_change_to_accel(Eigen::Vector3d(0.1, 0.0, 0.0), icp_step, kDt, accel, 0.3, 0.3, 1.0).corrected);
  EXPECT_FALSE(
      clamp_velocity_change_to_accel(Eigen::Vector3d(1.7, 0.0, 0.0), icp_step, 0.0, accel, 0.3, 0.3, 1.0).corrected);
  EXPECT_FALSE(
      clamp_velocity_change_to_accel(Eigen::Vector3d(1.7, 0.0, 0.0), icp_step, kDt, accel, 0.3, 0.3, 0.0).corrected);
}

TEST(KinematicVelocityGate, WeightScalesTheCorrection) {
  const Eigen::Vector3d v_prev(1.7, 0.0, 0.0);
  const Eigen::Vector3d icp_step(0.0, 0.0, 0.0);
  const Eigen::Vector3d accel(0.0, 0.0, 0.5);
  const auto full = clamp_velocity_change_to_accel(v_prev, icp_step, kDt, accel, 0.3, 0.3, 1.0);
  const auto half = clamp_velocity_change_to_accel(v_prev, icp_step, kDt, accel, 0.3, 0.3, 0.5);
  ASSERT_TRUE(full.corrected);
  ASSERT_TRUE(half.corrected);
  EXPECT_NEAR(half.correction.norm(), 0.5 * full.correction.norm(), 1.0e-12);
}

} // namespace
