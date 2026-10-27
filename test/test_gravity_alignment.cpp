// Tests for the sliding-window absolute gravity alignment helper.
// MIT License. Copyright (c) 2026 Sasaki.

#include "rko_lio/core/gravity_alignment.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using rko_lio::core::compute_gravity_alignment_correction;
using rko_lio::core::estimate_world_up_while_moving;
using rko_lio::core::limit_tilt_correction;
using rko_lio::core::WorldSpecificForceInterval;

constexpr double kGravity = 9.8107;

TEST(GravityAlignment, AlignedMeanYieldsIdentity) {
  const Eigen::Vector3d mean(0.0, 0.0, kGravity);
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, 0.05, 0.002);
  EXPECT_TRUE(result.valid);
  EXPECT_NEAR(result.tilt_rad, 0.0, 1e-12);
  EXPECT_NEAR(result.correction.log().norm(), 0.0, 1e-12);
}

TEST(GravityAlignment, TiltedMeanCorrectsTowardZ) {
  const double tilt = 0.05; // rad, ~2.9 deg
  const Eigen::Vector3d mean(kGravity * std::sin(tilt), 0.0, kGravity * std::cos(tilt));
  const double gain = 0.1;
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, gain, 0.1);
  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.tilt_rad, tilt, 1e-9);
  EXPECT_NEAR(result.correction.log().norm(), gain * tilt, 1e-9);
  // Applying the correction must reduce the measured tilt.
  const Eigen::Vector3d corrected = result.correction * mean;
  const double corrected_tilt = std::acos(corrected.normalized().z());
  EXPECT_NEAR(corrected_tilt, (1.0 - gain) * tilt, 1e-6);
}

TEST(GravityAlignment, CorrectionAngleIsCapped) {
  const double tilt = 0.5; // large drift
  const Eigen::Vector3d mean(kGravity * std::sin(tilt), 0.0, kGravity * std::cos(tilt));
  const double cap = 0.002;
  // Plausibility bound raised explicitly so the cap itself is what is tested.
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, 1.0, cap, 1.0);
  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.correction.log().norm(), cap, 1e-12);
}

TEST(GravityAlignment, MagnitudeGateRejectsSustainedAcceleration) {
  // 20% away from |g|: sustained braking/accelerating, not tilt.
  const Eigen::Vector3d mean(0.0, 0.0, 1.2 * kGravity);
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, 0.05, 0.002);
  EXPECT_FALSE(result.valid);
  EXPECT_NEAR(result.correction.log().norm(), 0.0, 1e-12);
}

TEST(GravityAlignment, ImplausiblyLargeTiltRejected) {
  // A ~30 deg "tilt" cannot be drift while the correction is active; it is a
  // contaminated window and must be rejected outright.
  const double tilt = 0.5;
  const Eigen::Vector3d mean(kGravity * std::sin(tilt), 0.0, kGravity * std::cos(tilt));
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, 1.0, 0.002, 0.175);
  EXPECT_FALSE(result.valid);
  EXPECT_NEAR(result.tilt_rad, tilt, 1e-9);
  EXPECT_NEAR(result.correction.log().norm(), 0.0, 1e-12);
}

TEST(GravityAlignment, ZeroMeanRejected) {
  const auto result = compute_gravity_alignment_correction(Eigen::Vector3d::Zero(), kGravity, 0.05, 0.05, 0.002);
  EXPECT_FALSE(result.valid);
}

TEST(GravityAlignment, YawIsNeverTouched) {
  const double tilt = 0.05;
  const Eigen::Vector3d mean(kGravity * std::sin(tilt), 0.0, kGravity * std::cos(tilt));
  const auto result = compute_gravity_alignment_correction(mean, kGravity, 0.05, 0.5, 0.1);
  ASSERT_TRUE(result.valid);
  // The correction axis lies in the horizontal plane: no z (yaw) component.
  EXPECT_NEAR(result.correction.log().z(), 0.0, 1e-12);
}

TEST(TiltCorrectionLimit, SmallCorrectionPassesThrough) {
  const Sophus::SO3d predicted = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3));
  const Sophus::SO3d registered = Sophus::SO3d::exp(Eigen::Vector3d(1e-4, 0.0, 0.01)) * predicted;
  bool limited = true;
  const Sophus::SO3d result = limit_tilt_correction(predicted, registered, 1e-3, &limited);
  EXPECT_FALSE(limited);
  EXPECT_NEAR((result.inverse() * registered).log().norm(), 0.0, 1e-12);
}

TEST(TiltCorrectionLimit, TiltIsCappedAndYawKept) {
  const Sophus::SO3d predicted = Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.02, -1.0));
  const Eigen::Vector3d correction(0.01, -0.02, 0.05);  // world frame: pitch/roll pull and a turn
  const Sophus::SO3d registered = Sophus::SO3d::exp(correction) * predicted;
  bool limited = false;
  const double cap = 3e-4;
  const Sophus::SO3d result = limit_tilt_correction(predicted, registered, cap, &limited);
  EXPECT_TRUE(limited);
  const Eigen::Vector3d applied = (result * predicted.inverse()).log();
  EXPECT_NEAR(applied.head<2>().norm(), cap, 1e-9);
  EXPECT_NEAR(applied.z(), correction.z(), 1e-9);
  // The capped tilt keeps the direction of the registration's pull.
  EXPECT_NEAR(applied.head<2>().normalized().dot(correction.head<2>().normalized()), 1.0, 1e-9);
}

// A walk that speeds up from 0.2 to 0.8 m/s with a 2-Hz gait sway (world frame).
Eigen::Vector3d walk_velocity(const double t) {
  const double w = 2.0 * M_PI * 2.0;
  return {0.2 + 0.1 * t + 0.08 * std::cos(w * t), 0.05 * std::cos(1.7 * w * t), 0.04 * std::cos(w * t)};
}

Eigen::Vector3d walk_position(const double t) {
  const double w = 2.0 * M_PI * 2.0;
  return {0.2 * t + 0.05 * t * t + 0.08 / w * std::sin(w * t), 0.05 / (1.7 * w) * std::sin(1.7 * w * t),
          0.04 / w * std::sin(w * t)};
}

TEST(GravityAlignment, WalkingStartRemovesTheVelocityChange) {
  // World frame tilted 17 deg from level, as a walking start's initialization can leave it.
  const Eigen::Vector3d up = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.22, 0.0)) * Eigen::Vector3d::UnitZ();
  std::vector<WorldSpecificForceInterval> intervals;
  for (int i = 0; i < 80; ++i) {
    const double t = 0.1 * i;
    // the interval's mean acceleration is its velocity change over its length
    const Eigen::Vector3d mean_acceleration = (walk_velocity(t + 0.1) - walk_velocity(t)) / 0.1;
    intervals.push_back({t + 0.05, 0.1, mean_acceleration + kGravity * up});
  }
  const auto estimate = estimate_world_up_while_moving(intervals, walk_position, 1.0, 6.0, 0.5);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_LT(std::acos(std::min(1.0, estimate->normalized().dot(up))), 0.05 * M_PI / 180.0);
  EXPECT_NEAR(estimate->norm(), kGravity, 0.01);

  // The plain mean of the specific force keeps the walk's speed-up (0.1 m/s^2, about 0.6 deg).
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& interval : intervals) mean += interval.specific_force / static_cast<double>(intervals.size());
  EXPECT_GT(std::acos(std::min(1.0, mean.normalized().dot(up))), 0.3 * M_PI / 180.0);
}

TEST(GravityAlignment, WalkingStartNeedsIntervalsCoveringTheWindow) {
  std::vector<WorldSpecificForceInterval> intervals;
  for (int i = 0; i < 60; ++i) intervals.push_back({0.1 * i + 0.05, 0.1, kGravity * Eigen::Vector3d::UnitZ()});
  // The intervals end at 6 s: a window ending at 6 s needs them up to 6.5 s.
  EXPECT_FALSE(estimate_world_up_while_moving(intervals, walk_position, 1.0, 6.0, 0.5).has_value());
  EXPECT_TRUE(estimate_world_up_while_moving(intervals, walk_position, 1.0, 5.5, 0.5).has_value());
}

} // namespace
