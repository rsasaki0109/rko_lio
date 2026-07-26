// Tests for the sliding-window absolute gravity alignment helper.
// MIT License. Copyright (c) 2026 Sasaki.

#include "rko_lio/core/gravity_alignment.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using rko_lio::core::compute_gravity_alignment_correction;

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

} // namespace
