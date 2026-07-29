#include <gtest/gtest.h>

#include "rko_lio/core/kinematic_velocity_blend.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace {

using rko_lio::core::blend_icp_with_propagated_velocity;
using rko_lio::core::rotate_kinematic_velocity_prior;
using rko_lio::core::should_bridge_low_speed_with_inertial_activity;

constexpr double kDt = 0.1;

TEST(KinematicVelocityBlend, AgreementRefreshesAnchorWithoutChangingIcp) {
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d(0.169, 0.0, 0.0), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  ASSERT_TRUE(blend.valid);
  EXPECT_TRUE(blend.anchor_agrees);
  EXPECT_DOUBLE_EQ(blend.propagation_weight, 0.0);
  EXPECT_TRUE(blend.correction.isZero());
}

TEST(KinematicVelocityBlend, FreshPriorSoftlyBridgesFreeze) {
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  ASSERT_TRUE(blend.valid);
  EXPECT_FALSE(blend.anchor_agrees);
  EXPECT_NEAR(blend.propagation_weight, 0.9, 1.0e-12);
  EXPECT_NEAR(blend.correction.x(), 0.9 * 1.7 * kDt, 1.0e-12);
}

TEST(KinematicVelocityBlend, AuthorityDecaysSinceLastAnchor) {
  const auto fresh = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 1.0, 0.05, 2.0, 0.3);
  const auto stale = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 3.0, 1.0, 9.0, 1.0, 0.05, 2.0, 0.3);
  EXPECT_GT(fresh.propagation_weight, stale.propagation_weight);
  EXPECT_LT(stale.propagation_weight, 0.4);
}

TEST(KinematicVelocityBlend, CrossAxisIcpVelocityIsAlsoFusedTowardZero) {
  const Eigen::Vector3d icp_step(0.0, 0.07, -0.03);
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), icp_step, kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  ASSERT_GT(blend.propagation_weight, 0.0);
  const Eigen::Vector3d corrected = icp_step + blend.correction;
  EXPECT_NEAR(corrected.y(), 0.1 * icp_step.y(), 1.0e-12);
  EXPECT_NEAR(corrected.z(), 0.1 * icp_step.z(), 1.0e-12);
}

TEST(KinematicVelocityBlend, UsesCallerPropagatedVelocityPrediction) {
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.1, 0.0, 0.0), kDt,
      Eigen::Vector3d(1.2, 0.0, 0.0), 0.0, 1.0, 1.0, 2.0, 0.05, 2.0, 0.3);
  ASSERT_FALSE(blend.anchor_agrees);
  EXPECT_NEAR(blend.correction.x(), 0.5 * 0.2 * kDt, 1.0e-12);
}

TEST(KinematicVelocityBlend, HugeIcpOutlierIsRobustifiedBeforeFusion) {
  const Eigen::Vector3d icp_step(10.0, 0.0, 0.0); // raw ICP claims 100 m/s
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), icp_step, kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  ASSERT_NEAR(blend.propagation_weight, 0.9, 1.0e-12);
  const double fused_speed = (icp_step.x() + blend.correction.x()) / kDt;
  // prior + (1 - weight) * capped innovation = 1.7 + 0.1 * 2.0
  EXPECT_NEAR(fused_speed, 1.9, 1.0e-12);
}

TEST(KinematicVelocityBlend, HugeCrossAxisOutlierCannotMasqueradeAsAgreement) {
  const Eigen::Vector3d icp_step(0.17, 10.0, 0.0);
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), icp_step, kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  EXPECT_FALSE(blend.anchor_agrees);
  const Eigen::Vector3d fused_velocity = (icp_step + blend.correction) / kDt;
  EXPECT_NEAR(fused_velocity.x(), 1.7, 1.0e-9);
  EXPECT_NEAR(fused_velocity.y(), 0.2, 1.0e-9);
}

TEST(KinematicVelocityBlend, BelowMinSpeedDoesNotCreateStartupDirection) {
  const auto blend = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(0.1, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(0.6, 0.0, 0.0), 0.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  EXPECT_FALSE(blend.valid);
}

TEST(KinematicVelocityBlend, MissingAnchorOrInvalidInformationCannotCorrect) {
  const auto no_anchor = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), -1.0, 1.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  // The comparison itself is valid (and could establish an agreement anchor),
  // but a disagreement without an existing anchor has no authority to correct.
  EXPECT_TRUE(no_anchor.valid);
  EXPECT_TRUE(no_anchor.correction.isZero());

  const auto bad_information = blend_icp_with_propagated_velocity(
      Eigen::Vector3d(1.7, 0.0, 0.0), Eigen::Vector3d::Zero(), kDt,
      Eigen::Vector3d(1.7, 0.0, 0.0), 0.0, 0.0, 9.0, 2.0, 0.05, 2.0, 0.3);
  EXPECT_TRUE(bad_information.valid);
  EXPECT_TRUE(bad_information.correction.isZero());
}

TEST(KinematicVelocityBlend, WalkingActivityBridgesAnAnchoredLowSpeedLock) {
  EXPECT_TRUE(should_bridge_low_speed_with_inertial_activity(
      0.08, 0.3, 0.19, 0.03, true, true));
  EXPECT_FALSE(should_bridge_low_speed_with_inertial_activity(
      0.08, 0.3, 0.0015, 0.03, true, true));
}

TEST(KinematicVelocityBlend, ActivityBridgeFailsClosedWithoutTrustedState) {
  EXPECT_FALSE(should_bridge_low_speed_with_inertial_activity(
      0.08, 0.3, 0.19, 0.03, false, true));
  EXPECT_FALSE(should_bridge_low_speed_with_inertial_activity(
      0.08, 0.3, 0.19, 0.03, true, false));
  EXPECT_FALSE(should_bridge_low_speed_with_inertial_activity(
      0.08, 0.3, 0.19, 0.0, true, true));
}

TEST(KinematicVelocityBlend, YawRotationPreservesPriorSpeed) {
  constexpr double kHalfPi = 1.5707963267948966;
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(kHalfPi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d rotated =
      rotate_kinematic_velocity_prior(rotation, Eigen::Vector3d(1.7, 0.0, 0.0));
  EXPECT_NEAR(rotated.x(), 0.0, 1.0e-12);
  EXPECT_NEAR(rotated.y(), 1.7, 1.0e-12);
  EXPECT_NEAR(rotated.z(), 0.0, 1.0e-12);
  EXPECT_NEAR(rotated.norm(), 1.7, 1.0e-12);
}

} // namespace
