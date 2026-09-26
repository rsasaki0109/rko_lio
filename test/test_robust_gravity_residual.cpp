#include <gtest/gtest.h>
#include "rko_lio/core/robust_gravity_residual.hpp"
#include <cmath>

using namespace rko_lio::core;

TEST(RobustGravityResidual, ZeroAndSmallResidualRetainInformation) {
  const auto zero = robust_gravity_residual(0.0);
  EXPECT_DOUBLE_EQ(zero.weight, 1.0);
  EXPECT_DOUBLE_EQ(zero.cost, 0.0);
  const auto small = robust_gravity_residual(0.01);
  EXPECT_GT(small.weight, 0.999);
  EXPECT_NEAR(small.cost, 0.005, 1e-6);
}
TEST(RobustGravityResidual, RejectsDirectionOutlierEvenAtCorrectMagnitude) {
  const double angle = 72.0 * std::acos(-1.0) / 180.0;
  const double squared_error = 2 * GRAVITY_MAG * GRAVITY_MAG * (1 - std::cos(angle));
  EXPECT_DOUBLE_EQ(robust_gravity_residual(squared_error).weight, 0.0);
}
TEST(RobustGravityResidual, CostGradientMatchesIRLSWeight) {
  for (double fraction : {0.01, 0.2, 0.5, 0.9, 1.1}) {
    const double residual = fraction * GRAVITY_MAG;
    const double eps = 1e-5;
    const double derivative = (robust_gravity_residual((residual + eps) * (residual + eps)).cost -
                               robust_gravity_residual((residual - eps) * (residual - eps)).cost) / (2 * eps);
    EXPECT_NEAR(derivative, robust_gravity_residual(residual * residual).weight * residual, 1e-7);
  }
}
TEST(RobustGravityResidual, CutoffIsContinuousAndCostBounded) {
  const double limit = GRAVITY_MAG * GRAVITY_MAG;
  const auto at = robust_gravity_residual(limit);
  const auto below = robust_gravity_residual(limit * (1.0 - 1e-8));
  EXPECT_DOUBLE_EQ(at.weight, 0.0);
  EXPECT_NEAR(below.weight, 0.0, 1e-14);
  EXPECT_NEAR(below.cost, at.cost, 1e-12);
  EXPECT_DOUBLE_EQ(robust_gravity_residual(100 * limit).cost, at.cost);
}

TEST(RobustGravityDirection, DirectionConsistentReferenceIgnoresMagnitude) {
  const Eigen::Vector3d up(0, 0, GRAVITY_MAG);
  const auto reference = robust_gravity_direction(up, Eigen::Vector3d(0.3, 0, GRAVITY_MAG));
  for (double scale : {0.25, 0.5, 1.0, 2.0, 4.0}) {
    const auto actual = robust_gravity_direction(up, scale * Eigen::Vector3d(0.3, 0, GRAVITY_MAG));
    EXPECT_TRUE(actual.residual.isApprox(reference.residual, 1e-12));
    EXPECT_NEAR(actual.loss.weight, reference.loss.weight, 1e-12);
    EXPECT_GT(actual.loss.weight, 0.99);
  }
}
TEST(RobustGravityDirection, RejectsLargeAngleRegardlessOfMagnitude) {
  const Eigen::Vector3d up(0, 0, GRAVITY_MAG);
  const auto tilted = Sophus::SO3d::exp(Eigen::Vector3d(1.3, 0, 0)) * up;
  for (double scale : {0.5, 1.0, 3.0})
    EXPECT_DOUBLE_EQ(robust_gravity_direction(up, scale * tilted).loss.weight, 0.0);
}
TEST(RobustGravityDirection, ZeroMeasurementProvidesNoConstraint) {
  const auto loss = robust_gravity_direction(Eigen::Vector3d(0, 0, GRAVITY_MAG), Eigen::Vector3d::Zero());
  EXPECT_DOUBLE_EQ(loss.loss.weight, 0.0);
  EXPECT_TRUE(loss.residual.isZero());
}
TEST(RobustGravityDirection, LeftRotationCostGradientMatchesLinearSystem) {
  const Eigen::Vector3d up(0, 0, GRAVITY_MAG);
  const auto R = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.3));
  const Eigen::Vector3d measured = 2.0 * (Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.1, 0)) * up);
  const auto term = robust_gravity_direction(R.inverse() * up, measured);
  const Eigen::Matrix3d J = R.inverse().matrix() * Sophus::SO3d::hat(up);
  const Eigen::Vector3d gradient = term.loss.weight * J.transpose() * term.residual;
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero(); delta[axis] = 1e-6;
    const auto plus = robust_gravity_direction((Sophus::SO3d::exp(delta) * R).inverse() * up, measured);
    const auto minus = robust_gravity_direction((Sophus::SO3d::exp(-delta) * R).inverse() * up, measured);
    EXPECT_NEAR((plus.loss.cost - minus.loss.cost) / 2e-6, gradient[axis], 1e-7);
  }
}
