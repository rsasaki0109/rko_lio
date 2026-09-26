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
