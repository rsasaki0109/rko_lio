/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <gtest/gtest.h>

#include <vector>

#include "rko_lio/core/fixed_lag_pose_optimizer.hpp"

namespace {

using rko_lio::core::FixedLagPoseOptimizerConfig;
using rko_lio::core::FixedLagRelativeConstraint;
using rko_lio::core::optimize_fixed_lag_poses;

Sophus::SE3d translated(const double x, const double y = 0.0) {
  return Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(x, y, 0.0));
}

}  // namespace

TEST(FixedLagPoseOptimizer, CorrectsRedundantTranslationChain) {
  const std::vector<Sophus::SE3d> poses{
      translated(0.0), translated(1.2), translated(2.4)};
  const std::vector<FixedLagRelativeConstraint> constraints{
      {0U, 1U, translated(1.0), 1.0},
      {1U, 2U, translated(1.0), 1.0},
      {0U, 2U, translated(2.0), 1.0}};

  const auto result = optimize_fixed_lag_poses(poses, constraints);

  ASSERT_TRUE(result.valid);
  EXPECT_LT(result.final_cost, result.initial_cost);
  EXPECT_NEAR(result.poses[1].translation().x(), 1.0, 1.0e-6);
  EXPECT_NEAR(result.poses[2].translation().x(), 2.0, 1.0e-6);
}

TEST(FixedLagPoseOptimizer, KeepsOldestPoseExactlyFixed) {
  const Sophus::SE3d anchor = translated(4.0, -2.0);
  const std::vector<Sophus::SE3d> poses{anchor, translated(5.3, -2.0)};
  const std::vector<FixedLagRelativeConstraint> constraints{
      {0U, 1U, translated(1.0), 1.0}};

  const auto result = optimize_fixed_lag_poses(poses, constraints);

  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(result.poses.front().matrix().isApprox(anchor.matrix(), 0.0));
  EXPECT_NEAR(result.poses.back().translation().x(), 5.0, 1.0e-6);
  EXPECT_NEAR(result.poses.back().translation().y(), -2.0, 1.0e-6);
}

TEST(FixedLagPoseOptimizer, CanFixNewestPoseWhileCorrectingHistory) {
  const std::vector<Sophus::SE3d> poses{
      translated(0.0), translated(1.3), translated(2.0)};
  const std::vector<FixedLagRelativeConstraint> constraints{
      {0U, 1U, translated(1.0), 1.0},
      {1U, 2U, translated(1.0), 1.0}};
  FixedLagPoseOptimizerConfig config;
  config.fix_latest_pose = true;

  const auto result = optimize_fixed_lag_poses(poses, constraints, config);

  ASSERT_TRUE(result.valid);
  EXPECT_LT(result.final_cost, result.initial_cost);
  EXPECT_TRUE(result.poses.front().matrix().isApprox(poses.front().matrix(), 0.0));
  EXPECT_TRUE(result.poses.back().matrix().isApprox(poses.back().matrix(), 0.0));
  EXPECT_NEAR(result.poses[1].translation().x(), 1.0, 1.0e-6);
}

TEST(FixedLagPoseOptimizer, RobustlyReducesAnInconsistentWindow) {
  const std::vector<Sophus::SE3d> poses{
      translated(0.0), translated(1.1), translated(2.2), translated(3.3)};
  const std::vector<FixedLagRelativeConstraint> constraints{
      {0U, 1U, translated(1.0), 2.0},
      {1U, 2U, translated(1.0), 2.0},
      {2U, 3U, translated(1.0), 2.0},
      {0U, 3U, translated(3.05), 1.0},
      {0U, 3U, translated(5.0), 0.05}};

  const auto result = optimize_fixed_lag_poses(poses, constraints);

  ASSERT_TRUE(result.valid);
  EXPECT_LT(result.final_cost, result.initial_cost);
  EXPECT_LT(std::abs(result.poses.back().translation().x() - 3.0), 0.1);
}

TEST(FixedLagPoseOptimizer, RejectsInvalidConstraintIndices) {
  const std::vector<Sophus::SE3d> poses{translated(0.0), translated(1.0)};
  const std::vector<FixedLagRelativeConstraint> constraints{
      {0U, 2U, translated(1.0), 1.0}};

  EXPECT_FALSE(optimize_fixed_lag_poses(poses, constraints).valid);
}
