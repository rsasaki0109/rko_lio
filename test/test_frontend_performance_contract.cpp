// MIT License
//
// Copyright (c) 2026 Sasaki
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <vector>

#include "rko_lio/core/sparse_voxel_grid.hpp"

TEST(FrontendPerformanceContract, FindsExactNearestPointAcrossVoxelBoundaries) {
  rko_lio::core::SparseVoxelGrid grid(1.0, 100.0, 20U);
  const std::vector<Eigen::Vector3d> points{
      {-0.95, 0.10, 0.10}, {0.95, 0.10, 0.10}, {1.05, 0.10, 0.10}, {2.10, 0.10, 0.10}};
  grid.AddPoints(points);

  const auto [nearest, distance] = grid.GetClosestNeighbor({0.99, 0.10, 0.10}, 2);
  EXPECT_TRUE(nearest.isApprox(Eigen::Vector3d(0.95, 0.10, 0.10), 1e-12));
  EXPECT_NEAR(distance, 0.04, 1e-12);
}

TEST(FrontendPerformanceContract, ClearCanBeFollowedByInsertionAndSearch) {
  rko_lio::core::SparseVoxelGrid grid(0.5, 100.0, 20U);
  grid.AddPoints({Eigen::Vector3d(1.0, 0.0, 0.0)});
  grid.Clear();
  grid.AddPoints({Eigen::Vector3d(2.0, 0.0, 0.0)});

  const auto [nearest, distance] = grid.GetClosestNeighbor({2.1, 0.0, 0.0});
  EXPECT_TRUE(nearest.isApprox(Eigen::Vector3d(2.0, 0.0, 0.0), 1e-12));
  EXPECT_NEAR(distance, 0.1, 1e-12);
}

TEST(FrontendPerformanceContract, BranchAndBoundMatchesBruteForceAcrossSignedBoundaries) {
  constexpr int radius = 3;
  rko_lio::core::SparseVoxelGrid grid(1.0, 100.0, 20U);
  std::vector<Eigen::Vector3d> points;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) {
      for (int z = -4; z <= 4; ++z) {
        points.emplace_back(x + 0.13, y + 0.37, z + 0.61);
      }
    }
  }
  grid.AddPoints(points);

  const std::vector<Eigen::Vector3d> queries{
      {-1.000001, -0.000001, 0.999999},
      {-0.999999, 0.000001, 1.000001},
      {-0.01, -0.99, 0.50},
      {0.99, 1.01, -1.01},
      {2.40, -2.60, 0.20}};
  for (const auto& query : queries) {
    const Eigen::Vector3i query_voxel = query.array().floor().cast<int>();
    Eigen::Vector3d expected = Eigen::Vector3d::Zero();
    double expected_squared_distance = std::numeric_limits<double>::max();
    for (const auto& point : points) {
      const Eigen::Vector3i point_voxel = point.array().floor().cast<int>();
      if ((point_voxel - query_voxel).cwiseAbs().maxCoeff() > radius) {
        continue;
      }
      const double squared_distance = (point - query).squaredNorm();
      if (squared_distance < expected_squared_distance) {
        expected = point;
        expected_squared_distance = squared_distance;
      }
    }

    const auto [actual, distance] = grid.GetClosestNeighbor(query, radius);
    EXPECT_TRUE(actual.isApprox(expected, 1e-12)) << "query: " << query.transpose();
    EXPECT_NEAR(distance, std::sqrt(expected_squared_distance), 1e-12)
        << "query: " << query.transpose();
  }
}
