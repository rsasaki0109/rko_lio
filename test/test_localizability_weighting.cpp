// Unit tests for the localizability-aware ICP weighting: the pure weight
// function (localizability_weighting.hpp) and the per-voxel normal cache it
// consumes (VoxelHashMap::voxel_normal / set_maintain_normals).
#include <gtest/gtest.h>

#include "rko_lio/core/localizability_weighting.hpp"
#include "rko_lio/core/voxel_hash_map.hpp"

#include <Eigen/Core>

#include <cmath>
#include <optional>
#include <vector>

namespace {

using rko_lio::core::localizability_weight;
using rko_lio::core::VoxelHashMap;

TEST(LocalizabilityWeight, NeutralWithoutNormalOrBoostOrAxis) {
  const Eigen::Vector3d axis(1.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(localizability_weight(std::nullopt, axis, 30.0), 1.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(1.0, 0.0, 0.0), axis, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(1.0, 0.0, 0.0), axis, -1.0), 1.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Zero(), 30.0), 1.0);
}

TEST(LocalizabilityWeight, AxisAlignedNormalGetsFullBoost) {
  const Eigen::Vector3d axis(1.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(1.0, 0.0, 0.0), axis, 30.0), 31.0);
  // Orientation of the normal must not matter.
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(-1.0, 0.0, 0.0), axis, 30.0), 31.0);
}

TEST(LocalizabilityWeight, AxisNeutralWallNormalStaysAtOne) {
  const Eigen::Vector3d axis(1.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(0.0, 1.0, 0.0), axis, 30.0), 1.0);
  EXPECT_DOUBLE_EQ(localizability_weight(Eigen::Vector3d(0.0, 0.0, 1.0), axis, 30.0), 1.0);
}

TEST(LocalizabilityWeight, ObliqueNormalScalesWithSquaredCosine) {
  const Eigen::Vector3d axis(1.0, 0.0, 0.0);
  const Eigen::Vector3d oblique = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
  EXPECT_NEAR(localizability_weight(oblique, axis, 30.0), 1.0 + 30.0 * 0.5, 1.0e-12);
}

TEST(VoxelHashMapNormals, PlanarVoxelYieldsSurfaceNormal) {
  VoxelHashMap grid(1.0, 100.0, 20);
  grid.set_maintain_normals(true);
  // An x = const wall patch inside one voxel: normal must be +/-x.
  std::vector<Eigen::Vector3d> wall;
  for (int y = 0; y < 3; ++y) {
    for (int z = 0; z < 3; ++z) {
      wall.emplace_back(0.5, 0.1 + 0.3 * y, 0.1 + 0.3 * z);
    }
  }
  grid.add_points(wall);
  const auto normal = grid.voxel_normal(wall.front());
  ASSERT_TRUE(normal.has_value());
  EXPECT_NEAR(std::abs(normal->x()), 1.0, 1.0e-9);
  EXPECT_NEAR(normal->norm(), 1.0, 1.0e-9);
}

TEST(VoxelHashMapNormals, TooFewOrNonPlanarPointsYieldNoNormal) {
  VoxelHashMap grid(1.0, 100.0, 20);
  grid.set_maintain_normals(true);
  // Fewer than min_normal_points_ samples.
  grid.add_points({{0.1, 0.1, 0.1}, {0.4, 0.4, 0.4}, {0.7, 0.2, 0.6}});
  EXPECT_FALSE(grid.voxel_normal(Eigen::Vector3d(0.1, 0.1, 0.1)).has_value());
  // A line (rank-1 spread) must not produce a surface normal. Spacing must clear
  // add_points()'s dedup radius (sqrt(voxel_size^2 / max_points) = ~0.45 here) so
  // enough collinear samples actually enter the voxel.
  VoxelHashMap line_grid(2.0, 100.0, 20);
  line_grid.set_maintain_normals(true);
  std::vector<Eigen::Vector3d> line;
  for (int k = 0; k < 5; ++k) {
    line.emplace_back(0.1 + 0.45 * k, 0.5, 0.5);
  }
  line_grid.add_points(line);
  EXPECT_FALSE(line_grid.voxel_normal(line.front()).has_value());
}

TEST(VoxelHashMapNormals, DisabledMaintenanceKeepsCacheEmpty) {
  VoxelHashMap grid(1.0, 100.0, 20);
  std::vector<Eigen::Vector3d> wall;
  for (int y = 0; y < 3; ++y) {
    for (int z = 0; z < 3; ++z) {
      wall.emplace_back(0.5, 0.1 + 0.3 * y, 0.1 + 0.3 * z);
    }
  }
  grid.add_points(wall);
  EXPECT_FALSE(grid.voxel_normal(wall.front()).has_value());
}

TEST(VoxelHashMapNormals, ClippingErasesNormalsWithVoxels) {
  VoxelHashMap grid(1.0, 5.0, 20);
  grid.set_maintain_normals(true);
  std::vector<Eigen::Vector3d> wall;
  for (int y = 0; y < 3; ++y) {
    for (int z = 0; z < 3; ++z) {
      wall.emplace_back(10.5, 0.1 + 0.3 * y, 0.1 + 0.3 * z);
    }
  }
  grid.add_points(wall);
  ASSERT_TRUE(grid.voxel_normal(wall.front()).has_value());
  grid.remove_points_far_from_location(Eigen::Vector3d::Zero());
  EXPECT_FALSE(grid.voxel_normal(wall.front()).has_value());
}

} // namespace
