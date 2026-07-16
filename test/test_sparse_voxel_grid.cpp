#include <gtest/gtest.h>

#include "rko_lio/core/sparse_voxel_grid.hpp"

#include <vector>

namespace {

using rko_lio::core::SparseVoxelGrid;

// Regression test for the kidnap-recovery segfault: LIO::recover_with_scan()
// calls SparseVoxelGrid::Clear() (map_.clear(CLEAR_MEMORY), which destroys
// every inner/leaf grid node) and then immediately AddPoints() again to
// re-seed the local map at the relocalized pose. AddPoints() used to reuse a
// persistent Bonxai::VoxelGrid::Accessor class member across calls for its
// O(1) repeat-lookup cache. That cache holds *raw pointers* to the last
// visited inner/leaf grid nodes and is only invalidated on a cache miss (a
// different inner-grid coordinate than last time) -- Clear() has no way to
// reach into and invalidate an external accessor's cache. So the first
// AddPoints() call after a Clear() whose point happened to land in the same
// inner-grid block (a several-meter cube) as whatever was last touched
// before the clear would skip the lookup entirely and write through a
// dangling pointer: a use-after-free that either segfaults (if the freed
// memory has already been reclaimed) or silently drops the point (the write
// never reaches the new, empty grid).
TEST(SparseVoxelGrid, AddPointsAfterClearIsVisibleForSameInnerBlock) {
  constexpr double voxel_size = 0.2;
  constexpr double clipping_distance = 8.0;
  constexpr unsigned int max_points_per_voxel = 20;
  SparseVoxelGrid grid(voxel_size, clipping_distance, max_points_per_voxel);

  const Eigen::Vector3d first_point(1.0, 1.0, 1.0);
  grid.AddPoints({first_point});
  ASSERT_FALSE(grid.Empty());
  {
    const auto [neighbor, distance] = grid.GetClosestNeighbor(first_point);
    EXPECT_LT(distance, 1e-9) << "sanity check: point should be found before Clear()";
    (void)neighbor;
  }

  grid.Clear();
  ASSERT_TRUE(grid.Empty());

  // Deliberately reuses a point in the same several-meter inner-grid block as
  // `first_point` (inner blocks span 2^(inner_bits+leaf_bits) * voxel_size =
  // 2^5 * 0.2 = 6.4m per axis here), which is exactly the scenario that hit
  // the stale-accessor cache with the old persistent-accessor implementation.
  const Eigen::Vector3d second_point(1.05, 1.0, 1.0);
  grid.AddPoints({second_point});

  EXPECT_FALSE(grid.Empty()) << "AddPoints() after Clear() must actually insert into the live grid";
  const auto [neighbor, distance] = grid.GetClosestNeighbor(second_point);
  EXPECT_LT(distance, 1e-9) << "point added after Clear() must be queryable back out of the grid";
  EXPECT_TRUE(neighbor.isApprox(second_point, 1e-9));
}

// Same scenario, but repeated across many distinct inner-grid blocks and
// several Clear()/AddPoints() cycles, so the test does not rely on a single
// lucky coordinate collision to exercise the stale-cache path.
TEST(SparseVoxelGrid, RepeatedClearAndAddPointsRoundTrips) {
  constexpr double voxel_size = 0.2;
  constexpr double clipping_distance = 8.0;
  constexpr unsigned int max_points_per_voxel = 20;
  SparseVoxelGrid grid(voxel_size, clipping_distance, max_points_per_voxel);

  for (int cycle = 0; cycle < 5; ++cycle) {
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 20; ++i) {
      points.emplace_back(static_cast<double>(i) * 0.5, static_cast<double>(cycle), 0.0);
    }
    grid.AddPoints(points);
    grid.Clear();
    ASSERT_TRUE(grid.Empty());
    grid.AddPoints(points);
    ASSERT_FALSE(grid.Empty());
    for (const auto& point : points) {
      const auto [neighbor, distance] = grid.GetClosestNeighbor(point);
      EXPECT_LT(distance, 1e-9);
      (void)neighbor;
    }
    grid.Clear();
  }
}

} // namespace
