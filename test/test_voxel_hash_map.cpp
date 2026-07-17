#include <gtest/gtest.h>

#include "rko_lio/core/voxel_hash_map.hpp"

#include <vector>

namespace {

using rko_lio::core::VoxelHashMap;

// Regression coverage for the kidnap-recovery clear()/add_points() sequence used by
// LIO::recover_with_scan(): clear() the local map, then immediately add_points() again to
// re-seed it at the relocalized pose.
//
// [Merge note, upstream-master-integration] This test used to guard a real use-after-free in
// this fork's old SparseVoxelGrid (Bonxai-backed): AddPoints() reused a persistent
// Bonxai::VoxelGrid::Accessor class member across calls for its O(1) repeat-lookup cache, which
// held raw pointers to the last-visited inner/leaf grid nodes. Clear() (map_.clear(CLEAR_MEMORY))
// destroyed those nodes but had no way to invalidate the external accessor's cache, so the next
// AddPoints() call for a point landing in the same inner-grid block as before the clear would
// write through a dangling pointer.
//
// Upstream's replacement, VoxelHashMap (rko_lio/core/voxel_hash_map.{hpp,cpp}), is backed
// directly by tsl::robin_map and never stores a persistent accessor/iterator as a class member:
// every add_points()/get_closest_neighbor() call does a fresh map_.try_emplace()/map_.find()
// against the live map. There is no cache to go stale across clear(), so this specific
// use-after-free pattern does not exist upstream -- verified by inspection of voxel_hash_map.cpp
// during the merge. This test is kept (renamed to the new API) as a general regression guard for
// the clear()+add_points() sequence, not because the UAF risk still applies.
TEST(VoxelHashMap, AddPointsAfterClearIsVisibleForSameBlock) {
  constexpr double voxel_size = 0.2;
  constexpr double clipping_distance = 8.0;
  constexpr unsigned int max_points_per_voxel = 20;
  VoxelHashMap grid(voxel_size, clipping_distance, max_points_per_voxel);

  const Eigen::Vector3d first_point(1.0, 1.0, 1.0);
  grid.add_points({first_point});
  ASSERT_FALSE(grid.empty());
  {
    const auto [neighbor, distance] = grid.get_closest_neighbor(first_point);
    EXPECT_LT(distance, 1e-9) << "sanity check: point should be found before clear()";
    (void)neighbor;
  }

  grid.clear();
  ASSERT_TRUE(grid.empty());

  const Eigen::Vector3d second_point(1.05, 1.0, 1.0);
  grid.add_points({second_point});

  EXPECT_FALSE(grid.empty()) << "add_points() after clear() must actually insert into the live grid";
  const auto [neighbor, distance] = grid.get_closest_neighbor(second_point);
  EXPECT_LT(distance, 1e-9) << "point added after clear() must be queryable back out of the grid";
  EXPECT_TRUE(neighbor.isApprox(second_point, 1e-9));
}

// Same scenario, but repeated across many distinct coordinates and several clear()/add_points()
// cycles, so the test does not rely on a single lucky coordinate collision.
TEST(VoxelHashMap, RepeatedClearAndAddPointsRoundTrips) {
  constexpr double voxel_size = 0.2;
  constexpr double clipping_distance = 8.0;
  constexpr unsigned int max_points_per_voxel = 20;
  VoxelHashMap grid(voxel_size, clipping_distance, max_points_per_voxel);

  for (int cycle = 0; cycle < 5; ++cycle) {
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 20; ++i) {
      points.emplace_back(static_cast<double>(i) * 0.5, static_cast<double>(cycle), 0.0);
    }
    grid.add_points(points);
    grid.clear();
    ASSERT_TRUE(grid.empty());
    grid.add_points(points);
    ASSERT_FALSE(grid.empty());
    for (const auto& point : points) {
      const auto [neighbor, distance] = grid.get_closest_neighbor(point);
      EXPECT_LT(distance, 1e-9);
      (void)neighbor;
    }
    grid.clear();
  }
}

} // namespace
