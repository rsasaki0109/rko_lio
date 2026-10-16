#include <gtest/gtest.h>

#include "rko_lio/core/voxel_hash_map.hpp"

#include <sophus/se3.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

TEST(VoxelHashMap, FindsExactNearestPointAcrossVoxelBoundaries) {
  VoxelHashMap grid(1.0, 100.0, 20U);
  const std::vector<Eigen::Vector3d> points{
      {-0.95, 0.10, 0.10}, {0.95, 0.10, 0.10}, {1.05, 0.10, 0.10}, {2.10, 0.10, 0.10}};
  grid.add_points(points);

  const auto [nearest, distance] = grid.get_closest_neighbor({0.99, 0.10, 0.10}, 2);
  EXPECT_TRUE(nearest.isApprox(Eigen::Vector3d(0.95, 0.10, 0.10), 1e-12));
  EXPECT_NEAR(distance, 0.04, 1e-12);
}

TEST(VoxelHashMap, BranchAndBoundMatchesBruteForceAcrossSignedBoundaries) {
  VoxelHashMap grid(1.0, 100.0, 20U);
  std::vector<Eigen::Vector3d> points;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) {
      for (int z = -4; z <= 4; ++z) {
        points.emplace_back(x + 0.13, y + 0.37, z + 0.61);
      }
    }
  }
  grid.add_points(points);

  const std::vector<Eigen::Vector3d> queries{
      {-1.000001, -0.000001, 0.999999},
      {-0.999999, 0.000001, 1.000001},
      {-0.01, -0.99, 0.50},
      {0.99, 1.01, -1.01},
      {2.40, -2.60, 0.20}};
  for (const int radius : {1, 3}) {
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

      const auto [actual, distance] = grid.get_closest_neighbor(query, radius);
      EXPECT_TRUE(actual.isApprox(expected, 1e-12)) << "query: " << query.transpose();
      EXPECT_NEAR(distance, std::sqrt(expected_squared_distance), 1e-12)
          << "query: " << query.transpose();
    }
  }
}

// LIO::relevel_local_frame relies on transform() moving every stored point rigidly and
// re-bucketing it, so queries in the new frame find the moved points.
TEST(VoxelHashMap, TransformMovesEveryPointRigidly) {
  VoxelHashMap grid(0.5, 100.0, 20);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 40; ++i) {
    points.emplace_back(0.37 * i, -0.21 * i, 0.05 * i);
  }
  grid.add_points(points);

  const Sophus::SE3d transform(Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.02, 0.3)), Eigen::Vector3d(1.0, 2.0, -0.5));
  grid.transform(transform);

  EXPECT_EQ(grid.pointcloud().size(), points.size());
  for (const auto& point : points) {
    const Eigen::Vector3d moved = transform * point;
    const auto [neighbor, distance] = grid.get_closest_neighbor(moved);
    EXPECT_LT(distance, 1e-9) << "point " << point.transpose();
    EXPECT_TRUE(neighbor.isApprox(moved, 1e-9));
  }
}

TEST(VoxelHashMap, RadiusOnePreservesTieOrderAndEmptySentinel) {
  VoxelHashMap grid(1.0, 100.0, 20U);
  const auto [empty_point, empty_distance] = grid.get_closest_neighbor(Eigen::Vector3d::Zero());
  EXPECT_EQ(empty_distance, std::numeric_limits<double>::max());
  EXPECT_TRUE(empty_point.isZero());
  grid.add_points({{0.25, 0.0, 0.0}, {-0.25, 0.0, 0.0}, {0.0, -0.25, 0.0}});
  const auto [nearest, distance] = grid.get_closest_neighbor(Eigen::Vector3d::Zero());
  EXPECT_TRUE(nearest == Eigen::Vector3d(-0.25, 0.0, 0.0));
  EXPECT_DOUBLE_EQ(distance, 0.25);
  const auto [exact, zero_distance] = grid.get_closest_neighbor(nearest);
  EXPECT_TRUE(exact == nearest);
  EXPECT_DOUBLE_EQ(zero_distance, 0.0);
}

// remove_points_far_from_location() skips its walk over the map when no voxel can be far enough.
// Move the origin so that walks are both skipped and run, and add a voxel after a skipped walk.
TEST(VoxelHashMap, RemovesExactlyTheFarVoxelsAsTheOriginMoves) {
  constexpr double clipping_distance = 10.0;
  VoxelHashMap grid(0.5, clipping_distance, 20U);
  std::vector<Eigen::Vector3d> expected;
  const auto add = [&](const std::vector<Eigen::Vector3d>& points) {
    grid.add_points(points);
    expected.insert(expected.end(), points.begin(), points.end());
  };
  const auto lexicographic = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::lexicographical_compare(a.data(), a.data() + 3, b.data(), b.data() + 3);
  };
  const auto remove = [&](const Eigen::Vector3d& origin) {
    grid.remove_points_far_from_location(origin);
    std::erase_if(expected, [&](const Eigen::Vector3d& point) {
      return (point - origin).squaredNorm() > clipping_distance * clipping_distance;
    });
    std::vector<Eigen::Vector3d> actual = grid.pointcloud();
    std::vector<Eigen::Vector3d> sorted_expected = expected;
    std::sort(actual.begin(), actual.end(), lexicographic);
    std::sort(sorted_expected.begin(), sorted_expected.end(), lexicographic);
    EXPECT_TRUE(actual == sorted_expected) << "origin: " << origin.transpose();
  };

  std::vector<Eigen::Vector3d> line; // one point per voxel, x from 0.25 to 19.75
  for (int i = 0; i < 40; ++i) {
    line.emplace_back(0.5 * i + 0.25, 0.25, 0.25);
  }
  add(line);
  remove({0.0, 0.0, 0.0});
  remove({0.5, 0.0, 0.0});
  add({{-8.75, 0.25, 0.25}});
  remove({1.0, 0.0, 0.0});
  remove({4.0, 0.0, 0.0});
  remove({-3.0, 0.0, 0.0});
  EXPECT_FALSE(grid.empty());
  remove({30.0, 0.0, 0.0});
  EXPECT_TRUE(grid.empty());
}

// The map must end up exactly as the historical walk over map_ (erasing far voxels in iteration
// order) leaves it, including its iteration order, which VoxelHashMap::transform() depends on.
TEST(VoxelHashMap, PruningLeavesTheMapAsAWalkOverItWould) {
  constexpr double clipping_distance = 15.0;
  const auto walk_and_prune = [](VoxelHashMap& map, const Eigen::Vector3d& origin) {
    const double clipping_distance_sq = map.clipping_distance_ * map.clipping_distance_;
    for (auto it = map.map_.begin(); it != map.map_.end();) {
      if ((it->second.front() - origin).squaredNorm() > clipping_distance_sq) {
        map.normal_map_.erase(it->first);
        it = map.map_.erase(it);
      } else {
        ++it;
      }
    }
  };
  std::uint64_t state = 7;
  const auto uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / static_cast<double>(std::uint64_t{1} << 53);
  };
  VoxelHashMap grid(0.5, clipping_distance, 5U);
  VoxelHashMap reference(0.5, clipping_distance, 5U);
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  for (int step = 0; step < 200; ++step) {
    origin += Eigen::Vector3d(4.0 * uniform() - 1.0, 4.0 * uniform() - 2.0, 0.2 * uniform() - 0.1);
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 500; ++i) {
      points.push_back(origin + Eigen::Vector3d(30.0 * uniform() - 15.0, 30.0 * uniform() - 15.0, 4.0 * uniform()));
    }
    grid.add_points(points);
    reference.add_points(points);
    grid.remove_points_far_from_location(origin);
    walk_and_prune(reference, origin);
    ASSERT_TRUE(grid.pointcloud() == reference.pointcloud()) << "step " << step;
  }
}

} // namespace
