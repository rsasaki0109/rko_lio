// MIT License

// Copyright (c) 2024 Tiziano Guadagnino, Benedikt Mersch, Ignacio Vizzo, Cyrill
// Stachniss.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "sparse_voxel_grid.hpp"

#include <Eigen/Core>
#include <array>
#include <bonxai/bonxai.hpp>
#include <cstdint>
#include <sophus/se3.hpp>

#include "bonxai/grid_coord.hpp"
// [instrumentation, additive-only] scoped timers for map-maintenance costs.
#include "profiler.hpp"

namespace {
using Bonxai::CoordT;

constexpr std::array<Bonxai::CoordT, 27> shifts{
    CoordT{.x = -1, .y = -1, .z = -1}, CoordT{.x = -1, .y = -1, .z = 0}, CoordT{.x = -1, .y = -1, .z = 1},
    CoordT{.x = -1, .y = 0, .z = -1},  CoordT{.x = -1, .y = 0, .z = 0},  CoordT{.x = -1, .y = 0, .z = 1},
    CoordT{.x = -1, .y = 1, .z = -1},  CoordT{.x = -1, .y = 1, .z = 0},  CoordT{.x = -1, .y = 1, .z = 1},

    CoordT{.x = 0, .y = -1, .z = -1},  CoordT{.x = 0, .y = -1, .z = 0},  CoordT{.x = 0, .y = -1, .z = 1},
    CoordT{.x = 0, .y = 0, .z = -1},   CoordT{.x = 0, .y = 0, .z = 0},   CoordT{.x = 0, .y = 0, .z = 1},
    CoordT{.x = 0, .y = 1, .z = -1},   CoordT{.x = 0, .y = 1, .z = 0},   CoordT{.x = 0, .y = 1, .z = 1},

    CoordT{.x = 1, .y = -1, .z = -1},  CoordT{.x = 1, .y = -1, .z = 0},  CoordT{.x = 1, .y = -1, .z = 1},
    CoordT{.x = 1, .y = 0, .z = -1},   CoordT{.x = 1, .y = 0, .z = 0},   CoordT{.x = 1, .y = 0, .z = 1},
    CoordT{.x = 1, .y = 1, .z = -1},   CoordT{.x = 1, .y = 1, .z = 0},   CoordT{.x = 1, .y = 1, .z = 1}};

constexpr uint8_t inner_grid_log2_dim = 2;
constexpr uint8_t leaf_grid_log2_dim = 3;

// [NN-search optimization, additive] Branch-and-bound helper: given the
// query's fractional offset `frac` (in [0, voxel_size), the distance from
// the query to the near face, along one axis, of its own voxel toward the
// -1 direction) and an integer voxel `offset` along that same axis relative
// to the query's own voxel, returns a true lower bound on the squared
// per-axis distance from the query to any point inside the offset voxel's
// slab along that axis. offset == 0 always returns 0 exactly (the query is
// inside that slab). This is only ever a *lower* bound (never an
// overestimate) up to ordinary floating-point rounding of a single
// subtraction, so it is safe to use for pruning: see the comment above
// GetClosestNeighbor for why pruning with this bound cannot change which
// neighbor is selected.
inline double axis_bound_sq(const double frac, const double voxel_size, const int offset) {
  if (offset == 0) {
    return 0.0;
  }
  const double dist = offset > 0 ? (static_cast<double>(offset) * voxel_size - frac)
                                  : (static_cast<double>(-offset - 1) * voxel_size + frac);
  const double clamped = std::max(0.0, dist);
  return clamped * clamped;
}
} // namespace

namespace rko_lio::core {

SparseVoxelGrid::SparseVoxelGrid(const double voxel_size,
                                 const double clipping_distance,
                                 const unsigned int max_points_per_voxel)
    : voxel_size_(voxel_size),
      clipping_distance_(clipping_distance),
      max_points_per_voxel_(max_points_per_voxel),
      map_(voxel_size, inner_grid_log2_dim, leaf_grid_log2_dim) {}

std::tuple<Eigen::Vector3d, double> SparseVoxelGrid::GetClosestNeighbor(const Eigen::Vector3d& query) const {
  return GetClosestNeighbor(query, 1);
}

// [NN-search optimization] GetClosestNeighbor scans a fixed voxel
// neighborhood around `query` (27 voxels for the common voxel_search_radius
// == 1 case, (2r+1)^3 for larger r) and returns the single closest point
// under squared Euclidean distance, taking sqrt once at the end (this was
// already true before this optimization pass).
//
// Added here: per-voxel branch-and-bound pruning. Before touching the hash
// map (const_accessor.value(...), the actual expensive step) or scanning a
// voxel's up-to-max_points_per_voxel_ points, we first compute a cheap,
// closed-form lower bound on the squared distance from `query` to that
// voxel's axis-aligned bounding box (axis_bound_sq above) and skip the
// voxel entirely if that bound already meets or exceeds the best squared
// distance found so far.
//
// Why this cannot change the result (exactness argument):
// The voxel (and within-voxel point) visitation ORDER is left completely
// unchanged from the original implementation (still the fixed `shifts`
// array order / the same nested dx,dy,dz loop order). This matters: if a
// voxel's lower bound >= the current best-so-far, then EVERY point in that
// voxel has squared_distance >= bound >= best-so-far, i.e. none of them can
// satisfy the strict '<' that the original code uses to update the
// running winner. So skipping such a voxel leaves the running
// (closest_neighbor, closest_squared_distance) state identical to what the
// unpruned original would have after processing that voxel. This holds by
// induction, voxel by voxel, in the (unchanged) scan order, so the final
// winner -- including which of two exactly-tied candidates is kept, since
// that is entirely determined by which one the fixed scan order reaches
// first -- is bit-for-bit identical to the original. Reordering voxel
// visitation (e.g. to visit low-bound voxels first, which would trigger
// pruning earlier) was deliberately NOT done here: it would change which
// voxel is scanned first when two points from different voxels tie exactly
// on distance, which could change the selected winner. Keeping the
// original order sacrifices a bit of pruning opportunity (pruning still
// triggers plenty once a handful of nearby voxels have been scanned) in
// exchange for a fully rigorous exactness guarantee.
//
// Residual risk: axis_bound_sq is computed via a different floating-point
// operation sequence than the direct (point - query).squaredNorm() used for
// real candidates. In principle this could, at the sub-ULP level, produce a
// bound that is a few ULPs different from the mathematically exact minimum
// for a point sitting exactly on a voxel boundary. This is guarded with a
// std::max(0.0, ...) clamp and is expected to be unobservable in practice
// (points are floating-point sensor measurements essentially never exactly
// coincident with a voxel boundary), and is exactly the kind of risk this
// task asked to flag and let the trajectory byte-diff catch.
std::tuple<Eigen::Vector3d, double> SparseVoxelGrid::GetClosestNeighbor(const Eigen::Vector3d& query,
                                                                        int voxel_search_radius) const {
  Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
  double closest_squared_distance = std::numeric_limits<double>::max();
  const auto const_accessor = map_.createConstAccessor();
  const Bonxai::CoordT voxel = map_.posToCoord(query);
  const int radius = std::max(1, voxel_search_radius);
  // Fractional offset of the query within its own voxel, along each axis:
  // the distance from the query to the near (-axis-direction) face of its
  // own voxel. Computed once per query (hoisted out of the per-candidate
  // inner loops) via the same coordinate transform Bonxai uses for cell
  // boundaries (coordToPos), so it is consistent with how the grid itself
  // defines voxel boundaries. Mathematically frac_* is in [0, voxel_size_);
  // clamped defensively against floating-point noise at that boundary.
  const Bonxai::Point3D origin = map_.coordToPos(voxel);
  const double frac_x = std::max(0.0, query.x() - origin.x);
  const double frac_y = std::max(0.0, query.y() - origin.y);
  const double frac_z = std::max(0.0, query.z() - origin.z);
  if (radius == 1) {
    std::for_each(shifts.cbegin(), shifts.cend(), [&](const Bonxai::CoordT& voxel_shift) {
      const double bound_sq = axis_bound_sq(frac_x, voxel_size_, voxel_shift.x) +
                              axis_bound_sq(frac_y, voxel_size_, voxel_shift.y) +
                              axis_bound_sq(frac_z, voxel_size_, voxel_shift.z);
      if (bound_sq >= closest_squared_distance) {
        return;
      }
      const Bonxai::CoordT query_voxel = voxel + voxel_shift;
      const VoxelBlock* voxel_points = const_accessor.value(query_voxel);
      if (voxel_points != nullptr) {
        for (const Eigen::Vector3d& point : *voxel_points) {
          const double squared_distance = (point - query).squaredNorm();
          if (squared_distance < closest_squared_distance) {
            closest_neighbor = point;
            closest_squared_distance = squared_distance;
          }
        }
      }
    });
    const double closest_distance =
        closest_squared_distance == std::numeric_limits<double>::max()
            ? std::numeric_limits<double>::max()
            : std::sqrt(closest_squared_distance);
    return std::make_tuple(closest_neighbor, closest_distance);
  }

  for (int dx = -radius; dx <= radius; ++dx) {
    const double bound_x = axis_bound_sq(frac_x, voxel_size_, dx);
    if (bound_x >= closest_squared_distance) {
      continue;
    }
    for (int dy = -radius; dy <= radius; ++dy) {
      const double bound_xy = bound_x + axis_bound_sq(frac_y, voxel_size_, dy);
      if (bound_xy >= closest_squared_distance) {
        continue;
      }
      for (int dz = -radius; dz <= radius; ++dz) {
        const double bound_sq = bound_xy + axis_bound_sq(frac_z, voxel_size_, dz);
        if (bound_sq >= closest_squared_distance) {
          continue;
        }
        const Bonxai::CoordT query_voxel = voxel + CoordT{.x = dx, .y = dy, .z = dz};
        const VoxelBlock* voxel_points = const_accessor.value(query_voxel);
        if (voxel_points == nullptr) {
          continue;
        }
        for (const Eigen::Vector3d& point : *voxel_points) {
          const double squared_distance = (point - query).squaredNorm();
          if (squared_distance < closest_squared_distance) {
            closest_neighbor = point;
            closest_squared_distance = squared_distance;
          }
        }
      }
    }
  }
  const double closest_distance =
      closest_squared_distance == std::numeric_limits<double>::max()
          ? std::numeric_limits<double>::max()
          : std::sqrt(closest_squared_distance);
  return std::make_tuple(closest_neighbor, closest_distance);
}

void SparseVoxelGrid::AddPoints(const std::vector<Eigen::Vector3d>& points) {
  SCOPED_PROFILER("SparseVoxelGrid::AddPoints");
  const double map_resolution = std::sqrt(voxel_size_ * voxel_size_ / max_points_per_voxel_);
  auto accessor = map_.createAccessor();
  std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d& p) {
    const auto voxel_coordinates = map_.posToCoord(p);
    VoxelBlock* voxel_points = accessor.value(voxel_coordinates, /*create_if_missing=*/true);
    if (voxel_points->size() == max_points_per_voxel_ ||
        std::any_of(voxel_points->cbegin(), voxel_points->cend(),
                    [&](const auto& voxel_point) { return (voxel_point - p).norm() < map_resolution; })) {
      return;
    }
    voxel_points->reserve(max_points_per_voxel_);
    voxel_points->emplace_back(p);
  });
}

void SparseVoxelGrid::RemovePointsFarFromLocation(const Eigen::Vector3d& origin) {
  SCOPED_PROFILER("SparseVoxelGrid::RemovePointsFarFromLocation");
  auto is_too_far_away = [&](const VoxelBlock& block) { return (block.front() - origin).norm() > clipping_distance_; };

  std::vector<Bonxai::CoordT> keys_to_delete;
  auto& root_map = map_.rootMap();
  for (auto& [key, inner_grid] : root_map) {
    for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      auto& leaf_grid = inner_grid.cell(inner_index);
      const auto& voxel_block = leaf_grid->cell(leaf_grid->mask().findFirstOn());
      if (is_too_far_away(voxel_block)) {
        inner_grid.mask().setOff(inner_index);
        leaf_grid.reset();
      }
    }
    if (inner_grid.mask().isOff()) {
      keys_to_delete.push_back(key);
    }
  }
  for (const auto& key : keys_to_delete) {
    root_map.erase(key);
  }
}

void SparseVoxelGrid::Update(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& pose) {
  SCOPED_PROFILER("SparseVoxelGrid::Update");
  std::vector<Eigen::Vector3d> points_transformed(points.size());
  std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                 [&](const auto& point) { return pose * point; });
  const Eigen::Vector3d& origin = pose.translation();
  AddPoints(points_transformed);
  RemovePointsFarFromLocation(origin);
}

std::vector<Eigen::Vector3d> SparseVoxelGrid::Pointcloud() const {
  std::vector<Eigen::Vector3d> point_cloud;
  point_cloud.reserve(map_.activeCellsCount() * max_points_per_voxel_);
  map_.forEachCell([&point_cloud, this](const VoxelBlock& block, const auto&) {
    point_cloud.insert(point_cloud.end(), block.cbegin(), block.cend());
  });
  return point_cloud;
}

// [instrumentation, additive-only] Counts stored points by summing per-voxel
// block sizes only (no point data is copied), unlike Pointcloud(). Read-only,
// does not affect map state.
std::size_t SparseVoxelGrid::ActivePointCount() const {
  std::size_t total = 0;
  map_.forEachCell([&total](const VoxelBlock& block, const auto&) { total += block.size(); });
  return total;
}

} // namespace rko_lio::core
