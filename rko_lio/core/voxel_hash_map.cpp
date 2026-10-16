// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub.
//
// Ported to rko_lio from kiss-icp (kiss_icp/cpp/kiss_icp/core/VoxelHashMap.{hpp,cpp}).

#include "voxel_hash_map.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sophus/se3.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tuple>
#include <utility>
#include <vector>
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace {
using rko_lio::core::Voxel;

// A hint to load `address` into the cache. It changes nothing that is read or computed.
inline void prefetch(const void* address) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#else
  static_cast<void>(address);
#endif
}

// The first cache lines (up to three, or eight points) of the `count` points at `points`.
inline void prefetch_points(const Eigen::Vector3d* points, const std::uint32_t count) {
  if (points == nullptr) {
    return;
  }
  const char* bytes = reinterpret_cast<const char*>(points);
  const std::size_t size = std::min<std::size_t>(std::size_t{count} * sizeof(Eigen::Vector3d), 3 * 64);
  for (std::size_t offset = 0; offset < size; offset += 64) {
    prefetch(bytes + offset);
  }
}

static const std::array<Voxel, 27> shifts{
    Voxel{-1, -1, -1}, Voxel{-1, -1, 0}, Voxel{-1, -1, 1}, //
    Voxel{-1, 0, -1},  Voxel{-1, 0, 0},  Voxel{-1, 0, 1},  //
    Voxel{-1, 1, -1},  Voxel{-1, 1, 0},  Voxel{-1, 1, 1},  //

    Voxel{0, -1, -1},  Voxel{0, -1, 0},  Voxel{0, -1, 1},  //
    Voxel{0, 0, -1},   Voxel{0, 0, 0},   Voxel{0, 0, 1},   //
    Voxel{0, 1, -1},   Voxel{0, 1, 0},   Voxel{0, 1, 1},   //

    Voxel{1, -1, -1},  Voxel{1, -1, 0},  Voxel{1, -1, 1}, //
    Voxel{1, 0, -1},   Voxel{1, 0, 0},   Voxel{1, 0, 1},  //
    Voxel{1, 1, -1},   Voxel{1, 1, 0},   Voxel{1, 1, 1}};

inline double axis_bound_sq(const double frac, const double voxel_size, const int offset) {
  if (offset == 0) {
    return 0.0;
  }
  const double distance = offset > 0 ? static_cast<double>(offset) * voxel_size - frac
                                     : static_cast<double>(-offset - 1) * voxel_size + frac;
  const double clamped = std::max(0.0, distance);
  return clamped * clamped;
}

// The 3x3x3 search around `voxel`. points_of(i) returns the points of voxel + shifts[i] as a pointer
// and a count (nullptr and 0 when that voxel does not exist); it is called only for the voxels
// that the distance bound does not rule out, in the order of `shifts`.
template <typename PointsOf>
std::tuple<Eigen::Vector3d, double> closest_neighbor_around(const Eigen::Vector3d& query,
                                                            const Voxel& voxel,
                                                            const double voxel_size,
                                                            const PointsOf& points_of) {
  Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
  double closest_squared_distance = std::numeric_limits<double>::max();
  const Eigen::Vector3d fractional = query - voxel.cast<double>() * voxel_size;
  std::array<std::array<double, 3>, 3> axis_bounds;
  for (int axis = 0; axis < 3; ++axis) {
    for (int offset = -1; offset <= 1; ++offset) {
      axis_bounds[axis][offset + 1] = axis_bound_sq(fractional[axis], voxel_size, offset);
    }
  }
  for (std::size_t shift = 0; shift < shifts.size(); ++shift) {
    const Voxel& voxel_shift = shifts[shift];
    const double bound_sq = axis_bounds[0][voxel_shift.x() + 1] +
                            axis_bounds[1][voxel_shift.y() + 1] +
                            axis_bounds[2][voxel_shift.z() + 1];
    if (bound_sq >= closest_squared_distance) {
      continue;
    }
    const auto [points, count] = points_of(shift);
    for (std::size_t i = 0; i < count; ++i) {
      const double squared_distance = (points[i] - query).squaredNorm();
      if (squared_distance < closest_squared_distance) {
        closest_neighbor = points[i];
        closest_squared_distance = squared_distance;
      }
    }
  }
  const double closest_distance = closest_squared_distance == std::numeric_limits<double>::max()
                                      ? std::numeric_limits<double>::max()
                                      : std::sqrt(closest_squared_distance);
  return std::make_tuple(closest_neighbor, closest_distance);
}

} // namespace

namespace rko_lio::core {

VoxelHashMap::VoxelHashMap(const double voxel_size,
                           const double clipping_distance,
                           const unsigned int max_points_per_voxel)
    : voxel_size_(voxel_size),
      inv_voxel_size_(1.0 / voxel_size),
      clipping_distance_(clipping_distance),
      max_points_per_voxel_(max_points_per_voxel) {}

std::tuple<Eigen::Vector3d, double> VoxelHashMap::get_closest_neighbor(const Eigen::Vector3d& query) const {
  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {
    const auto search = map_.find(voxel + shifts[shift]);
    return search != map_.end() ? std::pair(search.value().data(), search.value().size())
                                : std::pair<const Eigen::Vector3d*, std::size_t>(nullptr, 0);
  });
}

std::tuple<Eigen::Vector3d, double> VoxelHashMap::get_closest_neighbor(const Eigen::Vector3d& query,
                                                                        NeighborhoodCache& cache) const {
  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  if (voxel != cache.voxel) {
    cache.voxel = voxel;
    cache.looked_up = 0;
  } else {
    // The search below reads again the points of the voxels found by the earlier calls, and those
    // are rarely still in the cache by then. Requesting them all first lets the misses overlap
    // instead of stalling the search one voxel at a time.
    for (std::uint32_t pending = cache.looked_up; pending != 0; pending &= pending - 1) {
      const auto shift = static_cast<std::size_t>(std::countr_zero(pending));
      prefetch_points(cache.points[shift], cache.sizes[shift]);
    }
  }
  return closest_neighbor_around(query, voxel, voxel_size_, [&](const std::size_t shift) {
    const std::uint32_t bit = std::uint32_t{1} << shift;
    if ((cache.looked_up & bit) == 0) {
      const auto search = map_.find(voxel + shifts[shift]);
      cache.points[shift] = search != map_.end() ? search.value().data() : nullptr;
      cache.sizes[shift] = search != map_.end() ? static_cast<std::uint32_t>(search.value().size()) : 0U;
      cache.looked_up |= bit;
    }
    return std::pair<const Eigen::Vector3d*, std::size_t>(cache.points[shift], cache.sizes[shift]);
  });
}

std::tuple<Eigen::Vector3d, double> VoxelHashMap::get_closest_neighbor(const Eigen::Vector3d& query,
                                                                        int voxel_search_radius) const {
  const int radius = std::max(1, voxel_search_radius);
  if (radius == 1) {
    return get_closest_neighbor(query);
  }
  Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
  double closest_squared_distance = std::numeric_limits<double>::max();
  const Voxel voxel = point_to_voxel(query, inv_voxel_size_);
  const Eigen::Vector3d fractional = query - voxel.cast<double>() * voxel_size_;
  for (int dx = -radius; dx <= radius; ++dx) {
    const double bound_x = axis_bound_sq(fractional.x(), voxel_size_, dx);
    if (bound_x >= closest_squared_distance) {
      continue;
    }
    for (int dy = -radius; dy <= radius; ++dy) {
      const double bound_xy = bound_x + axis_bound_sq(fractional.y(), voxel_size_, dy);
      if (bound_xy >= closest_squared_distance) {
        continue;
      }
      for (int dz = -radius; dz <= radius; ++dz) {
        const double bound_sq = bound_xy + axis_bound_sq(fractional.z(), voxel_size_, dz);
        if (bound_sq >= closest_squared_distance) {
          continue;
        }
        const Voxel query_voxel = voxel + Voxel{dx, dy, dz};
        const auto search = map_.find(query_voxel);
        if (search == map_.end()) {
          continue;
        }
        const VoxelBlock& voxel_points = search.value();
        for (const Eigen::Vector3d& point : voxel_points) {
          const double squared_distance = (point - query).squaredNorm();
          if (squared_distance < closest_squared_distance) {
            closest_neighbor = point;
            closest_squared_distance = squared_distance;
          }
        }
      }
    }
  }
  const double closest_distance = closest_squared_distance == std::numeric_limits<double>::max()
                                       ? std::numeric_limits<double>::max()
                                       : std::sqrt(closest_squared_distance);
  return std::make_tuple(closest_neighbor, closest_distance);
}

void VoxelHashMap::add_points(const std::vector<Eigen::Vector3d>& points) {
  const double map_resolution_sq = voxel_size_ * voxel_size_ / max_points_per_voxel_;
  std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d& p) {
    const Voxel voxel = point_to_voxel(p, inv_voxel_size_);
    auto it = map_.try_emplace(voxel).first;
    VoxelBlock& voxel_points = it.value();
    if (voxel_points.size() == max_points_per_voxel_ ||
        std::any_of(voxel_points.cbegin(), voxel_points.cend(),
                    [&](const auto& voxel_point) { return (voxel_point - p).squaredNorm() < map_resolution_sq; })) {
      return;
    }
    if (voxel_points.empty()) {
      voxel_fronts_.push_back({p, voxel});
      far_bound_sq_ = std::max(far_bound_sq_, (p - far_bound_origin_).squaredNorm());
    }
    voxel_points.reserve(max_points_per_voxel_);
    voxel_points.emplace_back(p);
    if (maintain_normals_) {
      refresh_normal(voxel, voxel_points);
    }
  });
}

void VoxelHashMap::refresh_normal(const Voxel& voxel, const VoxelBlock& points) {
  if (points.size() < min_normal_points_) {
    return; // too few samples: keep any stale normal erased
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    mean += p;
  }
  mean /= static_cast<double>(points.size());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    const Eigen::Vector3d d = p - mean;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<double>(points.size());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  const Eigen::Vector3d& eigenvalues = solver.eigenvalues(); // ascending
  // Surface-like requires a genuine two-directional spread: the mid eigenvalue must
  // dominate the smallest (flat) AND be a real fraction of the largest (rules out
  // lines, whose two small eigenvalues are both ~0 and whose "normal" direction is
  // arbitrary within the perpendicular plane).
  if (eigenvalues(1) < planarity_ratio_ * eigenvalues(0) ||
      eigenvalues(1) < min_mid_to_major_ratio_ * eigenvalues(2)) {
    normal_map_.erase(voxel); // not surface-like (line/blob): no trustworthy normal
    return;
  }
  normal_map_.insert_or_assign(voxel, solver.eigenvectors().col(0));
}

std::optional<Eigen::Vector3d> VoxelHashMap::voxel_normal(const Eigen::Vector3d& point) const {
  const auto search = normal_map_.find(point_to_voxel(point, inv_voxel_size_));
  if (search == normal_map_.end()) {
    return std::nullopt;
  }
  return search->second;
}

void VoxelHashMap::remove_points_far_from_location(const Eigen::Vector3d& origin) {
  // When no first point can be beyond the clipping distance, nothing is erased. The relative
  // margin is far larger than the rounding of these distances.
  if (std::sqrt(far_bound_sq_) + (origin - far_bound_origin_).norm() < clipping_distance_ * (1.0 - 1e-9)) {
    return;
  }
  const double clipping_distance_sq = clipping_distance_ * clipping_distance_;
  const auto distance_sq = [&](const VoxelFront& front) { return (front.point - origin).squaredNorm(); };
  // The largest distance among the voxels that stay, and whether any voxel goes. Both reductions
  // are exact, so the split across threads does not matter.
  using FarScan = std::pair<double, bool>;
  const auto [kept_sq, any_far] = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>{0, voxel_fronts_.size()}, FarScan{0.0, false},
      [&](const tbb::blocked_range<std::size_t>& r, FarScan scan) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const double d = distance_sq(voxel_fronts_[i]);
          if (d > clipping_distance_sq) {
            scan.second = true;
          } else {
            scan.first = std::max(scan.first, d);
          }
        }
        return scan;
      },
      [](const FarScan& a, const FarScan& b) { return FarScan{std::max(a.first, b.first), a.second || b.second}; });
  if (any_far) {
    // Erasing the far voxels in this order instead of map_'s iteration order leaves map_ the same:
    // backward-shift deletion keeps the remaining entries in their order, each as close to its
    // ideal bucket as that order allows, whatever the order of the deletions.
    for (std::size_t i = 0; i < voxel_fronts_.size();) {
      if (distance_sq(voxel_fronts_[i]) > clipping_distance_sq) {
        normal_map_.erase(voxel_fronts_[i].voxel);
        map_.erase(voxel_fronts_[i].voxel);
        voxel_fronts_[i] = voxel_fronts_.back();
        voxel_fronts_.pop_back();
      } else {
        ++i;
      }
    }
  }
  far_bound_origin_ = origin;
  far_bound_sq_ = kept_sq;
}

void VoxelHashMap::update(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& pose) {
  std::vector<Eigen::Vector3d> points_transformed(points.size());
  // Each point is transformed on its own; add_points() below keeps the order of the points.
  tbb::parallel_for(tbb::blocked_range<std::size_t>{0, points.size()}, [&](const tbb::blocked_range<std::size_t>& r) {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      points_transformed[i] = pose * points[i];
    }
  });
  const Eigen::Vector3d& origin = pose.translation();
  add_points(points_transformed);
  remove_points_far_from_location(origin);
}

void VoxelHashMap::transform(const Sophus::SE3d& transform) {
  std::vector<Eigen::Vector3d> points = pointcloud();
  for (Eigen::Vector3d& point : points) {
    point = transform * point;
  }
  clear();
  add_points(points);
}

std::vector<Eigen::Vector3d> VoxelHashMap::pointcloud() const {
  std::vector<Eigen::Vector3d> point_cloud;
  point_cloud.reserve(map_.size() * max_points_per_voxel_);
  for (const auto& [_, block] : map_) {
    point_cloud.insert(point_cloud.end(), block.cbegin(), block.cend());
  }
  return point_cloud;
}

} // namespace rko_lio::core
