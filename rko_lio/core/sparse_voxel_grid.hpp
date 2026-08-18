// straight copy from kinematic_icp, thanks Tiziano
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
#pragma once
#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cstddef>
#include <sophus/se3.hpp>

namespace rko_lio::core {

using VoxelBlock = std::vector<Eigen::Vector3d>;

struct SparseVoxelGrid {
  explicit SparseVoxelGrid(const double voxel_size,
                           const double clipping_distance,
                           const unsigned int max_points_per_voxel);

  void Clear() { map_.clear(Bonxai::ClearOption::CLEAR_MEMORY); }
  bool Empty() const { return map_.activeCellsCount() == 0; }
  void Update(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& pose);
  void AddPoints(const std::vector<Eigen::Vector3d>& points);
  void RemovePointsFarFromLocation(const Eigen::Vector3d& origin);
  std::vector<Eigen::Vector3d> Pointcloud() const;
  // [instrumentation, additive-only] Cheap map-growth gauge helper: sums
  // per-voxel point counts without copying any point data (unlike
  // Pointcloud()), so it is safe to call periodically for logging.
  std::size_t ActivePointCount() const;
  // [NN-search optimization] Returns the exact closest point to `query`
  // within the fixed voxel neighborhood (searched by squared distance, one
  // final sqrt on the winner). Uses branch-and-bound AABB pruning to skip
  // voxels that cannot improve on the current best without touching the
  // hash map or scanning their points; voxel visitation order is left
  // unchanged from the pre-optimization implementation so the result
  // (including exact-tie resolution) is bit-for-bit identical to a naive
  // exhaustive scan. See the .cpp for the full exactness argument.
  std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d& query) const;
  std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d& query, int voxel_search_radius) const;

  double voxel_size_;
  double clipping_distance_;
  unsigned int max_points_per_voxel_;
  Bonxai::VoxelGrid<VoxelBlock> map_;

};

} // namespace rko_lio::core
