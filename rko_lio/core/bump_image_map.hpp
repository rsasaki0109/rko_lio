/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, Autonomous Systems Lab, ETH Zurich
 * Copyright (c) 2026, rsasaki0109
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Voxel-wise oriented height ("bump") images and the registration residual on them.
//
// Port of the map, residual and map-informed sampling of BIEVR-LIO (Pfreundschuh et
// al., arXiv 2604.14421; https://github.com/ethz-asl/BIEVR-LIO, BSD-3-Clause). Each
// voxel fits a plane to its points and stores, on a fine pixel grid in that plane, the
// height of the surface above it. A scan point is registered against the height at its
// pixel, so subtle relief such as grass or road texture constrains the pose where a
// plane fit or a point-to-point match sees only a flat surface.
#pragma once

#include "voxel_down_sample.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <sophus/se3.hpp>
#include <tsl/robin_map.h>
#include <vector>

namespace rko_lio::core {

struct BumpImageMapConfig {
  double voxel_size = 0.5;   // voxel side length (m)
  double pixel_size = 0.05;  // bump image pixel side length (m)
  bool weighted = true;      // weight pixel updates by inverse range
  bool smooth = true;        // Gaussian-smooth the bump image over valid pixels
  double normal_tolerance_deg = 3.0;  // reproject a voxel's image when its normal turns further
  std::size_t max_voxels = 1500000;   // least recently updated voxels are dropped beyond this
};

struct BumpVoxel {
  bool observed = false;
  /** Plane frame (z along the normal, origin at the mean) and image frame (origin at the
   *  image corner), both from world. */
  Sophus::SE3d plane_from_world;
  Sophus::SE3d image_from_world;
  Eigen::MatrixXf heights;
  Eigen::MatrixXf smoothed;
  Eigen::MatrixXf weights;
  Eigen::Matrix3d outer_sum = Eigen::Matrix3d::Zero();
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  std::size_t num_points = 0;
  /** Mean absolute height of the valid pixels: how much relief the voxel holds. */
  double relief = 0.0;
  std::uint64_t last_update = 0;
  /** Points kept until the voxel has enough of them for a normal. */
  std::vector<Eigen::Vector4d> pending;
};

class BumpImageMap {
public:
  explicit BumpImageMap(BumpImageMapConfig config = {});

  /** Adds world points; `ranges` (same size) weight the pixel updates. */
  void integrate(const std::vector<Eigen::Vector3d>& world_points, const std::vector<double>& ranges);

  /** The observed voxel containing `point`, else the observed voxel among its six face
   *  neighbours whose centroid is closest; nullptr when there is none. */
  const BumpVoxel* find(const Eigen::Vector3d& point) const;
  /** The observed voxel containing `point`, without the neighbour fallback. */
  const BumpVoxel* at(const Eigen::Vector3d& point) const { return observed(voxel_of(point)); }

  bool empty() const { return map_.empty(); }
  std::size_t size() const { return map_.size(); }
  void clear() { map_.clear(); }
  const BumpImageMapConfig& config() const { return config_; }

private:
  Eigen::Vector3i voxel_of(const Eigen::Vector3d& point) const;
  const BumpVoxel* observed(const Eigen::Vector3i& voxel) const;
  bool update_normal(BumpVoxel& voxel) const;
  void update_image(const std::vector<Eigen::Vector4d>& points, BumpVoxel& voxel, bool normal_changed,
                    const Eigen::Vector3i& index) const;

  BumpImageMapConfig config_;
  double cos_normal_tolerance_ = 1.0;
  Eigen::MatrixXf gauss_kernel_;
  std::uint64_t update_counter_ = 0;
  tsl::robin_map<Eigen::Vector3i, BumpVoxel, VoxelHash> map_;
};

/** Height of the bump image at continuous pixel (x, y) by validity-weighted bilinear
 *  interpolation, and its central-difference gradient (per pixel; zero where a
 *  neighbour is missing). Returns false when no valid pixel surrounds (x, y). */
bool bump_height_and_gradient(const BumpVoxel& voxel, double x, double y, double& height, double& dx, double& dy);

struct BumpImageSystem {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  double chi = 0.0;
  int residuals = 0;
};

/**
 * Gauss-Newton terms of the point-to-bump-image residual r = z - h(x, y) for `points`
 * (base frame) at `world_from_base`, Huber-weighted with `huber_delta`. The update is
 * applied as exp(dx) * pose with dx = [translation, rotation], as in rko_lio's ICP.
 */
BumpImageSystem build_bump_image_system(const BumpImageMap& map,
                                        const std::vector<Eigen::Vector3d>& points,
                                        const Sophus::SE3d& world_from_base,
                                        double huber_delta);

/**
 * Map-informed sampling: keeps every point of the `informed_voxels` map voxels with the
 * most relief (among those the scan hits with more than one point) and one point per
 * remaining voxel. `points` are base frame, placed with `world_from_base`.
 */
std::vector<Eigen::Vector3d> sample_informed_points(const BumpImageMap& map,
                                                    const std::vector<Eigen::Vector3d>& points,
                                                    const Sophus::SE3d& world_from_base,
                                                    std::size_t informed_voxels);

} // namespace rko_lio::core
