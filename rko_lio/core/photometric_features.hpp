/*
 * MIT License
 *
 * Copyright (c) 2026 rsasaki0109.
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

// Photometric patch features on LiDAR intensity images and the Gauss-Newton
// terms they add to scan registration.
//
// Follows COIN-LIO's complementary feature selection, patch tracking and
// photometric error (Pfreundschuh et al., ICRA 2024; reference code
// https://github.com/ethz-asl/COIN-LIO, BSD-3-Clause, Copyright (c) 2024
// Patrick Pfreundschuh). Patches are anchored to world points, keep the
// intensities of the scan they were created in, and are scored against each
// new scan's image. New patches are chosen where the image gradient moves
// along the directions the geometric registration observes worst.

#pragma once

#include "photometric_image.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <vector>

namespace rko_lio::core {

struct PhotometricFeatureConfig {
  int patch_size = 5;
  int num_features = 60;
  int max_lifetime = 25;
  int margin = 10;
  int suppression_radius = 10;
  double grad_min = 16.5;
  double ncc_threshold = 0.7075;
  double range_threshold_m = 0.2;
  double min_range_m = 0.7;
  double max_range_m = 30.0;
};

struct PhotometricFeature {
  /** World point and reference intensity of every patch pixel (row-major patch). */
  std::vector<Eigen::Vector3d> world_points;
  std::vector<float> intensities;
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  int lifetime = 1;
};

/** Sum of the photometric Gauss-Newton terms over the residual pixels. */
struct PhotometricSystem {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  double chi = 0.0;
  int residuals = 0;
  int patches = 0;
};

class PhotometricFeatureManager {
public:
  explicit PhotometricFeatureManager(PhotometricFeatureConfig config = {}) : config_(std::move(config)) {}

  const std::vector<PhotometricFeature>& features() const { return features_; }
  const PhotometricFeatureConfig& config() const { return config_; }
  void clear() { features_.clear(); }

  /**
   * After a scan is registered: drop patches that left the image, got occluded,
   * stopped matching (NCC) or aged out, then add patches along the weak directions.
   *
   * @param world_from_cloud pose of the cloud frame at the scan end
   * @param weak_directions_cloud unit directions (cloud frame) the geometry observes worst
   */
  void update(const PhotometricFrame& frame,
              const LidarImageModel& model,
              const Sophus::SE3d& world_from_cloud,
              const std::vector<Eigen::Vector3d>& weak_directions_cloud);

  /** Number of patches removed and added by the last update. */
  int removed_last() const { return removed_last_; }
  int added_last() const { return added_last_; }

private:
  void track(const PhotometricFrame& frame, const LidarImageModel& model, const Sophus::SE3d& world_from_cloud);
  void detect(const PhotometricFrame& frame,
              const LidarImageModel& model,
              const Sophus::SE3d& world_from_cloud,
              const std::vector<Eigen::Vector3d>& weak_directions_cloud);

  PhotometricFeatureConfig config_;
  std::vector<PhotometricFeature> features_;
  int removed_last_ = 0;
  int added_last_ = 0;
};

/**
 * Photometric terms for the pose perturbation used by rko_lio's ICP:
 * world_from_base <- exp(dx) * world_from_base, dx = [translation, rotation].
 * Residual per patch pixel: current image intensity at the reprojected world point
 * minus the patch's reference intensity. A patch contributes only when all of
 * its pixels project into valid image area.
 */
PhotometricSystem build_photometric_system(const std::vector<PhotometricFeature>& features,
                                           const PhotometricFrame& frame,
                                           const LidarImageModel& model,
                                           const PhotometricFeatureConfig& config,
                                           const Sophus::SE3d& world_from_base,
                                           const Sophus::SE3d& base_from_cloud);

/**
 * Surface normals (cloud frame) sampled on a grid of the organized image from
 * neighbouring deskewed points; rko_lio's point-to-point ICP has no normals of
 * its own, and its translation information is isotropic per correspondence.
 */
std::vector<Eigen::Vector3d> sample_surface_normals(const PhotometricFrame& frame, int row_step, int col_step);

/**
 * Directions (unit, frame of the Jacobian rows) along which few geometric
 * residuals constrain translation: eigenvectors of sum(J^T J) whose summed
 * per-row alignment (|cos| > 0.5) stays below `min_contribution`, as COIN-LIO
 * counts them. Returns the three axes when none is weak.
 */
std::vector<Eigen::Vector3d> weak_translation_directions(const std::vector<Eigen::Vector3d>& translation_rows,
                                                         double min_contribution);

/** The weak directions alone: empty when every direction is constrained. */
std::vector<Eigen::Vector3d> find_weak_translation_directions(const std::vector<Eigen::Vector3d>& translation_rows,
                                                              double min_contribution);

} // namespace rko_lio::core
