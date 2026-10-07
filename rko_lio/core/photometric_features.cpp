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

#include "photometric_features.hpp"

#include <Eigen/Eigenvalues>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace rko_lio::core {

namespace {

// Patch pixel offsets (du, dv), in COIN-LIO's order: columns outer, rows inner.
std::vector<std::pair<int, int>> patch_offsets(int patch_size) {
  std::vector<std::pair<int, int>> offsets;
  const int half = patch_size / 2;
  for (int du = -half; du <= half; ++du) {
    for (int dv = -half; dv <= half; ++dv) {
      offsets.emplace_back(du, dv);
    }
  }
  return offsets;
}

bool inside_margin(const Eigen::Vector2d& uv, int rows, int cols, int margin) {
  return uv.x() >= margin && uv.x() <= cols - margin && uv.y() >= margin && uv.y() <= rows - margin;
}

// Dominant gradient direction of the patch around (row, col): eigenvector of
// the larger eigenvalue of the 5x5 structure tensor of 3x3 Sobel derivatives
// (cv::cornerEigenValsAndVecs(blockSize = 5, ksize = 3)).
Eigen::Vector2d dominant_gradient_direction(const FloatImage& image, int row, int col) {
  double sxx = 0.0;
  double sxy = 0.0;
  double syy = 0.0;
  for (int dr = -2; dr <= 2; ++dr) {
    for (int dc = -2; dc <= 2; ++dc) {
      const int r = std::clamp(row + dr, 1, image.rows - 2);
      const int c = std::clamp(col + dc, 1, image.cols - 2);
      const double gx = (image.at(r - 1, c + 1) + 2 * image.at(r, c + 1) + image.at(r + 1, c + 1)) -
                        (image.at(r - 1, c - 1) + 2 * image.at(r, c - 1) + image.at(r + 1, c - 1));
      const double gy = (image.at(r + 1, c - 1) + 2 * image.at(r + 1, c) + image.at(r + 1, c + 1)) -
                        (image.at(r - 1, c - 1) + 2 * image.at(r - 1, c) + image.at(r - 1, c + 1));
      sxx += gx * gx;
      sxy += gx * gy;
      syy += gy * gy;
    }
  }
  Eigen::Matrix2d tensor;
  tensor << sxx, sxy, sxy, syy;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(tensor);
  return solver.eigenvectors().col(1); // eigenvalues ascend
}

double ncc(const std::vector<float>& a, const std::vector<double>& b) {
  const double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
  const double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
  double cross = 0.0;
  double var_a = 0.0;
  double var_b = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    cross += (a[i] - mean_a) * (b[i] - mean_b);
    var_a += (a[i] - mean_a) * (a[i] - mean_a);
    var_b += (b[i] - mean_b) * (b[i] - mean_b);
  }
  const double denominator = std::sqrt(var_a * var_b);
  return denominator > 0.0 ? cross / denominator : 0.0;
}

} // namespace

void PhotometricFeatureManager::update(const PhotometricFrame& frame,
                                       const LidarImageModel& model,
                                       const Sophus::SE3d& world_from_cloud,
                                       const std::vector<Eigen::Vector3d>& weak_directions_cloud) {
  removed_last_ = 0;
  added_last_ = 0;
  if (!frame.valid()) {
    return;
  }
  track(frame, model, world_from_cloud);
  detect(frame, model, world_from_cloud, weak_directions_cloud);
}

void PhotometricFeatureManager::track(const PhotometricFrame& frame,
                                      const LidarImageModel& model,
                                      const Sophus::SE3d& world_from_cloud) {
  const Sophus::SE3d cloud_from_world = world_from_cloud.inverse();
  std::vector<PhotometricFeature> kept;
  kept.reserve(features_.size());
  for (PhotometricFeature& feature : features_) {
    std::vector<double> current;
    current.reserve(feature.world_points.size());
    std::vector<Eigen::Vector2d> pixels;
    pixels.reserve(feature.world_points.size());
    bool visible = true;
    for (const Eigen::Vector3d& world_point : feature.world_points) {
      CapturedProjection projection;
      if (!project_captured(frame, model, cloud_from_world * world_point, false, projection) ||
          !inside_margin(projection.uv, frame.rows, frame.cols, config_.margin)) {
        visible = false;
        break;
      }
      const int row = static_cast<int>(projection.uv.y());
      const int col = static_cast<int>(projection.uv.x());
      if (frame.masked(row, col)) {
        visible = false;
        break;
      }
      // Occluded, or the surface is gone.
      const double expected_range = model.to_lidar_origin(projection.point_captured).norm();
      if (std::abs(expected_range - frame.range.at(row, col)) > config_.range_threshold_m) {
        visible = false;
        break;
      }
      current.push_back(frame.intensity.bilinear(projection.uv.x(), projection.uv.y()));
      pixels.push_back(projection.uv);
    }
    if (visible && feature.lifetime < config_.max_lifetime && ncc(feature.intensities, current) > config_.ncc_threshold) {
      feature.lifetime += 1;
      feature.center = pixels[pixels.size() / 2];
      kept.push_back(std::move(feature));
    } else {
      ++removed_last_;
    }
  }
  features_ = std::move(kept);
}

void PhotometricFeatureManager::detect(const PhotometricFrame& frame,
                                       const LidarImageModel& model,
                                       const Sophus::SE3d& world_from_cloud,
                                       const std::vector<Eigen::Vector3d>& weak_directions_cloud) {
  const int wanted = config_.num_features - static_cast<int>(features_.size());
  if (wanted <= 0 || weak_directions_cloud.empty()) {
    return;
  }
  const int rows = frame.rows;
  const int cols = frame.cols;
  // Selectable pixels: valid image area, away from the border and from existing patches.
  std::vector<std::uint8_t> selectable(static_cast<std::size_t>(rows) * cols, 0U);
  for (int r = config_.margin; r < rows - config_.margin; ++r) {
    for (int c = config_.margin; c < cols - config_.margin; ++c) {
      selectable[static_cast<std::size_t>(r) * cols + c] = frame.masked(r, c) ? 0U : 1U;
    }
  }
  auto suppress = [&](std::vector<std::uint8_t>& mask, const Eigen::Vector2d& center) {
    const int radius = config_.suppression_radius;
    const int cr = static_cast<int>(std::lround(center.y()));
    const int cc = static_cast<int>(std::lround(center.x()));
    for (int r = std::max(0, cr - radius); r <= std::min(rows - 1, cr + radius); ++r) {
      for (int c = std::max(0, cc - radius); c <= std::min(cols - 1, cc + radius); ++c) {
        if ((r - cr) * (r - cr) + (c - cc) * (c - cc) <= radius * radius) {
          mask[static_cast<std::size_t>(r) * cols + c] = 0U;
        }
      }
    }
  };
  for (const PhotometricFeature& feature : features_) {
    suppress(selectable, feature.center);
  }

  // Strong-gradient candidates, strongest first, thinned by non-maximum suppression.
  std::vector<std::pair<double, int>> scored;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t pixel = static_cast<std::size_t>(r) * cols + c;
      if (selectable[pixel] == 0U) {
        continue;
      }
      const double gradient =
          std::min(255.0, std::round(0.5 * std::abs(frame.dx.at(r, c)) + 0.5 * std::abs(frame.dy.at(r, c))));
      if (gradient > config_.grad_min) {
        scored.emplace_back(gradient, static_cast<int>(pixel));
      }
    }
  }
  std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<int> candidates;
  for (const auto& [score, pixel] : scored) {
    if (selectable[pixel] == 0U) {
      continue;
    }
    candidates.push_back(pixel);
    suppress(selectable, Eigen::Vector2d(pixel % cols, pixel / cols));
  }

  // Score each candidate per weak direction: how well its gradient sees motion along it.
  const std::size_t n_directions = weak_directions_cloud.size();
  std::vector<std::vector<std::pair<double, int>>> by_direction(n_directions);
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const int row = candidates[i] / cols;
    const int col = candidates[i] % cols;
    const int index = frame.pixel_point[candidates[i]];
    if (index < 0 || !frame.points_end[index].allFinite()) {
      continue;
    }
    const Eigen::Vector2d gradient = dominant_gradient_direction(frame.intensity, row, col);
    const Eigen::Matrix<double, 2, 3> du_dp = model.projection_jacobian(model.to_lidar_origin(frame.points_end[index]));
    for (std::size_t d = 0; d < n_directions; ++d) {
      Eigen::Vector2d flow = du_dp * weak_directions_cloud[d];
      if (flow.norm() < 1e-12) {
        continue;
      }
      flow.normalize();
      by_direction[d].emplace_back(std::abs(gradient.dot(flow)), i);
    }
  }
  for (auto& list : by_direction) {
    std::stable_sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  }
  // Round robin over the directions so every weak direction gets patches.
  std::vector<int> chosen;
  std::vector<std::uint8_t> taken(candidates.size(), 0U);
  std::size_t longest = 0;
  for (const auto& list : by_direction) {
    longest = std::max(longest, list.size());
  }
  for (std::size_t rank = 0; rank < longest && static_cast<int>(chosen.size()) < wanted; ++rank) {
    for (std::size_t d = 0; d < n_directions && static_cast<int>(chosen.size()) < wanted; ++d) {
      if (rank >= by_direction[d].size()) {
        continue;
      }
      const int i = by_direction[d][rank].second;
      if (taken[i] == 0U) {
        taken[i] = 1U;
        chosen.push_back(i);
      }
    }
  }

  const auto offsets = patch_offsets(config_.patch_size);
  for (const int i : chosen) {
    const int row = candidates[i] / cols;
    const int col = candidates[i] % cols;
    PhotometricFeature feature;
    feature.center = Eigen::Vector2d(col, row);
    bool complete = true;
    for (const auto& [du, dv] : offsets) {
      const int r = row + dv;
      const int c = col + du;
      const int index = frame.pixel_point[static_cast<std::size_t>(r) * cols + c];
      if (index < 0 || !frame.points_end[index].allFinite()) {
        complete = false;
        break;
      }
      feature.world_points.push_back(world_from_cloud * frame.points_end[index]);
      feature.intensities.push_back(frame.intensity.at(r, c));
    }
    if (complete) {
      features_.push_back(std::move(feature));
      ++added_last_;
    }
  }
}

PhotometricSystem build_photometric_system(const std::vector<PhotometricFeature>& features,
                                           const PhotometricFrame& frame,
                                           const LidarImageModel& model,
                                           const PhotometricFeatureConfig& config,
                                           const Sophus::SE3d& world_from_base,
                                           const Sophus::SE3d& base_from_cloud) {
  PhotometricSystem system;
  if (!frame.valid()) {
    return system;
  }
  const Sophus::SE3d cloud_from_world = (world_from_base * base_from_cloud).inverse();
  const Eigen::Matrix3d R_base_world = world_from_base.so3().inverse().matrix();
  const Eigen::Matrix3d R_cloud_base = base_from_cloud.so3().inverse().matrix();
  for (const PhotometricFeature& feature : features) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    double chi = 0.0;
    bool complete = true;
    for (std::size_t l = 0; l < feature.world_points.size(); ++l) {
      const Eigen::Vector3d& world_point = feature.world_points[l];
      CapturedProjection projection;
      if (!project_captured(frame, model, cloud_from_world * world_point, true, projection)) {
        complete = false;
        break;
      }
      const Eigen::Vector3d lidar_point = model.to_lidar_origin(projection.point_captured);
      const double range = lidar_point.norm();
      const Eigen::Vector2d& uv = projection.uv;
      if (range < config.min_range_m || range > config.max_range_m ||
          !(uv.x() > config.margin && uv.x() < frame.cols - config.margin && uv.y() > config.margin &&
            uv.y() < frame.rows - config.margin) ||
          frame.masked(static_cast<int>(uv.y()), static_cast<int>(uv.x()))) {
        complete = false;
        break;
      }
      const double dI_du = 0.5 * (frame.intensity.bilinear(uv.x() + 1, uv.y()) - frame.intensity.bilinear(uv.x() - 1, uv.y()));
      const double dI_dv = 0.5 * (frame.intensity.bilinear(uv.x(), uv.y() + 1) - frame.intensity.bilinear(uv.x(), uv.y() - 1));
      const Eigen::RowVector2d dI(dI_du, dI_dv);
      // d(point in the capture frame) / d(perturbation of world_from_base)
      const Eigen::Matrix3d R_capture = frame.capture_from_end[projection.capture_slot].so3().matrix();
      const Eigen::Matrix3d to_capture = R_capture * R_cloud_base * R_base_world;
      Eigen::Matrix<double, 3, 6> dp_dx;
      dp_dx.leftCols<3>() = -to_capture;
      dp_dx.rightCols<3>() = to_capture * Sophus::SO3d::hat(world_point);
      const Eigen::Matrix<double, 1, 6> J = dI * model.projection_jacobian(lidar_point) * dp_dx;
      const double residual = frame.intensity.bilinear(uv.x(), uv.y()) - feature.intensities[l];
      H += J.transpose() * J;
      b += J.transpose() * residual;
      chi += residual * residual;
    }
    if (complete) {
      system.H += H;
      system.b += b;
      system.chi += chi;
      system.residuals += static_cast<int>(feature.world_points.size());
      system.patches += 1;
    }
  }
  return system;
}

std::vector<Eigen::Vector3d> sample_surface_normals(const PhotometricFrame& frame, int row_step, int col_step) {
  std::vector<Eigen::Vector3d> normals;
  if (!frame.valid()) {
    return normals;
  }
  auto point_at = [&](int r, int c) -> const Eigen::Vector3d* {
    if (r < 0 || r >= frame.rows) {
      return nullptr;
    }
    c = (c % frame.cols + frame.cols) % frame.cols;
    const int index = frame.pixel_point[static_cast<std::size_t>(r) * frame.cols + c];
    if (index < 0 || !frame.points_end[index].allFinite()) {
      return nullptr;
    }
    return &frame.points_end[index];
  };
  for (int r = 2; r < frame.rows - 2; r += row_step) {
    for (int c = 0; c < frame.cols; c += col_step) {
      const Eigen::Vector3d* center = point_at(r, c);
      const Eigen::Vector3d* left = point_at(r, c - 2);
      const Eigen::Vector3d* right = point_at(r, c + 2);
      const Eigen::Vector3d* up = point_at(r - 2, c);
      const Eigen::Vector3d* down = point_at(r + 2, c);
      if (!center || !left || !right || !up || !down) {
        continue;
      }
      // Skip depth discontinuities: neighbours must lie within 10% of the centre range.
      const double range = center->norm();
      const double tolerance = 0.1 * range + 0.05;
      if ((*left - *center).norm() > tolerance || (*right - *center).norm() > tolerance ||
          (*up - *center).norm() > tolerance || (*down - *center).norm() > tolerance) {
        continue;
      }
      const Eigen::Vector3d normal = (*right - *left).cross(*down - *up);
      if (normal.norm() > 1e-9) {
        normals.push_back(normal.normalized());
      }
    }
  }
  return normals;
}

std::vector<Eigen::Vector3d> weak_translation_directions(const std::vector<Eigen::Vector3d>& translation_rows,
                                                         const double min_contribution) {
  const std::vector<Eigen::Vector3d> weak = find_weak_translation_directions(translation_rows, min_contribution);
  if (!weak.empty()) {
    return weak;
  }
  return {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()};
}

std::vector<Eigen::Vector3d> find_weak_translation_directions(const std::vector<Eigen::Vector3d>& translation_rows,
                                                              const double min_contribution) {
  if (translation_rows.size() <= 3) {
    return {};
  }
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d& row : translation_rows) {
    information += row * row.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(information);
  const Eigen::Matrix3d directions = solver.eigenvectors();
  std::vector<Eigen::Vector3d> weak;
  for (int d = 0; d < 3; ++d) {
    double contribution = 0.0;
    for (const Eigen::Vector3d& row : translation_rows) {
      const double alignment = std::abs(row.normalized().dot(directions.col(d)));
      if (alignment > 0.5) {
        contribution += alignment;
      }
    }
    if (contribution < min_contribution) {
      weak.push_back(directions.col(d));
    }
  }
  return weak;
}

} // namespace rko_lio::core
