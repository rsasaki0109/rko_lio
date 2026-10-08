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

#include "bump_image_map.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tuple>

namespace rko_lio::core {

namespace {

constexpr std::size_t kMinNormalPoints = 4;

bool voxel_less(const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
  return std::tie(a.x(), a.y(), a.z()) < std::tie(b.x(), b.y(), b.z());
}

Eigen::MatrixXf gaussian_kernel(const int radius) {
  const int size = 2 * radius + 1;
  // OpenCV's default sigma for this kernel size.
  const float sigma = 0.3f * ((size - 1) * 0.5f - 1.0f) + 0.8f;
  Eigen::MatrixXf kernel(size, size);
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      kernel(y + radius, x + radius) = std::exp(-static_cast<float>(x * x + y * y) / (2.0f * sigma * sigma));
    }
  }
  return kernel / kernel.sum();
}

// Validity-weighted bilinear interpolation of `image` over the 2x2 block at (x0, y0).
bool interpolate(const Eigen::MatrixXf& image,
                 const Eigen::MatrixXf& weights,
                 const int x0,
                 const int y0,
                 const double fx,
                 const double fy,
                 double& value) {
  const double w00 = (1.0 - fx) * (1.0 - fy) * (weights(y0, x0) > 0.0f);
  const double w01 = fx * (1.0 - fy) * (weights(y0, x0 + 1) > 0.0f);
  const double w10 = (1.0 - fx) * fy * (weights(y0 + 1, x0) > 0.0f);
  const double w11 = fx * fy * (weights(y0 + 1, x0 + 1) > 0.0f);
  const double sum = w00 + w01 + w10 + w11;
  if (sum == 0.0) return false;
  value = (w00 * image(y0, x0) + w01 * image(y0, x0 + 1) + w10 * image(y0 + 1, x0) +
           w11 * image(y0 + 1, x0 + 1)) /
          sum;
  return true;
}

} // namespace

BumpImageMap::BumpImageMap(BumpImageMapConfig config)
    : config_(config),
      cos_normal_tolerance_(std::cos(config.normal_tolerance_deg * M_PI / 180.0)),
      gauss_kernel_(gaussian_kernel(1)) {}

Eigen::Vector3i BumpImageMap::voxel_of(const Eigen::Vector3d& point) const {
  return point_to_voxel(point, 1.0 / config_.voxel_size);
}

const BumpVoxel* BumpImageMap::observed(const Eigen::Vector3i& voxel) const {
  const auto it = map_.find(voxel);
  return it != map_.end() && it->second.observed ? &it->second : nullptr;
}

const BumpVoxel* BumpImageMap::find(const Eigen::Vector3d& point) const {
  const Eigen::Vector3i center = voxel_of(point);
  if (const BumpVoxel* voxel = observed(center)) return voxel;
  static const std::array<Eigen::Vector3i, 6> kFaces = {
      Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(-1, 0, 0), Eigen::Vector3i(0, 1, 0),
      Eigen::Vector3i(0, -1, 0), Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(0, 0, -1)};
  const BumpVoxel* nearest = nullptr;
  double nearest_distance = std::numeric_limits<double>::max();
  for (const Eigen::Vector3i& offset : kFaces) {
    const BumpVoxel* voxel = observed(center + offset);
    if (voxel == nullptr) continue;
    const double distance = (point - voxel->sum / static_cast<double>(voxel->num_points)).squaredNorm();
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest = voxel;
    }
  }
  return nearest;
}

void BumpImageMap::integrate(const std::vector<Eigen::Vector3d>& world_points, const std::vector<double>& ranges) {
  if (world_points.empty()) return;
  ++update_counter_;

  // Group the points by voxel, in a deterministic order.
  std::vector<std::pair<Eigen::Vector3i, Eigen::Vector4d>> hashed(world_points.size());
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, world_points.size()), [&](const auto& r) {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      hashed[i] = {voxel_of(world_points[i]), Eigen::Vector4d(world_points[i].x(), world_points[i].y(),
                                                              world_points[i].z(), ranges[i])};
    }
  });
  std::sort(hashed.begin(), hashed.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return voxel_less(a.first, b.first);
    return a.second.x() < b.second.x();
  });

  std::vector<std::size_t> group_starts;
  for (std::size_t i = 0; i < hashed.size(); ++i) {
    if (i == 0 || hashed[i].first != hashed[i - 1].first) group_starts.push_back(i);
  }
  // Insert every voxel before taking pointers: an insertion may rehash the map.
  for (const std::size_t start : group_starts) map_.try_emplace(hashed[start].first);
  std::vector<BumpVoxel*> voxels(group_starts.size());
  for (std::size_t g = 0; g < group_starts.size(); ++g) voxels[g] = &map_.find(hashed[group_starts[g]].first).value();

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, group_starts.size()), [&](const auto& r) {
    std::vector<Eigen::Vector4d> points;
    for (std::size_t g = r.begin(); g != r.end(); ++g) {
      const std::size_t start = group_starts[g];
      const std::size_t end = g + 1 < group_starts.size() ? group_starts[g + 1] : hashed.size();
      BumpVoxel& voxel = *voxels[g];
      voxel.last_update = update_counter_;
      points.clear();
      for (std::size_t i = start; i < end; ++i) {
        const Eigen::Vector3d p = hashed[i].second.head<3>();
        voxel.sum += p;
        voxel.outer_sum += p * p.transpose();
        ++voxel.num_points;
        points.push_back(hashed[i].second);
      }
      const bool was_observed = voxel.observed;
      const bool normal_changed = update_normal(voxel);
      if (!voxel.observed) {
        // No normal yet, so the points cannot be projected; keep them until there is one.
        voxel.pending.insert(voxel.pending.end(), points.begin(), points.end());
        continue;
      }
      if (!was_observed) {
        points.insert(points.end(), voxel.pending.begin(), voxel.pending.end());
        voxel.pending.clear();
        voxel.pending.shrink_to_fit();
      }
      update_image(points, voxel, normal_changed, hashed[start].first);
    }
  });

  if (map_.size() > config_.max_voxels) {
    std::vector<std::pair<std::uint64_t, Eigen::Vector3i>> ages;
    ages.reserve(map_.size());
    for (const auto& [key, voxel] : map_) ages.emplace_back(voxel.last_update, key);
    const std::size_t remove = map_.size() - config_.max_voxels;
    std::nth_element(ages.begin(), ages.begin() + remove, ages.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return voxel_less(a.second, b.second);
    });
    for (std::size_t i = 0; i < remove; ++i) map_.erase(ages[i].second);
  }
}

bool BumpImageMap::update_normal(BumpVoxel& voxel) const {
  if (voxel.num_points < kMinNormalPoints) return false;
  const double n = static_cast<double>(voxel.num_points);
  const Eigen::Vector3d mean = voxel.sum / n;
  const Eigen::Matrix3d covariance = (voxel.outer_sum - voxel.sum * mean.transpose()) / (n - 1.0);
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  const Eigen::Vector3d normal = solver.eigenvectors().col(0);

  if (voxel.observed) {
    const Eigen::Vector3d previous = voxel.plane_from_world.so3().matrix().row(2).transpose();
    if (std::abs(previous.dot(normal)) >= cos_normal_tolerance_) return false;
  }
  const Eigen::Vector3d x_axis = normal.unitOrthogonal();
  Eigen::Matrix3d world_from_plane;
  world_from_plane.col(0) = x_axis;
  world_from_plane.col(1) = normal.cross(x_axis).normalized();
  world_from_plane.col(2) = normal;
  voxel.plane_from_world = Sophus::SE3d(Sophus::SO3d::fitToSO3(world_from_plane), mean).inverse();
  voxel.observed = true;
  return true;
}

void BumpImageMap::update_image(const std::vector<Eigen::Vector4d>& points,
                                BumpVoxel& voxel,
                                const bool normal_changed,
                                const Eigen::Vector3i& index) const {
  const double inv_pixel = 1.0 / config_.pixel_size;
  Eigen::MatrixXi changed;
  if (normal_changed) {
    // Size the image to the voxel's footprint on the new plane, then carry the old
    // surface over into it.
    double u_min = std::numeric_limits<double>::max(), u_max = std::numeric_limits<double>::lowest();
    double v_min = u_min, v_max = u_max;
    const Eigen::Vector3d origin = index.cast<double>() * config_.voxel_size;
    for (int corner = 0; corner < 8; ++corner) {
      const Eigen::Vector3d offset((corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
      const Eigen::Vector3d p = voxel.plane_from_world * (origin + config_.voxel_size * offset);
      u_min = std::min(u_min, p.x());
      u_max = std::max(u_max, p.x());
      v_min = std::min(v_min, p.y());
      v_max = std::max(v_max, p.y());
    }
    const int width = static_cast<int>(std::ceil((u_max - u_min) * inv_pixel)) + 1;
    const int height = static_cast<int>(std::ceil((v_max - v_min) * inv_pixel)) + 1;

    const Eigen::MatrixXf old_heights = voxel.heights;
    const Eigen::MatrixXf old_weights = voxel.weights;
    const Sophus::SE3d world_from_old_image = voxel.image_from_world.inverse();
    voxel.image_from_world = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(-u_min, -v_min, 0.0)) * voxel.plane_from_world;
    voxel.heights = Eigen::MatrixXf::Zero(height, width);
    voxel.smoothed = Eigen::MatrixXf::Zero(height, width);
    voxel.weights = Eigen::MatrixXf::Zero(height, width);
    changed = Eigen::MatrixXi::Zero(height, width);

    const Sophus::SE3d new_from_old = voxel.image_from_world * world_from_old_image;
    for (int i = 0; i < old_heights.rows(); ++i) {
      for (int j = 0; j < old_heights.cols(); ++j) {
        if (old_weights(i, j) == 0.0f) continue;
        const Eigen::Vector3d p =
            new_from_old * Eigen::Vector3d(j * config_.pixel_size, i * config_.pixel_size, old_heights(i, j));
        const int x = static_cast<int>(std::round(p.x() * inv_pixel));
        const int y = static_cast<int>(std::round(p.y() * inv_pixel));
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        voxel.heights(y, x) = static_cast<float>(p.z());
        voxel.weights(y, x) = old_weights(i, j);
        changed(y, x) = 1;
      }
    }
  } else {
    changed = Eigen::MatrixXi::Zero(voxel.heights.rows(), voxel.heights.cols());
  }

  for (const Eigen::Vector4d& point : points) {
    const Eigen::Vector3d p = voxel.image_from_world * Eigen::Vector3d(point.head<3>());
    const int x = static_cast<int>(std::round(p.x() * inv_pixel));
    const int y = static_cast<int>(std::round(p.y() * inv_pixel));
    if (x < 0 || x >= voxel.heights.cols() || y < 0 || y >= voxel.heights.rows()) continue;
    // Cap the weight so points close to the sensor do not dominate.
    const float weight = config_.weighted ? static_cast<float>(std::min(0.5, 1.0 / point.w())) : 1.0f;
    const float old_weight = voxel.weights(y, x);
    voxel.weights(y, x) += weight;
    voxel.heights(y, x) = (voxel.heights(y, x) * old_weight + weight * static_cast<float>(p.z())) / voxel.weights(y, x);
    changed(y, x) = 1;
  }

  // Smoothing also touches the valid neighbours of every changed pixel.
  Eigen::MatrixXi dirty = changed;
  if (!normal_changed) {
    for (int i = 0; i < changed.rows(); ++i) {
      for (int j = 0; j < changed.cols(); ++j) {
        if (changed(i, j) != 1) continue;
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            const int ii = i + di, jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= changed.rows() || jj >= changed.cols()) continue;
            if (voxel.weights(ii, jj) > 0.0f) dirty(ii, jj) = 1;
          }
        }
      }
    }
  }
  if (!config_.smooth) {
    voxel.smoothed = voxel.heights;
  } else {
    const int radius = static_cast<int>(gauss_kernel_.rows()) / 2;
    for (int y = 0; y < dirty.rows(); ++y) {
      for (int x = 0; x < dirty.cols(); ++x) {
        if (!dirty(y, x)) continue;
        float sum = 0.0f, weight_sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
          for (int kx = -radius; kx <= radius; ++kx) {
            const int yy = y + ky, xx = x + kx;
            if (xx < 0 || yy < 0 || xx >= dirty.cols() || yy >= dirty.rows()) continue;
            if (voxel.weights(yy, xx) <= 0.0f) continue;
            const float w = gauss_kernel_(ky + radius, kx + radius);
            sum += voxel.heights(yy, xx) * w;
            weight_sum += w;
          }
        }
        voxel.smoothed(y, x) = weight_sum > 0.0f ? sum / weight_sum : 0.0f;
      }
    }
  }

  int valid = 0;
  double relief = 0.0;
  for (int i = 0; i < voxel.heights.rows(); ++i) {
    for (int j = 0; j < voxel.heights.cols(); ++j) {
      if (voxel.weights(i, j) <= 0.0f) continue;
      ++valid;
      relief += std::abs(static_cast<double>(voxel.smoothed(i, j)));
    }
  }
  // Voxels with few observed pixels rarely give correspondences.
  voxel.relief = valid < 5 ? 0.0 : relief / valid;
}

bool bump_height_and_gradient(const BumpVoxel& voxel,
                              const double x,
                              const double y,
                              double& height,
                              double& dx,
                              double& dy) {
  const Eigen::MatrixXf& image = voxel.smoothed;
  const Eigen::MatrixXf& weights = voxel.weights;
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int max_x = static_cast<int>(image.cols()) - 1;
  const int max_y = static_cast<int>(image.rows()) - 1;
  if (x0 < 0 || y0 < 0 || x0 + 1 > max_x || y0 + 1 > max_y) return false;
  const double fx = x - x0;
  const double fy = y - y0;
  if (!interpolate(image, weights, x0, y0, fx, fy, height)) return false;

  // Central differences of the interpolated surface, one pixel either side.
  dx = 0.0;
  dy = 0.0;
  double minus = 0.0, plus = 0.0;
  if (x0 >= 1 && x0 + 2 <= max_x && interpolate(image, weights, x0 - 1, y0, fx, fy, minus) &&
      interpolate(image, weights, x0 + 1, y0, fx, fy, plus)) {
    dx = 0.5 * (plus - minus);
  }
  if (y0 >= 1 && y0 + 2 <= max_y && interpolate(image, weights, x0, y0 - 1, fx, fy, minus) &&
      interpolate(image, weights, x0, y0 + 1, fx, fy, plus)) {
    dy = 0.5 * (plus - minus);
  }
  return true;
}

BumpImageSystem build_bump_image_system(const BumpImageMap& map,
                                        const std::vector<Eigen::Vector3d>& points,
                                        const Sophus::SE3d& world_from_base,
                                        const double huber_delta) {
  const double inv_pixel = 1.0 / map.config().pixel_size;
  const auto add = [](BumpImageSystem lhs, const BumpImageSystem& rhs) {
    lhs.H += rhs.H;
    lhs.b += rhs.b;
    lhs.chi += rhs.chi;
    lhs.residuals += rhs.residuals;
    return lhs;
  };
  return tbb::parallel_deterministic_reduce(
      tbb::blocked_range<std::size_t>(0, points.size()), BumpImageSystem{},
      [&](const tbb::blocked_range<std::size_t>& r, BumpImageSystem system) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const Eigen::Vector3d p_world = world_from_base * points[i];
          const BumpVoxel* voxel = map.find(p_world);
          if (voxel == nullptr) continue;
          const Eigen::Vector3d p_image = voxel->image_from_world * p_world;
          double height = 0.0, gx = 0.0, gy = 0.0;
          if (!bump_height_and_gradient(*voxel, p_image.x() * inv_pixel, p_image.y() * inv_pixel, height, gx, gy)) {
            continue;
          }
          // d p_image / d dx for exp(dx) * pose, dx = [translation, rotation].
          Eigen::Matrix<double, 3, 6> J_world;
          J_world.leftCols<3>() = Eigen::Matrix3d::Identity();
          J_world.rightCols<3>() = -Sophus::SO3d::hat(p_world);
          const Eigen::Matrix<double, 3, 6> J_image = voxel->image_from_world.so3().matrix() * J_world;
          const Eigen::Matrix<double, 1, 6> J =
              J_image.row(2) - gx * inv_pixel * J_image.row(0) - gy * inv_pixel * J_image.row(1);
          const double residual = p_image.z() - height;
          const double abs_residual = std::abs(residual);
          const bool inlier = abs_residual <= huber_delta;
          const double weight = inlier ? 1.0 : huber_delta / abs_residual;
          system.H.noalias() += weight * J.transpose() * J;
          system.b.noalias() += weight * J.transpose() * residual;
          system.chi += inlier ? 0.5 * residual * residual : huber_delta * (abs_residual - 0.5 * huber_delta);
          ++system.residuals;
        }
        return system;
      },
      add);
}

std::vector<Eigen::Vector3d> sample_informed_points(const BumpImageMap& map,
                                                    const std::vector<Eigen::Vector3d>& points,
                                                    const Sophus::SE3d& world_from_base,
                                                    const std::size_t informed_voxels) {
  if (points.size() <= informed_voxels) return points;
  const double inv_voxel = 1.0 / map.config().voxel_size;
  std::vector<std::pair<Eigen::Vector3i, std::size_t>> entries(points.size());
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size()), [&](const auto& r) {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      entries[i] = {point_to_voxel(world_from_base * points[i], inv_voxel), i};
    }
  });
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return voxel_less(a.first, b.first);
    return a.second < b.second;
  });

  struct Group {
    std::size_t start = 0;
    std::size_t end = 0;
    double score = 0.0;
  };
  std::vector<Group> groups;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i == 0 || entries[i].first != entries[i - 1].first) groups.push_back({i, i, 0.0});
    groups.back().end = i + 1;
  }
  for (Group& group : groups) {
    // Only voxels the scan hits more than once can give more than the coarse point.
    const BumpVoxel* voxel = map.at(world_from_base * points[entries[group.start].second]);
    group.score = (voxel != nullptr && group.end - group.start > 1) ? voxel->relief : 0.0;
  }
  std::vector<std::size_t> order(groups.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](const std::size_t a, const std::size_t b) { return groups[a].score > groups[b].score; });

  std::vector<Eigen::Vector3d> sampled;
  sampled.reserve(points.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    const Group& group = groups[order[k]];
    if (k < informed_voxels) {
      for (std::size_t i = group.start; i < group.end; ++i) sampled.push_back(points[entries[i].second]);
    } else {
      sampled.push_back(points[entries[group.start].second]);
    }
  }
  return sampled;
}

} // namespace rko_lio::core
