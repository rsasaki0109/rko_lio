#include "direct_visual_odometry.hpp"

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rko_lio::core {
namespace {

constexpr double kMinimumDepth = 0.1;

struct Feature {
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
  double score = 0.0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct Linearization {
  Eigen::Matrix<double, 8, 8> H = Eigen::Matrix<double, 8, 8>::Zero();
  Eigen::Matrix<double, 8, 1> b = Eigen::Matrix<double, 8, 1>::Zero();
  int residuals = 0;
  int inliers = 0;
  int observed_features = 0;
  int inlier_features = 0;
  double squared_error = 0.0;
};

std::vector<Feature> select_features(
    const GrayImage& image, const SparseDepthImage& depth,
    const FisheyeCameraModel& camera,
    const DirectVisualOdometryConfig& config) {
  std::vector<Feature> features;
  const int margin = config.patch_radius + 2;
  for (int top = margin; top < image.height - margin;
       top += config.grid_cell_size) {
    for (int left = margin; left < image.width - margin;
         left += config.grid_cell_size) {
      Feature best;
      for (int y = top; y < std::min(top + config.grid_cell_size,
                                     image.height - margin); ++y) {
        for (int x = left; x < std::min(left + config.grid_cell_size,
                                        image.width - margin); ++x) {
          const float z = depth.depth[static_cast<std::size_t>(y * image.width + x)];
          if (!std::isfinite(z)) {
            continue;
          }
          const double score = image.gradient_squared(x, y);
          if (score <= best.score || score < config.min_gradient * config.min_gradient) {
            continue;
          }
          const std::size_t index = static_cast<std::size_t>(y * image.width + x);
          if (depth.camera_points[index].allFinite() &&
              depth.projected_pixels[index].allFinite()) {
            best = {depth.projected_pixels[index], score,
                    depth.camera_points[index]};
          }
        }
      }
      if (best.score > 0.0) {
        features.push_back(best);
      }
    }
  }
  std::stable_sort(features.begin(), features.end(),
                   [](const Feature& lhs, const Feature& rhs) {
                     return lhs.score > rhs.score;
                   });
  if (static_cast<int>(features.size()) > config.max_features) {
    features.resize(static_cast<std::size_t>(config.max_features));
  }
  return features;
}

Linearization linearize(
    const std::vector<Feature>& features,
    const GrayImage& previous_image,
    const GrayImage& current_image,
    const SparseDepthImage& current_depth,
    const FisheyeCameraModel& camera,
    const Sophus::SE3d& pose,
    double gain, double bias,
    const DirectVisualOdometryConfig& config,
    bool build_system) {
  Linearization output;
  const int radius = config.patch_radius;
  for (const Feature& feature : features) {
    int feature_residuals = 0;
    int feature_inliers = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        Eigen::Vector3d previous_point;
        if (dx == 0 && dy == 0) {
          previous_point = feature.point;
        } else {
          if (!camera.backproject(feature.pixel + Eigen::Vector2d(dx, dy),
                                  feature.point.z(), previous_point)) {
            continue;
          }
        }
        const Eigen::Vector3d current_point = pose * previous_point;
        Eigen::Vector2d pixel;
        if (!camera.project(current_point, pixel) ||
            pixel.x() < 1.0 || pixel.y() < 1.0 ||
            pixel.x() >= current_image.width - 2.0 ||
            pixel.y() >= current_image.height - 2.0) {
          continue;
        }
        const float observed_depth = current_depth.nearest_depth(
            static_cast<int>(std::lround(pixel.x())),
            static_cast<int>(std::lround(pixel.y())),
            config.occlusion_search_radius);
        if (std::isfinite(observed_depth) &&
            observed_depth + config.occlusion_tolerance_m < current_point.z()) {
          continue;
        }
        const double reference = previous_image.bilinear(
            feature.pixel.x() + dx, feature.pixel.y() + dy);
        const double measured = current_image.bilinear(pixel.x(), pixel.y());
        const double residual = measured - gain * reference - bias;
        const double absolute = std::abs(residual);
        const double weight = absolute <= config.huber_delta
                                  ? 1.0
                                  : config.huber_delta / absolute;
        ++output.residuals;
        output.inliers += absolute <= config.huber_delta ? 1 : 0;
        ++feature_residuals;
        feature_inliers += absolute <= config.huber_delta ? 1 : 0;
        output.squared_error += residual * residual;
        if (!build_system) {
          continue;
        }
        Eigen::Matrix<double, 2, 3> projection_jacobian;
        for (int axis = 0; axis < 3; ++axis) {
          Eigen::Vector3d perturbed_point = current_point;
          perturbed_point(axis) += config.finite_difference_epsilon;
          Eigen::Vector2d perturbed_pixel;
          if (!camera.project(perturbed_point, perturbed_pixel)) {
            projection_jacobian.col(axis).setZero();
          } else {
            projection_jacobian.col(axis) =
                (perturbed_pixel - pixel) / config.finite_difference_epsilon;
          }
        }
        Eigen::Matrix<double, 3, 6> point_jacobian =
            Eigen::Matrix<double, 3, 6>::Zero();
        point_jacobian.leftCols<3>().setIdentity();
        Eigen::Matrix3d skew;
        skew << 0.0, -current_point.z(), current_point.y(),
                current_point.z(), 0.0, -current_point.x(),
                -current_point.y(), current_point.x(), 0.0;
        point_jacobian.rightCols<3>() = -skew;
        Eigen::RowVector2d image_gradient;
        image_gradient <<
            0.5 * (current_image.bilinear(pixel.x() + 1.0, pixel.y()) -
                   current_image.bilinear(pixel.x() - 1.0, pixel.y())),
            0.5 * (current_image.bilinear(pixel.x(), pixel.y() + 1.0) -
                   current_image.bilinear(pixel.x(), pixel.y() - 1.0));
        Eigen::Matrix<double, 8, 1> jacobian;
        jacobian.head<6>() =
            (image_gradient * projection_jacobian * point_jacobian).transpose();
        jacobian(6) = -reference;
        jacobian(7) = -1.0;
        output.H.noalias() += weight * jacobian * jacobian.transpose();
        output.b.noalias() += weight * jacobian * residual;
      }
    }
    if (feature_residuals > 0) {
      ++output.observed_features;
      output.inlier_features += feature_inliers * 2 >= feature_residuals ? 1 : 0;
    }
  }
  return output;
}

double rmse(const Linearization& linearization) {
  return linearization.residuals > 0
             ? std::sqrt(linearization.squared_error / linearization.residuals)
             : std::numeric_limits<double>::infinity();
}

}  // namespace

bool FisheyeCameraModel::valid() const {
  return width > 0 && height > 0 && fx > 0.0 && fy > 0.0 &&
         std::isfinite(cx) && std::isfinite(cy);
}

bool FisheyeCameraModel::project(const Eigen::Vector3d& point,
                                 Eigen::Vector2d& pixel) const {
  if (!valid() || !point.allFinite() || point.z() <= kMinimumDepth) {
    return false;
  }
  const double x = point.x() / point.z();
  const double y = point.y() / point.z();
  if (distortion_model == CameraDistortionModel::plumb_bob) {
    const double radius2 = x * x + y * y;
    const double radial =
        1.0 + distortion[0] * radius2 + distortion[1] * radius2 * radius2;
    const double distorted_x =
        x * radial + 2.0 * distortion[2] * x * y +
        distortion[3] * (radius2 + 2.0 * x * x);
    const double distorted_y =
        y * radial + distortion[2] * (radius2 + 2.0 * y * y) +
        2.0 * distortion[3] * x * y;
    pixel = Eigen::Vector2d(fx * distorted_x + cx, fy * distorted_y + cy);
    return pixel.allFinite() && pixel.x() >= 0.0 && pixel.y() >= 0.0 &&
           pixel.x() < width && pixel.y() < height;
  }
  const double radius = std::hypot(x, y);
  double scale = 1.0;
  if (radius > 1.0e-12) {
    const double theta = std::atan(radius);
    const double theta2 = theta * theta;
    const double distorted = theta *
        (1.0 + distortion[0] * theta2 + distortion[1] * theta2 * theta2 +
         distortion[2] * theta2 * theta2 * theta2 +
         distortion[3] * theta2 * theta2 * theta2 * theta2);
    scale = distorted / radius;
  }
  pixel = Eigen::Vector2d(fx * x * scale + cx, fy * y * scale + cy);
  return pixel.allFinite() && pixel.x() >= 0.0 && pixel.y() >= 0.0 &&
         pixel.x() < width && pixel.y() < height;
}

bool FisheyeCameraModel::backproject(const Eigen::Vector2d& pixel, double depth,
                                     Eigen::Vector3d& point) const {
  if (!valid() || !pixel.allFinite() || !std::isfinite(depth) ||
      depth <= kMinimumDepth) {
    return false;
  }
  const double mx = (pixel.x() - cx) / fx;
  const double my = (pixel.y() - cy) / fy;
  if (distortion_model == CameraDistortionModel::plumb_bob) {
    Eigen::Vector2d normalized(mx, my);
    for (int iteration = 0; iteration < 12; ++iteration) {
      const double x = normalized.x();
      const double y = normalized.y();
      const double radius2 = x * x + y * y;
      const double radial =
          1.0 + distortion[0] * radius2 + distortion[1] * radius2 * radius2;
      const double radial_gradient =
          distortion[0] + 2.0 * distortion[1] * radius2;
      const Eigen::Vector2d estimate(
          x * radial + 2.0 * distortion[2] * x * y +
              distortion[3] * (radius2 + 2.0 * x * x),
          y * radial + distortion[2] * (radius2 + 2.0 * y * y) +
              2.0 * distortion[3] * x * y);
      Eigen::Matrix2d jacobian;
      jacobian <<
          radial + 2.0 * x * x * radial_gradient +
              2.0 * distortion[2] * y + 6.0 * distortion[3] * x,
          2.0 * x * y * radial_gradient + 2.0 * distortion[2] * x +
              2.0 * distortion[3] * y,
          2.0 * x * y * radial_gradient + 2.0 * distortion[2] * x +
              2.0 * distortion[3] * y,
          radial + 2.0 * y * y * radial_gradient +
              6.0 * distortion[2] * y + 2.0 * distortion[3] * x;
      const double determinant = jacobian.determinant();
      if (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-12) {
        return false;
      }
      const Eigen::Vector2d update = jacobian.inverse() *
          (estimate - Eigen::Vector2d(mx, my));
      normalized -= update;
      if (!normalized.allFinite()) {
        return false;
      }
      if (update.squaredNorm() < 1.0e-24) {
        break;
      }
    }
    point = Eigen::Vector3d(normalized.x() * depth,
                            normalized.y() * depth, depth);
    return point.allFinite();
  }
  const double distorted_radius = std::hypot(mx, my);
  double theta = distorted_radius;
  for (int iteration = 0; iteration < 8 && distorted_radius > 1.0e-12;
       ++iteration) {
    const double t2 = theta * theta;
    const double t4 = t2 * t2;
    const double t6 = t4 * t2;
    const double t8 = t4 * t4;
    const double value = theta *
        (1.0 + distortion[0] * t2 + distortion[1] * t4 +
         distortion[2] * t6 + distortion[3] * t8) - distorted_radius;
    const double derivative = 1.0 + 3.0 * distortion[0] * t2 +
        5.0 * distortion[1] * t4 + 7.0 * distortion[2] * t6 +
        9.0 * distortion[3] * t8;
    if (std::abs(derivative) < 1.0e-12) {
      return false;
    }
    theta -= value / derivative;
  }
  const double radius = std::tan(theta);
  const double scale = distorted_radius > 1.0e-12
                           ? radius / distorted_radius
                           : 1.0;
  point = Eigen::Vector3d(mx * scale * depth, my * scale * depth, depth);
  return point.allFinite();
}

bool GrayImage::valid() const {
  return width > 2 && height > 2 &&
         pixels.size() == static_cast<std::size_t>(width * height);
}

double GrayImage::bilinear(double x, double y) const {
  if (!valid() || x < 0.0 || y < 0.0 || x >= width - 1.0 ||
      y >= height - 1.0) {
    return 0.0;
  }
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const double wx = x - x0;
  const double wy = y - y0;
  const auto sample = [this](int px, int py) {
    return static_cast<double>(pixels[static_cast<std::size_t>(py * width + px)]);
  };
  return (1.0 - wy) * ((1.0 - wx) * sample(x0, y0) + wx * sample(x0 + 1, y0)) +
         wy * ((1.0 - wx) * sample(x0, y0 + 1) + wx * sample(x0 + 1, y0 + 1));
}

double GrayImage::gradient_squared(int x, int y) const {
  if (!valid() || x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
    return 0.0;
  }
  const double gx = bilinear(x + 1, y) - bilinear(x - 1, y);
  const double gy = bilinear(x, y + 1) - bilinear(x, y - 1);
  return 0.25 * (gx * gx + gy * gy);
}

bool SparseDepthImage::valid() const {
  return width > 0 && height > 0 &&
         depth.size() == static_cast<std::size_t>(width * height) &&
         camera_points.size() == depth.size() &&
         projected_pixels.size() == depth.size();
}

float SparseDepthImage::nearest_depth(int x, int y, int radius) const {
  float nearest = std::numeric_limits<float>::infinity();
  if (!valid()) {
    return nearest;
  }
  for (int py = std::max(0, y - radius); py <= std::min(height - 1, y + radius); ++py) {
    for (int px = std::max(0, x - radius); px <= std::min(width - 1, x + radius); ++px) {
      nearest = std::min(nearest, depth[static_cast<std::size_t>(py * width + px)]);
    }
  }
  return nearest;
}

SparseDepthImage project_sparse_depth(
    const std::vector<Eigen::Vector3d>& lidar_points,
    const Sophus::SE3d& camera_T_lidar,
    const FisheyeCameraModel& camera) {
  SparseDepthImage image;
  image.width = camera.width;
  image.height = camera.height;
  image.depth.assign(static_cast<std::size_t>(camera.width * camera.height),
                     std::numeric_limits<float>::infinity());
  image.camera_points.assign(
      image.depth.size(),
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()));
  image.projected_pixels.assign(
      image.depth.size(),
      Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity()));
  for (const auto& lidar_point : lidar_points) {
    const Eigen::Vector3d point = camera_T_lidar * lidar_point;
    Eigen::Vector2d pixel;
    if (!camera.project(point, pixel)) {
      continue;
    }
    const int x = static_cast<int>(std::lround(pixel.x()));
    const int y = static_cast<int>(std::lround(pixel.y()));
    if (x < 0 || y < 0 || x >= camera.width || y >= camera.height) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(y * camera.width + x);
    float& cell = image.depth[index];
    if (point.z() < cell) {
      cell = static_cast<float>(point.z());
      image.camera_points[index] = point;
      image.projected_pixels[index] = pixel;
    }
  }
  return image;
}

DirectVisualOdometryResult align_direct_visual(
    const GrayImage& previous_image,
    const SparseDepthImage& previous_depth,
    const GrayImage& current_image,
    const SparseDepthImage& current_depth,
    const FisheyeCameraModel& camera,
    const Sophus::SE3d& initial_current_T_previous_camera,
    const DirectVisualOdometryConfig& config) {
  if (!previous_image.valid() || !current_image.valid() ||
      !previous_depth.valid() || !current_depth.valid() || !camera.valid() ||
      previous_image.width != camera.width || previous_image.height != camera.height ||
      current_image.width != camera.width || current_image.height != camera.height) {
    throw std::invalid_argument("direct visual inputs have inconsistent dimensions");
  }
  DirectVisualOdometryResult result;
  result.current_T_previous_camera = initial_current_T_previous_camera;
  const auto features = select_features(previous_image, previous_depth, camera, config);
  result.features = static_cast<int>(features.size());
  if (features.empty()) {
    result.failure_reason = DirectVisualFailureReason::no_features;
    return result;
  }
  double gain = 1.0;
  double bias = 0.0;
  const Linearization initial = linearize(
      features, previous_image, current_image, current_depth, camera,
      result.current_T_previous_camera, gain, bias, config, false);
  result.initial_rmse = rmse(initial);
  result.tracked_features = initial.observed_features;
  result.residuals = initial.residuals;
  result.inliers = initial.inlier_features;
  result.inlier_ratio = static_cast<double>(initial.inlier_features) /
                        std::max(1, initial.observed_features);
  if (result.initial_rmse > config.max_initial_rmse) {
    result.failure_reason =
        DirectVisualFailureReason::poor_initial_photometric_fit;
    result.final_rmse = result.initial_rmse;
    return result;
  }
  for (int iteration = 0; iteration < config.max_iterations; ++iteration) {
    const Linearization system = linearize(
        features, previous_image, current_image, current_depth, camera,
        result.current_T_previous_camera, gain, bias, config, true);
    if (system.residuals < config.min_residuals) {
      result.failure_reason = DirectVisualFailureReason::insufficient_residuals;
      return result;
    }
    Eigen::Matrix<double, 8, 8> regularized = system.H;
    regularized.diagonal().array() += 1.0e-6;
    const Eigen::Matrix<double, 8, 1> update = regularized.ldlt().solve(-system.b);
    if (!update.allFinite()) {
      result.failure_reason = DirectVisualFailureReason::nonfinite_update;
      return result;
    }
    if (update.head<6>().norm() > config.max_pose_step) {
      result.failure_reason = DirectVisualFailureReason::oversized_update;
      return result;
    }
    const Sophus::SE3d candidate =
        Sophus::SE3d::exp(update.head<6>()) * result.current_T_previous_camera;
    Eigen::Matrix<double, 6, 1> correction =
        (candidate * initial_current_T_previous_camera.inverse()).log();
    const double translation_norm = correction.head<3>().norm();
    if (translation_norm > config.max_translation_correction_m) {
      correction.head<3>() *=
          config.max_translation_correction_m / translation_norm;
    }
    const double rotation_norm = correction.tail<3>().norm();
    if (rotation_norm > config.max_rotation_correction_rad) {
      correction.tail<3>() *=
          config.max_rotation_correction_rad / rotation_norm;
    }
    result.current_T_previous_camera =
        Sophus::SE3d::exp(correction) * initial_current_T_previous_camera;
    gain = std::clamp(gain + update(6), 0.5, 2.0);
    bias = std::clamp(bias + update(7), -80.0, 80.0);
    if (update.norm() < 1.0e-5) {
      break;
    }
  }
  const Linearization final = linearize(
      features, previous_image, current_image, current_depth, camera,
      result.current_T_previous_camera, gain, bias, config, true);
  result.residuals = final.residuals;
  result.tracked_features = final.observed_features;
  result.inliers = final.inlier_features;
  result.inlier_ratio = static_cast<double>(final.inlier_features) /
                        std::max(1, final.observed_features);
  result.final_rmse = rmse(final);
  result.exposure_gain = gain;
  result.exposure_bias = bias;
  Eigen::Matrix2d exposure_information = final.H.bottomRightCorner<2, 2>();
  exposure_information.diagonal().array() += 1.0e-9;
  const Eigen::Matrix<double, 6, 2> pose_exposure =
      final.H.topRightCorner<6, 2>();
  result.pose_information =
      final.H.topLeftCorner<6, 6>() -
      pose_exposure * exposure_information.ldlt().solve(pose_exposure.transpose());
  result.pose_information =
      0.5 * (result.pose_information + result.pose_information.transpose());
  if (final.residuals < config.min_residuals) {
    result.failure_reason = DirectVisualFailureReason::insufficient_residuals;
  } else if (result.inlier_ratio < config.min_inlier_ratio) {
    result.failure_reason = DirectVisualFailureReason::low_inlier_ratio;
  } else if (result.final_rmse > config.max_rmse) {
    result.failure_reason = DirectVisualFailureReason::high_rmse;
  } else if (result.final_rmse >= result.initial_rmse) {
    result.failure_reason = DirectVisualFailureReason::no_photometric_improvement;
  } else {
    result.failure_reason = DirectVisualFailureReason::none;
    result.valid = true;
  }
  return result;
}

}  // namespace rko_lio::core
