#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace rko_lio::core {

struct VisualConstraintConfidence {
  std::size_t tracks = 0;
  std::size_t inliers = 0;
  double rotation_error_deg = 0.0;
  double translation_cosine = -1.0;
  double baseline_m = 0.0;
  // Pose information expressed in the same left-tangent world coordinates as
  // the LiDAR ICP Hessian. Identity preserves the artifact-adapter behavior.
  Eigen::Matrix<double, 6, 6> visual_information =
      Eigen::Matrix<double, 6, 6>::Identity();
};

struct SelectiveVisualFusionConfig {
  bool enabled = false;
  std::size_t min_tracks = 80;
  std::size_t min_inliers = 50;
  double min_inlier_ratio = 0.2;
  double max_rotation_error_deg = 3.0;
  double min_translation_cosine = 0.5;
  double min_baseline_m = 0.03;
  double max_baseline_m = 2.0;
  double weak_information_ratio = 1.0e-4;
  double relative_information_weight = 0.1;
  double min_visual_directional_information_ratio = 0.0;
  std::size_t max_weak_directions = 2;
  double max_translation_update_m = 0.05;
  double max_rotation_update_rad = 0.02;
};

struct SelectiveVisualFusionResult {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  bool accepted = false;
  std::size_t fused_directions = 0;
  std::size_t lidar_weak_directions = 0;
  std::size_t visual_unobservable_directions = 0;
  std::array<double, 6> visual_directional_information_ratios{};
  std::size_t visual_directional_information_ratio_count = 0;
};

inline std::size_t count_weak_information_directions(
    const Eigen::Matrix<double, 6, 6>& H,
    const SelectiveVisualFusionConfig& config) {
  if (!H.allFinite() || config.max_weak_directions == 0) {
    return 0;
  }
  const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (H + H.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return 0;
  }
  const double trace = solver.eigenvalues().sum();
  if (!(trace > 0.0)) {
    return 0;
  }
  std::size_t count = 0;
  for (int index = 0;
       index < 6 && count < config.max_weak_directions; ++index) {
    if (solver.eigenvalues()(index) / trace <=
        config.weak_information_ratio) {
      ++count;
    }
  }
  return count;
}

inline bool visual_constraint_passes_gate(
    const VisualConstraintConfidence& confidence,
    const SelectiveVisualFusionConfig& config) {
  const double inlier_ratio = static_cast<double>(confidence.inliers) /
                              static_cast<double>(std::max<std::size_t>(1, confidence.tracks));
  return confidence.tracks >= config.min_tracks &&
         confidence.inliers >= config.min_inliers &&
         inlier_ratio >= config.min_inlier_ratio &&
         std::isfinite(confidence.rotation_error_deg) &&
         confidence.rotation_error_deg <= config.max_rotation_error_deg &&
         std::isfinite(confidence.translation_cosine) &&
         confidence.translation_cosine >= config.min_translation_cosine &&
         std::isfinite(confidence.baseline_m) &&
         confidence.baseline_m >= config.min_baseline_m &&
         confidence.baseline_m <= config.max_baseline_m;
}

inline Eigen::Matrix<double, 6, 1> clamp_visual_update(
    const Eigen::Matrix<double, 6, 1>& update,
    const SelectiveVisualFusionConfig& config) {
  Eigen::Matrix<double, 6, 1> clamped = update;
  const double translation_norm = clamped.head<3>().norm();
  if (translation_norm > config.max_translation_update_m) {
    clamped.head<3>() *= config.max_translation_update_m / translation_norm;
  }
  const double rotation_norm = clamped.tail<3>().norm();
  if (rotation_norm > config.max_rotation_update_rad) {
    clamped.tail<3>() *= config.max_rotation_update_rad / rotation_norm;
  }
  return clamped;
}

/**
 * Add a bounded visual pose prior only along weak LiDAR Hessian directions.
 *
 * The linear system convention is H dx = -b. `visual_update` is the desired
 * left-tangent update from the current ICP iterate. Rejected constraints and
 * well-conditioned LiDAR systems are returned byte-for-byte unchanged.
 */
inline SelectiveVisualFusionResult fuse_visual_in_weak_directions(
    const Eigen::Matrix<double, 6, 6>& H,
    const Eigen::Matrix<double, 6, 1>& b,
    const Eigen::Matrix<double, 6, 1>& visual_update,
    const VisualConstraintConfidence& confidence,
    const SelectiveVisualFusionConfig& config) {
  SelectiveVisualFusionResult result;
  result.H = H;
  result.b = b;
  if (!config.enabled || !H.allFinite() || !b.allFinite() ||
      !visual_update.allFinite() ||
      !visual_constraint_passes_gate(confidence, config) ||
      config.relative_information_weight <= 0.0 ||
      config.max_weak_directions == 0) {
    return result;
  }

  const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (H + H.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return result;
  }
  const double trace = solver.eigenvalues().sum();
  const double maximum = solver.eigenvalues().maxCoeff();
  if (!(trace > 0.0) || !(maximum > 0.0)) {
    return result;
  }

  const Eigen::Matrix<double, 6, 1> target = clamp_visual_update(visual_update, config);
  const double weight = config.relative_information_weight * maximum;
  const Eigen::Matrix<double, 6, 6> visual_symmetric =
      0.5 * (confidence.visual_information +
             confidence.visual_information.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> visual_solver(
      visual_symmetric);
  const double visual_maximum =
      visual_solver.info() == Eigen::Success &&
              visual_solver.eigenvalues().allFinite()
          ? visual_solver.eigenvalues().maxCoeff()
          : 0.0;
  for (int index = 0; index < 6 && result.fused_directions < config.max_weak_directions;
       ++index) {
    if (solver.eigenvalues()(index) / trace > config.weak_information_ratio) {
      continue;
    }
    ++result.lidar_weak_directions;
    const Eigen::Matrix<double, 6, 1> axis = solver.eigenvectors().col(index);
    const double visual_directional_ratio =
        visual_maximum > 0.0
            ? axis.dot(visual_symmetric * axis) / visual_maximum
            : 0.0;
    if (result.visual_directional_information_ratio_count <
        result.visual_directional_information_ratios.size()) {
      result.visual_directional_information_ratios
          [result.visual_directional_information_ratio_count++] =
              visual_directional_ratio;
    }
    if (!std::isfinite(visual_directional_ratio) ||
        visual_directional_ratio <
            config.min_visual_directional_information_ratio) {
      ++result.visual_unobservable_directions;
      continue;
    }
    result.H.noalias() += weight * axis * axis.transpose();
    result.b.noalias() -= weight * axis * axis.dot(target);
    ++result.fused_directions;
  }
  result.accepted = result.fused_directions > 0;
  return result;
}

}  // namespace rko_lio::core
