/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
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

#include "lio.hpp"
#include "degeneracy_aware_solve.hpp"
#include "gravity_alignment.hpp"
#include "kinematic_scene_range_gate.hpp"
#include "kinematic_velocity_blend.hpp"
#include "kinematic_velocity_gate.hpp"
#include "localizability_weighting.hpp"
#include "preprocess_scan.hpp"
#include "profiler.hpp"
#include "radar_ego_velocity.hpp"
#include "util.hpp"
// other
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
// stl
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace {
constexpr double EPSILON = 1e-8;
constexpr auto EPSILON_TIME = std::chrono::nanoseconds(10);
using namespace rko_lio::core;

inline void transform_points(const Sophus::SE3d& T, Vector3dVector& points) {
  std::transform(points.begin(), points.end(), points.begin(), [&](const auto& point) { return T * point; });
}

struct AccelInfo {
  double accel_mag_variance;
  Eigen::Vector3d local_gravity_estimate;
};

struct BodyAccelKF {
  Eigen::Vector3d mean;
  Eigen::Matrix3d covariance;
};

struct AccelFilterStep {
  std::optional<AccelInfo> info;
  BodyAccelKF updated;
};

AccelFilterStep step_body_accel_filter(const BodyAccelKF& prev,
                                       const IntervalStats& stats,
                                       const Sophus::SO3d& rotation_estimate,
                                       const Nsec dt,
                                       const double max_expected_jerk) {
  if (stats.imu_count <= 1) {
    std::cerr << "[WARNING] " << stats.imu_count
              << " IMU message(s) in interval between two lidar scans. Cannot compute "
                 "acceleration statistics for orientation regularisation. Please check your data and its "
                 "timestamping as likely there should not be so few IMU measurements between two LiDAR scans.\n";
    return {std::nullopt, prev};
  }

  const Eigen::Vector3d avg_imu_accel = stats.imu_acceleration_sum / stats.imu_count;
  const double accel_mag_variance = stats.welford_sum_of_squares / (stats.imu_count - 1);
  const Eigen::Vector3d body_accel_measurement = avg_imu_accel + rotation_estimate.inverse() * gravity();

  const double max_acceleration_change = max_expected_jerk * to_seconds(dt);
  // assume [j, -j] range for uniform dist. on jerk. variance is (2j)^2 / 12 = j^2/3. multiply by dt^2 for accel
  const Eigen::Matrix3d process_noise = square(max_acceleration_change) / 3 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d cov_pred = prev.covariance + process_noise;

  // isotropic accel mag variance
  const Eigen::Matrix3d measurement_noise = accel_mag_variance / 3 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = cov_pred + measurement_noise;
  const Eigen::Matrix3d kalman_gain = cov_pred * S.inverse();

  const Eigen::Vector3d new_mean = prev.mean + kalman_gain * (body_accel_measurement - prev.mean);
  const Eigen::Matrix3d new_cov = cov_pred - kalman_gain * cov_pred;

  const Eigen::Vector3d local_gravity_estimate = avg_imu_accel - new_mean; // points upwards
  return {AccelInfo{.accel_mag_variance = accel_mag_variance, .local_gravity_estimate = local_gravity_estimate},
          BodyAccelKF{new_mean, new_cov}};
}

template <typename Functor>
  requires requires(Functor f, Nsec stamp) {
    { f(stamp) } -> std::same_as<Sophus::SE3d>;
  }
Vector3dVector deskew_scan(const Vector3dVector& frame,
                           const TimestampVector& timestamps,
                           const Nsec end_time,
                           const Functor& relative_pose_at_time) {
  const Sophus::SE3d scan_to_scan_motion_inverse = relative_pose_at_time(end_time).inverse();
  Vector3dVector deskewed(frame.size());
  std::transform(frame.cbegin(), frame.cend(), timestamps.cbegin(), deskewed.begin(),
                 [&](const Eigen::Vector3d& point, const Nsec timestamp) {
                   return (scan_to_scan_motion_inverse * relative_pose_at_time(timestamp)) * point;
                 });
  return deskewed;
}

// Re-runs the same deskew formula and range-clip predicate used by preprocess_scan /
// deskew_scan above, keeping per-point intensity index-aligned to the output points.
// preprocess_scan()/deskew_scan() have no intensity channel, so this is the simplest way
// to get a points+intensities pair index-aligned to filtered_frame without touching them.
template <typename Functor>
  requires requires(Functor f, Nsec stamp) {
    { f(stamp) } -> std::same_as<Sophus::SE3d>;
  }
std::pair<Vector3dVector, std::vector<float>> deskew_and_clip_with_intensity(
    const Vector3dVector& scan,
    const TimestampVector& timestamps,
    const std::vector<float>& intensities,
    const Nsec end_time,
    const Functor& relative_pose_at_time,
    const LIO::Config& config) {
  Vector3dVector points_out;
  std::vector<float> intensities_out;
  points_out.reserve(scan.size());
  intensities_out.reserve(scan.size());
  if (config.deskew) {
    const Sophus::SE3d scan_to_scan_motion_inverse = relative_pose_at_time(end_time).inverse();
    for (std::size_t i = 0; i < scan.size(); ++i) {
      const Sophus::SE3d pose = scan_to_scan_motion_inverse * relative_pose_at_time(timestamps[i]);
      const Eigen::Vector3d point = pose * scan[i];
      const double range = point.norm();
      if (range > config.min_range && range < config.max_range) {
        points_out.push_back(point);
        intensities_out.push_back(intensities[i]);
      }
    }
  } else {
    for (std::size_t i = 0; i < scan.size(); ++i) {
      const double range = scan[i].norm();
      if (range > config.min_range && range < config.max_range) {
        points_out.push_back(scan[i]);
        intensities_out.push_back(intensities[i]);
      }
    }
  }
  return {points_out, intensities_out};
}

using LinearSystem = std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double>;

// Per-build diagnostics of the localizability weighting (see
// localizability_weighting.hpp): how many correspondences existed and how many
// received a boosted (> 1) weight from an axis-observing map normal.
struct IcpWeightingStats {
  int correspondences = 0;
  int boosted = 0;
};

LinearSystem build_icp_linear_system(const Sophus::SE3d& current_pose,
                                     const rko_lio::core::Vector3dVector& frame,
                                     const rko_lio::core::VoxelHashMap& voxel_map,
                                     const double& max_correspondence_distance,
                                     const int voxel_search_radius = 1,
                                     const std::optional<Eigen::Vector3d>& localizability_axis = std::nullopt,
                                     const double localizability_boost = 0.0,
                                     IcpWeightingStats* weighting_stats = nullptr) {
  // H, b, chi, weight sum. The weight sum replaces the correspondence count in
  // the H/b normalization; with weighting disabled every weight is exactly 1
  // and the result is bit-identical to the historical count-normalized system.
  using WeightedSystem = std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double, double>;
  auto linear_system_reduce = [](WeightedSystem lhs, const WeightedSystem& rhs) {
    auto& [lhs_H, lhs_b, lhs_chi, lhs_w] = lhs;
    const auto& [rhs_H, rhs_b, rhs_chi, rhs_w] = rhs;
    lhs_H += rhs_H;
    lhs_b += rhs_b;
    lhs_chi += rhs_chi;
    lhs_w += rhs_w;
    return lhs;
  };

  const bool weighting_enabled = localizability_axis.has_value() && localizability_boost > 0.0;
  std::atomic<int> boosted_counter = 0;

  auto linear_system_for_one_point = [&](const Eigen::Vector3d& source, const Eigen::Vector3d& target) {
    Eigen::Matrix3_6d J_r;
    J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
    const Eigen::Vector3d residual = source - target;
    double weight = 1.0;
    if (weighting_enabled) {
      weight = localizability_weight(voxel_map.voxel_normal(target), *localizability_axis, localizability_boost);
      if (weight > 1.0) {
        boosted_counter++;
      }
    }
    return WeightedSystem(weight * J_r.transpose() * J_r,      // JTJ
                          weight * J_r.transpose() * residual, // JTr
                          weight * residual.squaredNorm(),     // chi
                          weight);
  };

  // The only parallel part
  using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
  std::atomic<int> correspondences_counter = 0;
  const auto& [H_icp, b_icp, chi_icp, weight_sum] = tbb::parallel_reduce(
      // Range
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
      // Identity
      WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, WeightedSystem J) -> WeightedSystem {
        return std::transform_reduce(r.begin(), r.end(), J, linear_system_reduce, [&](const auto& point) {
          // Compute data association and linear system
          const Eigen::Vector3d transformed_point = current_pose * point;
          const auto& [closest_neighbor, distance] =
              voxel_map.get_closest_neighbor(transformed_point, voxel_search_radius);
          if (distance < max_correspondence_distance) {
            correspondences_counter++;
            return linear_system_for_one_point(transformed_point, closest_neighbor);
          }
          // TODO (meher): additional 0 add flops, which may hurt single threaded perf slightly
          return WeightedSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0, 0.0);
        });
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      linear_system_reduce);

  if (correspondences_counter == 0) {
    throw std::runtime_error("Number of correspondences are 0.");
  }
  if (weighting_stats != nullptr) {
    weighting_stats->correspondences = correspondences_counter;
    weighting_stats->boosted = boosted_counter;
  }

  return {H_icp / weight_sum, b_icp / weight_sum, 0.5 * chi_icp};
}

LinearSystem build_orientation_linear_system(const Sophus::SE3d& current_pose,
                                             const Eigen::Vector3d& local_gravity_estimate) {
  const Sophus::SO3d& current_rotation = current_pose.so3();
  const Eigen::Vector3d predicted_gravity =
      current_rotation.inverse() * (-1 * gravity()); // points upwards, same as local_gravity_estimate
  const Eigen::Vector3d residual = predicted_gravity - local_gravity_estimate;

  Eigen::Matrix3_6d J_ori = Eigen::Matrix3_6d::Zero();
  J_ori.block<3, 3>(0, 3) = current_rotation.inverse().matrix() * Sophus::SO3d::hat(-1 * gravity()).matrix();

  return LinearSystem{J_ori.transpose() * J_ori, J_ori.transpose() * residual, 0.5 * residual.squaredNorm()};
}

// [v0.8 Phase 1, diagnostic-only] icp()'s full result: the optimized pose
// (bit-for-bit the same value this function has always returned/applied to
// the pose estimate) plus the final iteration's Gauss-Newton linear system
// (H, b). H/b are exposed purely so callers can thread them into
// State::icp_diagnostics for downstream conditioning diagnostics -- they are
// never read back into the solve, so this struct changes nothing about how
// `pose` is computed.
struct IcpResult {
  Sophus::SE3d pose;
  Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
  Eigen::Vector6d b = Eigen::Vector6d::Zero();
  std::size_t degeneracy_intervention_count = 0;
  std::size_t visual_fused_directions = 0;
  std::size_t visual_unobservable_directions = 0;
  std::array<double, 6> visual_directional_information_ratios{};
  std::size_t visual_directional_information_ratio_count = 0;
  // Final-iteration localizability weighting stats (zeros when disabled).
  IcpWeightingStats weighting_stats;
};

IcpResult icp(const Vector3dVector& frame,
             const VoxelHashMap& voxel_map,
             const Sophus::SE3d& initial_guess,
             const LIO::Config& config,
             const std::optional<AccelInfo>& optional_accel_info,
             const int voxel_search_radius = 1,
             const PersistentWeakDirectionState& persistent_direction = {},
             const bool start_with_extended_iteration_budget = false,
             const std::optional<VisualPosePrior>& visual_pose_prior = std::nullopt,
             const std::optional<Sophus::SE3d>& degeneracy_prior_pose = std::nullopt,
             const std::optional<Eigen::Vector3d>& localizability_axis = std::nullopt,
             const double localizability_boost = 0.0) {
  // in case config disables it, or we don't have valid IMU information for this icp loop, beta is -1
  const double beta = (config.min_beta > 0 && optional_accel_info.has_value())
                          ? (config.min_beta * (1 + optional_accel_info->accel_mag_variance))
                          : -1;

  Sophus::SE3d current_pose = initial_guess;
  // [v0.8 Phase 1, diagnostic-only] last iteration's linear system, kept
  // around purely to hand back to the caller alongside the pose.
  Eigen::Matrix6d last_H = Eigen::Matrix6d::Zero();
  Eigen::Vector6d last_b = Eigen::Vector6d::Zero();
  std::size_t degeneracy_intervention_count = 0;

  std::size_t iteration_budget =
      start_with_extended_iteration_budget
          ? std::max(config.max_iterations, config.degeneracy_adaptive_max_iterations)
          : config.max_iterations;
  IcpWeightingStats weighting_stats;
  for (size_t i = 0; i < iteration_budget; ++i) {
    const auto& [H, b, chi] = std::invoke([&]() -> LinearSystem {
      const auto& [H_icp, b_icp, chi_icp] =
          build_icp_linear_system(current_pose, frame, voxel_map, config.max_correspondence_distance,
                                  voxel_search_radius, localizability_axis, localizability_boost,
                                  &weighting_stats);
      if (beta >= 0) {
        const auto& [H_ori, b_ori, chi_ori] =
            build_orientation_linear_system(current_pose, optional_accel_info->local_gravity_estimate);
        return {H_icp + H_ori / beta, b_icp + b_ori / beta, chi_icp + chi_ori / beta};
      }
      return {H_icp, b_icp, chi_icp};
    });
    last_H = H;
    last_b = b;
    if (config.degeneracy_adaptive_iteration_budget &&
        has_weak_information_direction(H, config.degeneracy_adaptive_iteration_ratio)) {
      iteration_budget = std::max(iteration_budget, config.degeneracy_adaptive_max_iterations);
    }

    Eigen::Vector6d dx;
    if (config.degeneracy_aware_solve) {
      // Ordinarily the geometric/IMU initial guess doubles as the degeneracy
      // prior. A radar velocity prior (see LIO::register_scan) replaces only
      // this prior pose; the ICP iterate above (current_pose) still starts
      // from and is regularized around `initial_guess` as before.
      const Sophus::SE3d& prior_pose = degeneracy_prior_pose.has_value() ? *degeneracy_prior_pose : initial_guess;
      const Eigen::Vector6d prior_update = (prior_pose * current_pose.inverse()).log();
      DegeneracyAwareSolveConfig solve_config;
      solve_config.well_conditioned_ratio = config.degeneracy_well_conditioned_ratio;
      solve_config.multiplicity_relative_gap = config.degeneracy_multiplicity_relative_gap;
      solve_config.degenerate_prior_weight = config.degeneracy_prior_weight;
      solve_config.require_persistent_direction = config.degeneracy_persistence_gate;
      solve_config.persistent_direction_min_absolute_cosine =
          config.degeneracy_persistence_min_absolute_cosine;
      solve_config.persistent_direction = persistent_direction;
      const DegeneracyAwareSolveResult solve_result = solve_degeneracy_aware(H, b, prior_update, solve_config);
      if (!solve_result.valid) {
        throw std::runtime_error("Degeneracy-aware ICP solve failed.");
      }
      dx = solve_result.update;
      degeneracy_intervention_count += solve_result.intervention_applied ? 1U : 0U;
    } else {
      dx = H.ldlt().solve(-b);
    }
    current_pose = Sophus::SE3d::exp(dx) * current_pose;

    if (dx.norm() < config.convergence_criterion || i == (iteration_budget - 1)) {
      // TODO: proper debug logging
      // std::cout << "iter " << i << ", beta: " << beta << ", chi: " << chi << ", num_assoc: " <<
      // correspondences.size() << "\n";
      break;
    }
  }
  std::size_t visual_fused_directions = 0;
  std::size_t visual_unobservable_directions = 0;
  std::array<double, 6> visual_directional_information_ratios{};
  std::size_t visual_directional_information_ratio_count = 0;
  if (visual_pose_prior.has_value()) {
    const Eigen::Vector6d visual_update = (visual_pose_prior->pose * current_pose.inverse()).log();
    const auto fused =
        fuse_visual_in_weak_directions(last_H, last_b, visual_update, visual_pose_prior->confidence, config.visual_fusion);
    visual_unobservable_directions = fused.visual_unobservable_directions;
    visual_directional_information_ratios = fused.visual_directional_information_ratios;
    visual_directional_information_ratio_count = fused.visual_directional_information_ratio_count;
    if (fused.accepted) {
      const Eigen::Vector6d dx = fused.H.ldlt().solve(-fused.b);
      if (!dx.allFinite()) {
        throw std::runtime_error("Selective visual fusion solve failed.");
      }
      current_pose = Sophus::SE3d::exp(dx) * current_pose;
      last_H = fused.H;
      last_b = fused.b;
      visual_fused_directions = fused.fused_directions;
    }
  }
  return {current_pose, last_H, last_b, degeneracy_intervention_count,
          visual_fused_directions, visual_unobservable_directions,
          visual_directional_information_ratios,
          visual_directional_information_ratio_count, weighting_stats};
}

struct AlignmentStats {
  int correspondences = 0;
  double inlier_ratio = 0.0;
  double mean_error = std::numeric_limits<double>::max();
};

AlignmentStats evaluate_alignment(const Sophus::SE3d& pose,
                                  const Vector3dVector& frame,
                                  const VoxelHashMap& voxel_map,
                                  const double max_correspondence_distance,
                                  const int voxel_search_radius) {
  if (frame.empty()) {
    return {};
  }
  int correspondences = 0;
  double error_sum = 0.0;
  for (const Eigen::Vector3d& point : frame) {
    const Eigen::Vector3d transformed_point = pose * point;
    const auto& [closest_neighbor, distance] = voxel_map.get_closest_neighbor(transformed_point, voxel_search_radius);
    (void)closest_neighbor;
    if (distance < max_correspondence_distance) {
      ++correspondences;
      error_sum += distance;
    }
  }
  if (correspondences == 0) {
    return {};
  }
  return {
      .correspondences = correspondences,
      .inlier_ratio = static_cast<double>(correspondences) / static_cast<double>(frame.size()),
      .mean_error = error_sum / static_cast<double>(correspondences),
  };
}

int voxel_search_radius_for_distance(const VoxelHashMap& voxel_map, const double max_correspondence_distance) {
  const double voxel_size = std::max(1e-6, voxel_map.voxel_size_);
  return std::max(1, static_cast<int>(std::ceil(max_correspondence_distance / voxel_size)) + 1);
}

Sophus::SO3d yaw_rotation(const double yaw_rad) {
  return Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, yaw_rad));
}

inline Sophus::SO3d align_accel_to_z_world(const Eigen::Vector3d& accel) {
  //  unobservable in the gravity direction, and the z in R.log() will always be 0
  const Eigen::Vector3d z_world = {0.0, 0.0, 1.0};
  const Eigen::Quaterniond quat_accel = Eigen::Quaterniond::FromTwoVectors(accel, z_world);
  return Sophus::SO3d(quat_accel);
}

Eigen::Vector3d centroid_of(const Vector3dVector& points) {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (const auto& point : points) {
    sum += point;
  }
  return points.empty() ? sum : Eigen::Vector3d(sum / static_cast<double>(points.size()));
}
} // namespace

// ==========================
//   actual LIO class stuff
// ==========================

namespace rko_lio::core {

LIO::LIO(const Config& config_)
    : config(config_),
      map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel),
      relocalization_map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel) {
  // Per-voxel normals are only maintained when the localizability weighting can
  // consume them; the default path pays nothing.
  map.set_maintain_normals(config.localizability_weighting);
  // Pin TBB's worker pool to config.max_num_threads (0 = leave at TBB default).
  [[maybe_unused]] static const auto tbb_thread_limit = [&] {
    const int threads = config.max_num_threads > 0 ? config.max_num_threads : tbb::this_task_arena::max_concurrency();
    return tbb::global_control(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads));
  }();
}

// ==========================
//          private
// ==========================

void LIO::initialize(const Nsec lidar_time) {
  if (interval_stats.imu_count == 0) {
    std::cerr << "[WARNING] Cannot initialize. No imu measurements received.\n";
    // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    _initialized = true;
    return;
  }

  const Eigen::Vector3d avg_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3d avg_gyro = interval_stats.angular_velocity_sum / interval_stats.imu_count;

  const Sophus::SO3d initial_rotation = align_accel_to_z_world(avg_accel);
  lidar_state.pose.so3() = initial_rotation;
  imu_state = lidar_state;

  // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);

  // the pose for the current time gets logged at the end of register_scan in the typical fashion
  lidar_state.time = lidar_time;
  imu_state.time = lidar_time;

  const Eigen::Vector3d local_gravity = imu_state.pose.so3().inverse() * gravity();
  imu_bias.accelerometer = avg_accel + local_gravity;
  imu_bias.gyroscope = avg_gyro;

  _initialized = true;
  std::cout << "[INFO] Odometry map frame initialized using " << interval_stats.imu_count
            << " IMU measurements. Estimated initial rotation [se(3)] is " << imu_state.pose.so3().log().transpose()
            << "\n";
  std::cout << "[INFO] Estimated accel bias: " << imu_bias.accelerometer.transpose()
            << ", gyro bias: " << imu_bias.gyroscope.transpose() << "\n";
}

Vector3dVector LIO::bootstrap_first_scan(const Vector3dVector& scan, const Nsec current_lidar_time) {
  lidar_state.time = current_lidar_time;
  imu_state = lidar_state;
  auto preproc = preprocess_scan(scan, config);
  if (!config.initialization_phase) {
    update_maps(config.double_downsample ? preproc.map_frame : preproc.keypoints, lidar_state.pose);
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    std::cout << "[INFO] Odometry map frame initialized with first lidar scan.\n";
  }
  return std::move(preproc.filtered_frame);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> LIO::motion_priors_from_imu(const Nsec current_lidar_time) {
  if (config.initialization_phase && !_initialized) {
    // assume static and
    initialize(current_lidar_time);
    return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  }
  if (interval_stats.imu_count == 0) {
    std::cerr << "[WARNING] No Imu measurements in interval to average. Assuming constant velocity motion.\n";
    return {Eigen::Vector3d::Zero(), lidar_state.angular_velocity};
  }
  const Eigen::Vector3d avg_body_accel = interval_stats.body_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3d avg_ang_vel = interval_stats.angular_velocity_sum / interval_stats.imu_count;
  if (avg_body_accel.norm() > 50.0) {
    std::cerr << "[WARNING] Erratic body acceleration computed, norm > 50 m/s2. Either IMU data is corrupted, or you "
                 "should report an issue.";
  }
  return {avg_body_accel, avg_ang_vel};
}

void LIO::update_maps(const Vector3dVector& map_update_frame, const Sophus::SE3d& pose) {
  map.update(map_update_frame, pose);

  // The unpruned global map exists only for the opt-in kidnap recovery path.
  // Avoid transforming, copying, and retaining every map point when recovery
  // is disabled (the normal SLAM-only configuration).
  if (config.enable_kidnap_relocalization) {
    Vector3dVector points_transformed(map_update_frame.size());
    std::transform(map_update_frame.cbegin(), map_update_frame.cend(), points_transformed.begin(),
                   [&](const auto& point) { return pose * point; });
    relocalization_map.add_points(points_transformed);
  }
}

Vector3dVector LIO::recover_with_scan(const Vector3dVector& filtered_frame,
                                      const Vector3dVector& map_update_frame,
                                      const Nsec& current_lidar_time,
                                      const Sophus::SE3d& recovery_pose,
                                      const std::string& reason) {
  map.clear();
  lidar_state.pose = recovery_pose;
  lidar_state.time = current_lidar_time;
  lidar_state.velocity.setZero();
  lidar_state.angular_velocity.setZero();
  lidar_state.linear_acceleration.setZero();
  // [v0.8 Phase 1, diagnostic-only] the local map was just cleared and
  // `recovery_pose` did not come from a normal incremental ICP solve against
  // it, so any previously-recorded H/b would describe a now-irrelevant map.
  lidar_state.icp_diagnostics = std::nullopt;
  _persistent_weak_direction_tracker.reset();
  _previous_intensity_profile.reset();
  _kinematic_blend_anchor_time.reset();
  _kinematic_blend_propagated_velocity_world.reset();
  _kinematic_blend_turning_streak = 0;
  _kinematic_blend_speeding_streak = 0;
  _kinematic_blend_scene_last_rejected_time_sec = -1.0;
  _kinematic_blend_speed_last_rejected_time_sec = -1.0;
  imu_state = lidar_state;
  interval_stats.reset();
  update_maps(map_update_frame, lidar_state.pose);
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  _consecutive_registration_failures = 0;
  std::cout << "[INFO] Kidnap recovery accepted scan at " << to_seconds(current_lidar_time) << "s via " << reason
            << ".\n";
  return filtered_frame;
}

Vector3dVector LIO::drop_failed_scan(const Nsec& current_lidar_time, const std::string& reason) {
  lidar_state.time = current_lidar_time;
  // [v0.8 Phase 1, diagnostic-only] no ICP solve happened for the dropped scan.
  lidar_state.icp_diagnostics = std::nullopt;
  _persistent_weak_direction_tracker.reset();
  _previous_intensity_profile.reset();
  _kinematic_blend_anchor_time.reset();
  _kinematic_blend_propagated_velocity_world.reset();
  _kinematic_blend_turning_streak = 0;
  _kinematic_blend_speeding_streak = 0;
  _kinematic_blend_scene_last_rejected_time_sec = -1.0;
  _kinematic_blend_speed_last_rejected_time_sec = -1.0;
  imu_state = lidar_state;
  interval_stats.reset();
  std::cerr << "[WARNING] Dropping scan during kidnap recovery: " << reason << "\n";
  return {};
}

std::optional<Sophus::SE3d> LIO::try_global_relocalization(const Vector3dVector& keypoints) const {
  if (!config.enable_kidnap_relocalization || relocalization_map.empty() || keypoints.empty()) {
    return std::nullopt;
  }
  const int usable_pose_count =
      static_cast<int>(poses_with_timestamps.size()) - std::max(0, config.relocalization_min_pose_separation);
  if (usable_pose_count <= 0) {
    return std::nullopt;
  }

  LIO::Config relocalization_config = config;
  relocalization_config.max_iterations = static_cast<size_t>(std::max(1, config.relocalization_max_iterations));
  relocalization_config.max_correspondence_distance = config.relocalization_max_correspondence_distance;
  relocalization_config.min_beta = -1.0;
  const int pose_stride = std::max(1, config.relocalization_pose_stride);
  const int yaw_samples = std::max(1, config.relocalization_yaw_samples);
  const int voxel_search_radius =
      voxel_search_radius_for_distance(relocalization_map, relocalization_config.max_correspondence_distance);
  const double pi = std::acos(-1.0);

  bool found = false;
  Sophus::SE3d best_pose;
  AlignmentStats best_stats;
  for (int pose_index = 0; pose_index < usable_pose_count; pose_index += pose_stride) {
    const Sophus::SE3d& historical_pose = poses_with_timestamps[static_cast<size_t>(pose_index)].second;
    for (int yaw_index = 0; yaw_index < yaw_samples; ++yaw_index) {
      const double yaw = (2.0 * pi * static_cast<double>(yaw_index)) / static_cast<double>(yaw_samples);
      const Sophus::SE3d initial_guess(yaw_rotation(yaw) * historical_pose.so3(), historical_pose.translation());
      Sophus::SE3d optimized_pose;
      try {
        // [v0.8 Phase 1] icp() now returns IcpResult; relocalization only
        // needs the pose, so the diagnostic H/b are discarded here (a fresh
        // local map is about to replace the current one, see
        // LIO::recover_with_scan, which invalidates any prior diagnostics
        // anyway).
        optimized_pose =
            icp(keypoints, relocalization_map, initial_guess, relocalization_config, std::nullopt, voxel_search_radius)
                .pose;
      } catch (const std::exception&) {
        continue;
      }
      const AlignmentStats stats = evaluate_alignment(optimized_pose, keypoints, relocalization_map,
                                                       relocalization_config.max_correspondence_distance,
                                                       voxel_search_radius);
      if (stats.correspondences < config.relocalization_min_correspondences ||
          stats.inlier_ratio < config.relocalization_min_inlier_ratio ||
          stats.mean_error > config.relocalization_max_mean_error) {
        continue;
      }
      if (!found || stats.mean_error < best_stats.mean_error ||
          (std::abs(stats.mean_error - best_stats.mean_error) < 1e-6 &&
           stats.correspondences > best_stats.correspondences)) {
        found = true;
        best_pose = optimized_pose;
        best_stats = stats;
      }
    }
  }

  if (!found) {
    return std::nullopt;
  }
  std::cout << "[INFO] Kidnap relocalization matched " << best_stats.correspondences << "/" << keypoints.size()
            << " keypoints, mean error " << best_stats.mean_error << " m.\n";
  return best_pose;
}

// ==========================
//          public
// ==========================

// ============================ imu ===============================

void LIO::add_imu_measurement(const ImuControl& base_imu) {
  if (lidar_state.time < EPSILON_TIME) {
    static bool warning_skip_till_first_lidar = false;
    if (!warning_skip_till_first_lidar) {
      std::cerr << "[WARNING - ONCE] Skipping IMU, waiting for first LiDAR message.\n";
      warning_skip_till_first_lidar = true;
    }
    _last_real_imu_time = base_imu.time;
    _last_real_base_imu_ang_vel = base_imu.angular_velocity;
    return;
  }

  if (imu_state.time < EPSILON_TIME) {
    imu_state = lidar_state;
  }

  const double dt = to_seconds(base_imu.time - imu_state.time);

  if (dt < 0.0) {
    // messages are out of sync. thats a problem, since we integrate gyro from last lidar time onwards
    std::cerr << "[WARNING] Received IMU message from the past. Can result in errors.\n";
    // maybe skip this imu reading?
  }

  const Eigen::Vector3d unbiased_ang_vel = base_imu.angular_velocity - imu_bias.gyroscope;
  const Eigen::Vector3d unbiased_accel = base_imu.acceleration - imu_bias.accelerometer;

  const Eigen::Vector3d local_gravity = imu_state.pose.so3().inverse() * gravity();
  const Eigen::Vector3d compensated_accel = unbiased_accel + local_gravity;

  // imu state update
  Eigen::Vector6d tau;
  tau.head<3>() = imu_state.velocity * dt + compensated_accel * square(dt) / 2;
  tau.tail<3>() = unbiased_ang_vel * dt;
  imu_state.pose = imu_state.pose * Sophus::SE3d::exp(tau);
  imu_state.velocity += compensated_accel * dt;
  imu_state.angular_velocity = unbiased_ang_vel;
  imu_state.time = base_imu.time;

  interval_stats.update(unbiased_ang_vel, unbiased_accel, compensated_accel);

  _last_real_imu_time = base_imu.time;
  _last_real_base_imu_ang_vel = base_imu.angular_velocity;
}

void LIO::add_imu_measurement(const Sophus::SE3d& extrinsic_imu2base, const ImuControl& raw_imu) {
  if (extrinsic_imu2base.log().norm() < EPSILON) {
    add_imu_measurement(raw_imu);
    return;
  }

  if (_last_real_imu_time < EPSILON_TIME) {
    // skip IMU message as we need a previous imu time for extrinsic compensation
    _last_real_imu_time = raw_imu.time;
    return;
  }

  // accounting for the transport-rate
  ImuControl base_imu = raw_imu;
  const Sophus::SO3d& extrinsic_rotation = extrinsic_imu2base.so3();
  base_imu.angular_velocity = extrinsic_rotation * raw_imu.angular_velocity;

  const Eigen::Vector3d& lever_arm = -1 * extrinsic_imu2base.translation();
  const Nsec dt = raw_imu.time - _last_real_imu_time;

  const Eigen::Vector3d angular_acceleration = std::invoke([&]() -> Eigen::Vector3d {
    if (std::chrono::abs(dt) < std::chrono::microseconds(200)) {
      // if dt is less than the equivalent of a 5000 Hz imu, assuming zero ang accel,
      // causes numerical issues otherwise
      static bool warning_imu_too_close = false;
      if (!warning_imu_too_close) {
        std::cerr << "[WARNING - ONCE] Received IMU message with a very short delta to previous IMU message. Ignoring "
                     "all such messages.\n";
        warning_imu_too_close = true;
      }
      return Eigen::Vector3d::Zero();
    } else {
      const Eigen::Vector3d angular_acceleration =
          (base_imu.angular_velocity - _last_real_base_imu_ang_vel) / to_seconds(dt);
      return angular_acceleration;
    }
  });

  base_imu.acceleration = extrinsic_rotation * raw_imu.acceleration + angular_acceleration.cross(lever_arm) +
                          base_imu.angular_velocity.cross(base_imu.angular_velocity.cross(lever_arm));

  this->add_imu_measurement(base_imu);
}

Sophus::SE3d LIO::predict_pose_at(const Nsec& time) const {
  const double dt = to_seconds(time - lidar_state.time);
  Eigen::Vector3d average_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d average_angular_velocity = Eigen::Vector3d::Zero();
  if (config.initialization_phase && !_initialized) {
    // The registration path assumes zero motion while collecting its
    // initialization window.
  } else if (interval_stats.imu_count > 0) {
    average_acceleration = interval_stats.body_acceleration_sum / interval_stats.imu_count;
    average_angular_velocity = interval_stats.angular_velocity_sum / interval_stats.imu_count;
  } else {
    average_angular_velocity = lidar_state.angular_velocity;
  }
  Eigen::Vector6d motion = Eigen::Vector6d::Zero();
  motion.head<3>() = lidar_state.velocity * dt + average_acceleration * square(dt) / 2.0;
  motion.tail<3>() = average_angular_velocity * dt;
  return lidar_state.pose * Sophus::SE3d::exp(motion);
}

// ============================ lidar ===============================

Vector3dVector LIO::register_scan(const Vector3dVector& scan,
                                  const TimestampVector& timestamps,
                                  const std::vector<float>* intensities) {
  if (timestamps.empty()) {
    throw std::invalid_argument("LIO::register_scan: timestamps must not be empty.");
  }
  // TODO: redundant max compute as its available after process_timestamps
  const Nsec current_lidar_time = *std::max_element(timestamps.cbegin(), timestamps.cend());

  if (lidar_state.time < EPSILON_TIME) {
    return bootstrap_first_scan(scan, current_lidar_time);
  }

  if (to_seconds(std::chrono::abs(current_lidar_time - lidar_state.time)) > config.max_scan_delta_sec) {
    const double diff_seconds = to_seconds(current_lidar_time - lidar_state.time);
    // Re-anchor instead of throwing: a throw leaves lidar_state.time behind,
    // so every later scan also exceeds the delta and the pipeline never
    // recovers from a single dropout. Dropping this scan while advancing the
    // internal clock lets the next scan register normally, and the failure
    // count lets relocalize_after_scan_gap trigger when enabled.
    ++_consecutive_registration_failures;
    return drop_failed_scan(current_lidar_time,
                            "LiDAR scan gap of " + std::to_string(diff_seconds) +
                                " seconds exceeds max_scan_delta_sec; re-anchoring at the new timestamp.");
  }

  const auto [avg_body_accel, avg_ang_vel] = motion_priors_from_imu(current_lidar_time);

  // compute relative motion using controls
  auto relative_pose_at_time = [&](const Nsec time) -> Sophus::SE3d {
    const double dt = to_seconds(time - lidar_state.time);
    Eigen::Vector6d tau;
    tau.head<3>() = lidar_state.velocity * dt + (avg_body_accel * square(dt) / 2);
    tau.tail<3>() = avg_ang_vel * dt;
    return Sophus::SE3d::exp(tau);
  };

  const Sophus::SE3d initial_guess = predict_pose_at(current_lidar_time);

  std::optional<VisualPosePrior> visual_pose_prior;
  if (_visual_pose_prior.has_value() && config.visual_fusion.enabled &&
      std::abs(to_seconds(_visual_pose_prior->time - current_lidar_time)) <= config.visual_prior_max_time_offset_sec) {
    visual_pose_prior = _visual_pose_prior;
  }
  // A camera measurement is single-use even when stale or rejected. This
  // prevents one visual edge from being silently applied to multiple scans.
  _visual_pose_prior.reset();

  // Radar-informed degeneracy prior: initial_guess with translation replaced
  // by integrating the radar ego-velocity estimate instead of the IMU/geometric
  // one. Only ever consumed inside solve_degeneracy_aware's prior blend (see
  // icp() above); the ICP initial guess itself is untouched.
  std::optional<Sophus::SE3d> radar_prior_pose;
  std::optional<Eigen::Vector3d> radar_velocity_base;
  std::optional<Eigen::Matrix3d> radar_info_base;
  if (_radar_velocity_prior.has_value() &&
      std::abs(to_seconds(_radar_velocity_prior->time - current_lidar_time)) <= config.radar_prior_max_time_offset_sec &&
      _radar_velocity_prior->velocity_base.norm() >= config.radar_min_speed) {
    radar_velocity_base = _radar_velocity_prior->velocity_base;
    radar_info_base = _radar_velocity_prior->info_base;
    if (config.radar_velocity_fusion && config.degeneracy_aware_solve) {
      const double radar_dt = to_seconds(current_lidar_time - lidar_state.time);
      const Eigen::Vector3d predicted_translation =
          lidar_state.pose.translation() + initial_guess.so3() * (_radar_velocity_prior->velocity_base * radar_dt);
      radar_prior_pose = Sophus::SE3d(initial_guess.so3(), predicted_translation);
    }
  }
  // A radar measurement is single-use even when stale or rejected, mirroring the visual prior.
  _radar_velocity_prior.reset();

  // Reflectivity/intensity constraint: scan-to-scan cross-correlation of a 1D reflectivity
  // profile along the confirmed persistent weak-direction axis, used to recover translation
  // ICP under-observes in self-similar environments (e.g. a straight tunnel). Points+
  // intensities are deskewed/clipped together once here; the same pair is reused after ICP
  // below to store this scan's own profile for the *next* scan. Radar has priority when both
  // are active: it directly measures velocity, a stronger signal than a correlation peak.
  std::optional<Sophus::SE3d> intensity_prior_pose;
  Vector3dVector intensity_aligned_points;
  std::vector<float> intensity_aligned_values;
  const bool intensity_alignment_available =
      config.intensity_constraint && intensities != nullptr && intensities->size() == scan.size();
  // The Hessian-gate-free velocity-disagreement gate (applied after ICP, below) reuses this
  // exact same deskewed points+intensities pair; compute it once up front if either feature
  // needs it, so enabling only intensity_disagreement_gate still gets it without duplicating
  // the deskew_and_clip_with_intensity call.
  const bool intensity_disagreement_alignment_available =
      config.intensity_disagreement_gate && intensities != nullptr && intensities->size() == scan.size();
  if (intensity_alignment_available || intensity_disagreement_alignment_available) {
    std::tie(intensity_aligned_points, intensity_aligned_values) = deskew_and_clip_with_intensity(
        scan, timestamps, *intensities, current_lidar_time, relative_pose_at_time, config);
  }
  if (intensity_alignment_available) {
    const PersistentWeakDirectionState& incoming_direction = _persistent_weak_direction_tracker.state();
    if (!radar_prior_pose.has_value() && incoming_direction.confirmed && _previous_intensity_profile.has_value() &&
        !intensity_aligned_points.empty()) {
      ++intensity_prior_attempt_count;
      IntensityProfileConfig profile_config;
      profile_config.bin_size_m = config.intensity_bin_size_m;
      profile_config.half_length_m = config.intensity_profile_half_length_m;
      profile_config.max_shift_m = config.intensity_max_shift_m;
      profile_config.min_correlation = config.intensity_min_correlation;
      profile_config.min_filled_bins = config.intensity_min_filled_bins;
      // Built at the initial guess (not yet ICP-corrected), against the same origin/axis as
      // the stored reference: the correlation peak then directly measures the translation-
      // along-axis error in the initial guess (see intensity_profile.hpp's doc comment).
      Vector3dVector current_world_points(intensity_aligned_points.size());
      std::transform(intensity_aligned_points.cbegin(), intensity_aligned_points.cend(), current_world_points.begin(),
                     [&](const auto& point) { return initial_guess * point; });
      const IntensityProfile current_profile =
          build_intensity_profile(current_world_points, intensity_aligned_values, _previous_intensity_profile->axis,
                                  _previous_intensity_profile->origin, profile_config);
      const ProfileShiftResult shift =
          estimate_profile_shift(_previous_intensity_profile->profile, current_profile, profile_config);
      if (shift.valid) {
        const Eigen::Vector3d predicted_translation =
            initial_guess.translation() + shift.shift_m * _previous_intensity_profile->axis;
        intensity_prior_pose = Sophus::SE3d(initial_guess.so3(), predicted_translation);
        ++intensity_prior_applied_count;
      }
    }
  }

  // body acceleration filter
  const auto kf_step = step_body_accel_filter({mean_body_acceleration, body_acceleration_covariance}, interval_stats,
                                              initial_guess.so3(), current_lidar_time - lidar_state.time,
                                              config.max_expected_jerk);
  mean_body_acceleration = kf_step.updated.mean;
  body_acceleration_covariance = kf_step.updated.covariance;

  auto preproc_result =
      config.deskew ? preprocess_scan(deskew_scan(scan, timestamps, current_lidar_time, relative_pose_at_time), config)
                    : preprocess_scan(scan, config);

  if (preproc_result.keypoints.size() < 10) {
    const std::string error_msg =
        "Keypoints for ICP registration = " + std::to_string(preproc_result.keypoints.size()) +
        ", this is too little for ICP and likely unintended. Input scan size = " + std::to_string(scan.size()) +
        ". Config voxel size = " + std::to_string(config.voxel_size) +
        ". Either the input scan is corrupt (empty) or the downsampling is too aggressive.";
    ++_consecutive_registration_failures;
    _persistent_weak_direction_tracker.reset();
    _previous_intensity_profile.reset();
    if (config.reset_on_registration_failure &&
        _consecutive_registration_failures >= std::max(1, config.recovery_min_failures)) {
      return drop_failed_scan(current_lidar_time, error_msg);
    }
    throw std::invalid_argument(error_msg);
  }

  const Vector3dVector& map_input = config.double_downsample ? preproc_result.map_frame : preproc_result.keypoints;

  if (config.enable_kidnap_relocalization && config.relocalize_after_scan_gap &&
      _consecutive_registration_failures >= std::max(1, config.recovery_min_failures)) {
    if (const auto relocalized_pose = try_global_relocalization(preproc_result.keypoints)) {
      return recover_with_scan(preproc_result.filtered_frame, map_input, current_lidar_time, relocalized_pose.value(),
                               "global relocalization after scan gap");
    }
  }

  // [v0.8 Phase 1, diagnostic-only] reset before attempting a fresh ICP
  // solve; overwritten below on success. Every early-return path above this
  // point (drop_failed_scan / recover_with_scan) already resets this field
  // itself, so this covers the remaining "about to attempt icp()" case.
  lidar_state.icp_diagnostics = std::nullopt;

  // Whether the kinematic velocity gate corrected this scan's pose; consumed by
  // the map-update skip at the end of this function.
  bool kinematic_gate_corrected_this_scan = false;
  bool kinematic_blend_suppress_map_update_this_scan = false;
  double kinematic_blend_map_update_fraction_this_scan = 1.0;

  if (!map.empty()) {
    SCOPED_PROFILER("ICP");
    Sophus::SE3d optimized_pose;
    // [v0.8 Phase 1, diagnostic-only] final ICP linear system, threaded into
    // lidar_state.icp_diagnostics below on success.
    Eigen::Matrix6d icp_H = Eigen::Matrix6d::Zero();
    Eigen::Vector6d icp_b = Eigen::Vector6d::Zero();
    std::size_t degeneracy_intervention_count = 0;
    std::size_t visual_fused_directions = 0;
    std::size_t visual_unobservable_directions = 0;
    std::array<double, 6> visual_directional_information_ratios{};
    std::size_t visual_directional_information_ratio_count = 0;
    // Radar has priority over the intensity constraint when both are active (see the
    // intensity-prior construction above for why); at most one ever reaches icp()'s shared
    // degeneracy_prior_pose slot.
    const std::optional<Sophus::SE3d>& degeneracy_prior_pose =
        radar_prior_pose.has_value() ? radar_prior_pose : intensity_prior_pose;
    // Localizability weighting axis: the predicted motion direction for this
    // scan (initial guess vs. previous pose), same convention as the intensity
    // disagreement gate's correlation axis. No Hessian gating on purpose --
    // soft degeneracy is precisely the regime the eigenvalue gates miss.
    std::optional<Eigen::Vector3d> localizability_axis;
    if (config.localizability_weighting) {
      localizability_axis = unit_axis_from_step(initial_guess.translation() - lidar_state.pose.translation(),
                                                config.localizability_min_step_m);
      if (localizability_axis.has_value()) {
        ++localizability_attempt_count;
      }
    }
    try {
      const IcpResult icp_result = icp(preproc_result.keypoints, map, initial_guess, config, kf_step.info, 1,
                                       _persistent_weak_direction_tracker.state(),
                                       config.degeneracy_adaptive_iteration_budget &&
                                           _adaptive_iteration_hold_remaining > 0,
                                       visual_pose_prior, degeneracy_prior_pose, localizability_axis,
                                       config.localizability_boost);
      if (icp_result.weighting_stats.boosted > 0) {
        ++localizability_weighted_scan_count;
        localizability_boosted_fraction_sum += static_cast<double>(icp_result.weighting_stats.boosted) /
                                               std::max(1, icp_result.weighting_stats.correspondences);
      }
      optimized_pose = icp_result.pose;
      icp_H = icp_result.H;
      icp_b = icp_result.b;
      degeneracy_intervention_count = icp_result.degeneracy_intervention_count;
      visual_fused_directions = icp_result.visual_fused_directions;
      visual_unobservable_directions = icp_result.visual_unobservable_directions;
      visual_directional_information_ratios = icp_result.visual_directional_information_ratios;
      visual_directional_information_ratio_count = icp_result.visual_directional_information_ratio_count;
    } catch (const std::exception&) {
      ++_consecutive_registration_failures;
      _persistent_weak_direction_tracker.reset();
      _previous_intensity_profile.reset();
      _adaptive_iteration_hold_remaining = 0;
      if (_consecutive_registration_failures < std::max(1, config.recovery_min_failures)) {
        throw;
      }
      if (const auto relocalized_pose = try_global_relocalization(preproc_result.keypoints)) {
        return recover_with_scan(preproc_result.filtered_frame, map_input, current_lidar_time,
                                 relocalized_pose.value(), "global relocalization");
      }
      if (config.reset_on_registration_failure) {
        return recover_with_scan(preproc_result.filtered_frame, map_input, current_lidar_time, lidar_state.pose,
                                 "local reset");
      }
      throw;
    }

    // Continuous, information-weighted radar/ICP velocity fusion (see Config::
    // radar_velocity_continuous_fusion doc comment and radar_ego_velocity.hpp's
    // blend_icp_radar_velocity for the pure math). Runs on every scan with a fresh radar prior,
    // no streak/threshold gate, and blends all three world-frame translation axes at once
    // (weighted by relative confidence), instead of radar_disagreement_gate's single-axis,
    // 10-scan-streak, post-hoc correction below. When enabled it subsumes that gate entirely
    // (see the `!config.radar_velocity_continuous_fusion` guard on the gate below) rather than
    // letting both paths correct the same scan.
    bool radar_continuous_fused_this_scan = false;
    if (config.radar_velocity_continuous_fusion && radar_velocity_base.has_value() && radar_info_base.has_value()) {
      const double radar_dt = to_seconds(current_lidar_time - lidar_state.time);
      if (radar_dt > 1.0e-6) {
        ++radar_continuous_attempt_count;
        const Eigen::Vector3d icp_velocity_world =
            (optimized_pose.translation() - lidar_state.pose.translation()) / radar_dt;
        // icp_H's translation block is world-frame: build_icp_linear_system's Jacobian sets
        // J_r.block(0,0) = I against a residual computed as (current_pose * point) - target,
        // i.e. directly in the map/world frame, with no further rotation applied. H is already
        // averaged over correspondences, so H_tt / dt^2 is only a proxy for a translation-rate
        // information matrix; radar_fusion_icp_information_scale lets the two sensors' relative
        // trust be tuned empirically.
        const Eigen::Matrix3d icp_information_world =
            config.radar_fusion_icp_information_scale * (icp_H.block<3, 3>(0, 0) / square(radar_dt));
        const Eigen::Vector3d radar_velocity_world = optimized_pose.so3() * (*radar_velocity_base);
        const Eigen::Matrix3d R_wb = optimized_pose.so3().matrix();
        const Eigen::Matrix3d radar_information_world = R_wb * (*radar_info_base) * R_wb.transpose();
        const RadarIcpVelocityBlendResult blend = blend_icp_radar_velocity(
            icp_velocity_world, icp_information_world, radar_velocity_world, radar_information_world);
        if (blend.valid) {
          const Eigen::Vector3d correction = (blend.velocity - icp_velocity_world) * radar_dt;
          optimized_pose.translation() += correction;
          ++radar_continuous_fused_scan_count;
          radar_continuous_correction_magnitude_sum += correction.norm();
          radar_continuous_fused_this_scan = true;
        }
      }
    }
    (void)radar_continuous_fused_this_scan; // reserved for future gating of downstream features

    // Radar-vs-ICP velocity disagreement gate: ICP locked onto clutter that
    // travels with the sensor (fog) yields a confident near-zero motion that
    // the eigenvalue gates cannot flag. After a persistent disagreement,
    // blend the translation toward the radar-observed displacement, but only
    // along the radar velocity direction. Skipped entirely when
    // radar_velocity_continuous_fusion is on -- that feature already corrects every axis on
    // every scan with a fresh radar prior, so this gate would either double-correct or fight the
    // continuous blend's own numbers with its own separate streak/threshold logic.
    bool radar_disagreement_corrected_this_scan = false;
    if (config.radar_disagreement_gate && !config.radar_velocity_continuous_fusion &&
        radar_velocity_base.has_value()) {
      const double radar_dt = to_seconds(current_lidar_time - lidar_state.time);
      if (radar_dt > 1.0e-6) {
        const Eigen::Vector3d icp_velocity_world =
            (optimized_pose.translation() - lidar_state.pose.translation()) / radar_dt;
        const Eigen::Vector3d radar_velocity_world = optimized_pose.so3() * (*radar_velocity_base);
        const Eigen::Vector3d velocity_gap = radar_velocity_world - icp_velocity_world;
        if (velocity_gap.norm() >= config.radar_disagreement_min_mps) {
          ++_radar_disagreement_streak;
        } else {
          _radar_disagreement_streak = 0;
        }
        constexpr double kMinRadarSpeed = 1.0e-3;
        if (_radar_disagreement_streak >= config.radar_disagreement_min_scans &&
            radar_velocity_world.norm() > kMinRadarSpeed) {
          const Eigen::Vector3d direction = radar_velocity_world.normalized();
          const double weight = std::max(0.0, std::min(1.0, config.radar_disagreement_weight));
          const double correction = weight * velocity_gap.dot(direction) * radar_dt;
          optimized_pose.translation() += correction * direction;
          ++radar_disagreement_corrected_scan_count;
          radar_disagreement_corrected_this_scan = true;
        }
      }
    } else if (config.radar_disagreement_gate && !config.radar_velocity_continuous_fusion) {
      _radar_disagreement_streak = 0;
    }

    // Intensity-vs-ICP velocity disagreement gate: same lesson as the radar gate just above
    // (gate on sensor-vs-ICP consistency, not the Hessian spectrum), but derives its own
    // "sensor" velocity from scan-to-scan reflectivity-texture correlation instead of Doppler,
    // so it also helps radar-less rigs. Unlike intensity_constraint (which only engages once
    // degeneracy_persistence_gate confirms a weak direction, missing "soft" degeneracy in
    // between confirmations), this gate tracks the scan's own ICP motion axis every scan.
    // Radar has priority: skip entirely on scans the radar gate already corrected.
    if (config.intensity_disagreement_gate && !radar_disagreement_corrected_this_scan &&
        intensity_disagreement_alignment_available && !intensity_aligned_points.empty()) {
      // Step 1 (see intensity_profile.hpp's unit_axis_from_step doc comment): this scan's own
      // motion direction, without any Hessian gating. lidar_state.pose is still the *previous*
      // scan's pose here (the state update happens further below), so this is exactly
      // optimized_pose.translation() - prev_translation.
      constexpr double kMinStepM = 0.01; // 1 cm: below this there's no reliable along-track signal.
      std::optional<Eigen::Vector3d> motion_axis =
          unit_axis_from_step(optimized_pose.translation() - lidar_state.pose.translation(), kMinStepM);
      if (!motion_axis.has_value()) {
        // ICP step too small to trust; fall back to the initial guess's own displacement.
        motion_axis = unit_axis_from_step(initial_guess.translation() - lidar_state.pose.translation(), kMinStepM);
      }
      if (!motion_axis.has_value()) {
        // Stationary rig: no along-track signal either way.
        _intensity_disagreement_streak = 0;
      } else {
        IntensityProfileConfig profile_config;
        profile_config.bin_size_m = config.intensity_bin_size_m;
        profile_config.half_length_m = config.intensity_profile_half_length_m;
        profile_config.max_shift_m = config.intensity_max_shift_m;
        profile_config.min_correlation = config.intensity_min_correlation;
        profile_config.min_filled_bins = config.intensity_min_filled_bins;
        // Step 2/3: correlate against the previous scan's stored profile, along *that* stored
        // profile's own axis/origin (not this scan's freshly chosen motion_axis) -- the shift
        // and the profiles it's built from must all describe the same fixed direction for the
        // correlation to have a clean physical meaning. motion_axis only decides what gets
        // stored below for the *next* scan's correlation, mirroring how the persistence-gated
        // intensity_constraint feature above lets the current scan's own axis take over only
        // once it becomes the new stored reference.
        bool disagreement_measured = false;
        double intensity_velocity_along_axis = 0.0;
        double icp_velocity_along_axis = 0.0;
        Eigen::Vector3d correction_axis = *motion_axis;
        const double motion_dt = to_seconds(current_lidar_time - lidar_state.time);
        if (_previous_intensity_velocity_profile.has_value() && motion_dt > 1.0e-6) {
          ++intensity_disagreement_attempt_count;
          const Eigen::Vector3d corr_axis = _previous_intensity_velocity_profile->axis;
          const Eigen::Vector3d corr_origin = _previous_intensity_velocity_profile->origin;
          // Built at the initial guess (pre-ICP-correction), exactly like intensity_constraint's
          // own correlation above: the shift then measures the initial guess's along-axis error.
          Vector3dVector initial_guess_world_points(intensity_aligned_points.size());
          std::transform(intensity_aligned_points.cbegin(), intensity_aligned_points.cend(),
                         initial_guess_world_points.begin(), [&](const auto& point) { return initial_guess * point; });
          const IntensityProfile current_profile = build_intensity_profile(
              initial_guess_world_points, intensity_aligned_values, corr_axis, corr_origin, profile_config);
          const ProfileShiftResult shift =
              estimate_profile_shift(_previous_intensity_velocity_profile->profile, current_profile, profile_config);
          // Reject shifts pinned at the search boundary: the true peak may lie beyond
          // max_shift_m, so the correlation result there is unreliable (item 5).
          const double saturation_margin = 0.5 * profile_config.bin_size_m;
          if (shift.valid && std::abs(shift.shift_m) < config.intensity_max_shift_m - saturation_margin) {
            ++intensity_disagreement_valid_shift_count;
            intensity_velocity_along_axis = intensity_implied_velocity_along_axis(
                corr_axis, initial_guess.translation(), corr_origin, shift.shift_m, motion_dt);
            icp_velocity_along_axis =
                corr_axis.dot(optimized_pose.translation() - lidar_state.pose.translation()) / motion_dt;
            correction_axis = corr_axis;
            disagreement_measured = true;
          }
        }
        if (disagreement_measured) {
          const double gap = std::abs(intensity_velocity_along_axis - icp_velocity_along_axis);
          if (gap >= config.intensity_disagreement_min_mps) {
            ++_intensity_disagreement_streak;
            ++intensity_disagreement_exceeded_threshold_count;
          } else {
            _intensity_disagreement_streak = 0;
          }
          if (_intensity_disagreement_streak >= config.intensity_disagreement_min_scans) {
            const double weight = std::max(0.0, std::min(1.0, config.intensity_disagreement_weight));
            const double correction = weight * (intensity_velocity_along_axis - icp_velocity_along_axis) * motion_dt;
            optimized_pose.translation() += correction * correction_axis;
            ++intensity_disagreement_corrected_scan_count;
          }
        }
        // No `else` here: when this scan produced no trustworthy shift (insufficient
        // texture/correlation -- the common case, unlike radar's near-every-scan Doppler
        // prior), that is an absence of evidence, not evidence of ICP agreement. Unlike the
        // radar gate's unconditional per-scan reset, the streak is deliberately left untouched
        // rather than zeroed; it only resets on a measured agreement (gap below threshold,
        // above) or a genuinely stationary rig (motion_axis missing, below).
        // Store this scan's own profile (world-anchored at its just-optimized pose, along its
        // own fresh motion_axis) for the *next* scan's correlation, mirroring
        // intensity_constraint's store block below but without any persistence gating.
        Vector3dVector store_world_points(intensity_aligned_points.size());
        std::transform(intensity_aligned_points.cbegin(), intensity_aligned_points.cend(), store_world_points.begin(),
                       [&](const auto& point) { return optimized_pose * point; });
        const IntensityProfile store_profile = build_intensity_profile(
            store_world_points, intensity_aligned_values, *motion_axis, optimized_pose.translation(), profile_config);
        if (store_profile.valid) {
          _previous_intensity_velocity_profile = StoredIntensityProfile{store_profile, optimized_pose.translation(), *motion_axis};
        } else {
          _previous_intensity_velocity_profile.reset();
        }
      }
    } else if (config.intensity_disagreement_gate) {
      _intensity_disagreement_streak = 0;
    }

    // Soft inertial velocity bridge (see kinematic_velocity_blend.hpp). ICP agreement
    // refreshes a trusted anchor. On disagreement, a propagation-speed prior aligned with
    // the current motion axis decays with time since that anchor. This prevents
    // both the tunnel zero-motion attractor and the indefinite hard-clamp runaway.
    //
    // This newer blend takes precedence if both it and the legacy hard gate are enabled.
    if (config.kinematic_velocity_blend && interval_stats.imu_count > 0) {
      const Eigen::Vector3d previous_velocity_world = lidar_state.pose.so3() * lidar_state.velocity;
      bool scene_rejected = false;
      if (config.kinematic_blend_range_scene_gate) {
        ++kinematic_blend_scene_evaluation_count;
        const KinematicSceneRangeSupport scene =
            evaluate_kinematic_scene_range_support(
                preproc_result.filtered_frame, config.min_range, config.max_range,
                config.kinematic_blend_scene_near_range_m,
                config.kinematic_blend_scene_max_near_fraction,
                config.kinematic_blend_scene_far_range_m,
                config.kinematic_blend_scene_min_far_fraction,
                config.kinematic_blend_scene_min_valid_points);
        if (scene.valid) {
          ++kinematic_blend_scene_valid_count;
          kinematic_blend_scene_near_fraction_sum += scene.near_fraction;
          kinematic_blend_scene_far_fraction_sum += scene.far_fraction;
        }
        const KinematicSceneReenableGate scene_gate =
            update_kinematic_scene_reenable_gate(
                scene.trusted, to_seconds(current_lidar_time),
                config.kinematic_blend_scene_reenable_delay_sec,
                _kinematic_blend_scene_last_rejected_time_sec);
        _kinematic_blend_scene_last_rejected_time_sec =
            scene_gate.last_rejected_time_sec;
        scene_rejected = scene_gate.rejected;
        kinematic_blend_scene_rejected_scan_count += !scene.trusted ? 1U : 0U;
        kinematic_blend_scene_cooldown_rejected_scan_count +=
            scene_gate.cooldown_active ? 1U : 0U;
      }
      const KinematicPersistentSpeedGate speed_gate =
          update_kinematic_persistent_speed_gate(
              previous_velocity_world.norm(),
              config.kinematic_blend_max_activation_speed_mps,
              config.kinematic_blend_speed_gate_min_scans,
              _kinematic_blend_speeding_streak);
      _kinematic_blend_speeding_streak = speed_gate.streak;
      const KinematicSceneReenableGate speed_reenable_gate =
          update_kinematic_speed_reenable_gate(
              !speed_gate.rejected, to_seconds(current_lidar_time),
              config.kinematic_blend_speed_reenable_delay_sec,
              _kinematic_blend_speed_last_rejected_time_sec);
      _kinematic_blend_speed_last_rejected_time_sec =
          speed_reenable_gate.last_rejected_time_sec;
      const bool speed_too_high = speed_reenable_gate.rejected;
      kinematic_blend_speed_rejected_scan_count += speed_gate.rejected ? 1U : 0U;
      kinematic_blend_speed_cooldown_rejected_scan_count +=
          speed_reenable_gate.cooldown_active ? 1U : 0U;
      const Eigen::Vector3d interval_mean_body_accel =
          interval_stats.body_acceleration_sum / interval_stats.imu_count;
      const Eigen::Vector3d interval_mean_angular_velocity_world =
          lidar_state.pose.so3() *
          (interval_stats.angular_velocity_sum / interval_stats.imu_count);
      const bool yaw_rate_exceeded =
          config.kinematic_blend_max_yaw_rate_rad_s > 0.0 &&
          std::abs(interval_mean_angular_velocity_world.z()) >
              config.kinematic_blend_max_yaw_rate_rad_s;
      if (yaw_rate_exceeded) {
        ++_kinematic_blend_turning_streak;
      } else {
        _kinematic_blend_turning_streak = 0;
      }
      const bool turning_too_fast =
          _kinematic_blend_turning_streak >=
          std::max<std::size_t>(1, config.kinematic_blend_yaw_gate_min_scans);
      const bool speed_too_low =
          previous_velocity_world.norm() < config.kinematic_blend_min_speed;
      kinematic_blend_low_speed_rejected_scan_count += speed_too_low ? 1U : 0U;
      kinematic_blend_yaw_rejected_scan_count += turning_too_fast ? 1U : 0U;
      if (speed_too_low || speed_too_high || scene_rejected || turning_too_fast) {
        _kinematic_blend_anchor_time.reset();
        _kinematic_blend_propagated_velocity_world.reset();
      } else {
        ++kinematic_blend_attempt_count;
        const double blend_dt = to_seconds(current_lidar_time - lidar_state.time);
        double anchor_age_sec =
            _kinematic_blend_anchor_time.has_value()
                ? to_seconds(current_lidar_time - _kinematic_blend_anchor_time.value())
                : -1.0;
        // Once the prior has decayed to exp(-5), discard it completely. The next
        // ICP/propagation agreement can then establish a clean anchor instead of
        // comparing forever against a long-drifted inertial state.
        if (anchor_age_sec > 5.0 * config.kinematic_blend_decay_time_sec) {
          _kinematic_blend_anchor_time.reset();
          _kinematic_blend_propagated_velocity_world.reset();
          anchor_age_sec = -1.0;
          ++kinematic_blend_anchor_expiration_count;
        }
        const Eigen::Vector3d propagation_seed_world =
            _kinematic_blend_propagated_velocity_world.value_or(previous_velocity_world);
        const Eigen::Vector3d measured_accel_world = lidar_state.pose.so3() * interval_mean_body_accel;
        Eigen::Vector3d propagated_velocity_world =
            propagation_seed_world + measured_accel_world * blend_dt;
        if (config.kinematic_blend_max_propagated_speed_mps > 0.0 &&
            propagated_velocity_world.norm() > config.kinematic_blend_max_propagated_speed_mps) {
          propagated_velocity_world *=
              config.kinematic_blend_max_propagated_speed_mps / propagated_velocity_world.norm();
          ++kinematic_blend_propagated_speed_clamp_count;
        }
        const KinematicVelocityBlend blend = blend_icp_with_propagated_velocity(
            previous_velocity_world, optimized_pose.translation() - lidar_state.pose.translation(), blend_dt,
            propagated_velocity_world, anchor_age_sec,
            config.kinematic_blend_icp_information_scale,
            config.kinematic_blend_propagation_information_scale,
            config.kinematic_blend_decay_time_sec, config.kinematic_blend_anchor_agreement_mps,
            config.kinematic_blend_max_icp_innovation_mps, config.kinematic_blend_min_speed);
        if (blend.valid) {
          kinematic_blend_max_disagreement_mps =
              std::max(kinematic_blend_max_disagreement_mps, blend.disagreement_mps);
        } else {
          ++kinematic_blend_invalid_result_count;
        }
        if (blend.valid && blend.anchor_agrees) {
          const bool establishing_first_anchor = !_kinematic_blend_anchor_time.has_value();
          _kinematic_blend_anchor_time = current_lidar_time;
          // Seed from ICP once, but never let later agreement refreshes feed ICP
          // velocity back into the independent prior. Repeated small (< agreement
          // threshold) ICP increments could otherwise accumulate into an arbitrarily
          // fast self-confirming map trajectory while every individual scan appeared
          // to agree.
          _kinematic_blend_propagated_velocity_world =
              establishing_first_anchor
                  ? (optimized_pose.translation() - lidar_state.pose.translation()) / blend_dt
                  : propagated_velocity_world;
          ++kinematic_blend_anchor_refresh_count;
          kinematic_blend_last_anchor_refresh_time_sec =
              to_seconds(current_lidar_time);
        } else if (_kinematic_blend_anchor_time.has_value() && blend.propagation_weight > 0.0) {
          _kinematic_blend_propagated_velocity_world = propagated_velocity_world;
          optimized_pose.translation() += blend.correction;
          ++kinematic_blend_corrected_scan_count;
          kinematic_blend_correction_m_sum += blend.correction.norm();
          kinematic_blend_max_correction_m =
              std::max(kinematic_blend_max_correction_m, blend.correction.norm());
          const double correction_time_sec = to_seconds(current_lidar_time);
          if (kinematic_blend_first_correction_time_sec < 0.0) {
            kinematic_blend_first_correction_time_sec = correction_time_sec;
          }
          kinematic_blend_last_correction_time_sec = correction_time_sec;
          kinematic_blend_propagation_weight_sum += blend.propagation_weight;
          kinematic_blend_max_propagation_weight =
              std::max(kinematic_blend_max_propagation_weight, blend.propagation_weight);
          kinematic_blend_max_anchor_age_sec =
              std::max(kinematic_blend_max_anchor_age_sec, anchor_age_sec);
          kinematic_blend_suppress_map_update_this_scan =
              blend.propagation_weight > config.kinematic_blend_map_update_max_propagation_weight;
          kinematic_blend_map_update_fraction_this_scan =
              std::max(std::clamp(config.kinematic_blend_map_update_min_fraction, 0.0, 1.0),
                       1.0 - blend.propagation_weight);
        }
      }
    }

    // Accelerometer-consistency velocity gate (see kinematic_velocity_gate.hpp): clamp the
    // per-scan velocity change along the previous motion direction to what the measured body
    // acceleration allows. Breaks the geometric zero-motion attractor (an ICP freeze implies
    // a deceleration the accelerometer flatly contradicts) while letting genuine braking
    // through, since braking raises the measured acceleration and with it the allowed change.
    // Applied after the radar/intensity gates so their corrections are part of the checked
    // translation, and before the velocity estimation below so the state update sees the
    // clamped pose.
    else if (config.kinematic_velocity_gate && interval_stats.imu_count > 0) {
      ++kinematic_gate_attempt_count;
      const Eigen::Vector3d interval_mean_body_accel =
          interval_stats.body_acceleration_sum / interval_stats.imu_count;
      const double gate_dt = to_seconds(current_lidar_time - lidar_state.time);
      const KinematicVelocityClamp clamp = clamp_velocity_change_to_accel(
          lidar_state.pose.so3() * lidar_state.velocity,
          optimized_pose.translation() - lidar_state.pose.translation(), gate_dt,
          lidar_state.pose.so3() * interval_mean_body_accel, config.kinematic_gate_accel_margin,
          config.kinematic_gate_min_speed, config.kinematic_gate_weight);
      if (clamp.corrected) {
        optimized_pose.translation() += clamp.correction;
        kinematic_gate_corrected_this_scan = true;
        ++kinematic_gate_corrected_scan_count;
        kinematic_gate_correction_m_sum += clamp.correction.norm();
      }
    }

    // Sliding-window absolute gravity alignment (see gravity_alignment.hpp for why the
    // min_beta regularization above cannot correct slow pitch/roll drift on its own). The
    // window collects each scan interval's raw (gravity-inclusive) accelerometer mean rotated
    // by that scan's optimized orientation; its long-window mean must point world-up, and the
    // residual tilt is fed back as a small, capped, roll/pitch-only left-multiplicative
    // correction. Applied before the velocity/acceleration estimation below so the state
    // update sees the corrected orientation.
    if (config.gravity_window_alignment && interval_stats.imu_count > 0) {
      const Eigen::Vector3d interval_mean_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
      const Eigen::Vector3d interval_mean_ang_vel_world =
          optimized_pose.so3() * (interval_stats.angular_velocity_sum / interval_stats.imu_count);
      // Sustained turning (world-frame yaw rate) adds centripetal acceleration that does not
      // average out over the window; skip such intervals instead of poisoning the mean.
      if (std::abs(interval_mean_ang_vel_world.z()) <= config.gravity_alignment_max_yaw_rate_rad_s) {
        _gravity_alignment_window.emplace_back(current_lidar_time, optimized_pose.so3() * interval_mean_accel);
      }
      while (!_gravity_alignment_window.empty() &&
             to_seconds(current_lidar_time - _gravity_alignment_window.front().first) > config.gravity_window_sec) {
        _gravity_alignment_window.pop_front();
      }
      const double window_span =
          _gravity_alignment_window.empty()
              ? 0.0
              : to_seconds(current_lidar_time - _gravity_alignment_window.front().first);
      // Only act on a mostly-full window: a short window has not yet averaged out the body
      // acceleration and would inject transient tilt as a false correction.
      if (window_span >= 0.5 * config.gravity_window_sec && _gravity_alignment_window.size() >= 10) {
        ++gravity_alignment_attempt_count;
        Eigen::Vector3d mean_world_accel = Eigen::Vector3d::Zero();
        for (const auto& [_, world_accel] : _gravity_alignment_window) {
          mean_world_accel += world_accel;
        }
        mean_world_accel /= static_cast<double>(_gravity_alignment_window.size());
        const GravityAlignmentCorrection alignment = compute_gravity_alignment_correction(
            mean_world_accel, GRAVITY_MAG, config.gravity_alignment_max_magnitude_deviation,
            config.gravity_alignment_gain, config.gravity_alignment_max_correction_rad,
            config.gravity_alignment_max_plausible_tilt_rad);
        if (alignment.valid) {
          ++gravity_alignment_applied_count;
          gravity_alignment_last_tilt_rad = alignment.tilt_rad;
          gravity_alignment_correction_rad_sum += alignment.correction.log().norm();
          optimized_pose.so3() = alignment.correction * optimized_pose.so3();
        }
      }
    }

    // estimate velocities and accelerations from the new pose
    const double dt = to_seconds(current_lidar_time - lidar_state.time);
    const Sophus::SE3d motion = lidar_state.pose.inverse() * optimized_pose;
    const Eigen::Vector6d local_velocity = motion.log() / dt;
    const Eigen::Vector3d local_linear_acceleration =
        (local_velocity.head<3>() - motion.so3().inverse() * lidar_state.velocity) / dt;

    // update
    lidar_state.pose = optimized_pose;
    lidar_state.velocity = local_velocity.head<3>();
    lidar_state.angular_velocity = local_velocity.tail<3>();
    lidar_state.linear_acceleration = local_linear_acceleration;

    // [v0.8 Phase 1, diagnostic-only] expose the final ICP linear system and
    // its eigen-summary. Purely additive: nothing above this line (the pose/
    // velocity/acceleration estimate) depends on this field.
    PersistentWeakDirectionState persistent_direction;
    if (config.degeneracy_adaptive_iteration_budget) {
      if (has_weak_information_direction(icp_H, config.degeneracy_adaptive_iteration_ratio)) {
        _adaptive_iteration_hold_remaining = config.degeneracy_adaptive_hold_scans;
      } else if (_adaptive_iteration_hold_remaining > 0) {
        --_adaptive_iteration_hold_remaining;
      }
    } else {
      _adaptive_iteration_hold_remaining = 0;
    }
    if (config.degeneracy_persistence_gate) {
      PersistentWeakDirectionConfig persistence_config;
      persistence_config.min_consecutive_scans = config.degeneracy_persistence_min_scans;
      persistence_config.min_absolute_cosine = config.degeneracy_persistence_min_absolute_cosine;
      persistence_config.min_translation_fraction = config.degeneracy_persistence_min_translation_fraction;
      persistence_config.require_multiscan_observability = config.degeneracy_multiscan_observability_gate;
      persistence_config.observability_window_scans = config.degeneracy_observability_window_scans;
      persistence_config.observability_min_scans = config.degeneracy_observability_min_scans;
      persistence_config.max_aggregate_directional_information_ratio =
          config.degeneracy_observability_max_directional_ratio;
      persistent_direction = _persistent_weak_direction_tracker.observe(
          icp_H, config.degeneracy_persistence_tracking_ratio, config.degeneracy_multiplicity_relative_gap,
          persistence_config);
    } else {
      _persistent_weak_direction_tracker.reset();
    }
    // Store this scan's own reflectivity profile (world-anchored via the just-optimized
    // pose) for the next scan's correlation. Kept alive as long as the persistence tracker
    // still has a candidate axis, even before it's confirmed, so a profile is already
    // available the moment confirmation fires. Dropped whenever the candidate direction is
    // lost (e.g. the Hessian became well-conditioned again): a stale axis/profile from an
    // unrelated degenerate direction must not be reused.
    if (intensity_alignment_available && persistent_direction.candidate_available) {
      Vector3dVector world_points(intensity_aligned_points.size());
      std::transform(intensity_aligned_points.cbegin(), intensity_aligned_points.cend(), world_points.begin(),
                     [&](const auto& point) { return optimized_pose * point; });
      IntensityProfileConfig profile_config;
      profile_config.bin_size_m = config.intensity_bin_size_m;
      profile_config.half_length_m = config.intensity_profile_half_length_m;
      profile_config.max_shift_m = config.intensity_max_shift_m;
      profile_config.min_correlation = config.intensity_min_correlation;
      profile_config.min_filled_bins = config.intensity_min_filled_bins;
      const Eigen::Vector3d origin = centroid_of(world_points);
      const Eigen::Vector3d axis = persistent_direction.axis.head<3>().normalized();
      const IntensityProfile profile =
          build_intensity_profile(world_points, intensity_aligned_values, axis, origin, profile_config);
      if (profile.valid) {
        _previous_intensity_profile = StoredIntensityProfile{profile, origin, axis};
      } else {
        _previous_intensity_profile.reset();
      }
    } else if (intensity_alignment_available) {
      _previous_intensity_profile.reset();
    }
    if (visual_pose_prior.has_value()) {
      ++visual_prior_attempt_count;
      visual_fused_direction_count += visual_fused_directions;
      visual_unobservable_direction_count += visual_unobservable_directions;
      visual_observability_diagnostics.push_back(
          {current_lidar_time, visual_directional_information_ratios, visual_directional_information_ratio_count});
      visual_fused_scan_count += visual_fused_directions > 0 ? 1U : 0U;
    }
    if (radar_prior_pose.has_value()) {
      ++radar_prior_attempt_count;
      radar_fused_scan_count += degeneracy_intervention_count > 0 ? 1U : 0U;
    }
    lidar_state.icp_diagnostics = IcpDiagnostics{icp_H, icp_b, LocalizabilitySummary::from_hessian(icp_H),
                                                 persistent_direction, degeneracy_intervention_count};
    if (config.degeneracy_persistence_gate) {
      degeneracy_persistence_diagnostics.push_back({current_lidar_time, persistent_direction, degeneracy_intervention_count});
    }

    imu_state = lidar_state; // correct the drift in imu integration
  }
  // even if map is empty, time should still update
  lidar_state.time = current_lidar_time;
  // reset imu to last know lidar state (redundant with the ICP branch's `imu_state = lidar_state` above, but
  // still needed on the initial map-still-empty scans before the local map has enough points for ICP to run).
  imu_state = lidar_state;

  // reset imu averages
  interval_stats.reset();

  // Gate-corrected scans stay out of the map (see Config::kinematic_gate_skip_map_update):
  // inserting a glided scan would let the next ICP anchor to its own correction.
  if (!(kinematic_gate_corrected_this_scan && config.kinematic_gate_skip_map_update) &&
      !kinematic_blend_suppress_map_update_this_scan) {
    if (kinematic_blend_map_update_fraction_this_scan >= 1.0) {
      update_maps(map_input, lidar_state.pose);
    } else {
      // Uniform deterministic thinning keeps some fresh geometry available without
      // letting a propagation-dominated pose fully rewrite the local map.
      Vector3dVector thinned_map_input;
      thinned_map_input.reserve(static_cast<std::size_t>(
          std::ceil(kinematic_blend_map_update_fraction_this_scan * map_input.size())));
      for (std::size_t i = 0; i < map_input.size(); ++i) {
        const auto previous_count = static_cast<std::size_t>(
            std::floor(kinematic_blend_map_update_fraction_this_scan * static_cast<double>(i)));
        const auto next_count = static_cast<std::size_t>(
            std::floor(kinematic_blend_map_update_fraction_this_scan * static_cast<double>(i + 1)));
        if (next_count > previous_count) {
          thinned_map_input.push_back(map_input[i]);
        }
      }
      update_maps(thinned_map_input, lidar_state.pose);
    }
  }

  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  _consecutive_registration_failures = 0;

  return preproc_result.filtered_frame;
}

Vector3dVector LIO::register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                                  const Vector3dVector& scan,
                                  const TimestampVector& timestamps,
                                  const std::vector<float>* intensities) {
  if (extrinsic_lidar2base.log().norm() < EPSILON) {
    return register_scan(scan, timestamps, intensities);
  }

  Vector3dVector transformed_scan(scan.size());
  std::transform(scan.cbegin(), scan.cend(), transformed_scan.begin(),
                 [&](const Eigen::Vector3d& p) { return extrinsic_lidar2base * p; });
  Vector3dVector frame = register_scan(transformed_scan, timestamps, intensities);
  transform_points(extrinsic_lidar2base.inverse(), frame);
  return frame;
}
} // namespace rko_lio::core
