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
#include "preprocess_scan.hpp"
#include "profiler.hpp"
#include "util.hpp"
// other
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
// stl
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace {
constexpr double EPSILON = 1e-8;
constexpr auto EPSILON_TIME = std::chrono::nanoseconds(10);
using namespace rko_lio::core;

inline void transform_points(const Sophus::SE3d& T, Vector3dVector& points) {
  std::transform(points.begin(), points.end(), points.begin(), [&](const auto& point) { return T * point; });
}

using LinearSystem = std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double>;
template <typename VoxelMap>
LinearSystem build_icp_linear_system(const Sophus::SE3d& current_pose,
                                     const rko_lio::core::Vector3dVector& frame,
                                     const VoxelMap& voxel_map,
                                     const double& max_correspondance_distance,
                                     const int voxel_search_radius = 1,
                                     // [instrumentation, additive-only] optional out-param exposing this
                                     // iteration's raw correspondence count, purely for the ICP iteration/
                                     // correspondence histogram (see IcpIterationHistogram in profiler.hpp).
                                     // Defaulted to nullptr so every existing call site is unaffected; the
                                     // value written here is never read back into the solve below.
                                     int* correspondences_out = nullptr) {
  auto linear_system_reduce = [](LinearSystem lhs, const LinearSystem& rhs) {
    auto& [lhs_H, lhs_b, lhs_chi] = lhs;
    const auto& [rhs_H, rhs_b, rhs_chi] = rhs;
    lhs_H += rhs_H;
    lhs_b += rhs_b;
    lhs_chi += rhs_chi;
    return lhs;
  };

  auto linear_system_for_one_point = [](const Eigen::Vector3d& source, const Eigen::Vector3d& target) {
    Eigen::Matrix3_6d J_r;
    J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
    const Eigen::Vector3d residual = source - target;
    return LinearSystem(J_r.transpose() * J_r,
                        J_r.transpose() * residual,
                        residual.squaredNorm());
  };

  // The only parallel part
  using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
  std::atomic<int> correspondances_counter = 0;
  // A fixed reduction tree keeps primary ICP invariant when optional
  // frontends add other TBB workloads between scans.
  const auto& [H_icp, b_icp, chi_icp] = tbb::parallel_deterministic_reduce(
      // Range
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
      // Identity
      LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, LinearSystem J) -> LinearSystem {
        return std::transform_reduce(r.begin(), r.end(), J, linear_system_reduce, [&](const auto& point) {
          // Compute data association and linear system
          const Eigen::Vector3d transformed_point = current_pose * point;
          const auto& [closest_neighbor, distance] = voxel_map.GetClosestNeighbor(transformed_point, voxel_search_radius);
          if (distance < max_correspondance_distance) {
            correspondances_counter++;
            return linear_system_for_one_point(transformed_point, closest_neighbor);
          }
          // TODO (meher): additional 0 add flops, which may hurt single threaded perf slightly
          return LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0);
        });
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      linear_system_reduce);

  if (correspondances_counter == 0) {
    throw std::runtime_error("Number of correspondences are 0.");
  }

  // [instrumentation, additive-only] see parameter comment above.
  if (correspondences_out != nullptr) {
    *correspondences_out = correspondances_counter.load();
  }

  return {H_icp / correspondances_counter, b_icp / correspondances_counter, 0.5 * chi_icp};
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
  // [instrumentation, additive-only] iterations actually taken by the Gauss-
  // Newton loop below (1..iteration_budget) and the mean per-iteration
  // correspondence count, purely for the ICP iteration histogram. Neither
  // value is read back into the pose/H/b computation.
  std::size_t iterations_used = 0;
  double avg_correspondences_per_iteration = 0.0;
};

template <typename VoxelMap>
IcpResult icp(const Vector3dVector& frame,
             const VoxelMap& voxel_map,
             const Sophus::SE3d& initial_guess,
             const LIO::Config& config,
             const std::optional<AccelInfo>& optional_accel_info,
             const int voxel_search_radius = 1,
             const PersistentWeakDirectionState& persistent_direction = {},
             const bool start_with_extended_iteration_budget = false,
             const std::optional<VisualPosePrior>& visual_pose_prior = std::nullopt) {
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
  // [instrumentation, additive-only] tallies feeding IcpResult::iterations_used
  // / avg_correspondences_per_iteration; see the struct comment above.
  std::size_t iterations_used = 0;
  long long correspondences_sum = 0;
  for (size_t i = 0; i < iteration_budget; ++i) {
    int correspondences_this_iteration = 0;
    const auto& [H, b, chi] = std::invoke([&]() -> LinearSystem {
      const auto& [H_icp, b_icp, chi_icp] =
          build_icp_linear_system(
              current_pose, frame, voxel_map, config.max_correspondance_distance, voxel_search_radius,
              &correspondences_this_iteration);
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
      const Eigen::Vector6d prior_update = (initial_guess * current_pose.inverse()).log();
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

    // [instrumentation, additive-only]
    iterations_used = i + 1;
    correspondences_sum += correspondences_this_iteration;

    if (dx.norm() < config.convergence_criterion || i == (iteration_budget - 1)) {
      // TODO: proper debug logging
      // std::cout << "iter " << i << ", beta: " << beta << ", chi: " << chi << ", num_assoc: " <<
      // correspondences.size() << "\n";
      break;
    }
  }
  // [instrumentation, additive-only]
  const double avg_correspondences_per_iteration =
      iterations_used > 0 ? static_cast<double>(correspondences_sum) / static_cast<double>(iterations_used) : 0.0;
  std::size_t visual_fused_directions = 0;
  std::size_t visual_unobservable_directions = 0;
  std::array<double, 6> visual_directional_information_ratios{};
  std::size_t visual_directional_information_ratio_count = 0;
  if (visual_pose_prior.has_value()) {
    const Eigen::Vector6d visual_update =
        (visual_pose_prior->pose * current_pose.inverse()).log();
    const auto fused = fuse_visual_in_weak_directions(
        last_H, last_b, visual_update,
        visual_pose_prior->confidence, config.visual_fusion);
    visual_unobservable_directions = fused.visual_unobservable_directions;
    visual_directional_information_ratios =
        fused.visual_directional_information_ratios;
    visual_directional_information_ratio_count =
        fused.visual_directional_information_ratio_count;
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
          visual_directional_information_ratio_count,
          iterations_used, avg_correspondences_per_iteration};
}

struct CompositeVoxelMap {
  const SparseVoxelGrid& frozen;
  const SparseVoxelGrid& active;

  std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(
      const Eigen::Vector3d& query,
      const int voxel_search_radius) const {
    const auto frozen_result =
        frozen.GetClosestNeighbor(query, voxel_search_radius);
    const auto active_result =
        active.GetClosestNeighbor(query, voxel_search_radius);
    return std::get<1>(active_result) < std::get<1>(frozen_result) ?
      active_result : frozen_result;
  }
};

struct AlignmentStats {
  int correspondences = 0;
  double inlier_ratio = 0.0;
  double mean_error = std::numeric_limits<double>::max();
};

template <typename VoxelMap>
AlignmentStats evaluate_alignment(const Sophus::SE3d& pose,
                                  const Vector3dVector& frame,
                                  const VoxelMap& voxel_map,
                                  const double max_correspondance_distance,
                                  const int voxel_search_radius) {
  if (frame.empty()) {
    return {};
  }
  int correspondences = 0;
  double error_sum = 0.0;
  for (const Eigen::Vector3d& point : frame) {
    const Eigen::Vector3d transformed_point = pose * point;
    const auto& [closest_neighbor, distance] = voxel_map.GetClosestNeighbor(transformed_point, voxel_search_radius);
    (void)closest_neighbor;
    if (distance < max_correspondance_distance) {
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

int voxel_search_radius_for_distance(const SparseVoxelGrid& voxel_map, const double max_correspondance_distance) {
  const double voxel_size = std::max(1e-6, voxel_map.voxel_size_);
  return std::max(1, static_cast<int>(std::ceil(max_correspondance_distance / voxel_size)) + 1);
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
} // namespace

// ==========================
//   actual LIO class stuff
// ==========================

namespace rko_lio::core {

// ==========================
//          private
// ==========================

void LIO::initialize(const Secondsd lidar_time) {
  if (interval_stats.imu_count == 0) {
    std::cerr << "[WARNING] Cannot initialize. No imu measurements received.\n";
    // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    tracking_poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    _initialized = true;
    return;
  }

  const Eigen::Vector3d avg_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3d avg_gyro = interval_stats.angular_velocity_sum / interval_stats.imu_count;

  _imu_local_rotation = align_accel_to_z_world(avg_accel);
  _imu_local_rotation_time = lidar_time;
  lidar_state.pose.so3() = _imu_local_rotation;

  // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  tracking_poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);

  // the pose for the current time gets logged at the end of register_scan in the typical fashion
  lidar_state.time = lidar_time;

  const Eigen::Vector3d local_gravity = _imu_local_rotation.inverse() * gravity();
  imu_bias.accelerometer = avg_accel + local_gravity;
  imu_bias.gyroscope = avg_gyro;

  _initialized = true;
  std::cout << "[INFO] Odometry map frame initialized using " << interval_stats.imu_count
            << " IMU measurements. Estimated initial rotation [se(3)] is " << _imu_local_rotation.log().transpose()
            << "\n";
  std::cout << "[INFO] Estimated accel bias: " << imu_bias.accelerometer.transpose()
            << ", gyro bias: " << imu_bias.gyroscope.transpose() << "\n";
}

// use the acceleration kalman filter to compute the two values we need for ori. reg.
std::optional<AccelInfo> LIO::get_accel_info(const Sophus::SO3d& rotation_estimate, const Secondsd& time) {
  if (interval_stats.imu_count <= 1) {
    std::cerr << "[WARNING] " << interval_stats.imu_count
              << " IMU message(s) in interval between two lidar scans. Cannot compute "
                 "acceleration statistics for orientation regularisation. Please check your data and its "
                 "timestamping as likely there should not be so few IMU measurements between two LiDAR scans.\n";
    return std::nullopt;
  }

  const Eigen::Vector3d avg_imu_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const double accel_mag_variance = interval_stats.welford_sum_of_squares / (interval_stats.imu_count - 1);
  const double dt = (time - lidar_state.time).count();

  const Eigen::Vector3d& body_accel_measurement = avg_imu_accel + rotation_estimate.inverse() * gravity();

  const double max_acceleration_change = config.max_expected_jerk * dt;
  // assume [j, -j] range for uniform dist. on jerk. variance is (2j)^2 / 12 = j^2/3. multiply by dt^2 for accel
  const Eigen::Matrix3d process_noise = square(max_acceleration_change) / 3 * Eigen::Matrix3d::Identity();
  body_acceleration_covariance += process_noise;

  // isotropic accel mag variance
  const Eigen::Matrix3d measurement_noise = accel_mag_variance / 3 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = body_acceleration_covariance + measurement_noise;
  const Eigen::Matrix3d kalman_gain = body_acceleration_covariance * S.inverse();

  const Eigen::Vector3d innovation = kalman_gain * (body_accel_measurement - mean_body_acceleration);
  mean_body_acceleration += innovation;
  body_acceleration_covariance -= kalman_gain * body_acceleration_covariance;

  const Eigen::Vector3d local_gravity_estimate = avg_imu_accel - mean_body_acceleration; // points upwards

  return AccelInfo{.accel_mag_variance = accel_mag_variance, .local_gravity_estimate = local_gravity_estimate};
}

void LIO::update_maps(const Vector3dVector& map_update_frame, const Sophus::SE3d& pose) {
  map.Update(map_update_frame, pose);

  // [instrumentation, additive-only] Periodic map-growth gauge: sample the
  // registration map's active voxel count and stored point count so growth
  // over a run is visible. Sampled every N scans (count-only, no per-point
  // work or point copies) to keep overhead negligible; does not affect the
  // pose/map computation above or below.
  ++_map_growth_scan_counter;
  constexpr std::size_t kMapGrowthSampleStride = 25;
  if (_map_growth_scan_counter == 1 || _map_growth_scan_counter % kMapGrowthSampleStride == 0) {
    MapGrowthGauge::sample(
        "RegistrationMap", _map_growth_scan_counter, map.map_.activeCellsCount(), map.ActivePointCount());
  }

  // The unpruned global map exists only for the opt-in kidnap recovery path.
  // Avoid transforming, copying, and retaining every map point when recovery
  // is disabled (the normal SLAM-only configuration).
  if (config.enable_kidnap_relocalization) {
    Vector3dVector points_transformed(map_update_frame.size());
    std::transform(map_update_frame.cbegin(), map_update_frame.cend(), points_transformed.begin(),
                   [&](const auto& point) { return pose * point; });
    relocalization_map.AddPoints(points_transformed);
  }
}

bool LIO::local_map_empty() const {
  return config.fixed_lag_multiscan ?
      (_fixed_lag_frozen_map.Empty() && _fixed_lag_active_map.Empty()) :
      map.Empty();
}

Vector3dVector LIO::local_map_pointcloud() const {
  Vector3dVector points = config.fixed_lag_multiscan ?
      _fixed_lag_frozen_map.Pointcloud() : map.Pointcloud();
  if (config.fixed_lag_multiscan) {
    Vector3dVector active_points = _fixed_lag_active_map.Pointcloud();
    points.insert(
        points.end(), std::make_move_iterator(active_points.begin()),
        std::make_move_iterator(active_points.end()));
  }
  return points;
}

void LIO::reset_fixed_lag_window() {
  finalize_fixed_lag_window();
  _fixed_lag_frozen_map.Clear();
  _fixed_lag_active_map.Clear();
  _fixed_lag_tracking_pose = Sophus::SE3d{};
  _fixed_lag_frames.clear();
  _fixed_lag_scan_constraints.clear();
}

std::vector<LIO::FinalizedFixedLagPose> LIO::take_finalized_fixed_lag_poses() {
  std::vector<FinalizedFixedLagPose> finalized;
  finalized.swap(_finalized_fixed_lag_poses);
  return finalized;
}

void LIO::finalize_fixed_lag_window() {
  for (FixedLagFrame& frame : _fixed_lag_frames) {
    _finalized_fixed_lag_poses.push_back({frame.time, frame.pose});
    _fixed_lag_frozen_map.Update(frame.map_update_frame, frame.pose);
  }
  _fixed_lag_frames.clear();
  _fixed_lag_scan_constraints.clear();
  _fixed_lag_active_map.Clear();
}

void LIO::rebuild_fixed_lag_active_map() {
  _fixed_lag_active_map.Clear();
  for (const FixedLagFrame& frame : _fixed_lag_frames) {
    _fixed_lag_active_map.Update(frame.map_update_frame, frame.pose);
  }
}

Sophus::SE3d LIO::update_fixed_lag_window(
    const Vector3dVector& map_update_frame,
    const Vector3dVector& keypoints,
    const Secondsd& time,
    const Sophus::SE3d& pose) {
  const std::size_t window_size = std::max<std::size_t>(2U, config.fixed_lag_window_size);
  const std::size_t new_index = _fixed_lag_frames.size();
  const std::size_t neighbor_count = std::min(
      config.fixed_lag_neighbor_scans, _fixed_lag_frames.size());
  const std::size_t first_neighbor = _fixed_lag_frames.size() - neighbor_count;

  LIO::Config pairwise_config = config;
  pairwise_config.max_iterations = std::max<std::size_t>(
      1U, config.fixed_lag_pairwise_max_iterations);
  pairwise_config.min_beta = -1.0;
  pairwise_config.degeneracy_aware_solve = false;
  pairwise_config.degeneracy_adaptive_iteration_budget = false;
  pairwise_config.visual_fusion.enabled = false;
  pairwise_config.fixed_lag_multiscan = false;

  for (std::size_t neighbor_index = first_neighbor;
       neighbor_index < _fixed_lag_frames.size(); ++neighbor_index) {
    ++fixed_lag_pairwise_attempt_count;
    const FixedLagFrame& neighbor = _fixed_lag_frames[neighbor_index];
    SparseVoxelGrid target(
        config.voxel_size, config.max_range, config.max_points_per_voxel);
    target.AddPoints(neighbor.keypoints);
    const Sophus::SE3d initial_relative = neighbor.pose.inverse() * pose;
    try {
      const AlignmentStats initial_alignment = evaluate_alignment(
          initial_relative, keypoints, target,
          config.max_correspondance_distance, 1);
      const IcpResult pairwise = icp(
          keypoints, target, initial_relative, pairwise_config, std::nullopt);
      const AlignmentStats refined_alignment = evaluate_alignment(
          pairwise.pose, keypoints, target,
          config.max_correspondance_distance, 1);
      const Eigen::Vector6d correction =
          (pairwise.pose * initial_relative.inverse()).log();
      const double rotation_deg = correction.tail<3>().norm() *
          180.0 / std::numbers::pi;
      if (correction.head<3>().norm() <=
              config.fixed_lag_pairwise_max_translation_m &&
          rotation_deg <= config.fixed_lag_pairwise_max_rotation_deg &&
          refined_alignment.correspondences >= static_cast<int>(
              config.fixed_lag_pairwise_min_correspondences) &&
          refined_alignment.inlier_ratio >=
              config.fixed_lag_pairwise_min_inlier_ratio &&
          refined_alignment.mean_error <= initial_alignment.mean_error *
              (1.0 - config.fixed_lag_pairwise_min_error_reduction)) {
        _fixed_lag_scan_constraints.push_back({
            neighbor_index, new_index, pairwise.pose,
            config.fixed_lag_scan_constraint_weight});
        ++fixed_lag_pairwise_accept_count;
      }
    } catch (const std::exception&) {
      // A missing pairwise overlap removes one optional factor. The primary
      // scan-to-map registration has already accepted this frame.
    }
  }

  _fixed_lag_frames.push_back({
      time, pose, pose, map_update_frame, keypoints, poses_with_timestamps.size()});

  std::vector<Sophus::SE3d> initial_poses;
  initial_poses.reserve(_fixed_lag_frames.size());
  for (const FixedLagFrame& frame : _fixed_lag_frames) {
    initial_poses.push_back(frame.pose);
  }
  std::vector<FixedLagRelativeConstraint> constraints =
      _fixed_lag_scan_constraints;
  for (std::size_t pose_index = 1U; pose_index < _fixed_lag_frames.size(); ++pose_index) {
    constraints.push_back({
        0U, pose_index,
        _fixed_lag_frames.front().odometry_pose.inverse() *
            _fixed_lag_frames[pose_index].odometry_pose,
        config.fixed_lag_odometry_prior_weight});
  }

  FixedLagPoseOptimizerConfig optimizer_config;
  optimizer_config.huber_delta_m = config.fixed_lag_huber_delta_m;
  optimizer_config.fix_latest_pose = config.fixed_lag_fix_latest_pose;
  const FixedLagPoseOptimizerResult optimized =
      optimize_fixed_lag_poses(initial_poses, constraints, optimizer_config);
  ++fixed_lag_window_attempt_count;
  bool accept_optimization = optimized.valid &&
      optimized.final_cost <= optimized.initial_cost;
  if (accept_optimization) {
    for (std::size_t pose_index = 0U; pose_index < optimized.poses.size(); ++pose_index) {
      // Gate total displacement from primary odometry, not merely this
      // iteration's increment. Otherwise many individually-small accepted
      // window solves can accumulate an unbounded trajectory deformation.
      const Eigen::Vector6d correction =
          (optimized.poses[pose_index] *
           _fixed_lag_frames[pose_index].odometry_pose.inverse()).log();
      const double rotation_deg = correction.tail<3>().norm() *
          180.0 / std::numbers::pi;
      const bool latest_pose = pose_index + 1U == optimized.poses.size();
      if (correction.head<3>().norm() > config.fixed_lag_max_pose_correction_m ||
          rotation_deg > config.fixed_lag_max_pose_correction_deg ||
          (latest_pose &&
           (correction.head<3>().norm() >
                config.fixed_lag_max_latest_pose_correction_m ||
            rotation_deg >
                config.fixed_lag_max_latest_pose_correction_deg))) {
        accept_optimization = false;
        break;
      }
    }
  }
  if (accept_optimization && !_fixed_lag_frames.empty()) {
    const CompositeVoxelMap target_map{
        _fixed_lag_frozen_map, _fixed_lag_active_map};
    if (!_fixed_lag_frozen_map.Empty() || !_fixed_lag_active_map.Empty()) {
      const AlignmentStats initial_alignment = evaluate_alignment(
          pose, keypoints, target_map, config.max_correspondance_distance, 1);
      const AlignmentStats refined_alignment = evaluate_alignment(
          optimized.poses.back(), keypoints, target_map,
          config.max_correspondance_distance, 1);
      const double minimum_correspondences =
          config.fixed_lag_map_min_correspondence_ratio *
          static_cast<double>(initial_alignment.correspondences);
      if (static_cast<double>(refined_alignment.correspondences) <
              minimum_correspondences ||
          refined_alignment.mean_error > initial_alignment.mean_error *
              config.fixed_lag_map_max_error_ratio) {
        accept_optimization = false;
      }
    }
  }
  if (accept_optimization) {
    ++fixed_lag_window_accept_count;
    for (std::size_t pose_index = 0U; pose_index < optimized.poses.size(); ++pose_index) {
      FixedLagFrame& frame = _fixed_lag_frames[pose_index];
      const Eigen::Vector6d applied =
          (optimized.poses[pose_index] * frame.pose.inverse()).log();
      const double applied_translation_m = applied.head<3>().norm();
      const double applied_rotation_deg =
          applied.tail<3>().norm() * 180.0 / std::numbers::pi;
      fixed_lag_max_applied_translation_m = std::max(
          fixed_lag_max_applied_translation_m, applied_translation_m);
      fixed_lag_max_applied_rotation_deg = std::max(
          fixed_lag_max_applied_rotation_deg, applied_rotation_deg);
      ++fixed_lag_applied_pose_count;
      fixed_lag_applied_translation_squared_sum +=
          applied_translation_m * applied_translation_m;
      fixed_lag_applied_rotation_deg_squared_sum +=
          applied_rotation_deg * applied_rotation_deg;
      if (pose_index + 1U == optimized.poses.size()) {
        fixed_lag_max_latest_translation_m = std::max(
            fixed_lag_max_latest_translation_m, applied_translation_m);
        fixed_lag_max_latest_rotation_deg = std::max(
            fixed_lag_max_latest_rotation_deg, applied_rotation_deg);
      }
      frame.pose = optimized.poses[pose_index];
      if (frame.pose_history_index < poses_with_timestamps.size()) {
        poses_with_timestamps[frame.pose_history_index].second = frame.pose;
      }
    }
  }

  while (_fixed_lag_frames.size() > window_size) {
    FixedLagFrame marginalized = std::move(_fixed_lag_frames.front());
    _fixed_lag_frames.pop_front();
    _finalized_fixed_lag_poses.push_back({marginalized.time, marginalized.pose});
    _fixed_lag_frozen_map.Update(
        marginalized.map_update_frame, marginalized.pose);
    std::vector<FixedLagRelativeConstraint> shifted_constraints;
    shifted_constraints.reserve(_fixed_lag_scan_constraints.size());
    for (FixedLagRelativeConstraint constraint : _fixed_lag_scan_constraints) {
      if (constraint.from == 0U || constraint.to == 0U) {
        continue;
      }
      --constraint.from;
      --constraint.to;
      shifted_constraints.push_back(std::move(constraint));
    }
    _fixed_lag_scan_constraints = std::move(shifted_constraints);
  }
  _fixed_lag_frozen_map.RemovePointsFarFromLocation(
      _fixed_lag_frames.back().pose.translation());
  rebuild_fixed_lag_active_map();

  return _fixed_lag_frames.back().pose;
}

Vector3dVector LIO::recover_with_scan(const Vector3dVector& filtered_frame,
                                      const Vector3dVector& map_update_frame,
                                      const Secondsd& current_lidar_time,
                                      const Sophus::SE3d& recovery_pose,
                                      const std::string& reason) {
  reset_fixed_lag_window();
  map.Clear();
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
  _imu_local_rotation = recovery_pose.so3();
  _imu_local_rotation_time = current_lidar_time;
  interval_stats.reset();
  _deskew_gyro_samples.clear();
  if (config.fixed_lag_multiscan) {
    _fixed_lag_tracking_pose = recovery_pose;
    update_maps(map_update_frame, recovery_pose);
    const Sophus::SE3d refined_pose = update_fixed_lag_window(
        map_update_frame, map_update_frame, current_lidar_time, recovery_pose);
    lidar_state.pose = config.fixed_lag_fix_latest_pose ?
        _fixed_lag_tracking_pose : refined_pose;
  } else {
    update_maps(map_update_frame, lidar_state.pose);
  }
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  tracking_poses_with_timestamps.emplace_back(
      lidar_state.time, config.fixed_lag_multiscan ?
          _fixed_lag_tracking_pose : lidar_state.pose);
  _consecutive_registration_failures = 0;
  std::cout << "[INFO] Kidnap recovery accepted scan at " << current_lidar_time.count() << "s via " << reason
            << ".\n";
  return filtered_frame;
}

Vector3dVector LIO::drop_failed_scan(const Secondsd& current_lidar_time, const std::string& reason) {
  lidar_state.time = current_lidar_time;
  // [v0.8 Phase 1, diagnostic-only] no ICP solve happened for the dropped scan.
  lidar_state.icp_diagnostics = std::nullopt;
  _persistent_weak_direction_tracker.reset();
  _imu_local_rotation_time = current_lidar_time;
  interval_stats.reset();
  _deskew_gyro_samples.clear();
  std::cerr << "[WARNING] Dropping scan during kidnap recovery: " << reason << "\n";
  return {};
}

std::optional<Sophus::SE3d> LIO::try_global_relocalization(const Vector3dVector& keypoints) const {
  if (!config.enable_kidnap_relocalization || relocalization_map.Empty() || keypoints.empty()) {
    return std::nullopt;
  }
  const int usable_pose_count =
      static_cast<int>(poses_with_timestamps.size()) - std::max(0, config.relocalization_min_pose_separation);
  if (usable_pose_count <= 0) {
    return std::nullopt;
  }

  LIO::Config relocalization_config = config;
  relocalization_config.max_iterations = static_cast<size_t>(std::max(1, config.relocalization_max_iterations));
  relocalization_config.max_correspondance_distance = config.relocalization_max_correspondance_distance;
  relocalization_config.min_beta = -1.0;
  const int pose_stride = std::max(1, config.relocalization_pose_stride);
  const int yaw_samples = std::max(1, config.relocalization_yaw_samples);
  const int voxel_search_radius =
      voxel_search_radius_for_distance(relocalization_map, relocalization_config.max_correspondance_distance);
  const double pi = std::acos(-1.0);

  bool found = false;
  Sophus::SE3d best_pose;
  AlignmentStats best_stats;
  for (int pose_index = 0; pose_index < usable_pose_count; pose_index += pose_stride) {
    const Sophus::SE3d& historical_pose = poses_with_timestamps[static_cast<size_t>(pose_index)].second;
    for (int yaw_index = 0; yaw_index < yaw_samples; ++yaw_index) {
      const double yaw = (2.0 * pi * static_cast<double>(yaw_index)) / static_cast<double>(yaw_samples);
      const Sophus::SE3d initial_guess(
          yaw_rotation(yaw) * historical_pose.so3(),
          historical_pose.translation());
      Sophus::SE3d optimized_pose;
      try {
        // [v0.8 Phase 1] icp() now returns IcpResult; relocalization only
        // needs the pose, so the diagnostic H/b are discarded here (a fresh
        // local map is about to replace the current one, see
        // LIO::recover_with_scan, which invalidates any prior diagnostics
        // anyway).
        optimized_pose = icp(
            keypoints,
            relocalization_map,
            initial_guess,
            relocalization_config,
            std::nullopt,
            voxel_search_radius).pose;
      } catch (const std::exception&) {
        continue;
      }
      const AlignmentStats stats = evaluate_alignment(
          optimized_pose,
          keypoints,
          relocalization_map,
          relocalization_config.max_correspondance_distance,
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

  if (_imu_local_rotation_time < EPSILON_TIME) {
    _imu_local_rotation_time = lidar_state.time;
  }

  const double dt = (base_imu.time - _imu_local_rotation_time).count();

  if (dt < 0.0) {
    // messages are out of sync. thats a problem, since we integrate gyro from last lidar time onwards
    std::cerr << "[WARNING] Received IMU message from the past. Can result in errors.\n";
    // maybe skip this imu reading?
  }

  const Eigen::Vector3d unbiased_ang_vel = base_imu.angular_velocity - imu_bias.gyroscope;
  const Eigen::Vector3d unbiased_accel = base_imu.acceleration - imu_bias.accelerometer;

  _deskew_gyro_samples.push_back({base_imu.time, unbiased_ang_vel});

  _imu_local_rotation = _imu_local_rotation * Sophus::SO3d::exp(unbiased_ang_vel * dt);
  _imu_local_rotation_time = base_imu.time;

  const Eigen::Vector3d local_gravity = _imu_local_rotation.inverse() * gravity();
  const Eigen::Vector3d compensated_accel = unbiased_accel + local_gravity;

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
  const Secondsd dt = raw_imu.time - _last_real_imu_time;

  const Eigen::Vector3d angular_acceleration = std::invoke([&]() -> Eigen::Vector3d {
    if (std::chrono::abs(dt) < Secondsd(1.0 / 5000.0)) {
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
          (base_imu.angular_velocity - _last_real_base_imu_ang_vel) / dt.count();
      return angular_acceleration;
    }
  });

  base_imu.acceleration = extrinsic_rotation * raw_imu.acceleration + angular_acceleration.cross(lever_arm) +
                          base_imu.angular_velocity.cross(base_imu.angular_velocity.cross(lever_arm));

  this->add_imu_measurement(base_imu);
}

Sophus::SE3d LIO::predict_pose_at(const Secondsd& time) const {
  const double dt = (time - lidar_state.time).count();
  Eigen::Vector3d average_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d average_angular_velocity = Eigen::Vector3d::Zero();
  if (config.initialization_phase && !_initialized) {
    // The registration path assumes zero motion while collecting its
    // initialization window.
  } else if (interval_stats.imu_count > 0) {
    average_acceleration =
        interval_stats.body_acceleration_sum / interval_stats.imu_count;
    average_angular_velocity =
        interval_stats.angular_velocity_sum / interval_stats.imu_count;
  } else {
    average_angular_velocity = lidar_state.angular_velocity;
  }
  Eigen::Vector6d motion = Eigen::Vector6d::Zero();
  motion.head<3>() = lidar_state.velocity * dt +
                     average_acceleration * square(dt) / 2.0;
  motion.tail<3>() = average_angular_velocity * dt;
  const Sophus::SE3d& base_pose =
      config.fixed_lag_multiscan && !config.fixed_lag_fix_latest_pose ?
          _fixed_lag_tracking_pose : lidar_state.pose;
  return base_pose * Sophus::SE3d::exp(motion);
}

// ============================ lidar ===============================

Vector3dVector LIO::register_scan(const Vector3dVector& scan, const TimestampVector& timestamps) {
  // TODO: redundant max compute as its available after process_timestamps
  const auto max = std::max_element(timestamps.cbegin(), timestamps.cend());
  const Secondsd current_lidar_time = *max;

  if (lidar_state.time < EPSILON_TIME) {
    lidar_state.time = current_lidar_time;
    const auto& preproc_result = preprocess_scan(scan, config);
    if (!config.initialization_phase) {
      // use the first frame for the map only if we're not initializing
      if (config.fixed_lag_multiscan) {
        _fixed_lag_tracking_pose = lidar_state.pose;
        update_maps(preproc_result.map_update_frame(), _fixed_lag_tracking_pose);
        const Sophus::SE3d refined_pose = update_fixed_lag_window(
            preproc_result.map_update_frame(), preproc_result.keypoints,
            lidar_state.time, lidar_state.pose);
        lidar_state.pose = config.fixed_lag_fix_latest_pose ?
            _fixed_lag_tracking_pose : refined_pose;
      } else {
        update_maps(preproc_result.map_update_frame(), lidar_state.pose);
      }
      poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
      tracking_poses_with_timestamps.emplace_back(
          lidar_state.time, config.fixed_lag_multiscan ?
              _fixed_lag_tracking_pose : lidar_state.pose);
      std::cout << "[INFO] Odometry map frame initialized with first lidar scan.\n";
    }
    return preproc_result.filtered_frame;
  }

  if (std::chrono::abs(current_lidar_time - lidar_state.time).count() > config.max_scan_delta_sec) {
    const double diff_seconds = (current_lidar_time - lidar_state.time).count();
    // TODO: std::expected with tl::expected (because ros humble)
    throw std::invalid_argument("Received LiDAR scan with " + std::to_string(diff_seconds) +
                                " seconds delta to previous scan.");
  }

  const auto& [avg_body_accel, avg_ang_vel] = std::invoke([&]() -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
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
  });

  // compute relative motion using controls
  auto relative_pose_at_time = [&](const Secondsd time) -> Sophus::SE3d {
    const double dt = (time - lidar_state.time).count();
    Eigen::Matrix<double, 6, 1> tau;
    tau.head<3>() = lidar_state.velocity * dt + (avg_body_accel * square(dt) / 2);
    tau.tail<3>() = avg_ang_vel * dt;
    const Sophus::SE3d constant_twist_pose = Sophus::SE3d::exp(tau);
    if (!config.piecewise_gyro_deskew || time < lidar_state.time) {
      return constant_twist_pose;
    }
    const Sophus::SO3d integrated_rotation = integrate_piecewise_angular_velocity(
        _deskew_gyro_samples, lidar_state.time, time, lidar_state.angular_velocity);
    return Sophus::SE3d(integrated_rotation, constant_twist_pose.translation());
  };

  const Sophus::SE3d initial_guess = predict_pose_at(current_lidar_time);

  std::optional<VisualPosePrior> visual_pose_prior;
  if (_visual_pose_prior.has_value() && config.visual_fusion.enabled &&
      std::abs((_visual_pose_prior->time - current_lidar_time).count()) <=
          config.visual_prior_max_time_offset_sec) {
    visual_pose_prior = _visual_pose_prior;
  }
  // A camera measurement is single-use even when stale or rejected. This
  // prevents one visual edge from being silently applied to multiple scans.
  _visual_pose_prior.reset();

  // body acceleration filter
  const auto& accel_filter_info = get_accel_info(initial_guess.so3(), current_lidar_time);

  const auto& preproc_result = preprocess_scan(scan, timestamps, current_lidar_time, relative_pose_at_time, config);

  if (preproc_result.keypoints.size() < 10) {
    const std::string error_msg =
        "Keypoints for ICP registration = " + std::to_string(preproc_result.keypoints.size()) +
        ", this is too little for ICP and likely unintended. Input scan size = " + std::to_string(scan.size()) +
        ". Config voxel size = " + std::to_string(config.voxel_size) +
        ". Either the input scan is corrupt (empty) or the downsampling is too aggressive.";
    ++_consecutive_registration_failures;
    _persistent_weak_direction_tracker.reset();
    if (config.reset_on_registration_failure &&
        _consecutive_registration_failures >= std::max(1, config.recovery_min_failures)) {
      return drop_failed_scan(current_lidar_time, error_msg);
    }
    throw std::invalid_argument(error_msg);
  }

  if (config.enable_kidnap_relocalization && config.relocalize_after_scan_gap &&
      _consecutive_registration_failures >= std::max(1, config.recovery_min_failures)) {
    if (const auto relocalized_pose = try_global_relocalization(preproc_result.keypoints)) {
      return recover_with_scan(preproc_result.filtered_frame,
                               preproc_result.map_update_frame(),
                               current_lidar_time,
                               relocalized_pose.value(),
                               "global relocalization after scan gap");
    }
  }

  // [v0.8 Phase 1, diagnostic-only] reset before attempting a fresh ICP
  // solve; overwritten below on success. Every early-return path above this
  // point (drop_failed_scan / recover_with_scan) already resets this field
  // itself, so this covers the remaining "about to attempt icp()" case.
  lidar_state.icp_diagnostics = std::nullopt;

  const bool registration_map_empty = map.Empty();
  if (!registration_map_empty) {
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
    try {
      const auto run_primary_icp = [&](const auto& target_map) {
        return icp(preproc_result.keypoints,
                   target_map,
                   initial_guess,
                   config,
                   accel_filter_info,
                   1,
                   _persistent_weak_direction_tracker.state(),
                   config.degeneracy_adaptive_iteration_budget &&
                       _adaptive_iteration_hold_remaining > 0,
                   visual_pose_prior);
      };
      const IcpResult icp_result = run_primary_icp(map);
      // [instrumentation, additive-only] see IcpIterationHistogram in
      // profiler.hpp; purely observational, does not affect optimized_pose.
      IcpIterationHistogram::record(icp_result.iterations_used, icp_result.avg_correspondences_per_iteration);
      optimized_pose = icp_result.pose;
      icp_H = icp_result.H;
      icp_b = icp_result.b;
      degeneracy_intervention_count = icp_result.degeneracy_intervention_count;
      visual_fused_directions = icp_result.visual_fused_directions;
      visual_unobservable_directions =
          icp_result.visual_unobservable_directions;
      visual_directional_information_ratios =
          icp_result.visual_directional_information_ratios;
      visual_directional_information_ratio_count =
          icp_result.visual_directional_information_ratio_count;
    } catch (const std::exception&) {
      ++_consecutive_registration_failures;
      _persistent_weak_direction_tracker.reset();
      _adaptive_iteration_hold_remaining = 0;
      if (_consecutive_registration_failures < std::max(1, config.recovery_min_failures)) {
        throw;
      }
      if (const auto relocalized_pose = try_global_relocalization(preproc_result.keypoints)) {
        return recover_with_scan(preproc_result.filtered_frame,
                                 preproc_result.map_update_frame(),
                                 current_lidar_time,
                                 relocalized_pose.value(),
                                 "global relocalization");
      }
      if (config.reset_on_registration_failure) {
        return recover_with_scan(preproc_result.filtered_frame,
                                 preproc_result.map_update_frame(),
                                 current_lidar_time,
                                 lidar_state.pose,
                                 "local reset");
      }
      throw;
    }

    const Sophus::SE3d previous_pose_for_motion =
        config.fixed_lag_multiscan && !config.fixed_lag_fix_latest_pose ?
            _fixed_lag_tracking_pose : lidar_state.pose;
    const Sophus::SE3d tracking_pose = optimized_pose;
    if (config.fixed_lag_multiscan) {
      update_maps(preproc_result.map_update_frame(), tracking_pose);
      _fixed_lag_tracking_pose = tracking_pose;
      const Sophus::SE3d refined_pose = update_fixed_lag_window(
          preproc_result.map_update_frame(), preproc_result.keypoints,
          current_lidar_time, optimized_pose);
      optimized_pose = config.fixed_lag_fix_latest_pose ?
          tracking_pose : refined_pose;
    }

    // estimate velocities and accelerations from the new pose
    const double dt = (current_lidar_time - lidar_state.time).count();
    const Sophus::SE3d motion =
        previous_pose_for_motion.inverse() * tracking_pose;
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
      persistence_config.require_multiscan_observability =
          config.degeneracy_multiscan_observability_gate;
      persistence_config.observability_window_scans = config.degeneracy_observability_window_scans;
      persistence_config.observability_min_scans = config.degeneracy_observability_min_scans;
      persistence_config.max_aggregate_directional_information_ratio =
          config.degeneracy_observability_max_directional_ratio;
      persistent_direction = _persistent_weak_direction_tracker.observe(icp_H,
                                                                         config.degeneracy_persistence_tracking_ratio,
                                                                         config.degeneracy_multiplicity_relative_gap,
                                                                         persistence_config);
    } else {
      _persistent_weak_direction_tracker.reset();
    }
    if (visual_pose_prior.has_value()) {
      ++visual_prior_attempt_count;
      visual_fused_direction_count += visual_fused_directions;
      visual_unobservable_direction_count += visual_unobservable_directions;
      visual_observability_diagnostics.push_back(
          {current_lidar_time, visual_directional_information_ratios,
           visual_directional_information_ratio_count});
      visual_fused_scan_count += visual_fused_directions > 0 ? 1U : 0U;
    }
    lidar_state.icp_diagnostics = IcpDiagnostics{icp_H,
                                                 icp_b,
                                                 LocalizabilitySummary::from_hessian(icp_H),
                                                 persistent_direction,
                                                 degeneracy_intervention_count};
    if (config.degeneracy_persistence_gate) {
      degeneracy_persistence_diagnostics.push_back(
          {current_lidar_time, persistent_direction, degeneracy_intervention_count});
    }

    // IMU propagation drives primary registration, so it must remain on the
    // unrefined tracking trajectory even when the published map pose changes.
    _imu_local_rotation = tracking_pose.so3();
  }
  // even if map is empty, time should still update
  lidar_state.time = current_lidar_time;
  _imu_local_rotation_time = current_lidar_time;

  // reset imu averages
  interval_stats.reset();
  _deskew_gyro_samples.clear();

  if (config.fixed_lag_multiscan) {
    if (registration_map_empty) {
      _fixed_lag_tracking_pose = lidar_state.pose;
      update_maps(preproc_result.map_update_frame(), _fixed_lag_tracking_pose);
      const Sophus::SE3d refined_pose = update_fixed_lag_window(
          preproc_result.map_update_frame(), preproc_result.keypoints,
          current_lidar_time, lidar_state.pose);
      lidar_state.pose = config.fixed_lag_fix_latest_pose ?
          _fixed_lag_tracking_pose : refined_pose;
    }
  } else {
    update_maps(preproc_result.map_update_frame(), lidar_state.pose);
  }

  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  tracking_poses_with_timestamps.emplace_back(
      lidar_state.time, config.fixed_lag_multiscan ?
          _fixed_lag_tracking_pose : lidar_state.pose);
  _consecutive_registration_failures = 0;

  return preproc_result.filtered_frame;
}

Vector3dVector LIO::register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                                  const Vector3dVector& scan,
                                  const TimestampVector& timestamps) {
  if (extrinsic_lidar2base.log().norm() < EPSILON) {
    return register_scan(scan, timestamps);
  }

  Vector3dVector transformed_scan = scan;
  transform_points(extrinsic_lidar2base, transformed_scan);
  Vector3dVector frame = register_scan(transformed_scan, timestamps);
  transform_points(extrinsic_lidar2base.inverse(), frame);
  return frame;
}
} // namespace rko_lio::core
