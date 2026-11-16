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

// Parameter reading moved verbatim from rko_lio::ros::BaseNode's constructor (node->declare_parameter
// -> params.declare), so the ROS 2 node and the ROS-free driver read identical configurations.

#include "rko_lio/driver/driver_config.hpp"
#include "rko_lio/core/imu_acceleration_unit.hpp"
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rko_lio::driver {

namespace {
Extrinsic parse_extrinsic(Parameters& params, const std::string& name) {
  const std::string param_name = "extrinsic_" + name + "2base_quat_xyzw_xyz";
  const std::vector<double> vec = params.declare<std::vector<double>>(param_name, std::vector<double>{});

  if (vec.size() != 7) {
    if (!vec.empty()) {
      std::ostringstream message;
      message << "Parameter 'extrinsic_" << name << "2base_quat_xyzw_xyz' is set but has wrong size: " << vec.size()
              << ". Expected 7 (qx, qy, qz, qw, x, y, z). check the value: "
              << Eigen::Map<const Eigen::VectorXd>(vec.data(), static_cast<Eigen::Index>(vec.size())).transpose();
      params.warn(message.str());
    }
    return {};
  }
  Eigen::Quaterniond q(vec[3], vec[0], vec[1], vec[2]); // qw, qx, qy, qz
  if (q.norm() < 1e-6) {
    throw std::runtime_error(name + " extrinsic quaternion has zero norm");
  }
  const Sophus::SE3d extrinsic(q, Eigen::Vector3d(vec[4], vec[5], vec[6]));
  std::ostringstream message;
  message << "Parsed " << name << " extrinsic as: " << extrinsic.log().transpose();
  params.info(message.str());
  return {true, extrinsic};
}
} // namespace

DriverConfig declare_driver_config(Parameters& params) {
  DriverConfig config;
  config.imu_topic = params.get_required_string("imu_topic");
  config.lidar_topic = params.get_required_string("lidar_topic");
  config.base_frame = params.get_required_string("base_frame");
  config.imu_frame = params.declare<std::string>("imu_frame", config.imu_frame);
  config.lidar_frame = params.declare<std::string>("lidar_frame", config.lidar_frame);
  config.odom_frame = params.declare<std::string>("odom_frame", config.odom_frame);
  config.odom_topic = params.declare<std::string>("odom_topic", config.odom_topic);
  config.invert_odom_tf = params.declare<bool>("invert_odom_tf", config.invert_odom_tf);

  config.publish_lidar_acceleration =
      params.declare<bool>("publish_lidar_acceleration", config.publish_lidar_acceleration);
  config.imu_acceleration_unit = params.declare<std::string>("imu_acceleration_unit", config.imu_acceleration_unit);
  core::acceleration_scale(config.imu_acceleration_unit, 0.0); // reject an unknown unit at startup
  config.publish_deskewed_scan = params.declare<bool>("publish_deskewed_scan", config.publish_deskewed_scan);
  if (config.publish_deskewed_scan) {
    config.deskewed_scan_topic = params.declare<std::string>("deskewed_scan_topic", config.deskewed_scan_topic);
    config.deskewed_scan_voxel_size =
        params.declare<double>("deskewed_scan_voxel_size", config.deskewed_scan_voxel_size);
  }
  config.publish_local_map = params.declare<bool>("publish_local_map", config.publish_local_map);
  if (config.publish_local_map) {
    config.map_topic = params.declare<std::string>("map_topic", config.map_topic);
    const double publish_map_after_seconds =
        params.declare<double>("publish_map_after", core::to_seconds(config.publish_map_after));
    config.publish_map_after =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(publish_map_after_seconds));
  }

  // lio params
  config.lio.deskew = params.declare<bool>("deskew", config.lio.deskew);
  config.lio.max_iterations =
      static_cast<size_t>(params.declare<int>("max_iterations", static_cast<int>(config.lio.max_iterations)));
  config.lio.voxel_size = params.declare<double>("voxel_size", config.lio.voxel_size);
  config.lio.max_points_per_voxel =
      static_cast<int>(params.declare<int>("max_points_per_voxel", config.lio.max_points_per_voxel));
  config.lio.local_map_compact_points =
      params.declare<bool>("local_map_compact_points", config.lio.local_map_compact_points);
  config.lio.max_range = params.declare<double>("max_range", config.lio.max_range);
  config.lio.min_range = params.declare<double>("min_range", config.lio.min_range);
  config.lio.convergence_criterion =
      params.declare<double>("convergence_criterion", config.lio.convergence_criterion);
  config.lio.max_correspondence_distance =
      params.declare<double>("max_correspondence_distance", config.lio.max_correspondence_distance);
  config.lio.start_standstill_radius =
      params.declare<double>("start_standstill_radius", config.lio.start_standstill_radius);
  config.lio.start_standstill_max_accel_std =
      params.declare<double>("start_standstill_max_accel_std", config.lio.start_standstill_max_accel_std);
  config.lio.start_standstill_hold =
      params.declare<bool>("start_standstill_hold", config.lio.start_standstill_hold);
  config.lio.start_standstill_max_gyro_rad_s = params.declare<double>(
      "start_standstill_max_gyro_rad_s", config.lio.start_standstill_max_gyro_rad_s);
  config.lio.start_standstill_max_sec =
      params.declare<double>("start_standstill_max_sec", config.lio.start_standstill_max_sec);
  config.lio.start_standstill_bias =
      params.declare<std::string>("start_standstill_bias", config.lio.start_standstill_bias);
  config.lio.start_standstill_hold_radius =
      params.declare<double>("start_standstill_hold_radius", config.lio.start_standstill_hold_radius);
  config.lio.start_standstill_moving_start_only = params.declare<bool>(
      "start_standstill_moving_start_only", config.lio.start_standstill_moving_start_only);
  if (config.lio.start_standstill_bias != "interval" && config.lio.start_standstill_bias != "average" &&
      config.lio.start_standstill_bias != "none") {
    throw std::invalid_argument("start_standstill_bias must be interval, average or none");
  }
  config.lio.close_scene_max_median_range =
      params.declare<double>("close_scene_max_median_range", config.lio.close_scene_max_median_range);
  config.lio.close_scene_max_correspondence_distance = params.declare<double>(
      "close_scene_max_correspondence_distance", config.lio.close_scene_max_correspondence_distance);
  config.lio.close_scene_localizability_boost =
      params.declare<double>("close_scene_localizability_boost", config.lio.close_scene_localizability_boost);
  config.lio.close_scene_localizability_normal_only = params.declare<bool>(
      "close_scene_localizability_normal_only", config.lio.close_scene_localizability_normal_only);
  config.lio.close_scene_min_inlier_fraction =
      params.declare<double>("close_scene_min_inlier_fraction", config.lio.close_scene_min_inlier_fraction);
  config.lio.max_num_threads =
      static_cast<int>(params.declare<int>("max_num_threads", config.lio.max_num_threads));
  config.lio.initialization_phase =
      params.declare<bool>("initialization_phase", config.lio.initialization_phase);
  config.lio.max_expected_jerk = params.declare<double>("max_expected_jerk", config.lio.max_expected_jerk);
  config.lio.double_downsample = params.declare<bool>("double_downsample", config.lio.double_downsample);
  config.lio.legacy_voxel_downsample =
      params.declare<bool>("legacy_voxel_downsample", config.lio.legacy_voxel_downsample);
  config.lio.samplewise_gyro_deskew =
      params.declare<bool>("samplewise_gyro_deskew", config.lio.samplewise_gyro_deskew);
  config.lio.min_beta = params.declare<double>("min_beta", config.lio.min_beta);
  config.lio.gait_speed_prior = params.declare<bool>("gait_speed_prior", config.lio.gait_speed_prior);
  config.lio.gait_window_sec = params.declare<double>("gait_window_sec", config.lio.gait_window_sec);
  config.lio.gait_min_peak_fraction =
      params.declare<double>("gait_min_peak_fraction", config.lio.gait_min_peak_fraction);
  config.lio.gait_min_magnitude_std =
      params.declare<double>("gait_min_magnitude_std", config.lio.gait_min_magnitude_std);
  config.lio.gait_min_step_length =
      params.declare<double>("gait_min_step_length", config.lio.gait_min_step_length);
  config.lio.gait_default_step_length =
      params.declare<double>("gait_default_step_length", config.lio.gait_default_step_length);
  config.lio.gait_max_step_length =
      params.declare<double>("gait_max_step_length", config.lio.gait_max_step_length);
  config.lio.gait_min_direction_speed =
      params.declare<double>("gait_min_direction_speed", config.lio.gait_min_direction_speed);
  config.lio.gait_reversal_lookback_sec =
      params.declare<double>("gait_reversal_lookback_sec", config.lio.gait_reversal_lookback_sec);
  config.lio.gait_max_correction_m =
      params.declare<double>("gait_max_correction_m", config.lio.gait_max_correction_m);
  config.lio.gait_sets_speed = params.declare<bool>("gait_sets_speed", config.lio.gait_sets_speed);
  config.lio.gait_hold_without_cadence =
      params.declare<bool>("gait_hold_without_cadence", config.lio.gait_hold_without_cadence);
  config.lio.gait_standing_accel_std =
      params.declare<double>("gait_standing_accel_std", config.lio.gait_standing_accel_std);
  config.lio.gait_correction_weak_ratio =
      params.declare<double>("gait_correction_weak_ratio", config.lio.gait_correction_weak_ratio);
  config.lio.icp_levenberg_marquardt =
      params.declare<bool>("icp_levenberg_marquardt", config.lio.icp_levenberg_marquardt);
  config.lio.icp_lm_initial_lambda_factor =
      params.declare<double>("icp_lm_initial_lambda_factor", config.lio.icp_lm_initial_lambda_factor);
  config.lio.icp_lm_max_attempts =
      params.declare<int>("icp_lm_max_attempts", config.lio.icp_lm_max_attempts);
  config.lio.velocity_window_sec =
      params.declare<double>("velocity_window_sec", config.lio.velocity_window_sec);
  config.lio.inertial_window = params.declare<bool>("inertial_window", config.lio.inertial_window);
  config.lio.inertial_window_sec =
      params.declare<double>("inertial_window_sec", config.lio.inertial_window_sec);
  config.lio.inertial_window_min_intervals =
      params.declare<int>("inertial_window_min_intervals", config.lio.inertial_window_min_intervals);
  config.lio.inertial_window_rotation_weight = params.declare<double>(
      "inertial_window_rotation_weight", config.lio.inertial_window_rotation_weight);
  config.lio.inertial_window_velocity_weight = params.declare<double>(
      "inertial_window_velocity_weight", config.lio.inertial_window_velocity_weight);
  config.lio.inertial_window_position_weight = params.declare<double>(
      "inertial_window_position_weight", config.lio.inertial_window_position_weight);
  config.lio.inertial_window_accelerometer_bias_prior_weight = params.declare<double>(
      "inertial_window_accelerometer_bias_prior_weight", config.lio.inertial_window_accelerometer_bias_prior_weight);
  config.lio.inertial_window_gyroscope_bias_prior_weight = params.declare<double>(
      "inertial_window_gyroscope_bias_prior_weight", config.lio.inertial_window_gyroscope_bias_prior_weight);
  config.lio.skip_registration_after_gap_sec = params.declare<double>(
      "skip_registration_after_gap_sec", config.lio.skip_registration_after_gap_sec);
  config.lio.icp_keypoint_voxel_multiplier =
      params.declare<double>("icp_keypoint_voxel_multiplier", config.lio.icp_keypoint_voxel_multiplier);
  config.lio.min_icp_keypoints = static_cast<size_t>(std::max<int64_t>(
      1, params.declare<int64_t>("min_icp_keypoints", static_cast<int64_t>(config.lio.min_icp_keypoints))));
  config.lio.keypoint_shortfall_voxel_multiplier = params.declare<double>(
      "keypoint_shortfall_voxel_multiplier", config.lio.keypoint_shortfall_voxel_multiplier);

  // ---- degeneracy-aware ICP solve (fork addition; default-off) ----
  config.lio.degeneracy_aware_solve =
      params.declare<bool>("degeneracy_aware_solve", config.lio.degeneracy_aware_solve);
  config.lio.degeneracy_well_conditioned_ratio = params.declare<double>(
      "degeneracy_well_conditioned_ratio", config.lio.degeneracy_well_conditioned_ratio);
  config.lio.degeneracy_multiplicity_relative_gap = params.declare<double>(
      "degeneracy_multiplicity_relative_gap", config.lio.degeneracy_multiplicity_relative_gap);
  config.lio.degeneracy_prior_weight =
      params.declare<double>("degeneracy_prior_weight", config.lio.degeneracy_prior_weight);
  config.lio.degeneracy_persistence_gate =
      params.declare<bool>("degeneracy_persistence_gate", config.lio.degeneracy_persistence_gate);
  const auto degeneracy_persistence_min_scans = params.declare<int>(
      "degeneracy_persistence_min_scans", static_cast<int>(config.lio.degeneracy_persistence_min_scans));
  config.lio.degeneracy_persistence_min_scans =
      static_cast<size_t>(degeneracy_persistence_min_scans > 0 ? degeneracy_persistence_min_scans : 1);
  config.lio.degeneracy_persistence_tracking_ratio = params.declare<double>(
      "degeneracy_persistence_tracking_ratio", config.lio.degeneracy_persistence_tracking_ratio);
  config.lio.degeneracy_persistence_min_absolute_cosine = params.declare<double>(
      "degeneracy_persistence_min_absolute_cosine", config.lio.degeneracy_persistence_min_absolute_cosine);
  config.lio.degeneracy_persistence_min_translation_fraction = params.declare<double>(
      "degeneracy_persistence_min_translation_fraction", config.lio.degeneracy_persistence_min_translation_fraction);
  config.lio.degeneracy_adaptive_iteration_budget = params.declare<bool>(
      "degeneracy_adaptive_iteration_budget", config.lio.degeneracy_adaptive_iteration_budget);
  const auto degeneracy_adaptive_max_iterations = params.declare<int>(
      "degeneracy_adaptive_max_iterations", static_cast<int>(config.lio.degeneracy_adaptive_max_iterations));
  config.lio.degeneracy_adaptive_max_iterations =
      static_cast<size_t>(degeneracy_adaptive_max_iterations > 0 ? degeneracy_adaptive_max_iterations : 1);
  config.lio.degeneracy_adaptive_iteration_ratio = params.declare<double>(
      "degeneracy_adaptive_iteration_ratio", config.lio.degeneracy_adaptive_iteration_ratio);
  const auto degeneracy_adaptive_hold_scans = params.declare<int>(
      "degeneracy_adaptive_hold_scans", static_cast<int>(config.lio.degeneracy_adaptive_hold_scans));
  config.lio.degeneracy_adaptive_hold_scans =
      static_cast<size_t>(degeneracy_adaptive_hold_scans > 0 ? degeneracy_adaptive_hold_scans : 1);
  config.lio.degeneracy_multiscan_observability_gate = params.declare<bool>(
      "degeneracy_multiscan_observability_gate", config.lio.degeneracy_multiscan_observability_gate);
  const auto degeneracy_observability_window_scans = params.declare<int>(
      "degeneracy_observability_window_scans", static_cast<int>(config.lio.degeneracy_observability_window_scans));
  config.lio.degeneracy_observability_window_scans =
      static_cast<size_t>(degeneracy_observability_window_scans > 0 ? degeneracy_observability_window_scans : 1);
  const auto degeneracy_observability_min_scans = params.declare<int>(
      "degeneracy_observability_min_scans", static_cast<int>(config.lio.degeneracy_observability_min_scans));
  config.lio.degeneracy_observability_min_scans =
      static_cast<size_t>(degeneracy_observability_min_scans > 0 ? degeneracy_observability_min_scans : 1);
  config.lio.degeneracy_observability_max_directional_ratio = params.declare<double>(
      "degeneracy_observability_max_directional_ratio", config.lio.degeneracy_observability_max_directional_ratio);

  // ---- selective visual fusion (fork addition; default-off) ----
  config.lio.visual_fusion.enabled =
      params.declare<bool>("selective_visual_fusion", config.lio.visual_fusion.enabled);
  const auto visual_min_tracks =
      params.declare<int>("visual_min_tracks", static_cast<int>(config.lio.visual_fusion.min_tracks));
  config.lio.visual_fusion.min_tracks = static_cast<std::size_t>(visual_min_tracks > 0 ? visual_min_tracks : 1);
  const auto visual_min_inliers =
      params.declare<int>("visual_min_inliers", static_cast<int>(config.lio.visual_fusion.min_inliers));
  config.lio.visual_fusion.min_inliers = static_cast<std::size_t>(visual_min_inliers > 0 ? visual_min_inliers : 1);
  config.lio.visual_fusion.min_inlier_ratio =
      params.declare<double>("visual_min_inlier_ratio", config.lio.visual_fusion.min_inlier_ratio);
  config.lio.visual_fusion.max_rotation_error_deg = params.declare<double>(
      "visual_max_rotation_error_deg", config.lio.visual_fusion.max_rotation_error_deg);
  config.lio.visual_fusion.min_translation_cosine = params.declare<double>(
      "visual_min_translation_cosine", config.lio.visual_fusion.min_translation_cosine);
  config.lio.visual_fusion.min_baseline_m =
      params.declare<double>("visual_min_baseline_m", config.lio.visual_fusion.min_baseline_m);
  config.lio.visual_fusion.max_baseline_m =
      params.declare<double>("visual_max_baseline_m", config.lio.visual_fusion.max_baseline_m);
  config.lio.visual_fusion.weak_information_ratio = params.declare<double>(
      "visual_weak_information_ratio", config.lio.visual_fusion.weak_information_ratio);
  config.lio.visual_fusion.relative_information_weight = params.declare<double>(
      "visual_relative_information_weight", config.lio.visual_fusion.relative_information_weight);
  config.lio.visual_fusion.min_visual_directional_information_ratio =
      params.declare<double>("visual_min_directional_information_ratio",
                                      config.lio.visual_fusion.min_visual_directional_information_ratio);
  const auto visual_max_weak_directions = params.declare<int>(
      "visual_max_weak_directions", static_cast<int>(config.lio.visual_fusion.max_weak_directions));
  config.lio.visual_fusion.max_weak_directions =
      static_cast<std::size_t>(visual_max_weak_directions > 0 ? visual_max_weak_directions : 0);
  config.lio.visual_fusion.max_translation_update_m = params.declare<double>(
      "visual_max_translation_update_m", config.lio.visual_fusion.max_translation_update_m);
  config.lio.visual_fusion.max_rotation_update_rad = params.declare<double>(
      "visual_max_rotation_update_rad", config.lio.visual_fusion.max_rotation_update_rad);
  config.lio.visual_prior_max_time_offset_sec = params.declare<double>(
      "visual_prior_max_time_offset_sec", config.lio.visual_prior_max_time_offset_sec);

  // ---- radar ego-velocity fusion (fork addition; default-off) ----
  config.lio.radar_velocity_fusion =
      params.declare<bool>("radar_velocity_fusion", config.lio.radar_velocity_fusion);
  config.lio.radar_prior_max_time_offset_sec =
      params.declare<double>("radar_prior_max_time_offset_sec", config.lio.radar_prior_max_time_offset_sec);
  config.lio.radar_min_speed = params.declare<double>("radar_min_speed", config.lio.radar_min_speed);
  config.lio.radar_disagreement_gate =
      params.declare<bool>("radar_disagreement_gate", config.lio.radar_disagreement_gate);
  config.lio.radar_disagreement_min_mps =
      params.declare<double>("radar_disagreement_min_mps", config.lio.radar_disagreement_min_mps);
  const auto radar_disagreement_min_scans = params.declare<int>(
      "radar_disagreement_min_scans", static_cast<int>(config.lio.radar_disagreement_min_scans));
  config.lio.radar_disagreement_min_scans =
      static_cast<size_t>(radar_disagreement_min_scans > 0 ? radar_disagreement_min_scans : 1);
  config.lio.radar_disagreement_weight =
      params.declare<double>("radar_disagreement_weight", config.lio.radar_disagreement_weight);
  config.lio.radar_velocity_continuous_fusion = params.declare<bool>(
      "radar_velocity_continuous_fusion", config.lio.radar_velocity_continuous_fusion);
  config.lio.radar_fusion_icp_information_scale = params.declare<double>(
      "radar_fusion_icp_information_scale", config.lio.radar_fusion_icp_information_scale);

  // ---- sliding-window absolute gravity alignment (fork addition; default-off) ----
  config.lio.max_tilt_correction_deg =
      params.declare<double>("max_tilt_correction_deg", config.lio.max_tilt_correction_deg);
  config.lio.max_yaw_correction_deg =
      params.declare<double>("max_yaw_correction_deg", config.lio.max_yaw_correction_deg);
  config.lio.max_tilt_correction_median_factor = params.declare<double>(
      "max_tilt_correction_median_factor", config.lio.max_tilt_correction_median_factor);
  config.lio.max_tilt_correction_median_min_scans = static_cast<int>(params.declare<int64_t>(
      "max_tilt_correction_median_min_scans", config.lio.max_tilt_correction_median_min_scans));
  config.lio.gyro_bias_time_constant_sec =
      params.declare<double>("gyro_bias_time_constant_sec", config.lio.gyro_bias_time_constant_sec);
  config.lio.gyro_bias_max_correction_deg =
      params.declare<double>("gyro_bias_max_correction_deg", config.lio.gyro_bias_max_correction_deg);
  config.lio.gyro_bias_standstill_max_speed = params.declare<double>(
      "gyro_bias_standstill_max_speed", config.lio.gyro_bias_standstill_max_speed);
  config.lio.gyro_bias_standstill_max_rate_rad_s = params.declare<double>(
      "gyro_bias_standstill_max_rate_rad_s", config.lio.gyro_bias_standstill_max_rate_rad_s);
  config.lio.gyro_bias_start_sec =
      params.declare<double>("gyro_bias_start_sec", config.lio.gyro_bias_start_sec);
  config.lio.vehicle_stall_guard =
      params.declare<bool>("vehicle_stall_guard", config.lio.vehicle_stall_guard);
  config.lio.stall_guard.window_sec =
      params.declare<double>("stall_guard.window_sec", config.lio.stall_guard.window_sec);
  config.lio.stall_guard.min_drop_mps =
      params.declare<double>("stall_guard.min_drop_mps", config.lio.stall_guard.min_drop_mps);
  config.lio.stall_guard.max_speed_ratio =
      params.declare<double>("stall_guard.max_speed_ratio", config.lio.stall_guard.max_speed_ratio);
  config.lio.stall_guard.min_speed_mps =
      params.declare<double>("stall_guard.min_speed_mps", config.lio.stall_guard.min_speed_mps);
  config.lio.stall_guard.exit_agreement_mps = params.declare<double>(
      "stall_guard.exit_agreement_mps", config.lio.stall_guard.exit_agreement_mps);
  config.lio.stall_guard.exit_scans = static_cast<int>(
      params.declare<int>("stall_guard.exit_scans", config.lio.stall_guard.exit_scans));
  config.lio.stall_guard.max_hold_sec =
      params.declare<double>("stall_guard.max_hold_sec", config.lio.stall_guard.max_hold_sec);
  config.lio.initialization_max_gyro_rad_s =
      params.declare<double>("initialization_max_gyro_rad_s", config.lio.initialization_max_gyro_rad_s);
  config.lio.moving_start_gravity_window_sec = params.declare<double>(
      "moving_start_gravity_window_sec", config.lio.moving_start_gravity_window_sec);
  config.lio.moving_start_gravity_output_only = params.declare<bool>(
      "moving_start_gravity_output_only", config.lio.moving_start_gravity_output_only);
  config.lio.output_filter_position_sigma =
      params.declare<double>("output_filter_position_sigma", config.lio.output_filter_position_sigma);
  config.lio.output_filter_acceleration_noise = params.declare<double>(
      "output_filter_acceleration_noise", config.lio.output_filter_acceleration_noise);
  config.lio.output_filter_bias_noise =
      params.declare<double>("output_filter_bias_noise", config.lio.output_filter_bias_noise);
  config.lio.gravity_window_alignment =
      params.declare<bool>("gravity_window_alignment", config.lio.gravity_window_alignment);
  config.lio.gravity_window_sec = params.declare<double>("gravity_window_sec", config.lio.gravity_window_sec);
  config.lio.gravity_alignment_gain =
      params.declare<double>("gravity_alignment_gain", config.lio.gravity_alignment_gain);
  config.lio.gravity_alignment_max_correction_rad = params.declare<double>(
      "gravity_alignment_max_correction_rad", config.lio.gravity_alignment_max_correction_rad);
  config.lio.gravity_alignment_max_magnitude_deviation = params.declare<double>(
      "gravity_alignment_max_magnitude_deviation", config.lio.gravity_alignment_max_magnitude_deviation);
  config.lio.gravity_alignment_max_plausible_tilt_rad = params.declare<double>(
      "gravity_alignment_max_plausible_tilt_rad", config.lio.gravity_alignment_max_plausible_tilt_rad);
  config.lio.gravity_alignment_max_yaw_rate_rad_s = params.declare<double>(
      "gravity_alignment_max_yaw_rate_rad_s", config.lio.gravity_alignment_max_yaw_rate_rad_s);
  config.lio.gravity_alignment_relevel_map =
      params.declare<bool>("gravity_alignment_relevel_map", config.lio.gravity_alignment_relevel_map);
  config.lio.gravity_alignment_relevel_min_tilt_rad = params.declare<double>(
      "gravity_alignment_relevel_min_tilt_rad", config.lio.gravity_alignment_relevel_min_tilt_rad);

  // ---- localizability-aware ICP weighting (fork addition; default-off) ----
  config.lio.localizability_weighting =
      params.declare<bool>("localizability_weighting", config.lio.localizability_weighting);
  config.lio.localizability_boost =
      params.declare<double>("localizability_boost", config.lio.localizability_boost);
  config.lio.localizability_min_step_m =
      params.declare<double>("localizability_min_step_m", config.lio.localizability_min_step_m);
  config.lio.localizability_normal_only =
      params.declare<bool>("localizability_normal_only", config.lio.localizability_normal_only);

  // ---- accelerometer-consistency velocity gate (fork addition; default-off) ----
  config.lio.kinematic_velocity_gate =
      params.declare<bool>("kinematic_velocity_gate", config.lio.kinematic_velocity_gate);
  config.lio.kinematic_gate_accel_margin =
      params.declare<double>("kinematic_gate_accel_margin", config.lio.kinematic_gate_accel_margin);
  config.lio.kinematic_gate_min_speed =
      params.declare<double>("kinematic_gate_min_speed", config.lio.kinematic_gate_min_speed);
  config.lio.kinematic_gate_weight =
      params.declare<double>("kinematic_gate_weight", config.lio.kinematic_gate_weight);
  config.lio.kinematic_gate_skip_map_update =
      params.declare<bool>("kinematic_gate_skip_map_update", config.lio.kinematic_gate_skip_map_update);

  // ---- anchor-decayed inertial velocity blend (fork addition; default-off) ----
  config.lio.kinematic_velocity_blend =
      params.declare<bool>("kinematic_velocity_blend", config.lio.kinematic_velocity_blend);
  config.lio.kinematic_blend_icp_information_scale = params.declare<double>(
      "kinematic_blend_icp_information_scale", config.lio.kinematic_blend_icp_information_scale);
  config.lio.kinematic_blend_propagation_information_scale = params.declare<double>(
      "kinematic_blend_propagation_information_scale", config.lio.kinematic_blend_propagation_information_scale);
  config.lio.kinematic_blend_decay_time_sec = params.declare<double>(
      "kinematic_blend_decay_time_sec", config.lio.kinematic_blend_decay_time_sec);
  config.lio.kinematic_blend_anchor_agreement_mps = params.declare<double>(
      "kinematic_blend_anchor_agreement_mps", config.lio.kinematic_blend_anchor_agreement_mps);
  config.lio.kinematic_blend_max_icp_innovation_mps = params.declare<double>(
      "kinematic_blend_max_icp_innovation_mps", config.lio.kinematic_blend_max_icp_innovation_mps);
  config.lio.kinematic_blend_max_propagated_speed_mps = params.declare<double>(
      "kinematic_blend_max_propagated_speed_mps", config.lio.kinematic_blend_max_propagated_speed_mps);
  config.lio.kinematic_blend_max_activation_speed_mps = params.declare<double>(
      "kinematic_blend_max_activation_speed_mps", config.lio.kinematic_blend_max_activation_speed_mps);
  const auto kinematic_blend_speed_gate_min_scans = params.declare<int>(
      "kinematic_blend_speed_gate_min_scans",
      static_cast<int>(config.lio.kinematic_blend_speed_gate_min_scans));
  config.lio.kinematic_blend_speed_gate_min_scans = static_cast<std::size_t>(
      kinematic_blend_speed_gate_min_scans > 0 ? kinematic_blend_speed_gate_min_scans : 1);
  config.lio.kinematic_blend_speed_reenable_delay_sec = params.declare<double>(
      "kinematic_blend_speed_reenable_delay_sec",
      config.lio.kinematic_blend_speed_reenable_delay_sec);
  config.lio.kinematic_blend_min_speed =
      params.declare<double>("kinematic_blend_min_speed", config.lio.kinematic_blend_min_speed);
  config.lio.kinematic_blend_min_accel_magnitude_variance = params.declare<double>(
      "kinematic_blend_min_accel_magnitude_variance",
      config.lio.kinematic_blend_min_accel_magnitude_variance);
  const auto kinematic_blend_inactivity_gate_min_scans =
      params.declare<int>(
          "kinematic_blend_inactivity_gate_min_scans",
          static_cast<int>(
              config.lio.kinematic_blend_inactivity_gate_min_scans));
  config.lio.kinematic_blend_inactivity_gate_min_scans =
      static_cast<std::size_t>(
          kinematic_blend_inactivity_gate_min_scans > 0
              ? kinematic_blend_inactivity_gate_min_scans
              : 1);
  config.lio.kinematic_blend_max_yaw_rate_rad_s = params.declare<double>(
      "kinematic_blend_max_yaw_rate_rad_s", config.lio.kinematic_blend_max_yaw_rate_rad_s);
  const auto kinematic_blend_yaw_gate_min_scans = params.declare<int>(
      "kinematic_blend_yaw_gate_min_scans",
      static_cast<int>(config.lio.kinematic_blend_yaw_gate_min_scans));
  config.lio.kinematic_blend_yaw_gate_min_scans = static_cast<std::size_t>(
      kinematic_blend_yaw_gate_min_scans > 0 ? kinematic_blend_yaw_gate_min_scans : 1);
  config.lio.kinematic_blend_range_scene_gate = params.declare<bool>(
      "kinematic_blend_range_scene_gate", config.lio.kinematic_blend_range_scene_gate);
  config.lio.kinematic_blend_scene_near_range_m = params.declare<double>(
      "kinematic_blend_scene_near_range_m", config.lio.kinematic_blend_scene_near_range_m);
  config.lio.kinematic_blend_scene_max_near_fraction = params.declare<double>(
      "kinematic_blend_scene_max_near_fraction", config.lio.kinematic_blend_scene_max_near_fraction);
  config.lio.kinematic_blend_scene_far_range_m = params.declare<double>(
      "kinematic_blend_scene_far_range_m", config.lio.kinematic_blend_scene_far_range_m);
  config.lio.kinematic_blend_scene_min_far_fraction = params.declare<double>(
      "kinematic_blend_scene_min_far_fraction", config.lio.kinematic_blend_scene_min_far_fraction);
  const auto kinematic_blend_scene_min_valid_points = params.declare<int>(
      "kinematic_blend_scene_min_valid_points",
      static_cast<int>(config.lio.kinematic_blend_scene_min_valid_points));
  config.lio.kinematic_blend_scene_min_valid_points = static_cast<std::size_t>(
      kinematic_blend_scene_min_valid_points > 0 ? kinematic_blend_scene_min_valid_points : 1);
  config.lio.kinematic_blend_scene_reenable_delay_sec = params.declare<double>(
      "kinematic_blend_scene_reenable_delay_sec",
      config.lio.kinematic_blend_scene_reenable_delay_sec);
  config.lio.kinematic_blend_map_update_max_propagation_weight = params.declare<double>(
      "kinematic_blend_map_update_max_propagation_weight",
      config.lio.kinematic_blend_map_update_max_propagation_weight);
  config.lio.kinematic_blend_map_update_min_fraction = params.declare<double>(
      "kinematic_blend_map_update_min_fraction", config.lio.kinematic_blend_map_update_min_fraction);

  // ---- intensity/reflectivity texture constraint (fork addition; default-off) ----
  config.lio.intensity_constraint =
      params.declare<bool>("intensity_constraint", config.lio.intensity_constraint);
  config.lio.intensity_bin_size_m =
      params.declare<double>("intensity_bin_size_m", config.lio.intensity_bin_size_m);
  config.lio.intensity_profile_half_length_m = params.declare<double>(
      "intensity_profile_half_length_m", config.lio.intensity_profile_half_length_m);
  config.lio.intensity_max_shift_m =
      params.declare<double>("intensity_max_shift_m", config.lio.intensity_max_shift_m);
  config.lio.intensity_min_correlation =
      params.declare<double>("intensity_min_correlation", config.lio.intensity_min_correlation);
  config.lio.intensity_min_peak_margin =
      params.declare<double>("intensity_min_peak_margin", config.lio.intensity_min_peak_margin);
  const auto intensity_peak_exclusion_radius_bins = params.declare<int>(
      "intensity_peak_exclusion_radius_bins",
      static_cast<int>(config.lio.intensity_peak_exclusion_radius_bins));
  config.lio.intensity_peak_exclusion_radius_bins = static_cast<size_t>(
      intensity_peak_exclusion_radius_bins > 0
          ? intensity_peak_exclusion_radius_bins
          : 0);
  const auto intensity_min_filled_bins = params.declare<int>(
      "intensity_min_filled_bins", static_cast<int>(config.lio.intensity_min_filled_bins));
  config.lio.intensity_min_filled_bins =
      static_cast<size_t>(intensity_min_filled_bins > 0 ? intensity_min_filled_bins : 1);
  config.lio.intensity_oriented_grid = params.declare<bool>(
      "intensity_oriented_grid", config.lio.intensity_oriented_grid);
  config.lio.intensity_grid_half_width_m = params.declare<double>(
      "intensity_grid_half_width_m", config.lio.intensity_grid_half_width_m);
  config.lio.intensity_grid_max_lateral_shift_m =
      params.declare<double>(
          "intensity_grid_max_lateral_shift_m",
          config.lio.intensity_grid_max_lateral_shift_m);
  config.lio.intensity_grid_height_weight = params.declare<double>(
      "intensity_grid_height_weight",
      config.lio.intensity_grid_height_weight);
  config.lio.intensity_disagreement_gate =
      params.declare<bool>("intensity_disagreement_gate", config.lio.intensity_disagreement_gate);
  config.lio.intensity_disagreement_min_mps =
      params.declare<double>("intensity_disagreement_min_mps", config.lio.intensity_disagreement_min_mps);
  const auto intensity_disagreement_min_scans = params.declare<int>(
      "intensity_disagreement_min_scans", static_cast<int>(config.lio.intensity_disagreement_min_scans));
  config.lio.intensity_disagreement_min_scans =
      static_cast<size_t>(intensity_disagreement_min_scans > 0 ? intensity_disagreement_min_scans : 1);
  config.lio.intensity_disagreement_weight = params.declare<double>(
      "intensity_disagreement_weight", config.lio.intensity_disagreement_weight);

  // ---- photometric registration on LiDAR intensity images (fork addition, default off) ----
  config.lio.photometric = params.declare<bool>("photometric", config.lio.photometric);
  config.lio.photometric_scale = params.declare<double>("photometric_scale", config.lio.photometric_scale);
  config.lio.photometric_free_weak_axis =
      params.declare<bool>("photometric_free_weak_axis", config.lio.photometric_free_weak_axis);
  config.lio.photometric_max_scan_interval_sec = params.declare<double>(
      "photometric_max_scan_interval_sec", config.lio.photometric_max_scan_interval_sec);
  config.lio.bump_image_registration =
      params.declare<bool>("bump_image_registration", config.lio.bump_image_registration);
  config.lio.bump_image_map.voxel_size =
      params.declare<double>("bump_image_map.voxel_size", config.lio.bump_image_map.voxel_size);
  config.lio.bump_image_map.pixel_size =
      params.declare<double>("bump_image_map.pixel_size", config.lio.bump_image_map.pixel_size);
  config.lio.bump_image_map.weighted =
      params.declare<bool>("bump_image_map.weighted", config.lio.bump_image_map.weighted);
  config.lio.bump_image_map.smooth =
      params.declare<bool>("bump_image_map.smooth", config.lio.bump_image_map.smooth);
  config.lio.bump_image_map.normal_tolerance_deg = params.declare<double>(
      "bump_image_map.normal_tolerance_deg", config.lio.bump_image_map.normal_tolerance_deg);
  config.lio.bump_image_map.max_voxels = static_cast<std::size_t>(params.declare<int>(
      "bump_image_map.max_voxels", static_cast<int>(config.lio.bump_image_map.max_voxels)));
  config.lio.bump_image_source_voxel_size = params.declare<double>(
      "bump_image_source_voxel_size", config.lio.bump_image_source_voxel_size);
  config.lio.bump_image_informed_voxels =
      params.declare<int>("bump_image_informed_voxels", config.lio.bump_image_informed_voxels);
  config.lio.bump_image_huber_delta =
      params.declare<double>("bump_image_huber_delta", config.lio.bump_image_huber_delta);
  config.lio.bump_image_point_to_point_weight = params.declare<double>(
      "bump_image_point_to_point_weight", config.lio.bump_image_point_to_point_weight);
  config.lio.bump_image_point_to_point_start_distance = params.declare<double>(
      "bump_image_point_to_point_start_distance", config.lio.bump_image_point_to_point_start_distance);
  config.lio.bump_image_max_rotation_correction_deg = params.declare<double>(
      "bump_image_max_rotation_correction_deg", config.lio.bump_image_max_rotation_correction_deg);
  config.lio.bump_image_rotation_fallback_window = params.declare<int>(
      "bump_image_rotation_fallback_window", config.lio.bump_image_rotation_fallback_window);
  config.lio.bump_image_rotation_fallback_min_count = params.declare<int>(
      "bump_image_rotation_fallback_min_count", config.lio.bump_image_rotation_fallback_min_count);
  config.lio.bump_image_rotation_hard_limit_deg = params.declare<double>(
      "bump_image_rotation_hard_limit_deg", config.lio.bump_image_rotation_hard_limit_deg);
  config.lio.photometric_weak_direction_min_contribution = params.declare<double>(
      "photometric_weak_direction_min_contribution", config.lio.photometric_weak_direction_min_contribution);
  config.lio.photometric_normal_row_step =
      params.declare<int>("photometric_normal_row_step", config.lio.photometric_normal_row_step);
  config.lio.photometric_normal_col_step =
      params.declare<int>("photometric_normal_col_step", config.lio.photometric_normal_col_step);
  config.photometric_channel = params.declare<std::string>("photometric_channel", config.photometric_channel);
  {
    // Beam geometry from the sensor metadata (Ouster: beam_altitude_angles, pixel_shift_by_row,
    // lidar_origin_to_beam_origin_mm, columns_per_frame).
    core::LidarImageModel& model = config.lio.photometric_model;
    const auto altitudes_deg =
        params.declare<std::vector<double>>("photometric_model.altitudes_deg", std::vector<double>{});
    model.altitudes_rad.clear();
    for (const double degrees : altitudes_deg) {
      model.altitudes_rad.push_back(degrees * M_PI / 180.0);
    }
    const auto pixel_shift =
        params.declare<std::vector<int64_t>>("photometric_model.pixel_shift_by_row", std::vector<int64_t>{});
    model.pixel_shift_by_row.assign(pixel_shift.begin(), pixel_shift.end());
    model.rows = static_cast<int>(model.altitudes_rad.size());
    model.cols = static_cast<int>(params.declare<int>("photometric_model.columns", 1024));
    model.beam_offset_m = params.declare<double>("photometric_model.beam_offset_mm", 0.0) * 1e-3;
    model.u_shift = static_cast<int>(params.declare<int>("photometric_model.u_shift", 0));
    model.cloud_to_lidar_z_m =
        params.declare<double>("photometric_model.cloud_to_lidar_z_m", model.cloud_to_lidar_z_m);
    if (config.lio.photometric && !model.valid()) {
      params.warn("photometric is enabled but photometric_model is incomplete (altitudes_deg, pixel_shift_by_row, "
                  "columns); photometric terms will not be used.");
    }

    core::IntensityImageConfig& image = config.lio.photometric_image;
    image.intensity_scale = params.declare<double>("photometric_image.intensity_scale", image.intensity_scale);
    image.line_removal = params.declare<bool>("photometric_image.line_removal", image.line_removal);
    image.line_highpass =
        params.declare<std::vector<double>>("photometric_image.line_highpass", std::vector<double>{});
    image.line_lowpass =
        params.declare<std::vector<double>>("photometric_image.line_lowpass", std::vector<double>{});
    image.brightness_filter =
        params.declare<bool>("photometric_image.brightness_filter", image.brightness_filter);
    image.brightness_window_cols =
        params.declare<int>("photometric_image.brightness_window_cols", image.brightness_window_cols);
    image.brightness_window_rows =
        params.declare<int>("photometric_image.brightness_window_rows", image.brightness_window_rows);
    image.blur = params.declare<bool>("photometric_image.blur", image.blur);
    image.min_range_m = params.declare<double>("photometric_image.min_range_m", image.min_range_m);
    image.max_range_m = params.declare<double>("photometric_image.max_range_m", image.max_range_m);
    image.erosion_margin = params.declare<int>("photometric_image.erosion_margin", image.erosion_margin);
    const auto masks =
        params.declare<std::vector<int64_t>>("photometric_image.masks", std::vector<int64_t>{});
    image.masks.clear();
    for (std::size_t i = 0; i + 3 < masks.size(); i += 4) {
      image.masks.push_back({static_cast<int>(masks[i]), static_cast<int>(masks[i + 1]),
                             static_cast<int>(masks[i + 2]), static_cast<int>(masks[i + 3])});
    }

    core::PhotometricFeatureConfig& features = config.lio.photometric_features;
    features.patch_size = params.declare<int>("photometric_features.patch_size", features.patch_size);
    features.num_features = params.declare<int>("photometric_features.num_features", features.num_features);
    features.max_lifetime = params.declare<int>("photometric_features.max_lifetime", features.max_lifetime);
    features.margin = params.declare<int>("photometric_features.margin", features.margin);
    features.suppression_radius =
        params.declare<int>("photometric_features.suppression_radius", features.suppression_radius);
    features.grad_min = params.declare<double>("photometric_features.grad_min", features.grad_min);
    features.ncc_threshold =
        params.declare<double>("photometric_features.ncc_threshold", features.ncc_threshold);
    features.range_threshold_m =
        params.declare<double>("photometric_features.range_threshold_m", features.range_threshold_m);
    features.min_range_m = image.min_range_m;
    features.max_range_m = image.max_range_m;
    image.patch_size = features.patch_size;
  }

  // ---- scan-gap re-anchor + kidnap relocalization (fork addition; default preserves legacy
  // behavior for max_scan_delta_sec, kidnap features themselves default-off) ----
  config.lio.max_scan_delta_sec =
      params.declare<double>("max_scan_delta_sec", config.lio.max_scan_delta_sec);
  config.lio.enable_kidnap_relocalization =
      params.declare<bool>("enable_kidnap_relocalization", config.lio.enable_kidnap_relocalization);
  config.lio.reset_on_registration_failure =
      params.declare<bool>("reset_on_registration_failure", config.lio.reset_on_registration_failure);
  config.lio.recovery_min_failures =
      params.declare<int>("recovery_min_failures", config.lio.recovery_min_failures);
  config.lio.relocalize_after_scan_gap =
      params.declare<bool>("relocalize_after_scan_gap", config.lio.relocalize_after_scan_gap);
  config.lio.relocalization_min_correspondences = params.declare<int>(
      "relocalization_min_correspondences", config.lio.relocalization_min_correspondences);
  config.lio.relocalization_min_inlier_ratio =
      params.declare<double>("relocalization_min_inlier_ratio", config.lio.relocalization_min_inlier_ratio);
  config.lio.relocalization_max_mean_error =
      params.declare<double>("relocalization_max_mean_error", config.lio.relocalization_max_mean_error);
  config.lio.relocalization_max_correspondence_distance = params.declare<double>(
      "relocalization_max_correspondence_distance", config.lio.relocalization_max_correspondence_distance);
  config.lio.relocalization_yaw_samples =
      params.declare<int>("relocalization_yaw_samples", config.lio.relocalization_yaw_samples);
  config.lio.relocalization_pose_stride =
      params.declare<int>("relocalization_pose_stride", config.lio.relocalization_pose_stride);
  config.lio.relocalization_min_pose_separation = params.declare<int>(
      "relocalization_min_pose_separation", config.lio.relocalization_min_pose_separation);
  config.lio.relocalization_max_iterations =
      params.declare<int>("relocalization_max_iterations", config.lio.relocalization_max_iterations);

  config.direct_visual_frontend = params.declare<bool>("direct_visual_frontend", config.direct_visual_frontend);
  config.direct_visual_require_previous_weak_direction = params.declare<bool>(
      "direct_visual_require_previous_weak_direction", config.direct_visual_require_previous_weak_direction);

  config.radar_topic = params.declare<std::string>("radar_topic", config.radar_topic);
  if (!config.radar_topic.empty()) {
    auto& radar = config.radar_ego_velocity_config;
    radar.doppler_sign = params.declare<double>("radar_doppler_sign", radar.doppler_sign);
    radar.ransac_inlier_threshold = params.declare<double>("radar_ransac_inlier_threshold", radar.ransac_inlier_threshold);
    const auto radar_ransac_iterations =
        params.declare<int>("radar_ransac_iterations", static_cast<int>(radar.ransac_iterations));
    radar.ransac_iterations = static_cast<std::size_t>(radar_ransac_iterations > 0 ? radar_ransac_iterations : 1);
    const auto radar_min_inliers = params.declare<int>("radar_min_inliers", static_cast<int>(radar.min_inliers));
    radar.min_inliers = static_cast<std::size_t>(radar_min_inliers > 0 ? radar_min_inliers : 1);
    config.radar_velocity_scale = params.declare<double>("radar_velocity_scale", config.radar_velocity_scale);
    // Per-axis radar velocity sigmas (radar sensor frame: forward/lateral/vertical), used only by
    // lio.radar_velocity_continuous_fusion to build each prior's info_base.
    config.radar_sigma_forward_mps = params.declare<double>("radar_sigma_forward_mps", config.radar_sigma_forward_mps);
    config.radar_sigma_lateral_mps = params.declare<double>("radar_sigma_lateral_mps", config.radar_sigma_lateral_mps);
    config.radar_sigma_vertical_mps =
        params.declare<double>("radar_sigma_vertical_mps", config.radar_sigma_vertical_mps);
  }

  auto& timestamps = config.timestamp_proc_config;
  timestamps.multiplier_to_seconds =
      params.declare<double>("lidar_timestamps.multiplier_to_seconds", timestamps.multiplier_to_seconds);
  timestamps.offset_seconds = params.declare<double>("lidar_timestamps.offset_seconds", timestamps.offset_seconds);
  timestamps.force_absolute = params.declare<bool>("lidar_timestamps.force_absolute", timestamps.force_absolute);
  timestamps.force_relative = params.declare<bool>("lidar_timestamps.force_relative", timestamps.force_relative);

  config.extrinsic_imu2base = parse_extrinsic(params, "imu");
  config.extrinsic_lidar2base = parse_extrinsic(params, "lidar");
  if (config.lio.visual_fusion.enabled) {
    config.extrinsic_cam2base = parse_extrinsic(params, "cam");
  }
  if (config.lio.radar_velocity_fusion || config.lio.radar_velocity_continuous_fusion) {
    config.extrinsic_radar2base = parse_extrinsic(params, "radar");
    if (!config.extrinsic_radar2base) {
      params.warn("radar_velocity_fusion/radar_velocity_continuous_fusion is enabled but "
                  "extrinsic_radar2base_quat_xyzw_xyz was not set; assuming identity rotation between "
                  "radar and base.");
    }
  }

  config.dump_results = params.declare<bool>("dump_results", config.dump_results);
  config.results_dir = params.declare<std::string>("results_dir", config.results_dir);
  config.run_name = params.declare<std::string>("run_name", config.run_name);
  return config;
}

} // namespace rko_lio::driver
