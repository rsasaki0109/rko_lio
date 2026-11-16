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

#include "base_node.hpp"
#include "rko_lio/core/imu_acceleration_unit.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/voxel_down_sample.hpp"
#include "rko_lio/ros/utils/utils.hpp"
// other
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <stdexcept>

namespace {
using namespace std::literals;

// The odometry pose covariance (driver::pose_covariance_from_state) lives in
// rko_lio/driver/offline_driver.cpp, shared with the ROS-free driver.

Sophus::SE3d desired_base_pose_from_monocular_relative(const Sophus::SE3d& world_T_previous_base,
                                                       const Sophus::SE3d& world_T_predicted_current_base,
                                                       const Sophus::SE3d& base_T_camera,
                                                       const Eigen::Matrix3d& current_R_previous_camera,
                                                       const Eigen::Vector3d& current_t_previous_direction) {
  const Sophus::SE3d world_T_previous_camera = world_T_previous_base * base_T_camera;
  const Sophus::SE3d world_T_predicted_current_camera = world_T_predicted_current_base * base_T_camera;
  const Sophus::SE3d predicted_current_T_previous_camera =
      world_T_predicted_current_camera.inverse() * world_T_previous_camera;
  const double baseline = predicted_current_T_previous_camera.translation().norm();
  const Sophus::SE3d measured_current_T_previous_camera(Sophus::SO3d(current_R_previous_camera),
                                                        baseline * current_t_previous_direction.normalized());
  return world_T_previous_camera * measured_current_T_previous_camera.inverse() * base_T_camera.inverse();
}

} // namespace

namespace rko_lio::core {
// necessary for serializing the config, including the namespacing
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SelectiveVisualFusionConfig,
                                   enabled,
                                   min_tracks,
                                   min_inliers,
                                   min_inlier_ratio,
                                   max_rotation_error_deg,
                                   min_translation_cosine,
                                   min_baseline_m,
                                   max_baseline_m,
                                   weak_information_ratio,
                                   relative_information_weight,
                                   min_visual_directional_information_ratio,
                                   max_weak_directions,
                                   max_translation_update_m,
                                   max_rotation_update_rad)
// NOTE: LIO::Config's field count exceeds nlohmann's NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE macro
// limit (63 fields: NLOHMANN_JSON_PASTE tops out at PASTE64, which is `func` + 63 values --
// confirmed by build failure when this list hit 65 entries), so to_json/from_json are written
// out manually here in the exact same per-field style the macro itself expands to
// (`j["field"] = t.field;` / `j.at("field").get_to(t.field);`), just without its argument-count
// ceiling. Keep new fields appended at the end; there is no other constraint to preserve.
template <typename BasicJsonType,
         nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void to_json(BasicJsonType& nlohmann_json_j, const LIO::Config& nlohmann_json_t) {
  nlohmann_json_j["deskew"] = nlohmann_json_t.deskew;
  nlohmann_json_j["max_iterations"] = nlohmann_json_t.max_iterations;
  nlohmann_json_j["voxel_size"] = nlohmann_json_t.voxel_size;
  nlohmann_json_j["max_points_per_voxel"] = nlohmann_json_t.max_points_per_voxel;
  nlohmann_json_j["max_range"] = nlohmann_json_t.max_range;
  nlohmann_json_j["min_range"] = nlohmann_json_t.min_range;
  nlohmann_json_j["convergence_criterion"] = nlohmann_json_t.convergence_criterion;
  nlohmann_json_j["max_correspondence_distance"] = nlohmann_json_t.max_correspondence_distance;
  nlohmann_json_j["max_num_threads"] = nlohmann_json_t.max_num_threads;
  nlohmann_json_j["initialization_phase"] = nlohmann_json_t.initialization_phase;
  nlohmann_json_j["max_expected_jerk"] = nlohmann_json_t.max_expected_jerk;
  nlohmann_json_j["double_downsample"] = nlohmann_json_t.double_downsample;
  nlohmann_json_j["icp_keypoint_voxel_multiplier"] = nlohmann_json_t.icp_keypoint_voxel_multiplier;
  nlohmann_json_j["min_icp_keypoints"] = nlohmann_json_t.min_icp_keypoints;
  nlohmann_json_j["keypoint_shortfall_voxel_multiplier"] = nlohmann_json_t.keypoint_shortfall_voxel_multiplier;
  nlohmann_json_j["min_beta"] = nlohmann_json_t.min_beta;
  nlohmann_json_j["degeneracy_aware_solve"] = nlohmann_json_t.degeneracy_aware_solve;
  nlohmann_json_j["degeneracy_well_conditioned_ratio"] = nlohmann_json_t.degeneracy_well_conditioned_ratio;
  nlohmann_json_j["degeneracy_multiplicity_relative_gap"] = nlohmann_json_t.degeneracy_multiplicity_relative_gap;
  nlohmann_json_j["degeneracy_prior_weight"] = nlohmann_json_t.degeneracy_prior_weight;
  nlohmann_json_j["degeneracy_persistence_gate"] = nlohmann_json_t.degeneracy_persistence_gate;
  nlohmann_json_j["degeneracy_persistence_min_scans"] = nlohmann_json_t.degeneracy_persistence_min_scans;
  nlohmann_json_j["degeneracy_persistence_tracking_ratio"] = nlohmann_json_t.degeneracy_persistence_tracking_ratio;
  nlohmann_json_j["degeneracy_persistence_min_absolute_cosine"] =
      nlohmann_json_t.degeneracy_persistence_min_absolute_cosine;
  nlohmann_json_j["degeneracy_persistence_min_translation_fraction"] =
      nlohmann_json_t.degeneracy_persistence_min_translation_fraction;
  nlohmann_json_j["degeneracy_adaptive_iteration_budget"] = nlohmann_json_t.degeneracy_adaptive_iteration_budget;
  nlohmann_json_j["degeneracy_adaptive_max_iterations"] = nlohmann_json_t.degeneracy_adaptive_max_iterations;
  nlohmann_json_j["degeneracy_adaptive_iteration_ratio"] = nlohmann_json_t.degeneracy_adaptive_iteration_ratio;
  nlohmann_json_j["degeneracy_adaptive_hold_scans"] = nlohmann_json_t.degeneracy_adaptive_hold_scans;
  nlohmann_json_j["degeneracy_multiscan_observability_gate"] =
      nlohmann_json_t.degeneracy_multiscan_observability_gate;
  nlohmann_json_j["degeneracy_observability_window_scans"] = nlohmann_json_t.degeneracy_observability_window_scans;
  nlohmann_json_j["degeneracy_observability_min_scans"] = nlohmann_json_t.degeneracy_observability_min_scans;
  nlohmann_json_j["degeneracy_observability_max_directional_ratio"] =
      nlohmann_json_t.degeneracy_observability_max_directional_ratio;
  nlohmann_json_j["visual_fusion"] = nlohmann_json_t.visual_fusion;
  nlohmann_json_j["visual_prior_max_time_offset_sec"] = nlohmann_json_t.visual_prior_max_time_offset_sec;
  nlohmann_json_j["radar_velocity_fusion"] = nlohmann_json_t.radar_velocity_fusion;
  nlohmann_json_j["radar_prior_max_time_offset_sec"] = nlohmann_json_t.radar_prior_max_time_offset_sec;
  nlohmann_json_j["radar_min_speed"] = nlohmann_json_t.radar_min_speed;
  nlohmann_json_j["radar_disagreement_gate"] = nlohmann_json_t.radar_disagreement_gate;
  nlohmann_json_j["radar_disagreement_min_mps"] = nlohmann_json_t.radar_disagreement_min_mps;
  nlohmann_json_j["radar_disagreement_min_scans"] = nlohmann_json_t.radar_disagreement_min_scans;
  nlohmann_json_j["radar_disagreement_weight"] = nlohmann_json_t.radar_disagreement_weight;
  nlohmann_json_j["radar_velocity_continuous_fusion"] = nlohmann_json_t.radar_velocity_continuous_fusion;
  nlohmann_json_j["radar_fusion_icp_information_scale"] = nlohmann_json_t.radar_fusion_icp_information_scale;
  nlohmann_json_j["gravity_window_alignment"] = nlohmann_json_t.gravity_window_alignment;
  nlohmann_json_j["gravity_window_sec"] = nlohmann_json_t.gravity_window_sec;
  nlohmann_json_j["gravity_alignment_gain"] = nlohmann_json_t.gravity_alignment_gain;
  nlohmann_json_j["gravity_alignment_max_correction_rad"] = nlohmann_json_t.gravity_alignment_max_correction_rad;
  nlohmann_json_j["gravity_alignment_max_magnitude_deviation"] =
      nlohmann_json_t.gravity_alignment_max_magnitude_deviation;
  nlohmann_json_j["gravity_alignment_max_plausible_tilt_rad"] = nlohmann_json_t.gravity_alignment_max_plausible_tilt_rad;
  nlohmann_json_j["gravity_alignment_max_yaw_rate_rad_s"] = nlohmann_json_t.gravity_alignment_max_yaw_rate_rad_s;
  nlohmann_json_j["gravity_alignment_relevel_map"] = nlohmann_json_t.gravity_alignment_relevel_map;
  nlohmann_json_j["gravity_alignment_relevel_min_tilt_rad"] = nlohmann_json_t.gravity_alignment_relevel_min_tilt_rad;
  nlohmann_json_j["localizability_weighting"] = nlohmann_json_t.localizability_weighting;
  nlohmann_json_j["localizability_boost"] = nlohmann_json_t.localizability_boost;
  nlohmann_json_j["localizability_min_step_m"] = nlohmann_json_t.localizability_min_step_m;
  nlohmann_json_j["localizability_normal_only"] = nlohmann_json_t.localizability_normal_only;
  nlohmann_json_j["kinematic_velocity_gate"] = nlohmann_json_t.kinematic_velocity_gate;
  nlohmann_json_j["kinematic_gate_accel_margin"] = nlohmann_json_t.kinematic_gate_accel_margin;
  nlohmann_json_j["kinematic_gate_min_speed"] = nlohmann_json_t.kinematic_gate_min_speed;
  nlohmann_json_j["kinematic_gate_weight"] = nlohmann_json_t.kinematic_gate_weight;
  nlohmann_json_j["kinematic_gate_skip_map_update"] = nlohmann_json_t.kinematic_gate_skip_map_update;
  nlohmann_json_j["kinematic_velocity_blend"] = nlohmann_json_t.kinematic_velocity_blend;
  nlohmann_json_j["kinematic_blend_icp_information_scale"] =
      nlohmann_json_t.kinematic_blend_icp_information_scale;
  nlohmann_json_j["kinematic_blend_propagation_information_scale"] =
      nlohmann_json_t.kinematic_blend_propagation_information_scale;
  nlohmann_json_j["kinematic_blend_decay_time_sec"] = nlohmann_json_t.kinematic_blend_decay_time_sec;
  nlohmann_json_j["kinematic_blend_anchor_agreement_mps"] =
      nlohmann_json_t.kinematic_blend_anchor_agreement_mps;
  nlohmann_json_j["kinematic_blend_max_icp_innovation_mps"] =
      nlohmann_json_t.kinematic_blend_max_icp_innovation_mps;
  nlohmann_json_j["kinematic_blend_max_propagated_speed_mps"] =
      nlohmann_json_t.kinematic_blend_max_propagated_speed_mps;
  nlohmann_json_j["kinematic_blend_max_activation_speed_mps"] =
      nlohmann_json_t.kinematic_blend_max_activation_speed_mps;
  nlohmann_json_j["kinematic_blend_speed_gate_min_scans"] =
      nlohmann_json_t.kinematic_blend_speed_gate_min_scans;
  nlohmann_json_j["kinematic_blend_speed_reenable_delay_sec"] =
      nlohmann_json_t.kinematic_blend_speed_reenable_delay_sec;
  nlohmann_json_j["kinematic_blend_min_speed"] = nlohmann_json_t.kinematic_blend_min_speed;
  nlohmann_json_j["kinematic_blend_min_accel_magnitude_variance"] =
      nlohmann_json_t.kinematic_blend_min_accel_magnitude_variance;
  nlohmann_json_j["kinematic_blend_inactivity_gate_min_scans"] =
      nlohmann_json_t.kinematic_blend_inactivity_gate_min_scans;
  nlohmann_json_j["kinematic_blend_max_yaw_rate_rad_s"] =
      nlohmann_json_t.kinematic_blend_max_yaw_rate_rad_s;
  nlohmann_json_j["kinematic_blend_yaw_gate_min_scans"] =
      nlohmann_json_t.kinematic_blend_yaw_gate_min_scans;
  nlohmann_json_j["kinematic_blend_range_scene_gate"] =
      nlohmann_json_t.kinematic_blend_range_scene_gate;
  nlohmann_json_j["kinematic_blend_scene_near_range_m"] =
      nlohmann_json_t.kinematic_blend_scene_near_range_m;
  nlohmann_json_j["kinematic_blend_scene_max_near_fraction"] =
      nlohmann_json_t.kinematic_blend_scene_max_near_fraction;
  nlohmann_json_j["kinematic_blend_scene_far_range_m"] =
      nlohmann_json_t.kinematic_blend_scene_far_range_m;
  nlohmann_json_j["kinematic_blend_scene_min_far_fraction"] =
      nlohmann_json_t.kinematic_blend_scene_min_far_fraction;
  nlohmann_json_j["kinematic_blend_scene_min_valid_points"] =
      nlohmann_json_t.kinematic_blend_scene_min_valid_points;
  nlohmann_json_j["kinematic_blend_scene_reenable_delay_sec"] =
      nlohmann_json_t.kinematic_blend_scene_reenable_delay_sec;
  nlohmann_json_j["kinematic_blend_map_update_max_propagation_weight"] =
      nlohmann_json_t.kinematic_blend_map_update_max_propagation_weight;
  nlohmann_json_j["kinematic_blend_map_update_min_fraction"] =
      nlohmann_json_t.kinematic_blend_map_update_min_fraction;
  nlohmann_json_j["intensity_constraint"] = nlohmann_json_t.intensity_constraint;
  nlohmann_json_j["intensity_bin_size_m"] = nlohmann_json_t.intensity_bin_size_m;
  nlohmann_json_j["intensity_profile_half_length_m"] = nlohmann_json_t.intensity_profile_half_length_m;
  nlohmann_json_j["intensity_max_shift_m"] = nlohmann_json_t.intensity_max_shift_m;
  nlohmann_json_j["intensity_min_correlation"] = nlohmann_json_t.intensity_min_correlation;
  nlohmann_json_j["intensity_min_peak_margin"] = nlohmann_json_t.intensity_min_peak_margin;
  nlohmann_json_j["intensity_peak_exclusion_radius_bins"] =
      nlohmann_json_t.intensity_peak_exclusion_radius_bins;
  nlohmann_json_j["intensity_min_filled_bins"] = nlohmann_json_t.intensity_min_filled_bins;
  nlohmann_json_j["intensity_oriented_grid"] = nlohmann_json_t.intensity_oriented_grid;
  nlohmann_json_j["intensity_grid_half_width_m"] = nlohmann_json_t.intensity_grid_half_width_m;
  nlohmann_json_j["intensity_grid_max_lateral_shift_m"] =
      nlohmann_json_t.intensity_grid_max_lateral_shift_m;
  nlohmann_json_j["intensity_grid_height_weight"] =
      nlohmann_json_t.intensity_grid_height_weight;
  nlohmann_json_j["intensity_disagreement_gate"] = nlohmann_json_t.intensity_disagreement_gate;
  nlohmann_json_j["intensity_disagreement_min_mps"] = nlohmann_json_t.intensity_disagreement_min_mps;
  nlohmann_json_j["intensity_disagreement_min_scans"] = nlohmann_json_t.intensity_disagreement_min_scans;
  nlohmann_json_j["intensity_disagreement_weight"] = nlohmann_json_t.intensity_disagreement_weight;
  nlohmann_json_j["max_scan_delta_sec"] = nlohmann_json_t.max_scan_delta_sec;
  nlohmann_json_j["enable_kidnap_relocalization"] = nlohmann_json_t.enable_kidnap_relocalization;
  nlohmann_json_j["reset_on_registration_failure"] = nlohmann_json_t.reset_on_registration_failure;
  nlohmann_json_j["recovery_min_failures"] = nlohmann_json_t.recovery_min_failures;
  nlohmann_json_j["relocalize_after_scan_gap"] = nlohmann_json_t.relocalize_after_scan_gap;
  nlohmann_json_j["relocalization_min_correspondences"] = nlohmann_json_t.relocalization_min_correspondences;
  nlohmann_json_j["relocalization_min_inlier_ratio"] = nlohmann_json_t.relocalization_min_inlier_ratio;
  nlohmann_json_j["relocalization_max_mean_error"] = nlohmann_json_t.relocalization_max_mean_error;
  nlohmann_json_j["relocalization_max_correspondence_distance"] =
      nlohmann_json_t.relocalization_max_correspondence_distance;
  nlohmann_json_j["relocalization_yaw_samples"] = nlohmann_json_t.relocalization_yaw_samples;
  nlohmann_json_j["relocalization_pose_stride"] = nlohmann_json_t.relocalization_pose_stride;
  nlohmann_json_j["relocalization_min_pose_separation"] = nlohmann_json_t.relocalization_min_pose_separation;
  nlohmann_json_j["relocalization_max_iterations"] = nlohmann_json_t.relocalization_max_iterations;
  nlohmann_json_j["legacy_voxel_downsample"] = nlohmann_json_t.legacy_voxel_downsample;
  nlohmann_json_j["samplewise_gyro_deskew"] = nlohmann_json_t.samplewise_gyro_deskew;
  nlohmann_json_j["start_standstill_radius"] = nlohmann_json_t.start_standstill_radius;
  nlohmann_json_j["start_standstill_max_accel_std"] = nlohmann_json_t.start_standstill_max_accel_std;
  nlohmann_json_j["start_standstill_hold"] = nlohmann_json_t.start_standstill_hold;
  nlohmann_json_j["start_standstill_max_gyro_rad_s"] = nlohmann_json_t.start_standstill_max_gyro_rad_s;
  nlohmann_json_j["start_standstill_max_sec"] = nlohmann_json_t.start_standstill_max_sec;
  nlohmann_json_j["start_standstill_bias"] = nlohmann_json_t.start_standstill_bias;
  nlohmann_json_j["start_standstill_hold_radius"] = nlohmann_json_t.start_standstill_hold_radius;
  nlohmann_json_j["start_standstill_moving_start_only"] = nlohmann_json_t.start_standstill_moving_start_only;
  nlohmann_json_j["close_scene_max_median_range"] = nlohmann_json_t.close_scene_max_median_range;
  nlohmann_json_j["close_scene_max_correspondence_distance"] = nlohmann_json_t.close_scene_max_correspondence_distance;
  nlohmann_json_j["close_scene_localizability_boost"] = nlohmann_json_t.close_scene_localizability_boost;
  nlohmann_json_j["close_scene_localizability_normal_only"] = nlohmann_json_t.close_scene_localizability_normal_only;
  nlohmann_json_j["close_scene_min_inlier_fraction"] = nlohmann_json_t.close_scene_min_inlier_fraction;
}
template <typename BasicJsonType,
         nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void from_json(const BasicJsonType& nlohmann_json_j, LIO::Config& nlohmann_json_t) {
  nlohmann_json_j.at("deskew").get_to(nlohmann_json_t.deskew);
  nlohmann_json_j.at("max_iterations").get_to(nlohmann_json_t.max_iterations);
  nlohmann_json_j.at("voxel_size").get_to(nlohmann_json_t.voxel_size);
  nlohmann_json_j.at("max_points_per_voxel").get_to(nlohmann_json_t.max_points_per_voxel);
  nlohmann_json_j.at("max_range").get_to(nlohmann_json_t.max_range);
  nlohmann_json_j.at("min_range").get_to(nlohmann_json_t.min_range);
  nlohmann_json_j.at("convergence_criterion").get_to(nlohmann_json_t.convergence_criterion);
  nlohmann_json_j.at("max_correspondence_distance").get_to(nlohmann_json_t.max_correspondence_distance);
  nlohmann_json_j.at("max_num_threads").get_to(nlohmann_json_t.max_num_threads);
  nlohmann_json_j.at("initialization_phase").get_to(nlohmann_json_t.initialization_phase);
  nlohmann_json_j.at("max_expected_jerk").get_to(nlohmann_json_t.max_expected_jerk);
  nlohmann_json_j.at("double_downsample").get_to(nlohmann_json_t.double_downsample);
  nlohmann_json_j.at("icp_keypoint_voxel_multiplier").get_to(nlohmann_json_t.icp_keypoint_voxel_multiplier);
  nlohmann_json_j.at("min_icp_keypoints").get_to(nlohmann_json_t.min_icp_keypoints);
  nlohmann_json_t.keypoint_shortfall_voxel_multiplier = nlohmann_json_j.value(
      "keypoint_shortfall_voxel_multiplier", nlohmann_json_t.keypoint_shortfall_voxel_multiplier);
  nlohmann_json_j.at("min_beta").get_to(nlohmann_json_t.min_beta);
  nlohmann_json_j.at("degeneracy_aware_solve").get_to(nlohmann_json_t.degeneracy_aware_solve);
  nlohmann_json_j.at("degeneracy_well_conditioned_ratio").get_to(nlohmann_json_t.degeneracy_well_conditioned_ratio);
  nlohmann_json_j.at("degeneracy_multiplicity_relative_gap")
      .get_to(nlohmann_json_t.degeneracy_multiplicity_relative_gap);
  nlohmann_json_j.at("degeneracy_prior_weight").get_to(nlohmann_json_t.degeneracy_prior_weight);
  nlohmann_json_j.at("degeneracy_persistence_gate").get_to(nlohmann_json_t.degeneracy_persistence_gate);
  nlohmann_json_j.at("degeneracy_persistence_min_scans").get_to(nlohmann_json_t.degeneracy_persistence_min_scans);
  nlohmann_json_j.at("degeneracy_persistence_tracking_ratio")
      .get_to(nlohmann_json_t.degeneracy_persistence_tracking_ratio);
  nlohmann_json_j.at("degeneracy_persistence_min_absolute_cosine")
      .get_to(nlohmann_json_t.degeneracy_persistence_min_absolute_cosine);
  nlohmann_json_j.at("degeneracy_persistence_min_translation_fraction")
      .get_to(nlohmann_json_t.degeneracy_persistence_min_translation_fraction);
  nlohmann_json_j.at("degeneracy_adaptive_iteration_budget")
      .get_to(nlohmann_json_t.degeneracy_adaptive_iteration_budget);
  nlohmann_json_j.at("degeneracy_adaptive_max_iterations")
      .get_to(nlohmann_json_t.degeneracy_adaptive_max_iterations);
  nlohmann_json_j.at("degeneracy_adaptive_iteration_ratio")
      .get_to(nlohmann_json_t.degeneracy_adaptive_iteration_ratio);
  nlohmann_json_j.at("degeneracy_adaptive_hold_scans").get_to(nlohmann_json_t.degeneracy_adaptive_hold_scans);
  nlohmann_json_j.at("degeneracy_multiscan_observability_gate")
      .get_to(nlohmann_json_t.degeneracy_multiscan_observability_gate);
  nlohmann_json_j.at("degeneracy_observability_window_scans")
      .get_to(nlohmann_json_t.degeneracy_observability_window_scans);
  nlohmann_json_j.at("degeneracy_observability_min_scans").get_to(nlohmann_json_t.degeneracy_observability_min_scans);
  nlohmann_json_j.at("degeneracy_observability_max_directional_ratio")
      .get_to(nlohmann_json_t.degeneracy_observability_max_directional_ratio);
  nlohmann_json_j.at("visual_fusion").get_to(nlohmann_json_t.visual_fusion);
  nlohmann_json_j.at("visual_prior_max_time_offset_sec").get_to(nlohmann_json_t.visual_prior_max_time_offset_sec);
  nlohmann_json_j.at("radar_velocity_fusion").get_to(nlohmann_json_t.radar_velocity_fusion);
  nlohmann_json_j.at("radar_prior_max_time_offset_sec").get_to(nlohmann_json_t.radar_prior_max_time_offset_sec);
  nlohmann_json_j.at("radar_min_speed").get_to(nlohmann_json_t.radar_min_speed);
  nlohmann_json_j.at("radar_disagreement_gate").get_to(nlohmann_json_t.radar_disagreement_gate);
  nlohmann_json_j.at("radar_disagreement_min_mps").get_to(nlohmann_json_t.radar_disagreement_min_mps);
  nlohmann_json_j.at("radar_disagreement_min_scans").get_to(nlohmann_json_t.radar_disagreement_min_scans);
  nlohmann_json_j.at("radar_disagreement_weight").get_to(nlohmann_json_t.radar_disagreement_weight);
  nlohmann_json_j.at("radar_velocity_continuous_fusion").get_to(nlohmann_json_t.radar_velocity_continuous_fusion);
  nlohmann_json_j.at("radar_fusion_icp_information_scale")
      .get_to(nlohmann_json_t.radar_fusion_icp_information_scale);
  nlohmann_json_j.at("gravity_window_alignment").get_to(nlohmann_json_t.gravity_window_alignment);
  nlohmann_json_j.at("gravity_window_sec").get_to(nlohmann_json_t.gravity_window_sec);
  nlohmann_json_j.at("gravity_alignment_gain").get_to(nlohmann_json_t.gravity_alignment_gain);
  nlohmann_json_j.at("gravity_alignment_max_correction_rad")
      .get_to(nlohmann_json_t.gravity_alignment_max_correction_rad);
  nlohmann_json_j.at("gravity_alignment_max_magnitude_deviation")
      .get_to(nlohmann_json_t.gravity_alignment_max_magnitude_deviation);
  nlohmann_json_j.at("gravity_alignment_max_plausible_tilt_rad")
      .get_to(nlohmann_json_t.gravity_alignment_max_plausible_tilt_rad);
  nlohmann_json_j.at("gravity_alignment_max_yaw_rate_rad_s")
      .get_to(nlohmann_json_t.gravity_alignment_max_yaw_rate_rad_s);
  nlohmann_json_j.at("gravity_alignment_relevel_map").get_to(nlohmann_json_t.gravity_alignment_relevel_map);
  nlohmann_json_j.at("gravity_alignment_relevel_min_tilt_rad")
      .get_to(nlohmann_json_t.gravity_alignment_relevel_min_tilt_rad);
  nlohmann_json_j.at("localizability_weighting").get_to(nlohmann_json_t.localizability_weighting);
  nlohmann_json_j.at("localizability_boost").get_to(nlohmann_json_t.localizability_boost);
  nlohmann_json_j.at("localizability_min_step_m").get_to(nlohmann_json_t.localizability_min_step_m);
  nlohmann_json_j.at("localizability_normal_only").get_to(nlohmann_json_t.localizability_normal_only);
  nlohmann_json_j.at("kinematic_velocity_gate").get_to(nlohmann_json_t.kinematic_velocity_gate);
  nlohmann_json_j.at("kinematic_gate_accel_margin").get_to(nlohmann_json_t.kinematic_gate_accel_margin);
  nlohmann_json_j.at("kinematic_gate_min_speed").get_to(nlohmann_json_t.kinematic_gate_min_speed);
  nlohmann_json_j.at("kinematic_gate_weight").get_to(nlohmann_json_t.kinematic_gate_weight);
  nlohmann_json_j.at("kinematic_gate_skip_map_update").get_to(nlohmann_json_t.kinematic_gate_skip_map_update);
  nlohmann_json_j.at("kinematic_velocity_blend").get_to(nlohmann_json_t.kinematic_velocity_blend);
  nlohmann_json_j.at("kinematic_blend_icp_information_scale")
      .get_to(nlohmann_json_t.kinematic_blend_icp_information_scale);
  nlohmann_json_j.at("kinematic_blend_propagation_information_scale")
      .get_to(nlohmann_json_t.kinematic_blend_propagation_information_scale);
  nlohmann_json_j.at("kinematic_blend_decay_time_sec").get_to(nlohmann_json_t.kinematic_blend_decay_time_sec);
  nlohmann_json_j.at("kinematic_blend_anchor_agreement_mps")
      .get_to(nlohmann_json_t.kinematic_blend_anchor_agreement_mps);
  nlohmann_json_j.at("kinematic_blend_max_icp_innovation_mps")
      .get_to(nlohmann_json_t.kinematic_blend_max_icp_innovation_mps);
  nlohmann_json_j.at("kinematic_blend_max_propagated_speed_mps")
      .get_to(nlohmann_json_t.kinematic_blend_max_propagated_speed_mps);
  nlohmann_json_j.at("kinematic_blend_max_activation_speed_mps")
      .get_to(nlohmann_json_t.kinematic_blend_max_activation_speed_mps);
  nlohmann_json_j.at("kinematic_blend_speed_gate_min_scans")
      .get_to(nlohmann_json_t.kinematic_blend_speed_gate_min_scans);
  nlohmann_json_j.at("kinematic_blend_speed_reenable_delay_sec")
      .get_to(nlohmann_json_t.kinematic_blend_speed_reenable_delay_sec);
  nlohmann_json_j.at("kinematic_blend_min_speed").get_to(nlohmann_json_t.kinematic_blend_min_speed);
  if (nlohmann_json_j.contains("kinematic_blend_min_accel_magnitude_variance")) {
    nlohmann_json_j.at("kinematic_blend_min_accel_magnitude_variance")
        .get_to(nlohmann_json_t.kinematic_blend_min_accel_magnitude_variance);
  }
  if (nlohmann_json_j.contains("kinematic_blend_inactivity_gate_min_scans")) {
    nlohmann_json_j.at("kinematic_blend_inactivity_gate_min_scans")
        .get_to(nlohmann_json_t.kinematic_blend_inactivity_gate_min_scans);
  }
  nlohmann_json_j.at("kinematic_blend_max_yaw_rate_rad_s")
      .get_to(nlohmann_json_t.kinematic_blend_max_yaw_rate_rad_s);
  nlohmann_json_j.at("kinematic_blend_yaw_gate_min_scans")
      .get_to(nlohmann_json_t.kinematic_blend_yaw_gate_min_scans);
  nlohmann_json_j.at("kinematic_blend_range_scene_gate")
      .get_to(nlohmann_json_t.kinematic_blend_range_scene_gate);
  nlohmann_json_j.at("kinematic_blend_scene_near_range_m")
      .get_to(nlohmann_json_t.kinematic_blend_scene_near_range_m);
  nlohmann_json_j.at("kinematic_blend_scene_max_near_fraction")
      .get_to(nlohmann_json_t.kinematic_blend_scene_max_near_fraction);
  nlohmann_json_j.at("kinematic_blend_scene_far_range_m")
      .get_to(nlohmann_json_t.kinematic_blend_scene_far_range_m);
  nlohmann_json_j.at("kinematic_blend_scene_min_far_fraction")
      .get_to(nlohmann_json_t.kinematic_blend_scene_min_far_fraction);
  nlohmann_json_j.at("kinematic_blend_scene_min_valid_points")
      .get_to(nlohmann_json_t.kinematic_blend_scene_min_valid_points);
  nlohmann_json_j.at("kinematic_blend_scene_reenable_delay_sec")
      .get_to(nlohmann_json_t.kinematic_blend_scene_reenable_delay_sec);
  nlohmann_json_j.at("kinematic_blend_map_update_max_propagation_weight")
      .get_to(nlohmann_json_t.kinematic_blend_map_update_max_propagation_weight);
  nlohmann_json_j.at("kinematic_blend_map_update_min_fraction")
      .get_to(nlohmann_json_t.kinematic_blend_map_update_min_fraction);
  nlohmann_json_j.at("intensity_constraint").get_to(nlohmann_json_t.intensity_constraint);
  nlohmann_json_j.at("intensity_bin_size_m").get_to(nlohmann_json_t.intensity_bin_size_m);
  nlohmann_json_j.at("intensity_profile_half_length_m").get_to(nlohmann_json_t.intensity_profile_half_length_m);
  nlohmann_json_j.at("intensity_max_shift_m").get_to(nlohmann_json_t.intensity_max_shift_m);
  nlohmann_json_j.at("intensity_min_correlation").get_to(nlohmann_json_t.intensity_min_correlation);
  nlohmann_json_j.at("intensity_min_peak_margin").get_to(nlohmann_json_t.intensity_min_peak_margin);
  nlohmann_json_j.at("intensity_peak_exclusion_radius_bins")
      .get_to(nlohmann_json_t.intensity_peak_exclusion_radius_bins);
  nlohmann_json_j.at("intensity_min_filled_bins").get_to(nlohmann_json_t.intensity_min_filled_bins);
  nlohmann_json_j.at("intensity_oriented_grid").get_to(nlohmann_json_t.intensity_oriented_grid);
  nlohmann_json_j.at("intensity_grid_half_width_m").get_to(nlohmann_json_t.intensity_grid_half_width_m);
  nlohmann_json_j.at("intensity_grid_max_lateral_shift_m")
      .get_to(nlohmann_json_t.intensity_grid_max_lateral_shift_m);
  nlohmann_json_j.at("intensity_grid_height_weight")
      .get_to(nlohmann_json_t.intensity_grid_height_weight);
  nlohmann_json_j.at("intensity_disagreement_gate").get_to(nlohmann_json_t.intensity_disagreement_gate);
  nlohmann_json_j.at("intensity_disagreement_min_mps").get_to(nlohmann_json_t.intensity_disagreement_min_mps);
  nlohmann_json_j.at("intensity_disagreement_min_scans").get_to(nlohmann_json_t.intensity_disagreement_min_scans);
  nlohmann_json_j.at("intensity_disagreement_weight").get_to(nlohmann_json_t.intensity_disagreement_weight);
  nlohmann_json_j.at("max_scan_delta_sec").get_to(nlohmann_json_t.max_scan_delta_sec);
  nlohmann_json_j.at("enable_kidnap_relocalization").get_to(nlohmann_json_t.enable_kidnap_relocalization);
  nlohmann_json_j.at("reset_on_registration_failure").get_to(nlohmann_json_t.reset_on_registration_failure);
  nlohmann_json_j.at("recovery_min_failures").get_to(nlohmann_json_t.recovery_min_failures);
  nlohmann_json_j.at("relocalize_after_scan_gap").get_to(nlohmann_json_t.relocalize_after_scan_gap);
  nlohmann_json_j.at("relocalization_min_correspondences")
      .get_to(nlohmann_json_t.relocalization_min_correspondences);
  nlohmann_json_j.at("relocalization_min_inlier_ratio").get_to(nlohmann_json_t.relocalization_min_inlier_ratio);
  nlohmann_json_j.at("relocalization_max_mean_error").get_to(nlohmann_json_t.relocalization_max_mean_error);
  nlohmann_json_j.at("relocalization_max_correspondence_distance")
      .get_to(nlohmann_json_t.relocalization_max_correspondence_distance);
  nlohmann_json_j.at("relocalization_yaw_samples").get_to(nlohmann_json_t.relocalization_yaw_samples);
  nlohmann_json_j.at("relocalization_pose_stride").get_to(nlohmann_json_t.relocalization_pose_stride);
  nlohmann_json_j.at("relocalization_min_pose_separation").get_to(nlohmann_json_t.relocalization_min_pose_separation);
  nlohmann_json_j.at("relocalization_max_iterations").get_to(nlohmann_json_t.relocalization_max_iterations);
  nlohmann_json_t.legacy_voxel_downsample =
      nlohmann_json_j.value("legacy_voxel_downsample", nlohmann_json_t.legacy_voxel_downsample);
  nlohmann_json_t.samplewise_gyro_deskew =
      nlohmann_json_j.value("samplewise_gyro_deskew", nlohmann_json_t.samplewise_gyro_deskew);
  nlohmann_json_t.start_standstill_radius =
      nlohmann_json_j.value("start_standstill_radius", nlohmann_json_t.start_standstill_radius);
  nlohmann_json_t.start_standstill_max_accel_std =
      nlohmann_json_j.value("start_standstill_max_accel_std", nlohmann_json_t.start_standstill_max_accel_std);
  nlohmann_json_t.start_standstill_hold =
      nlohmann_json_j.value("start_standstill_hold", nlohmann_json_t.start_standstill_hold);
  nlohmann_json_t.start_standstill_max_gyro_rad_s =
      nlohmann_json_j.value("start_standstill_max_gyro_rad_s", nlohmann_json_t.start_standstill_max_gyro_rad_s);
  nlohmann_json_t.start_standstill_max_sec =
      nlohmann_json_j.value("start_standstill_max_sec", nlohmann_json_t.start_standstill_max_sec);
  nlohmann_json_t.start_standstill_bias =
      nlohmann_json_j.value("start_standstill_bias", nlohmann_json_t.start_standstill_bias);
  nlohmann_json_t.start_standstill_hold_radius =
      nlohmann_json_j.value("start_standstill_hold_radius", nlohmann_json_t.start_standstill_hold_radius);
  nlohmann_json_t.start_standstill_moving_start_only = nlohmann_json_j.value(
      "start_standstill_moving_start_only", nlohmann_json_t.start_standstill_moving_start_only);
  nlohmann_json_t.close_scene_max_median_range =
      nlohmann_json_j.value("close_scene_max_median_range", nlohmann_json_t.close_scene_max_median_range);
  nlohmann_json_t.close_scene_max_correspondence_distance = nlohmann_json_j.value(
      "close_scene_max_correspondence_distance", nlohmann_json_t.close_scene_max_correspondence_distance);
  nlohmann_json_t.close_scene_localizability_boost =
      nlohmann_json_j.value("close_scene_localizability_boost", nlohmann_json_t.close_scene_localizability_boost);
  nlohmann_json_t.close_scene_localizability_normal_only = nlohmann_json_j.value(
      "close_scene_localizability_normal_only", nlohmann_json_t.close_scene_localizability_normal_only);
  nlohmann_json_t.close_scene_min_inlier_fraction =
      nlohmann_json_j.value("close_scene_min_inlier_fraction", nlohmann_json_t.close_scene_min_inlier_fraction);
}
} // namespace rko_lio::core

namespace rko_lio::ros {

core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg) {
  core::ImuControl imu_data;
  imu_data.time = utils::to_ns(imu_msg.header.stamp);
  imu_data.angular_velocity = utils::ros_xyz_to_eigen_vector3d(imu_msg.angular_velocity);
  imu_data.acceleration = utils::ros_xyz_to_eigen_vector3d(imu_msg.linear_acceleration);
  return imu_data;
}

core::ImuControl BaseNode::imu_data_in_mps2(const sensor_msgs::msg::Imu& imu_msg) {
  core::ImuControl imu_data = imu_msg_to_imu_data(imu_msg);
  if (!(imu_acceleration_scale > 0.0)) {
    imu_acceleration_scale = core::acceleration_scale(imu_acceleration_unit, imu_data.acceleration.norm());
    RCLCPP_INFO(node->get_logger(), "IMU acceleration unit %s: first sample |a| = %.3f, scaling by %.5f",
                imu_acceleration_unit.c_str(), imu_data.acceleration.norm(), imu_acceleration_scale);
  }
  imu_data.acceleration *= imu_acceleration_scale;
  return imu_data;
}

namespace {
// rclcpp parameters behind the ROS-free driver::Parameters interface: each read declares the
// parameter on the node, exactly as the node's constructor did before declare_driver_config.
class RosParameters : public driver::Parameters {
public:
  explicit RosParameters(rclcpp::Node& node) : node_(node) {}
  bool get_bool(const std::string& name, bool default_value) override {
    return node_.declare_parameter<bool>(name, default_value);
  }
  int64_t get_int(const std::string& name, int64_t default_value) override {
    return node_.declare_parameter<int64_t>(name, default_value);
  }
  double get_double(const std::string& name, double default_value) override {
    return node_.declare_parameter<double>(name, default_value);
  }
  std::string get_string(const std::string& name, const std::string& default_value) override {
    return node_.declare_parameter<std::string>(name, default_value);
  }
  std::vector<double> get_double_array(const std::string& name, const std::vector<double>& default_value) override {
    return node_.declare_parameter<std::vector<double>>(name, default_value);
  }
  std::vector<int64_t> get_int_array(const std::string& name, const std::vector<int64_t>& default_value) override {
    return node_.declare_parameter<std::vector<int64_t>>(name, default_value);
  }
  std::string get_required_string(const std::string& name) override {
    return node_.declare_parameter<std::string>(name);
  }
  void info(const std::string& message) override { RCLCPP_INFO_STREAM(node_.get_logger(), message); }
  void warn(const std::string& message) override { RCLCPP_WARN_STREAM(node_.get_logger(), message); }

private:
  rclcpp::Node& node_;
};
} // namespace

BaseNode::BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options) {
  node = rclcpp::Node::make_shared(node_name, options);
  RosParameters params(*node);
  const driver::DriverConfig config = driver::declare_driver_config(params);
  imu_topic = config.imu_topic;
  lidar_topic = config.lidar_topic;
  base_frame = config.base_frame;
  imu_frame = config.imu_frame;
  lidar_frame = config.lidar_frame;
  odom_frame = config.odom_frame;
  odom_topic = config.odom_topic;
  invert_odom_tf = config.invert_odom_tf;
  publish_lidar_acceleration = config.publish_lidar_acceleration;
  imu_acceleration_unit = config.imu_acceleration_unit;
  publish_deskewed_scan = config.publish_deskewed_scan;
  deskewed_scan_topic = config.deskewed_scan_topic;
  deskewed_scan_voxel_size = config.deskewed_scan_voxel_size;
  publish_local_map = config.publish_local_map;
  map_topic = config.map_topic;
  publish_map_after = config.publish_map_after;
  photometric_channel = config.photometric_channel;
  direct_visual_frontend = config.direct_visual_frontend;
  direct_visual_require_previous_weak_direction = config.direct_visual_require_previous_weak_direction;
  radar_topic = config.radar_topic;
  radar_ego_velocity_config = config.radar_ego_velocity_config;
  radar_velocity_scale = config.radar_velocity_scale;
  radar_sigma_forward_mps = config.radar_sigma_forward_mps;
  radar_sigma_lateral_mps = config.radar_sigma_lateral_mps;
  radar_sigma_vertical_mps = config.radar_sigma_vertical_mps;
  timestamp_proc_config = config.timestamp_proc_config;
  dump_results = config.dump_results;
  results_dir = config.results_dir;
  run_name = config.run_name;

  // tf
  tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);

  // publishing
  const int publisher_queue_depth = node->declare_parameter<int>("publisher_queue_depth", 1);
  if (publisher_queue_depth < 1) {
    throw std::invalid_argument("publisher_queue_depth must be positive");
  }
  const rclcpp::QoS publisher_qos(
      rclcpp::SystemDefaultsQoS().keep_last(static_cast<std::size_t>(publisher_queue_depth)).durability_volatile());
  odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>(odom_topic, publisher_qos);
  if (publish_lidar_acceleration) {
    lidar_accel_publisher =
        node->create_publisher<geometry_msgs::msg::AccelStamped>("rko_lio/lidar_acceleration", publisher_qos);
  }
  if (publish_deskewed_scan) {
    frame_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_topic, publisher_qos);
  }
  if (publish_local_map) {
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
    map_publish_thead = std::jthread([this]() { publish_map_loop(); });
  }

  lio = std::make_unique<core::LIO>(config.lio);

  // Extrinsics given as parameters; the rest come from TF on the first messages.
  if (config.extrinsic_imu2base) {
    extrinsic_imu2base = *config.extrinsic_imu2base;
  }
  if (config.extrinsic_lidar2base) {
    extrinsic_lidar2base = *config.extrinsic_lidar2base;
  }
  if (config.extrinsic_cam2base) {
    extrinsic_cam2base = *config.extrinsic_cam2base;
    visual_extrinsic_set = true;
  }
  if (config.extrinsic_radar2base) {
    extrinsic_radar2base = *config.extrinsic_radar2base;
    radar_extrinsic_set = true;
  }
  extrinsics_set = config.extrinsic_imu2base.has_value() && config.extrinsic_lidar2base.has_value();
  if (!extrinsics_set) {
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  }
  if (lio->config.visual_fusion.enabled) {
    if (direct_visual_frontend) {
      configure_direct_visual_frontend();
    } else {
      load_visual_constraints();
    }
  }

  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Subscribed to IMU: "
                         << imu_topic << (!imu_frame.empty() ? " (frame " + imu_frame + ")" : "") << " and LiDAR: "
                         << lidar_topic << (!lidar_frame.empty() ? " (frame " + lidar_frame + ")" : "")
                         << ". Max number of threads: " << config.lio.max_num_threads << ". Publishing odometry to "
                         << odom_topic << " ( " << odom_frame
                         << " ) and acceleration "
                            "estimates to rko_lio/lidar_acceleration. Deskewing is "
                         << (lio->config.deskew ? "enabled" : "disabled") << "."
                         << (publish_deskewed_scan ? (" Publishing deskewed_cloud to " + deskewed_scan_topic + ".")
                                                   : ""));

  shutdown_callback_handle = node->get_node_base_interface()->get_context()->add_on_shutdown_callback([this]() {
    if (dump_results) {
      dump_results_to_disk(results_dir, run_name);
    }
  });

  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

bool BaseNode::ensure_frame_and_extrinsics(std::string& target_frame,
                                           const std::string& msg_frame,
                                           std::string_view kind) {
  if (target_frame.empty()) {
    if (msg_frame.empty() && !extrinsics_set) {
      throw std::runtime_error(std::string(kind) +
                               " message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    target_frame = msg_frame;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the " << kind << " frame id as: " << target_frame);
  }
  return check_and_set_extrinsics();
}

bool BaseNode::check_and_set_extrinsics() {
  if (extrinsics_set) {
    return true;
  }
  const std::optional<Sophus::SE3d> imu_transform = utils::get_transform(tf_buffer, imu_frame, base_frame, 0s);
  if (!imu_transform) {
    return false;
  }
  const std::optional<Sophus::SE3d> lidar_transform = utils::get_transform(tf_buffer, lidar_frame, base_frame, 0s);
  if (!lidar_transform) {
    return false;
  }
  extrinsic_imu2base = imu_transform.value();
  extrinsic_lidar2base = lidar_transform.value();
  extrinsics_set = true;
  return true;
}

std::tuple<core::Timestamps, core::Vector3dVector>
BaseNode::process_lidar_msg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const {
  const core::Nsec header_stamp = utils::to_ns(lidar_msg->header.stamp);
  if (!lio->config.deskew) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Deskewing is disabled. Populating timestamps with static header time.");
  }
  return driver::process_lidar_frame(utils::point_cloud_view(*lidar_msg), header_stamp, lio->config.deskew,
                                     timestamp_proc_config);
}

std::vector<float>
BaseNode::process_lidar_intensity(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const {
  // Only parsed when an intensity-based feature is enabled: point_cloud2_to_intensity() still
  // iterates the whole cloud, so skip it entirely on the (default) common path.
  if (lio->config.photometric) {
    // The photometric images take precedence over the reflectivity-profile features.
    std::vector<float> values = utils::point_cloud2_field_as_float(lidar_msg, photometric_channel);
    if (values.empty()) {
      RCLCPP_WARN_STREAM_ONCE(node->get_logger(), "photometric is enabled but the point cloud on "
                                                      << lidar_topic << " has no '" << photometric_channel
                                                      << "' field; photometric terms will not be used.");
    }
    return values;
  }
  const bool intensity_needed = lio->config.intensity_constraint || lio->config.intensity_disagreement_gate;
  if (!intensity_needed) {
    return {};
  }
  std::vector<float> intensities = utils::point_cloud2_to_intensity(lidar_msg);
  if (intensities.empty()) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "intensity_constraint/intensity_disagreement_gate is enabled but the point cloud on "
                                << lidar_topic
                                << " has no usable 'reflectivity' (uint16) or 'intensity' (float32) field; the "
                                   "intensity feature(s) will not fire.");
  }
  return intensities;
}

core::Vector3dVector BaseNode::register_scan_locked(const core::Vector3dVector& scan,
                                                    const core::TimestampVector& time_vector,
                                                    const std::vector<float>* intensities) {
  if (publish_local_map) {
    std::lock_guard lock(local_map_mutex); // map publish thread may access map simultaneously
    return lio->register_scan(extrinsic_lidar2base, scan, time_vector, intensities);
  }
  return lio->register_scan(extrinsic_lidar2base, scan, time_vector, intensities);
}

void BaseNode::publish_lidar_outputs(const core::Vector3dVector& deskewed_frame) const {
  if (publish_deskewed_scan) {
    std_msgs::msg::Header header;
    header.frame_id = lidar_frame;
    header.stamp = utils::to_ros_time(lio->lidar_state.time);
    frame_publisher->publish(utils::eigen_to_point_cloud2(
        deskewed_scan_voxel_size > 0.0 ? core::voxel_down_sample(deskewed_frame, deskewed_scan_voxel_size)
                                       : deskewed_frame,
        header));
  }
  if (lio->output_poses_with_timestamps.empty()) {
    publish_odometry(lio->lidar_state, odom_publisher);
  } else {
    // output_filter_position_sigma: the filtered position with the registered attitude
    core::State output_state = lio->lidar_state;
    output_state.pose = lio->output_poses_with_timestamps.back().second;
    publish_odometry(output_state, odom_publisher);
  }
  if (publish_lidar_acceleration) {
    publish_lidar_accel(lio->lidar_state);
  }
}

void BaseNode::publish_odometry(const core::State& state,
                                const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& publisher) const {
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = utils::to_ros_time(state.time);
  odom_msg.header.frame_id = odom_frame;
  odom_msg.child_frame_id = base_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  // [v0.8 Phase 1, diagnostic-only] anisotropic covariance from the final ICP Hessian; see
  // pose_covariance_from_state's doc comment above. Only this field is new -- pose.pose and
  // twist.twist are set exactly as before.
  odom_msg.pose.covariance = driver::pose_covariance_from_state(state);
  utils::eigen_vector3d_to_ros_xyz(state.velocity, odom_msg.twist.twist.linear);
  utils::eigen_vector3d_to_ros_xyz(state.angular_velocity, odom_msg.twist.twist.angular);
  publisher->publish(odom_msg);
}

void BaseNode::publish_tf(const core::State& state) const {
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = utils::to_ros_time(state.time);
  if (invert_odom_tf) {
    transform_msg.header.frame_id = base_frame;
    transform_msg.child_frame_id = odom_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = odom_frame;
    transform_msg.child_frame_id = base_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);
}

void BaseNode::publish_lidar_accel(const core::State& state) const {
  auto accel_msg = geometry_msgs::msg::AccelStamped();
  accel_msg.header.stamp = utils::to_ros_time(state.time);
  accel_msg.header.frame_id = base_frame;
  utils::eigen_vector3d_to_ros_xyz(state.linear_acceleration, accel_msg.accel.linear);
  lidar_accel_publisher->publish(accel_msg);
}

void BaseNode::publish_map_loop() {
  while (atomic_node_running) {
    std::this_thread::sleep_for(publish_map_after);
    std::unique_lock lock(local_map_mutex);
    if (lio->map.empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3dVector map_points = lio->map.pointcloud();
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = node->now();
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
  }
}

BaseNode::~BaseNode() {
  atomic_node_running = false;
  // Nodes can be destroyed before their context shuts down (including failed derived constructors).
  node->get_node_base_interface()->get_context()->remove_on_shutdown_callback(shutdown_callback_handle);
}

// ---- radar ego-velocity fusion (fork addition) ----

void BaseNode::radar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& radar_msg) {
  if (!lio->config.radar_velocity_fusion && !lio->config.radar_velocity_continuous_fusion) {
    return;
  }
  const bool has_velocity_field =
      std::any_of(radar_msg->fields.cbegin(), radar_msg->fields.cend(), [](const auto& field) {
        return field.name == "velocity" && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
      });
  if (!has_velocity_field) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Radar point cloud on " << radar_topic << " has no float32 'velocity' field; ignoring.");
    return;
  }

  const std::size_t point_count = static_cast<std::size_t>(radar_msg->height) * radar_msg->width;
  std::vector<core::RadarDopplerMeasurement> measurements;
  measurements.reserve(point_count);
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*radar_msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*radar_msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*radar_msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> msg_velocity(*radar_msg, "velocity");
  for (std::size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_velocity) {
    const Eigen::Vector3d point(*msg_x, *msg_y, *msg_z);
    const double range = point.norm();
    if (!(range > 1.0e-6) || !std::isfinite(*msg_velocity)) {
      continue;
    }
    measurements.push_back({point / range, static_cast<double>(*msg_velocity)});
  }

  const core::RadarEgoVelocityResult velocity_result =
      core::estimate_radar_ego_velocity(measurements, radar_ego_velocity_config);
  if (!velocity_result.valid) {
    return;
  }

  core::RadarVelocityPrior prior;
  prior.time = utils::to_ns(radar_msg->header.stamp);
  prior.velocity_base = radar_velocity_scale * (extrinsic_radar2base.so3() * velocity_result.velocity);
  if (lio->config.radar_velocity_continuous_fusion) {
    // Per-axis (forward/lateral/vertical, radar sensor frame) information, rotated into the
    // base frame so LIO::register_scan can rotate it once more into world using the current
    // pose estimate (see radar_velocity_continuous_fusion in lio.cpp). Kept in sync with
    // radar_velocity_scale conceptually: a scaled velocity with an unscaled sigma would
    // silently mistrust/overtrust the scaled estimate, but the scale here is a systematic bias
    // correction (narrow-FOV underestimate), not an added noise source, so the sigmas are left
    // as directly configured.
    const Eigen::Vector3d inverse_variance(1.0 / (radar_sigma_forward_mps * radar_sigma_forward_mps),
                                           1.0 / (radar_sigma_lateral_mps * radar_sigma_lateral_mps),
                                           1.0 / (radar_sigma_vertical_mps * radar_sigma_vertical_mps));
    const Eigen::Matrix3d R_radar2base = extrinsic_radar2base.so3().matrix();
    prior.info_base = R_radar2base * inverse_variance.asDiagonal() * R_radar2base.transpose();
  }
  {
    std::lock_guard lock(radar_prior_mutex);
    radar_prior_queue.push_back(prior);
    // Bound the queue for online use; offline the registration loop drains it.
    constexpr std::size_t kMaxBufferedRadarPriors = 2000;
    if (radar_prior_queue.size() > kMaxBufferedRadarPriors) {
      radar_prior_queue.pop_front();
    }
  }
}

void BaseNode::prepare_radar_prior(const core::Nsec& lidar_time) {
  if (!lio->config.radar_velocity_fusion && !lio->config.radar_velocity_continuous_fusion) {
    return;
  }
  const double tolerance = lio->config.radar_prior_max_time_offset_sec;
  std::optional<core::RadarVelocityPrior> best;
  {
    std::lock_guard lock(radar_prior_mutex);
    // Drop estimates that can never match a future scan again.
    while (!radar_prior_queue.empty() &&
           core::to_seconds(lidar_time - radar_prior_queue.front().time) > tolerance) {
      radar_prior_queue.pop_front();
    }
    for (const auto& candidate : radar_prior_queue) {
      const double offset = core::to_seconds(candidate.time - lidar_time);
      if (offset > tolerance) {
        break; // queue is time-ordered; later entries are only further away
      }
      if (!best.has_value() || std::abs(offset) < std::abs(core::to_seconds(best->time - lidar_time))) {
        best = candidate;
      }
    }
  }
  if (best.has_value()) {
    lio->set_radar_velocity_prior(*best);
  } else {
    lio->clear_radar_velocity_prior();
  }
}

// ---- selective visual fusion / direct visual odometry frontend (fork addition) ----

void BaseNode::load_visual_constraints() {
  const std::string path = node->declare_parameter<std::string>("visual_constraints_path", "");
  if (path.empty()) {
    throw std::runtime_error("selective_visual_fusion requires visual_constraints_path during artifact-driven "
                             "development");
  }
  if (!visual_extrinsic_set) {
    throw std::runtime_error("selective_visual_fusion requires extrinsic_cam2base_quat_xyzw_xyz");
  }
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open visual constraint artifact: " + path);
  }
  nlohmann::json report;
  stream >> report;
  if (report.at("schema_version").get<int>() != 1) {
    throw std::runtime_error("unsupported visual constraint schema");
  }
  for (const auto& record : report.at("constraints")) {
    if (!record.value("accepted", false)) {
      continue;
    }
    VisualRelativeConstraint constraint;
    constraint.first_time =
        std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(record.at("first_stamp").get<double>()));
    constraint.second_time = std::chrono::duration_cast<core::Nsec>(
        std::chrono::duration<double>(record.at("second_stamp").get<double>()));
    const auto rotation = record.at("rotation").get<std::vector<std::vector<double>>>();
    const auto direction = record.at("translation_direction").get<std::vector<double>>();
    if (rotation.size() != 3 || rotation[0].size() != 3 || rotation[1].size() != 3 || rotation[2].size() != 3 ||
        direction.size() != 3) {
      throw std::runtime_error("invalid visual rotation or translation direction shape");
    }
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        constraint.current_R_previous_camera(row, col) =
            rotation[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
      }
      constraint.current_t_previous_direction(row) = direction[static_cast<std::size_t>(row)];
    }
    constraint.confidence.tracks = record.at("tracks").get<std::size_t>();
    constraint.confidence.inliers = record.at("inliers").get<std::size_t>();
    constraint.confidence.rotation_error_deg = record.at("rotation_error_deg").get<double>();
    constraint.confidence.translation_cosine = record.at("translation_cosine").get<double>();
    constraint.confidence.baseline_m = record.at("predicted_translation_m").get<double>();
    visual_constraints.push_back(constraint);
  }
  std::sort(visual_constraints.begin(), visual_constraints.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second_time < rhs.second_time; });
  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Loaded " << visual_constraints.size() << " accepted visual constraints from " << path);
}

void BaseNode::configure_direct_visual_frontend() {
  visual_image_topic = node->declare_parameter<std::string>("visual_image_topic", visual_image_topic);
  visual_camera_time_offset_sec =
      node->declare_parameter<double>("visual_camera_time_offset_sec", visual_camera_time_offset_sec);
  visual_max_image_lidar_delta_sec =
      node->declare_parameter<double>("visual_max_image_lidar_delta_sec", visual_max_image_lidar_delta_sec);
  visual_keyframe_interval_sec =
      node->declare_parameter<double>("visual_keyframe_interval_sec", visual_keyframe_interval_sec);
  visual_camera.width = node->declare_parameter<int>("visual_camera_width", 0);
  visual_camera.height = node->declare_parameter<int>("visual_camera_height", 0);
  visual_camera.fx = node->declare_parameter<double>("visual_camera_fx", 0.0);
  visual_camera.fy = node->declare_parameter<double>("visual_camera_fy", 0.0);
  visual_camera.cx = node->declare_parameter<double>("visual_camera_cx", 0.0);
  visual_camera.cy = node->declare_parameter<double>("visual_camera_cy", 0.0);
  const auto distortion_model = node->declare_parameter<std::string>("visual_camera_distortion_model", "equidistant");
  if (distortion_model == "equidistant" || distortion_model == "fisheye") {
    visual_camera.distortion_model = core::CameraDistortionModel::equidistant;
  } else if (distortion_model == "plumb_bob" || distortion_model == "radtan") {
    visual_camera.distortion_model = core::CameraDistortionModel::plumb_bob;
  } else {
    throw std::runtime_error("direct_visual_frontend camera distortion model must be equidistant or plumb_bob");
  }
  const auto distortion = node->declare_parameter<std::vector<double>>("visual_camera_distortion", std::vector<double>{});
  if (distortion.size() != 4 || !visual_camera.valid()) {
    throw std::runtime_error("direct_visual_frontend requires valid intrinsics and four distortion coefficients");
  }
  std::copy(distortion.begin(), distortion.end(), visual_camera.distortion.begin());
  direct_visual_config.max_features =
      node->declare_parameter<int>("direct_visual_max_features", direct_visual_config.max_features);
  direct_visual_config.grid_cell_size =
      node->declare_parameter<int>("direct_visual_grid_cell_size", direct_visual_config.grid_cell_size);
  direct_visual_config.patch_radius =
      node->declare_parameter<int>("direct_visual_patch_radius", direct_visual_config.patch_radius);
  direct_visual_config.max_iterations =
      node->declare_parameter<int>("direct_visual_max_iterations", direct_visual_config.max_iterations);
  direct_visual_config.min_gradient =
      node->declare_parameter<double>("direct_visual_min_gradient", direct_visual_config.min_gradient);
  direct_visual_config.huber_delta =
      node->declare_parameter<double>("direct_visual_huber_delta", direct_visual_config.huber_delta);
  direct_visual_config.occlusion_tolerance_m = node->declare_parameter<double>(
      "direct_visual_occlusion_tolerance_m", direct_visual_config.occlusion_tolerance_m);
  direct_visual_config.min_residuals =
      node->declare_parameter<int>("direct_visual_min_residuals", direct_visual_config.min_residuals);
  direct_visual_config.min_inlier_ratio =
      node->declare_parameter<double>("direct_visual_min_inlier_ratio", direct_visual_config.min_inlier_ratio);
  direct_visual_config.max_rmse =
      node->declare_parameter<double>("direct_visual_max_rmse", direct_visual_config.max_rmse);
  direct_visual_config.max_initial_rmse =
      node->declare_parameter<double>("direct_visual_max_initial_rmse", direct_visual_config.max_initial_rmse);
  direct_visual_config.max_translation_correction_m = node->declare_parameter<double>(
      "direct_visual_max_translation_correction_m", direct_visual_config.max_translation_correction_m);
  direct_visual_config.max_rotation_correction_rad = node->declare_parameter<double>(
      "direct_visual_max_rotation_correction_rad", direct_visual_config.max_rotation_correction_rad);
  RCLCPP_INFO_STREAM(node->get_logger(), "Live direct visual frontend: "
                                             << visual_image_topic << " " << visual_camera.width << "x"
                                             << visual_camera.height << " distortion=" << distortion_model
                                             << " keyframe interval " << visual_keyframe_interval_sec << " s");
}

void BaseNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) {
  if (!direct_visual_frontend) {
    return;
  }
  if (image_msg->encoding != "mono8" || static_cast<int>(image_msg->width) != visual_camera.width ||
      static_cast<int>(image_msg->height) != visual_camera.height || image_msg->step < image_msg->width) {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Direct visual frontend requires mono8 images matching configured dimensions");
    return;
  }
  VisualImageFrame frame;
  frame.time = utils::to_ns(image_msg->header.stamp) +
               std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(visual_camera_time_offset_sec));
  frame.image.width = static_cast<int>(image_msg->width);
  frame.image.height = static_cast<int>(image_msg->height);
  frame.image.pixels.resize(static_cast<std::size_t>(frame.image.width * frame.image.height));
  for (int row = 0; row < frame.image.height; ++row) {
    std::copy_n(image_msg->data.begin() + static_cast<std::ptrdiff_t>(row * image_msg->step), frame.image.width,
               frame.image.pixels.begin() + static_cast<std::ptrdiff_t>(row * frame.image.width));
  }
  std::lock_guard lock(visual_buffer_mutex);
  visual_image_buffer.push(std::move(frame));
  while (visual_image_buffer.size() > 400) {
    visual_image_buffer.pop();
  }
}

std::optional<LiveVisualKeyframe> BaseNode::prepare_direct_visual_prior(const core::Vector3dVector& scan,
                                                                        const core::Nsec& lidar_time) {
  if (!direct_visual_frontend) {
    return std::nullopt;
  }
  std::optional<VisualImageFrame> selected;
  {
    std::lock_guard lock(visual_buffer_mutex);
    while (!visual_image_buffer.empty() && visual_image_buffer.front().time <= lidar_time) {
      selected = std::move(visual_image_buffer.front());
      visual_image_buffer.pop();
    }
  }
  if (!selected.has_value() || core::to_seconds(lidar_time - selected->time) > visual_max_image_lidar_delta_sec ||
      (live_visual_keyframe.has_value() &&
       core::to_seconds(selected->time - live_visual_keyframe->time) < visual_keyframe_interval_sec)) {
    return std::nullopt;
  }
  LiveVisualKeyframe pending;
  pending.time = selected->time;
  pending.image = std::move(selected->image);
  const Sophus::SE3d camera_T_lidar = extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  pending.depth = core::project_sparse_depth(scan, camera_T_lidar, visual_camera);
  if (!live_visual_keyframe.has_value()) {
    return pending;
  }
  if (direct_visual_require_previous_weak_direction &&
      (!lio->lidar_state.icp_diagnostics.has_value() ||
       core::count_weak_information_directions(lio->lidar_state.icp_diagnostics->H, lio->config.visual_fusion) ==
           0)) {
    ++direct_visual_weak_gate_skip_count;
    return pending;
  }

  const Sophus::SE3d predicted_image_base = lio->predict_pose_at(pending.time);
  const Sophus::SE3d predicted_lidar_base = lio->predict_pose_at(lidar_time);
  const Sophus::SE3d previous_camera = live_visual_keyframe->world_T_base * extrinsic_cam2base;
  const Sophus::SE3d predicted_camera = predicted_image_base * extrinsic_cam2base;
  const Sophus::SE3d initial_relative = predicted_camera.inverse() * previous_camera;
  ++direct_visual_attempt_count;
  const auto direct = core::align_direct_visual(live_visual_keyframe->image, live_visual_keyframe->depth,
                                                pending.image, pending.depth, visual_camera, initial_relative,
                                                direct_visual_config);
  if (direct_visual_attempt_count % 50 == 0) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Direct visual probe "
                                               << direct_visual_attempt_count << ": valid=" << direct.valid
                                               << " features=" << direct.features
                                               << " tracked=" << direct.tracked_features
                                               << " inlier_ratio=" << direct.inlier_ratio
                                               << " rmse=" << direct.final_rmse);
  }
  DirectVisualDiagnosticsSample diagnostics;
  diagnostics.time = pending.time;
  diagnostics.solver_valid = direct.valid;
  diagnostics.failure_reason = static_cast<int>(direct.failure_reason);
  diagnostics.features = direct.features;
  diagnostics.tracked_features = direct.tracked_features;
  diagnostics.inliers = direct.inliers;
  diagnostics.inlier_ratio = direct.inlier_ratio;
  diagnostics.initial_rmse = direct.initial_rmse;
  diagnostics.final_rmse = direct.final_rmse;
  diagnostics.exposure_gain = direct.exposure_gain;
  diagnostics.exposure_bias = direct.exposure_bias;
  diagnostics.predicted_baseline_m = initial_relative.translation().norm();
  if (!direct.valid) {
    direct_visual_diagnostics.push_back(diagnostics);
    return pending;
  }
  ++direct_visual_solver_valid_count;
  const Eigen::Matrix3d rotation_difference =
      direct.current_T_previous_camera.rotationMatrix() * initial_relative.rotationMatrix().transpose();
  const double rotation_cosine = std::clamp((rotation_difference.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double rotation_error_deg = std::acos(rotation_cosine) * 180.0 / std::numbers::pi;
  const Eigen::Vector3d predicted_translation = initial_relative.translation();
  const Eigen::Vector3d measured_translation = direct.current_T_previous_camera.translation();
  const double predicted_norm = predicted_translation.norm();
  const double measured_norm = measured_translation.norm();
  const double translation_cosine = predicted_norm > 1.0e-9 && measured_norm > 1.0e-9
                                        ? predicted_translation.dot(measured_translation) /
                                              (predicted_norm * measured_norm)
                                        : -1.0;
  diagnostics.rotation_error_deg = rotation_error_deg;
  diagnostics.translation_cosine = translation_cosine;
  diagnostics.baseline_m = measured_norm;
  diagnostics.predicted_baseline_m = predicted_norm;
  core::VisualConstraintConfidence confidence;
  confidence.tracks = static_cast<std::size_t>(direct.tracked_features);
  confidence.inliers = static_cast<std::size_t>(direct.inliers);
  confidence.rotation_error_deg = rotation_error_deg;
  confidence.translation_cosine = translation_cosine;
  confidence.baseline_m = measured_norm;
  const Eigen::Matrix6d camera_adjoint_inverse = predicted_camera.Adj().inverse();
  confidence.visual_information = camera_adjoint_inverse.transpose() * direct.pose_information * camera_adjoint_inverse;
  diagnostics.confidence_gate_passed = core::visual_constraint_passes_gate(confidence, lio->config.visual_fusion);
  direct_visual_diagnostics.push_back(diagnostics);
  if (!diagnostics.confidence_gate_passed) {
    return pending;
  }
  const Sophus::SE3d desired_image_camera = previous_camera * direct.current_T_previous_camera.inverse();
  const Sophus::SE3d desired_image_base = desired_image_camera * extrinsic_cam2base.inverse();
  const Sophus::SE3d image_T_lidar_prediction = predicted_image_base.inverse() * predicted_lidar_base;
  const Sophus::SE3d desired_lidar_base = desired_image_base * image_T_lidar_prediction;
  lio->set_visual_pose_prior({lidar_time, desired_lidar_base, confidence});
  ++direct_visual_valid_count;
  return pending;
}

void BaseNode::commit_direct_visual_keyframe(LiveVisualKeyframe frame,
                                             const core::Nsec& lidar_time,
                                             const core::Vector3dVector& deskewed_lidar_frame) {
  const double dt = core::to_seconds(frame.time - lidar_time);
  Eigen::Vector6d correction = Eigen::Vector6d::Zero();
  correction.head<3>() = lio->lidar_state.velocity * dt;
  correction.tail<3>() = lio->lidar_state.angular_velocity * dt;
  frame.world_T_base = lio->lidar_state.pose * Sophus::SE3d::exp(correction);
  const Sophus::SE3d camera_T_lidar = extrinsic_cam2base.inverse() * extrinsic_lidar2base;
  frame.depth = core::project_sparse_depth(deskewed_lidar_frame, camera_T_lidar, visual_camera);
  live_visual_keyframe = std::move(frame);
}

void BaseNode::prepare_visual_prior(const core::Nsec& lidar_time) {
  if (!lio->config.visual_fusion.enabled || visual_constraints.empty() || lio->poses_with_timestamps.empty()) {
    return;
  }
  const double tolerance = lio->config.visual_prior_max_time_offset_sec;
  while (next_visual_constraint < visual_constraints.size() &&
         core::to_seconds(visual_constraints[next_visual_constraint].second_time - lidar_time) < -tolerance) {
    ++next_visual_constraint;
  }
  if (next_visual_constraint >= visual_constraints.size()) {
    return;
  }
  const auto& constraint = visual_constraints[next_visual_constraint];
  if (std::abs(core::to_seconds(constraint.second_time - lidar_time)) > tolerance) {
    return;
  }

  const auto nearest = std::min_element(
      lio->poses_with_timestamps.begin(), lio->poses_with_timestamps.end(), [&constraint](const auto& lhs, const auto& rhs) {
        return std::abs(core::to_seconds(lhs.first - constraint.first_time)) <
               std::abs(core::to_seconds(rhs.first - constraint.first_time));
      });
  if (nearest == lio->poses_with_timestamps.end() ||
      std::abs(core::to_seconds(nearest->first - constraint.first_time)) > tolerance) {
    ++next_visual_constraint;
    return;
  }
  const double prediction_dt = core::to_seconds(constraint.second_time - lio->lidar_state.time);
  Eigen::Vector6d prediction = Eigen::Vector6d::Zero();
  prediction.head<3>() = lio->lidar_state.velocity * prediction_dt;
  prediction.tail<3>() = lio->lidar_state.angular_velocity * prediction_dt;
  const Sophus::SE3d predicted_current_pose = lio->lidar_state.pose * Sophus::SE3d::exp(prediction);
  const Sophus::SE3d desired_pose =
      desired_base_pose_from_monocular_relative(nearest->second, predicted_current_pose, extrinsic_cam2base,
                                                constraint.current_R_previous_camera,
                                                constraint.current_t_previous_direction);
  auto confidence = constraint.confidence;
  confidence.baseline_m =
      ((predicted_current_pose * extrinsic_cam2base).inverse() * (nearest->second * extrinsic_cam2base))
          .translation()
          .norm();
  lio->set_visual_pose_prior({constraint.second_time, desired_pose, confidence});
  ++next_visual_constraint;
}

void BaseNode::dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const {
  try {
    std::filesystem::create_directories(results_dir); // no error if exists
    int index = 0;
    std::filesystem::path output_dir = results_dir / (run_name + "_" + std::to_string(index));
    while (std::filesystem::exists(output_dir)) {
      ++index;
      output_dir = results_dir / (run_name + "_" + std::to_string(index));
    }
    std::filesystem::create_directory(output_dir);
    const std::filesystem::path output_file = output_dir / (run_name + "_tum_" + std::to_string(index) + ".txt");
    // dump poses
    if (std::ofstream file(output_file); file.is_open()) {
      // the filtered output when output_filter_position_sigma is set
      const auto& poses = lio->output_poses_with_timestamps.empty() ? lio->poses_with_timestamps
                                                                     : lio->output_poses_with_timestamps;
      for (const auto& [timestamp, pose] : poses) {
        const Eigen::Vector3d& translation = pose.translation();
        const Eigen::Quaterniond& quaternion = pose.so3().unit_quaternion();
        file << std::fixed << std::setprecision(6) << core::to_seconds(timestamp) << " " << translation.x() << " "
             << translation.y() << " " << translation.z() << " " << quaternion.x() << " " << quaternion.y() << " "
             << quaternion.z() << " " << quaternion.w() << "\n";
      }
      std::cout << "Poses written to " << std::filesystem::absolute(output_file) << "\n";
    }
    // dump config
    const nlohmann::json json_config = {{"config", lio->config}};
    const std::filesystem::path config_file = output_dir / "config.json";
    if (std::ofstream file(config_file); file.is_open()) {
      file << json_config.dump(4);
      std::cout << "Configuration written to " << config_file << "\n";
    }
    // [v0.8 Phase 1] opt-in per-scan persistence-gate diagnostics (degeneracy_persistence_gate).
    if (!lio->degeneracy_persistence_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file = output_dir / "degeneracy_persistence.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,candidate_available,consecutive_scans,matched_absolute_cosine,confirmed,"
                "observability_window_scans,aggregate_directional_information_ratio,"
                "multiscan_observability_confirmed,"
                "intervention_count,axis_tx,axis_ty,axis_tz,axis_rx,axis_ry,axis_rz\n";
        for (const auto& sample : lio->degeneracy_persistence_diagnostics) {
          const auto& state = sample.persistent_weak_direction;
          file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << ","
               << static_cast<int>(state.candidate_available) << "," << state.consecutive_scans << ","
               << state.matched_absolute_cosine << "," << static_cast<int>(state.confirmed) << ","
               << state.observability_window_scans << "," << state.aggregate_directional_information_ratio << ","
               << static_cast<int>(state.multiscan_observability_confirmed) << "," << sample.intervention_count;
          for (int axis = 0; axis < 6; ++axis) {
            file << "," << state.axis(axis);
          }
          file << "\n";
        }
        std::cout << "Degeneracy persistence diagnostics written to " << diagnostics_file << "\n";
      }
    }
    // Selective visual fusion / direct visual odometry summary.
    if (lio->config.visual_fusion.enabled) {
      const nlohmann::json visual_summary = {{"prior_attempt_count", lio->visual_prior_attempt_count},
                                             {"fused_scan_count", lio->visual_fused_scan_count},
                                             {"fused_direction_count", lio->visual_fused_direction_count},
                                             {"visual_unobservable_direction_count",
                                              lio->visual_unobservable_direction_count},
                                             {"direct_attempt_count", direct_visual_attempt_count},
                                             {"direct_weak_gate_skip_count", direct_visual_weak_gate_skip_count},
                                             {"direct_solver_valid_count", direct_visual_solver_valid_count},
                                             {"direct_valid_count", direct_visual_valid_count}};
      const std::filesystem::path visual_file = output_dir / "visual_fusion_summary.json";
      if (std::ofstream file(visual_file); file.is_open()) {
        file << visual_summary.dump(4) << "\n";
        std::cout << "Visual fusion summary written to " << visual_file << "\n";
      }
    }
    // Radar ego-velocity fusion summary.
    if (lio->config.radar_velocity_fusion || lio->config.radar_velocity_continuous_fusion) {
      nlohmann::json radar_summary = {
          {"prior_attempt_count", lio->radar_prior_attempt_count},
          {"fused_scan_count", lio->radar_fused_scan_count},
          {"disagreement_corrected_scan_count", lio->radar_disagreement_corrected_scan_count}};
      if (lio->config.radar_velocity_continuous_fusion) {
        radar_summary["continuous_attempt_count"] = lio->radar_continuous_attempt_count;
        radar_summary["continuous_fused_scan_count"] = lio->radar_continuous_fused_scan_count;
        radar_summary["continuous_mean_abs_correction_m"] =
            lio->radar_continuous_fused_scan_count > 0
                ? lio->radar_continuous_correction_magnitude_sum /
                      static_cast<double>(lio->radar_continuous_fused_scan_count)
                : 0.0;
      }
      const std::filesystem::path radar_file = output_dir / "radar_velocity_fusion_summary.json";
      if (std::ofstream file(radar_file); file.is_open()) {
        file << radar_summary.dump(4) << "\n";
        std::cout << "Radar velocity fusion summary written to " << radar_file << "\n";
      }
    }
    // Photometric registration summary.
    if (lio->config.photometric) {
      const nlohmann::json photometric_summary = {
          {"frame_count", lio->photometric_frame_count},
          {"scan_count", lio->photometric_scan_count},
          {"mean_patches_per_scan",
           lio->photometric_scan_count > 0
               ? static_cast<double>(lio->photometric_patch_sum) / static_cast<double>(lio->photometric_scan_count)
               : 0.0},
          {"gap_skip_count", lio->photometric_gap_skip_count},
          {"tracked_patches_at_end", lio->photometric_features().features().size()}};
      const std::filesystem::path photometric_file = output_dir / "photometric_summary.json";
      if (std::ofstream file(photometric_file); file.is_open()) {
        file << photometric_summary.dump(4) << "\n";
        std::cout << "Photometric summary written to " << photometric_file << "\n";
      }
    }
    // Gait speed summary.
    if (lio->config.gait_speed_prior) {
      nlohmann::json gait_summary = {{"applied_count", lio->gait_speed_applied_count},
                                     {"correction_limited_count", lio->gait_correction_limited_count},
                                     {"reversal_count", lio->gait_reversal_count},
                                     {"hold_count", lio->gait_hold_count},
                                     {"standing_count", lio->gait_standing_count}};
      if (const auto step = lio->gait_step_length()) {
        gait_summary["step_length_m"] = *step;
      }
      const std::filesystem::path gait_file = output_dir / "gait_speed_summary.json";
      if (std::ofstream file(gait_file); file.is_open()) {
        file << gait_summary.dump(4) << "\n";
      }
    }
    // Inertial window summary.
    if (lio->config.inertial_window) {
      const auto& window = lio->inertial_window();
      const nlohmann::json window_summary = {
          {"solve_count", lio->inertial_window_solve_count},
          {"final_accelerometer_bias_mps2",
           {window.accelerometer_bias()(0), window.accelerometer_bias()(1), window.accelerometer_bias()(2)}},
          {"final_gyroscope_bias_radps",
           {window.gyroscope_bias()(0), window.gyroscope_bias()(1), window.gyroscope_bias()(2)}}};
      const std::filesystem::path window_file = output_dir / "inertial_window_summary.json";
      if (std::ofstream file(window_file); file.is_open()) {
        file << window_summary.dump(4) << "\n";
      }
    }
    // Bump image registration summary.
    if (lio->config.bump_image_registration) {
      const nlohmann::json bump_summary = {{"rotation_fallback_count", lio->bump_rotation_fallback_count}};
      const std::filesystem::path bump_file = output_dir / "bump_image_summary.json";
      if (std::ofstream file(bump_file); file.is_open()) {
        file << bump_summary.dump(4) << "\n";
        std::cout << "Bump image summary written to " << bump_file << "\n";
      }
      if (!lio->bump_rotation_corrections.empty()) {
        const std::filesystem::path corrections_file = output_dir / "bump_rotation_corrections.csv";
        if (std::ofstream file(corrections_file); file.is_open()) {
          file << "time_sec,rotation_correction_deg\n" << std::setprecision(17);
          for (const auto& [time_sec, rotation_deg] : lio->bump_rotation_corrections) {
            file << time_sec << "," << rotation_deg << "\n";
          }
        }
      }
    }
    if (lio->config.close_scene_max_median_range > 0.0) {
      std::cout << "Close-scene settings used on " << lio->close_scene_scan_count << " registered scans ("
                << lio->close_scene_fallback_count << " registered again with the base settings)\n";
    }
    if (lio->config.start_standstill_radius > 0.0) {
      std::cout << "Standing still near the start on " << lio->start_standstill_count << " scan intervals\n";
    }
    if (lio->config.keypoint_shortfall_voxel_multiplier > 0.0) {
      std::cout << "Keypoints taken again at the finer voxel on " << lio->keypoint_shortfall_scan_count << " scans\n";
    }
    // Sliding-window gravity alignment summary.
    if (lio->config.gravity_window_alignment) {
      const nlohmann::json gravity_summary = {
          {"attempt_count", lio->gravity_alignment_attempt_count},
          {"applied_count", lio->gravity_alignment_applied_count},
          {"last_tilt_deg", lio->gravity_alignment_last_tilt_rad * 180.0 / M_PI},
          {"mean_correction_deg_per_applied_scan",
           lio->gravity_alignment_applied_count > 0
               ? (lio->gravity_alignment_correction_rad_sum /
                  static_cast<double>(lio->gravity_alignment_applied_count)) *
                     180.0 / M_PI
               : 0.0}};
      const std::filesystem::path gravity_file = output_dir / "gravity_alignment_summary.json";
      if (std::ofstream file(gravity_file); file.is_open()) {
        file << gravity_summary.dump(4) << "\n";
        std::cout << "Gravity alignment summary written to " << gravity_file << "\n";
      }
    }
    if (lio->config.vehicle_stall_guard) {
      std::cout << "Stall guard held the speed on " << lio->stall_guard.held_scan_count << " scans ("
                << lio->stall_guard.hold_count << " stalls)\n";
    }
    if (lio->config.max_tilt_correction_deg > 0.0) {
      std::cout << "Tilt correction limited on " << lio->tilt_correction_limited_count << " scans";
      if (lio->config.max_tilt_correction_median_factor > 0.0) {
        std::cout << ", last limit " << lio->tilt_correction_last_limit_rad * 180.0 / M_PI << " deg";
      }
      std::cout << "\n";
    }
    if (lio->config.max_yaw_correction_deg > 0.0) {
      std::cout << "Yaw correction limited on " << lio->yaw_correction_limited_count << " scans\n";
    }
    if (lio->config.gyro_bias_time_constant_sec > 0.0) {
      std::cout << "Gyro bias tracked on " << lio->gyro_bias_update_count << " scans, final "
                << lio->imu_bias.gyroscope.transpose() << "\n";
    }
    // Kinematic velocity gate summary.
    if (lio->config.kinematic_velocity_gate) {
      const nlohmann::json kinematic_summary = {
          {"attempt_count", lio->kinematic_gate_attempt_count},
          {"corrected_scan_count", lio->kinematic_gate_corrected_scan_count},
          {"mean_correction_m",
           lio->kinematic_gate_corrected_scan_count > 0
               ? lio->kinematic_gate_correction_m_sum /
                     static_cast<double>(lio->kinematic_gate_corrected_scan_count)
               : 0.0}};
      const std::filesystem::path kinematic_file = output_dir / "kinematic_velocity_gate_summary.json";
      if (std::ofstream file(kinematic_file); file.is_open()) {
        file << kinematic_summary.dump(4) << "\n";
        std::cout << "Kinematic velocity gate summary written to " << kinematic_file << "\n";
      }
    }
    // Anchor-decayed inertial velocity blend summary.
    if (lio->config.kinematic_velocity_blend) {
      const nlohmann::json blend_summary = {
          {"attempt_count", lio->kinematic_blend_attempt_count},
          {"anchor_refresh_count", lio->kinematic_blend_anchor_refresh_count},
          {"corrected_scan_count", lio->kinematic_blend_corrected_scan_count},
          {"mean_correction_m",
           lio->kinematic_blend_corrected_scan_count > 0
               ? lio->kinematic_blend_correction_m_sum /
                     static_cast<double>(lio->kinematic_blend_corrected_scan_count)
               : 0.0},
          {"mean_propagation_weight",
           lio->kinematic_blend_corrected_scan_count > 0
               ? lio->kinematic_blend_propagation_weight_sum /
                     static_cast<double>(lio->kinematic_blend_corrected_scan_count)
               : 0.0},
          {"max_propagation_weight", lio->kinematic_blend_max_propagation_weight},
          {"max_anchor_age_sec", lio->kinematic_blend_max_anchor_age_sec},
          {"scene_evaluation_count", lio->kinematic_blend_scene_evaluation_count},
          {"scene_valid_count", lio->kinematic_blend_scene_valid_count},
          {"scene_rejected_scan_count", lio->kinematic_blend_scene_rejected_scan_count},
          {"scene_cooldown_rejected_scan_count",
           lio->kinematic_blend_scene_cooldown_rejected_scan_count},
          {"speed_rejected_scan_count", lio->kinematic_blend_speed_rejected_scan_count},
          {"speed_cooldown_rejected_scan_count",
           lio->kinematic_blend_speed_cooldown_rejected_scan_count},
          {"low_speed_rejected_scan_count",
           lio->kinematic_blend_low_speed_rejected_scan_count},
          {"yaw_rejected_scan_count", lio->kinematic_blend_yaw_rejected_scan_count},
          {"anchor_expiration_count", lio->kinematic_blend_anchor_expiration_count},
          {"propagated_speed_clamp_count",
           lio->kinematic_blend_propagated_speed_clamp_count},
          {"invalid_result_count", lio->kinematic_blend_invalid_result_count},
          {"activity_retained_low_speed_scan_count",
           lio->kinematic_blend_activity_retained_low_speed_scan_count},
          {"inactivity_rejected_scan_count",
           lio->kinematic_blend_inactivity_rejected_scan_count},
          {"max_disagreement_mps", lio->kinematic_blend_max_disagreement_mps},
          {"max_correction_m", lio->kinematic_blend_max_correction_m},
          {"first_correction_time_sec",
           lio->kinematic_blend_first_correction_time_sec >= 0.0
               ? nlohmann::json(lio->kinematic_blend_first_correction_time_sec)
               : nlohmann::json(nullptr)},
          {"last_correction_time_sec",
           lio->kinematic_blend_last_correction_time_sec >= 0.0
               ? nlohmann::json(lio->kinematic_blend_last_correction_time_sec)
               : nlohmann::json(nullptr)},
          {"last_anchor_refresh_time_sec",
           lio->kinematic_blend_last_anchor_refresh_time_sec >= 0.0
               ? nlohmann::json(lio->kinematic_blend_last_anchor_refresh_time_sec)
               : nlohmann::json(nullptr)},
          {"mean_scene_near_fraction",
           lio->kinematic_blend_scene_valid_count > 0
               ? lio->kinematic_blend_scene_near_fraction_sum /
                     static_cast<double>(lio->kinematic_blend_scene_valid_count)
               : 0.0},
          {"mean_scene_far_fraction",
           lio->kinematic_blend_scene_valid_count > 0
               ? lio->kinematic_blend_scene_far_fraction_sum /
                     static_cast<double>(lio->kinematic_blend_scene_valid_count)
               : 0.0}};
      const std::filesystem::path blend_file = output_dir / "kinematic_velocity_blend_summary.json";
      if (std::ofstream file(blend_file); file.is_open()) {
        file << blend_summary.dump(4) << "\n";
        std::cout << "Kinematic velocity blend summary written to " << blend_file << "\n";
      }
    }
    // Localizability-aware ICP weighting summary.
    if (lio->config.localizability_weighting) {
      const nlohmann::json localizability_summary = {
          {"attempt_count", lio->localizability_attempt_count},
          {"weighted_scan_count", lio->localizability_weighted_scan_count},
          {"mean_boosted_correspondence_fraction",
           lio->localizability_weighted_scan_count > 0
               ? lio->localizability_boosted_fraction_sum /
                     static_cast<double>(lio->localizability_weighted_scan_count)
               : 0.0}};
      const std::filesystem::path localizability_file = output_dir / "localizability_weighting_summary.json";
      if (std::ofstream file(localizability_file); file.is_open()) {
        file << localizability_summary.dump(4) << "\n";
        std::cout << "Localizability weighting summary written to " << localizability_file << "\n";
      }
    }
    // Intensity/reflectivity texture constraint summary.
    if (lio->config.intensity_constraint || lio->config.intensity_disagreement_gate) {
      nlohmann::json intensity_summary = {{"prior_attempt_count", lio->intensity_prior_attempt_count},
                                          {"prior_applied_count", lio->intensity_prior_applied_count}};
      if (lio->config.intensity_disagreement_gate) {
        intensity_summary["disagreement_corrected_scan_count"] = lio->intensity_disagreement_corrected_scan_count;
        intensity_summary["disagreement_attempt_count"] = lio->intensity_disagreement_attempt_count;
        intensity_summary["disagreement_valid_shift_count"] = lio->intensity_disagreement_valid_shift_count;
        intensity_summary["ambiguous_shift_count"] = lio->intensity_ambiguous_shift_count;
        intensity_summary["peak_margin_sample_count"] = lio->intensity_peak_margin_sample_count;
        intensity_summary["peak_margin_mean"] =
            lio->intensity_peak_margin_sample_count > 0
                ? lio->intensity_peak_margin_sum /
                      static_cast<double>(lio->intensity_peak_margin_sample_count)
                : 0.0;
        intensity_summary["peak_margin_min"] =
            lio->intensity_peak_margin_sample_count > 0
                ? lio->intensity_peak_margin_min
                : 0.0;
        intensity_summary["disagreement_exceeded_threshold_count"] =
            lio->intensity_disagreement_exceeded_threshold_count;
      }
      const std::filesystem::path intensity_file = output_dir / "intensity_constraint_summary.json";
      if (std::ofstream file(intensity_file); file.is_open()) {
        file << intensity_summary.dump(4) << "\n";
        std::cout << "Intensity constraint summary written to " << intensity_file << "\n";
      }
      const std::filesystem::path peak_file =
          output_dir / "intensity_peak_diagnostics.csv";
      if (std::ofstream file(peak_file); file.is_open()) {
        file << "timestamp,source,correlation,second_best_correlation,"
                "peak_margin,longitudinal_shift_m,lateral_shift_m,"
                "overlap_bins,base_qualified,has_competing_peak,"
                "ambiguous,accepted,motion_dt_s,"
                "intensity_velocity_longitudinal_mps,"
                "intensity_velocity_lateral_mps,"
                "icp_velocity_longitudinal_mps,"
                "icp_velocity_lateral_mps,velocity_disagreement_mps,"
                "candidate_correction_m,"
                "applied_correction_longitudinal_m,"
                "applied_correction_lateral_m,applied_correction_m,"
                "disagreement_streak,disagreement_measured,"
                "correction_applied,intensity_channel_correlation,"
                "height_channel_correlation\n";
        for (const auto& diagnostic : lio->intensity_peak_diagnostics) {
          file << std::fixed << std::setprecision(9)
               << core::to_seconds(diagnostic.time) << ","
               << (diagnostic.source ==
                           core::IntensityPeakSource::persistent_prior
                       ? "persistent_prior"
                       : diagnostic.source ==
                                 core::IntensityPeakSource::oriented_grid
                             ? "oriented_grid"
                             : "disagreement_gate")
               << "," << diagnostic.correlation << ","
               << diagnostic.second_best_correlation << ","
               << diagnostic.peak_margin << ","
               << diagnostic.longitudinal_shift_m << ","
               << diagnostic.lateral_shift_m << ","
               << diagnostic.overlap_bins << ","
               << (diagnostic.base_qualified ? 1 : 0) << ","
               << (diagnostic.has_competing_peak ? 1 : 0) << ","
               << (diagnostic.ambiguous ? 1 : 0) << ","
               << (diagnostic.accepted ? 1 : 0) << ","
               << diagnostic.motion_dt_s << ","
               << diagnostic.intensity_velocity_longitudinal_mps
               << ","
               << diagnostic.intensity_velocity_lateral_mps << ","
               << diagnostic.icp_velocity_longitudinal_mps << ","
               << diagnostic.icp_velocity_lateral_mps << ","
               << diagnostic.velocity_disagreement_mps << ","
               << diagnostic.candidate_correction_m << ","
               << diagnostic.applied_correction_longitudinal_m << ","
               << diagnostic.applied_correction_lateral_m << ","
               << diagnostic.applied_correction_m << ","
               << diagnostic.disagreement_streak << ","
               << (diagnostic.disagreement_measured ? 1 : 0) << ","
               << (diagnostic.correction_applied ? 1 : 0) << ","
               << diagnostic.intensity_channel_correlation << ","
               << diagnostic.height_channel_correlation << "\n";
        }
        std::cout << "Intensity peak diagnostics written to " << peak_file
                  << "\n";
      }
    }
    if (!lio->visual_observability_diagnostics.empty()) {
      const std::filesystem::path observability_file = output_dir / "visual_directional_observability.csv";
      if (std::ofstream file(observability_file); file.is_open()) {
        file << "timestamp,direction_index,directional_information_ratio\n";
        for (const auto& sample : lio->visual_observability_diagnostics) {
          for (std::size_t index = 0; index < sample.ratio_count; ++index) {
            file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << "," << index << ","
                 << sample.directional_information_ratios[index] << "\n";
          }
        }
      }
    }
    if (!direct_visual_diagnostics.empty()) {
      const std::filesystem::path diagnostics_file = output_dir / "direct_visual_diagnostics.csv";
      if (std::ofstream file(diagnostics_file); file.is_open()) {
        file << "timestamp,solver_valid,confidence_gate_passed,failure_reason,features,tracked_features,inliers,"
                "inlier_ratio,initial_rmse,final_rmse,exposure_gain,exposure_bias,"
                "rotation_error_deg,translation_cosine,baseline_m,predicted_baseline_m\n";
        for (const auto& sample : direct_visual_diagnostics) {
          file << std::fixed << std::setprecision(9) << core::to_seconds(sample.time) << ","
               << static_cast<int>(sample.solver_valid) << "," << static_cast<int>(sample.confidence_gate_passed)
               << "," << sample.failure_reason << "," << sample.features << "," << sample.tracked_features << ","
               << sample.inliers << "," << sample.inlier_ratio << "," << sample.initial_rmse << ","
               << sample.final_rmse << "," << sample.exposure_gain << "," << sample.exposure_bias << ","
               << sample.rotation_error_deg << "," << sample.translation_cosine << "," << sample.baseline_m << ","
               << sample.predicted_baseline_m << "\n";
        }
      }
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "[WARNING] Cannot write files to disk, encountered filesystem error: " << ex.what() << "\n";
  }
}

} // namespace rko_lio::ros
