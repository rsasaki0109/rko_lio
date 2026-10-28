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

/**
 * @file lio.hpp
 * Core LIO class and utilities for RKO-LIO.
 */

#pragma once
#include "intensity_profile.hpp"
#include "gyro_deskew_history.hpp"
#include "oriented_intensity_grid.hpp"
#include "bump_image_map.hpp"
#include "bump_rotation_gate.hpp"
#include "inertial_window.hpp"
#include "gait_speed.hpp"
#include "gravity_alignment.hpp"
#include "output_position_filter.hpp"
#include "photometric_features.hpp"
#include "photometric_image.hpp"
#include "persistent_weak_direction.hpp"
#include "scan_statistics.hpp"
#include "selective_visual_fusion.hpp"
#include "stall_guard.hpp"
#include "voxel_hash_map.hpp"
#include "util.hpp"
#include <algorithm>
#include <array>
#include <deque>
#include <tuple>
#include <limits>
#include <optional>
#include <string>
#include <utility>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {

/** `position` moved back onto the sphere of `radius` around `anchor` if it lies outside
 *  (start_standstill_hold_radius). */
inline Eigen::Vector3d clamp_to_radius(const Eigen::Vector3d& position,
                                       const Eigen::Vector3d& anchor,
                                       const double radius) {
  const Eigen::Vector3d offset = position - anchor;
  const double distance = offset.norm();
  if (distance <= radius) return position;
  return anchor + offset * (std::max(radius, 0.0) / distance);
}

struct VisualPosePrior {
  Nsec time{0};
  Sophus::SE3d pose;
  VisualConstraintConfidence confidence;
};

/** One-shot base-frame ego-velocity prior derived from a radar Doppler scan. */
struct RadarVelocityPrior {
  Nsec time{0};
  Eigen::Vector3d velocity_base = Eigen::Vector3d::Zero();

  /** Base-frame translation information (inverse covariance) of `velocity_base`, used only by
   *  Config::radar_velocity_continuous_fusion (see LIO::register_scan). Computed by the ROS
   *  node from the radar-to-base extrinsic rotation and the per-axis radar sigmas
   *  (radar_sigma_forward/lateral/vertical_mps), i.e. info_base = R_radar2base *
   *  diag(1/sigma_fwd^2, 1/sigma_lat^2, 1/sigma_vert^2) * R_radar2base^T. Defaults to an
   *  isotropic 0.3 m/s sigma so a producer that never sets this field (or the continuous
   *  fusion feature being off) still yields a well-defined, harmless information matrix. */
  Eigen::Matrix3d info_base = Eigen::Matrix3d::Identity() * (1.0 / (0.3 * 0.3));
};

enum class IntensityPeakSource {
  persistent_prior,
  disagreement_gate,
  oriented_grid,
};

/** Raw correlation-peak evidence retained for offline policy analysis.
 *
 * Keeping observations instead of a fixed histogram lets benchmark tooling
 * change bins or compare future ambiguity policies without replaying a bag.
 */
struct IntensityPeakDiagnostic {
  Nsec time{0};
  IntensityPeakSource source = IntensityPeakSource::disagreement_gate;
  double correlation = -1.0;
  double second_best_correlation = -1.0;
  double peak_margin = 0.0;
  double longitudinal_shift_m = 0.0;
  double lateral_shift_m = 0.0;
  std::size_t overlap_bins = 0;
  bool base_qualified = false;
  bool has_competing_peak = false;
  bool ambiguous = false;
  bool accepted = false;
  double motion_dt_s = 0.0;
  double intensity_velocity_longitudinal_mps = 0.0;
  double intensity_velocity_lateral_mps = 0.0;
  double icp_velocity_longitudinal_mps = 0.0;
  double icp_velocity_lateral_mps = 0.0;
  double velocity_disagreement_mps = 0.0;
  double candidate_correction_m = 0.0;
  double applied_correction_longitudinal_m = 0.0;
  double applied_correction_lateral_m = 0.0;
  double applied_correction_m = 0.0;
  std::size_t disagreement_streak = 0;
  bool disagreement_measured = false;
  bool correction_applied = false;
  double intensity_channel_correlation = -2.0;
  double height_channel_correlation = -2.0;
};

struct VisualObservabilityDiagnosticsSample {
  Nsec time{0};
  std::array<double, 6> directional_information_ratios{};
  std::size_t ratio_count = 0;
};

/** Core LiDAR-inertial odometry algorithm class. */
class LIO {
public:
  /** Configuration parameters for odometry. */
  struct Config {
    /** Enable scan deskewing. */
    bool deskew = true;

    /** Maximum number of ICP iterations. */
    size_t max_iterations = 100;

    /** Size of voxel grid (m). */
    double voxel_size = 1.0;

    /** Max points per voxel. */
    int max_points_per_voxel = 20;

    /** Maximum lidar range (m). */
    double max_range = 100.0;

    /** Minimum lidar range (m). */
    double min_range = 1.0;

    /** ICP convergence threshold. */
    double convergence_criterion = 1e-5;

    /** Take Levenberg-Marquardt steps in the registration instead of plain Gauss-Newton:
     *  damp the step in body coordinates and accept it only when the cost drops (as
     *  BIEVR-LIO does). Along a direction the scan barely constrains, such as the axis of a
     *  tunnel, Gauss-Newton turns residual noise into large moves; in the GEODE shield
     *  tunnels those moves add up to the registration sliding along the lining. */
    bool icp_levenberg_marquardt = false;

    /** Handheld walking only: keep the speed of the prediction at cadence times a learned
     *  step length (GaitSpeedEstimator) along the direction of travel. Where the scan says
     *  nothing along that direction the registration otherwise lets the speed decay to zero
     *  or run away. */
    bool gait_speed_prior = false;
    double gait_window_sec = 4.0;
    double gait_min_peak_fraction = 0.25;
    double gait_min_magnitude_std = 0.5;
    double gait_min_step_length = 0.5;
    double gait_max_step_length = 1.0;
    /** Step length (m) used until enough has been learned (0: wait). Walkers in the GEODE
     *  shield tunnels take 0.58-0.79 m steps. */
    double gait_default_step_length = 0.7;
    /** Registered speed (m/s) needed to update the direction of travel. */
    double gait_min_direction_speed = 0.5;
    /** While the prediction runs at the gait speed, limit the registration's correction
     *  along the direction of travel to this much per scan (m, 0: no limit). A stalled
     *  scan otherwise enters the map at the stalled pose, and the map then holds the next
     *  scans back; the gait is the independent speed that tells the two apart. */
    double gait_max_correction_m = 0.0;
    /** Flip the direction of travel when the IMU, integrated from the velocity this long
     *  ago (s), says the walker now moves the other way faster than gait_min_direction_speed
     *  (walking back without turning the sensor). 0 disables. */
    double gait_reversal_lookback_sec = 4.0;
    /** Scale the whole velocity to the gait speed, keeping its direction, instead of only
     *  replacing its component along the direction of travel (see apply_gait_speed). */
    bool gait_sets_speed = false;
    /** Once the step is learned, a scan in a gap of the cadence (turning, stopping) whose
     *  registered speed exceeds the last gait speed keeps the walker's bounds (see
     *  hold_gait_speed) and the along-travel correction limit. */
    bool gait_hold_without_cadence = false;
    /** Standard deviation (m/s^2) of the specific force magnitude over the last second below
     *  which a walker without a cadence is standing. */
    double gait_standing_accel_std = 0.3;
    /** Limit the registration's correction along the travel (gait_max_correction_m) only while
     *  its weakest translation information is below this share of the trace (a tunnel's axis);
     *  where it sees every direction (a stairwell, a room) its correction stands. 0 limits it
     *  everywhere. */
    double gait_correction_weak_ratio = 0.0;
    /** Initial damping as a fraction of the largest diagonal entry of the system. */
    double icp_lm_initial_lambda_factor = 1e-9;
    /** Damping increases tried per iteration before giving up on a step. */
    int icp_lm_max_attempts = 20;

    /** Max distance for correspondences (m). */
    double max_correspondence_distance = 0.5;

    /** When positive, every correspondence of an ICP iteration is weighted with the
     *  Geman-McClure kernel (s^2 / (s^2 + r^2))^2, whose scale s is this factor times the
     *  median correspondence distance of that iteration (at least 1 mm), so that a few far
     *  correspondences no longer dominate the least-squares system. With 2.0 it roughly halved
     *  the error of the indoor MapsHD humanoid trials but made the outdoor ones (MapsHD
     *  outdoor-urk, Oxford Spires) worse. 0 disables it (plain least squares). */
    double icp_robust_kernel_scale_factor = 0.0;

    /** Weight of the along-surface residual of every correspondence whose map voxel has a
     *  normal: W = t (I - n n^T) + w n n^T (w the localizability weight, at least 1). 1 keeps
     *  the point-to-point ICP (the normals are not even looked up); 0 is point-to-plane. The
     *  rings a spinning LiDAR leaves on the ground pull a point-to-point registration along the
     *  ground while the map is young. */
    double icp_tangential_weight = 1.0;

    /** Thread count for data association (0 = automatic). */
    int max_num_threads = 0;

    /** Enable initialization phase. */
    bool initialization_phase = false;

    /** Maximum expected jerk (m/s³). */
    double max_expected_jerk = 3;

    /** Enable double downsampling. */
    bool double_downsample = true;

    /** Preserve the complete pre-v0.3 voxel sampling pipeline. */
    bool legacy_voxel_downsample = false;

    /** Deskew rotation from the integrated gyro samples. When false, the rotation uses the
     * interval's mean angular velocity, as before rko_lio #14. */
    bool samplewise_gyro_deskew = true;

    /** Voxel-size multiplier used only for ICP keypoints in double-downsample mode. */
    double icp_keypoint_voxel_multiplier = 1.5;

    /** Scans with fewer ICP keypoints are not registered. A mostly occluded scan (sensor
     *  covered by a hand or the body) can still yield a few dozen keypoints, and ICP on those
     *  can slide along a weakly observed direction and write the error into the map. */
    size_t min_icp_keypoints = 10;

    /** When positive, a scan with fewer than min_icp_keypoints keypoints takes its keypoints
     *  again at voxel_size times this multiplier (from the map-resolution downsample in
     *  double-downsample mode) before it is dropped. A small scene, such as a stairwell, can
     *  leave too few keypoints at the regular keypoint voxel while the scan itself is full. */
    double keypoint_shortfall_voxel_multiplier = 0.0;

    /** Minimum weight for orientation regularization. */
    double min_beta = 200;

    /** Estimate the velocity from the pose at least this long ago (s) instead of the
     *  previous scan's (0 keeps the previous scan). Over one 0.1 s scan a single wrong
     *  correction of a few decimetres becomes a velocity error of metres per second, and
     *  a registration that constrains the translation weakly (open ground while turning
     *  fast) cannot pull the next prediction back. */
    double velocity_window_sec = 0.0;

    /** BIEVR-LIO style inertial window (InertialWindow): hold the registered poses of the
     *  last inertial_window_sec fixed and solve the velocity at every scan and the IMU
     *  biases from the IMU between them. The newest velocity replaces the estimate above
     *  for the next prediction and deskew, and the biases replace imu_bias. The
     *  registration itself never sees the IMU. On ENWIDE this halves RunwayD's error but
     *  costs TunnelD and RunwayS; in the GEODE shield tunnels the window fits the
     *  registration's slip along the axis as an accelerometer bias and makes it worse,
     *  so it stays off by default. */
    bool inertial_window = false;
    double inertial_window_sec = 10.0;
    int inertial_window_min_intervals = 10;
    /** Residual weights (1 / sigma): rotation (rad), velocity (m/s), position (m). */
    double inertial_window_rotation_weight = 1.0;
    double inertial_window_velocity_weight = 1.0;
    double inertial_window_position_weight = 1.0;
    /** Weights (1 / sigma) of the priors keeping the biases near their last estimates. */
    double inertial_window_accelerometer_bias_prior_weight = 1.0;
    double inertial_window_gyroscope_bias_prior_weight = 10.0;

    /** Do not register a scan that follows a LiDAR gap longer than this (s); take the IMU
     *  prediction as its pose and keep it out of the map (0 disables). The first scan
     *  after a gap is often partial and its prediction coarse, and a tunnel constrains
     *  it so weakly that the registration can turn it by degrees (ENWIDE TunnelD). Use
     *  it with velocity_window_sec, so the skipped scan does not set the velocity. */
    double skip_registration_after_gap_sec = 0.0;

    /** Replace the legacy ICP solve with direction-aware prior blending. */
    bool degeneracy_aware_solve = false;

    /** Minimum normalized Hessian contribution considered well-conditioned. */
    double degeneracy_well_conditioned_ratio = 1.0e-6;

    /** Maximum contribution gap merged into one non-observable eigenspace. */
    double degeneracy_multiplicity_relative_gap = 1.0e-8;

    /** Motion-prior weight in an isolated degenerate direction. */
    double degeneracy_prior_weight = 0.25;

    /** Require a weak world-frame direction to persist across scans before intervention. */
    bool degeneracy_persistence_gate = false;

    /** Consecutive matching weak-direction scans required for confirmation. */
    size_t degeneracy_persistence_min_scans = 3;

    /** Broader diagnostic threshold used only to maintain the direction track. */
    double degeneracy_persistence_tracking_ratio = 1.5e-5;

    /** Minimum absolute axis cosine for persistence matching. */
    double degeneracy_persistence_min_absolute_cosine = 0.98;

    /** Minimum translational energy fraction for a tracked weak direction. */
    double degeneracy_persistence_min_translation_fraction = 0.99;

    /** Extend the ICP iteration budget only on scans with a weak Hessian direction. */
    bool degeneracy_adaptive_iteration_budget = false;

    /** Maximum ICP iterations used after a weak first-iteration Hessian. */
    size_t degeneracy_adaptive_max_iterations = 100;

    /** Normalized minimum-eigenvalue threshold that triggers the extended budget. */
    double degeneracy_adaptive_iteration_ratio = 1.5e-5;

    /** Scans retaining the extended budget after observing weak information. */
    size_t degeneracy_adaptive_hold_scans = 5;

    /** Require low information along the tracked direction over a multi-scan window. */
    bool degeneracy_multiscan_observability_gate = false;

    /** Maximum number of normalized scan Hessians retained by the observability gate. */
    size_t degeneracy_observability_window_scans = 10;

    /** Minimum accumulated scans required before the observability gate can confirm. */
    size_t degeneracy_observability_min_scans = 5;

    /** Maximum directional information contribution in the accumulated Hessian. */
    double degeneracy_observability_max_directional_ratio = 1.0e-6;

    /** Add a confidence-gated visual prior only in weak LiDAR directions. */
    SelectiveVisualFusionConfig visual_fusion;

    /** Maximum camera/LiDAR timestamp difference for a visual prior. */
    double visual_prior_max_time_offset_sec = 0.08;

    /** Blend a one-shot radar Doppler ego-velocity prior into weak (degenerate) Hessian
     *  directions only. Requires degeneracy_aware_solve; reuses degeneracy_prior_weight
     *  for the blend fraction, same as the geometric/IMU prior. */
    bool radar_velocity_fusion = false;

    /** Maximum radar/LiDAR timestamp difference for a radar velocity prior. */
    double radar_prior_max_time_offset_sec = 0.15;

    /** Minimum ego speed (m/s) required to trust a radar velocity prior; 0 disables the gate. */
    double radar_min_speed = 0.0;

    /** Correct translation along the radar-observed direction when radar and
     *  ICP velocities persistently disagree. Catches fog clutter-lock, where
     *  aerosol returns travel with the sensor and ICP confidently reports
     *  near-zero motion with a well-conditioned Hessian, so the eigenvalue
     *  gates above never fire. Reuses degeneracy_prior_weight as the blend. */
    bool radar_disagreement_gate = false;

    /** Minimum radar-vs-ICP velocity gap treated as a disagreement (m/s). */
    double radar_disagreement_min_mps = 0.2;

    /** Consecutive disagreeing scans required before correction starts. */
    size_t radar_disagreement_min_scans = 10;

    /** Blend fraction toward the radar displacement once the disagreement
     *  gate is open. Unlike the Hessian-weak prior, clutter-locked ICP is
     *  actively wrong along this direction, so full replacement is the
     *  default. */
    double radar_disagreement_weight = 1.0;

    /** Continuous, information-weighted radar/ICP velocity fusion. Unlike radar_disagreement_gate
     *  (which needs a 10-scan disagreement streak, only corrects along the single radar-heading
     *  direction, and only above radar_disagreement_min_mps), this blends every scan with a
     *  fresh radar prior, per-axis, weighted by the relative confidence (information) of the
     *  ICP solve vs. the radar measurement in that world-frame axis -- see
     *  radar_ego_velocity.hpp's blend_icp_radar_velocity for the pure math. Fixes fog
     *  clutter-lock from scan 1 instead of after 10 scans of accumulated drift. When enabled,
     *  this subsumes radar_disagreement_gate for scans with a fresh radar prior: that gate's
     *  block is skipped entirely (see LIO::register_scan) rather than double-correcting. */
    bool radar_velocity_continuous_fusion = false;

    /** Multiplies the ICP translation information (H_tt / dt^2) before blending against the
     *  radar information. H is already averaged over correspondences by build_icp_linear_system,
     *  so its absolute scale does not correspond to a physical covariance; this factor lets the
     *  relative trust between ICP and radar be tuned empirically without touching the radar
     *  sigmas below. 1.0 preserves H's own scale unmodified. */
    double radar_fusion_icp_information_scale = 1.0;

    /** Sliding-window absolute gravity alignment (see gravity_alignment.hpp). Treats the
     *  long-window mean of raw accelerometer samples, rotated into the world frame, as an
     *  absolute up reference and feeds the residual tilt back as a small, capped,
     *  roll/pitch-only correction after each registration. Complements min_beta's per-scan
     *  regularization, whose Kalman-filter reference is built with the current rotation
     *  estimate and therefore tracks -- instead of observing -- slow pitch/roll drift. */
    bool gravity_window_alignment = false;
    /** Largest correction (deg) per scan the registration may make to the gyro-predicted
     *  roll and pitch (0: no limit). Yaw stays with the registration. A 16-beam LiDAR on a
     *  vehicle pulls the pitch by about 1 deg/s in tunnels and on bridges; the gyro holds
     *  it, and gravity_window_alignment corrects the absolute tilt. */
    double max_tilt_correction_deg = 0.0;
    /** Treat the start as moving, and leave the IMU biases at zero, when the mean gyro rate
     *  of the initialization interval exceeds this (rad/s; 0: always static). */
    double initialization_max_gyro_rad_s = 0.0;
    /** After a moving start (initialization_max_gyro_rad_s), estimate the world-frame gravity
     *  again from this many seconds of motion (0: off), starting 1 s after the initialization:
     *  the specific force rotated into the world frame, less the velocity change measured by the
     *  registered positions (estimate_world_up_while_moving). The 0.1-s initialization average of a
     *  walking start contains the gait's acceleration (GrandTour ARC-5: the world frame came out
     *  16 deg from level). The frame and the map are kept, so the trajectory stays continuous;
     *  only the gravity vector the filter, the propagation and the regularization use changes. */
    double moving_start_gravity_window_sec = 0.0;
    /** Give the gravity of moving_start_gravity_window_sec to the output filter only: the
     *  registration, the propagation and the regularization keep the initial gravity (and so give
     *  the same trajectory as without the window), and the filter's IMU acceleration takes the
     *  estimate, so that the output is filtered after a moving start as well. */
    bool moving_start_gravity_output_only = false;
    /** Filter the output positions (published odometry, dumped trajectory) with the IMU between
     *  scans: registered position noise in m (0: off; see output_position_filter.hpp). The
     *  registration chain and the map are unchanged. Its prediction needs the gravity direction:
     *  after a moving start the output stays unfiltered until moving_start_gravity_window_sec has
     *  estimated the gravity again, and for the whole run without it. */
    double output_filter_position_sigma = 0.0;
    /** White acceleration noise of the output filter's motion model (m^2/s^3). */
    double output_filter_acceleration_noise = 0.001;
    /** Random walk of the output filter's acceleration bias (m^2/s^5). */
    double output_filter_bias_noise = 1.0e-4;
    /** Record the output filter's inputs in output_filter_events (IMU world accelerations,
     *  registered positions, restarts), so the filter can be replayed offline with other settings.
     *  Changes no output. */
    bool output_filter_log = false;

    /** Hold a vehicle's speed while its registration stalls (see stall_guard.hpp). */
    bool vehicle_stall_guard = false;
    StallGuardConfig stall_guard;

    /** Sliding window length (s) for the world-frame accelerometer mean. Long enough that
     *  quasi-steady body acceleration averages out; short enough that the drift being
     *  measured is approximately constant across the window. */
    double gravity_window_sec = 20.0;

    /** Fraction of the measured window tilt corrected per registered scan. */
    double gravity_alignment_gain = 0.05;

    /** Hard cap (rad) on the correction angle applied per registered scan. */
    double gravity_alignment_max_correction_rad = 0.002;

    /** Reject a window whose mean magnitude deviates from |g| by more than this fraction
     *  (sustained acceleration, e.g. constant braking, would masquerade as tilt). */
    double gravity_alignment_max_magnitude_deviation = 0.05;

    /** Reject a window whose measured tilt exceeds this (rad). Permissive by default
     *  (45 deg): the per-scan correction cap is the primary safety mechanism, and on
     *  sequences with fast genuine attitude error (fog clutter-lock) large measured tilts
     *  still carry real signal. Tighten on rigs where large tilts are known-impossible. */
    double gravity_alignment_max_plausible_tilt_rad = 0.785;

    /** Do not add a scan interval to the window while the world-frame yaw rate exceeds this
     *  (rad/s): sustained turning adds centripetal acceleration that does not average out
     *  and would masquerade as tilt (measured: fog loop walk p50 yaw rate 0.10 rad/s vs.
     *  tunnel straight traversal p90 0.04 rad/s). During long turns the window drains and
     *  the alignment goes dormant instead of injecting a false correction. */
    double gravity_alignment_max_yaw_rate_rad_s = 0.05;

    /** Apply the gravity alignment as a rigid re-levelling of the local frame instead of
     *  rotating only the new pose: the pose, the previous pose, the local map and the window
     *  are all rotated about the current position. A pose-only correction disagrees with the
     *  still-tilted local map, so the next registrations pull most of it back. The full
     *  measured tilt is applied at once whenever it reaches gravity_alignment_relevel_min_tilt_rad
     *  (gravity_alignment_gain and gravity_alignment_max_correction_rad are unused). Each
     *  re-level rebuilds the local map once (~150 ms for 0.85 M points). The kidnap-recovery
     *  map is not re-levelled. */
    bool gravity_alignment_relevel_map = false;

    /** Measured window tilt (rad) that triggers a re-level when gravity_alignment_relevel_map
     *  is on. */
    double gravity_alignment_relevel_min_tilt_rad = 0.005;

    /** Accelerometer-consistency velocity gate (see kinematic_velocity_gate.hpp). Clamps the
     *  per-scan velocity change along the previous motion direction to the measured body
     *  acceleration plus kinematic_gate_accel_margin, breaking the geometric zero-motion
     *  attractor of self-similar corridors (an ICP freeze implies a physically impossible
     *  deceleration the accelerometer contradicts). Genuine stops pass: braking raises the
     *  measured acceleration and with it the allowed change. */
    bool kinematic_velocity_gate = false;

    /** Slack (m/s^2) added to the axis-projected measured acceleration when bounding the
     *  per-scan velocity change (covers bias and accelerometer noise). Only the projection
     *  of the measured acceleration onto the motion direction opens the bound: vibration is
     *  mostly perpendicular to travel and must not loosen it. */
    double kinematic_gate_accel_margin = 0.3;

    /** Below this previous speed (m/s) the motion direction is untrustworthy and the gate
     *  stays out of the way (also lets the rig accelerate from rest). */
    double kinematic_gate_min_speed = 0.3;

    /** Fraction of the computed clamp correction actually applied (1 = full). */
    double kinematic_gate_weight = 1.0;

    /** Skip the local-map insertion on scans the gate corrected. Without this the glided
     *  scans enter the map and the next ICP anchors to its own correction (self-confirming
     *  feedback: measured 3.1x runaway overshoot on the NTNU tunnel). With it, ICP keeps
     *  disagreeing against the stale map while the gate bridges the degenerate stretch, and
     *  re-anchors as soon as genuinely informative structure comes back into range. */
    bool kinematic_gate_skip_map_update = true;

    /** Soft information-weighted fusion of ICP velocity and IMU propagation along the
     *  previous direction of travel. Unlike kinematic_velocity_gate, the propagation
     *  authority decays since the last scan where ICP and propagation agreed. */
    bool kinematic_velocity_blend = false;

    /** Fixed isotropic ICP velocity information scale. The point-to-point translation
     *  Hessian is intentionally not used because its N*I block hides tunnel degeneracy. */
    double kinematic_blend_icp_information_scale = 1.0;

    /** Initial rank-one IMU propagation information along the motion axis. */
    double kinematic_blend_propagation_information_scale = 9.0;

    /** Exponential time constant (s) for propagation information after the last anchor. */
    double kinematic_blend_decay_time_sec = 10.0;

    /** Maximum 3D ICP/propagation velocity difference (m/s) that refreshes the
     *  anchor. Agreement scans are left bit-identical to the ICP result. */
    double kinematic_blend_anchor_agreement_mps = 0.3;

    /** Huber-style norm cap (m/s) on the 3D ICP innovation before fusion.
     *  Prevents an arbitrarily large ICP outlier leaking through its finite
     *  information weight. Set <= 0 to disable robustification. */
    double kinematic_blend_max_icp_innovation_mps = 2.0;

    /** Maximum speed (m/s) of the independently integrated IMU pseudo-sensor.
     *  This caps only prior authority, not the final ICP pose. Set <= 0 to disable. */
    double kinematic_blend_max_propagated_speed_mps = 3.0;

    /** Clear the anchor when the previous LiDAR velocity exceeds this application
     *  envelope. Prevents a walking-speed preset from engaging on a driving rig.
     *  Set <= 0 to disable. */
    double kinematic_blend_max_activation_speed_mps = 0.0;

    /** Consecutive above-envelope scans required before the activation-speed
     *  gate clears the anchor. Rejects sustained driving without reacting to
     *  isolated walking-speed ICP noise. */
    std::size_t kinematic_blend_speed_gate_min_scans = 1;

    /** Time the platform speed must remain inside the activation envelope
     *  after a persistent rejection before the bridge may re-enable. */
    double kinematic_blend_speed_reenable_delay_sec = 0.0;

    /** Below this previous speed (m/s), clear the anchor and leave startup to ICP. */
    double kinematic_blend_min_speed = 0.3;

    /** Minimum within-scan raw acceleration-magnitude variance ((m/s^2)^2)
     *  required to treat a low LiDAR speed as an ICP lock rather than a real
     *  stop. Set <= 0 to disable activity-based low-speed bridging. */
    double kinematic_blend_min_accel_magnitude_variance = 0.0;

    /** Consecutive inactive scans required before clearing the prior. */
    std::size_t kinematic_blend_inactivity_gate_min_scans = 1;

    /** Suspend translation correction above this absolute world yaw rate
     *  (rad/s) and rotate the retained prior by the observed orientation
     *  change. Set <= 0 to disable the yaw suspension. */
    double kinematic_blend_max_yaw_rate_rad_s = 0.05;

    /** Consecutive above-threshold scans required before yaw disables the blend. */
    std::size_t kinematic_blend_yaw_gate_min_scans = 5;

    /** Require a tunnel-like scan range distribution before the inertial bridge
     *  can anchor or correct. Intended to reject near-field fog/clutter and
     *  close-range handheld sequences. */
    bool kinematic_blend_range_scene_gate = false;

    /** A return below this range counts as near-field scene support. */
    double kinematic_blend_scene_near_range_m = 3.0;

    /** Maximum fraction of valid returns allowed inside the near range. */
    double kinematic_blend_scene_max_near_fraction = 0.5;

    /** A return above this range counts as long-range structural support. */
    double kinematic_blend_scene_far_range_m = 10.0;

    /** Minimum fraction of valid returns required beyond the far range. */
    double kinematic_blend_scene_min_far_fraction = 0.05;

    /** Minimum valid scan points required to evaluate the range scene gate. */
    std::size_t kinematic_blend_scene_min_valid_points = 100;

    /** Time a structurally trusted scene must remain clear after a range-scene
     *  rejection before the inertial bridge may re-enable. */
    double kinematic_blend_scene_reenable_delay_sec = 0.0;

    /** Suppress map insertion only when propagation weight exceeds this threshold.
     *  1.0 keeps every scan (the default); values below 1 enable the map-policy A/B. */
    double kinematic_blend_map_update_max_propagation_weight = 1.0;

    /** Minimum fraction of a propagation-corrected scan inserted into the map.
     *  1.0 keeps the full scan; lower values thin it in proportion to
     *  (1 - propagation_weight), while retaining this safety floor. */
    double kinematic_blend_map_update_min_fraction = 1.0;

    /** Localizability-aware ICP correspondence weighting (X-ICP-inspired; see
     *  localizability_weighting.hpp). Multiplies every correspondence by
     *  1 + localizability_boost * (n . axis)^2, where n is a cached per-voxel map
     *  surface normal and axis is the current motion direction. Amplifies the small
     *  minority of axis-observing surfaces (door frames, niches: measured 1.3-7.8% of
     *  planar points on the NTNU tunnel) that soft along-corridor degeneracy otherwise
     *  drowns under the axis-neutral wall majority. Also enables per-voxel normal
     *  maintenance in the local map (skipped entirely when off). */
    bool localizability_weighting = false;

    /** Correspondence weight boost for a fully axis-aligned surface normal. */
    double localizability_boost = 30.0;

    /** Minimum predicted per-scan translation (m) required to trust a motion axis;
     *  below this the weighting is skipped for the scan (a near-stationary rig has
     *  no trustworthy along-track direction). */
    double localizability_min_step_m = 0.05;

    /** Boost only the residual component along the matched map normal (W = I + (w - 1) n n^T)
     *  instead of the whole point-to-point residual: the along-surface part, mostly the spacing
     *  of the map points, keeps its unit weight. */
    bool localizability_normal_only = false;

    /** Correct translation along a confirmed persistent weak direction using scan-to-scan
     *  cross-correlation of a 1D reflectivity/intensity profile built along that axis.
     *  Recovers along-tunnel translation that geometric ICP under-observes in self-similar
     *  environments. Requires degeneracy_aware_solve and degeneracy_persistence_gate; reuses
     *  degeneracy_prior_weight as the blend fraction. Only fires when no radar velocity prior
     *  is active (radar has priority; see LIO::register_scan). */
    bool intensity_constraint = false;

    /** Reflectivity profile bin width along the weak-direction axis (m). */
    double intensity_bin_size_m = 0.25;

    /** Reflectivity profile half-length around the reference scan's centroid (m). */
    double intensity_profile_half_length_m = 30.0;

    /** Maximum scan-to-scan shift searched by the profile cross-correlation (m). */
    double intensity_max_shift_m = 1.5;

    /** Minimum peak normalized cross-correlation required to trust a measured shift. */
    double intensity_min_correlation = 0.6;

    /** Minimum correlation margin between the best shift and the strongest
     *  non-adjacent shift. Zero preserves the historic best-peak-only gate. */
    double intensity_min_peak_margin = 0.0;

    /** Discrete bins around the best shift treated as the same peak when
     *  searching for a competing correlation hypothesis. */
    size_t intensity_peak_exclusion_radius_bins = 1;

    /** Minimum populated profile bins required for both scans in a correlation attempt. */
    size_t intensity_min_filled_bins = 40;

    /** Replace the 1D profile inside intensity_disagreement_gate with a
     *  longitudinal/lateral reflectivity+height grid. Default-off. */
    bool intensity_oriented_grid = false;

    /** Grid half-width perpendicular to the current motion axis (m). */
    double intensity_grid_half_width_m = 5.0;

    /** Maximum lateral shift searched by the oriented grid matcher (m). */
    double intensity_grid_max_lateral_shift_m = 0.5;

    /** Relative weight of the grid height channel; intensity weight is one. */
    double intensity_grid_height_weight = 0.25;

    /** Correct translation along the current motion direction when the reflectivity-profile-
     *  implied velocity and the ICP velocity persistently disagree. Unlike intensity_constraint
     *  above (which requires degeneracy_aware_solve + a confirmed Hessian-weak direction),
     *  this gate needs neither: it tracks the scan's own ICP motion axis every scan, so it
     *  also catches "soft" degeneracy that never triggers the persistence gate. Mirrors
     *  radar_disagreement_gate's structure/lesson (gate on sensor-vs-ICP consistency) but uses
     *  the LiDAR's own reflectivity texture as the "sensor" instead of radar Doppler, so it
     *  works on radar-less rigs. Reuses the intensity_* profile settings above. Radar has
     *  priority: skipped on scans radar_disagreement_gate already corrected.
     */
    bool intensity_disagreement_gate = false;

    /** Minimum intensity-vs-ICP velocity gap treated as a disagreement (m/s). */
    double intensity_disagreement_min_mps = 0.2;

    /** Consecutive disagreeing scans required before correction starts. */
    size_t intensity_disagreement_min_scans = 10;

    /** Blend fraction toward the intensity-implied displacement once the disagreement gate is
     *  open. Defaults to full replacement, matching radar_disagreement_weight's reasoning. */
    double intensity_disagreement_weight = 1.0;

    /** Maximum delta between adjacent LiDAR scan timestamps (s).
     *  Frames whose stamp is further than this from the previous LiDAR
     *  state time are dropped. Default 1.0 preserves the historic check;
     *  raise it to tolerate kidnap-style recordings with longer scan gaps. */
    double max_scan_delta_sec = 1.0;

    /** Enable recovery after kidnap-style ICP failures. */
    bool enable_kidnap_relocalization = false;

    /** If relocalization fails, start a new local map at the last known pose. */
    bool reset_on_registration_failure = false;

    /** Consecutive registration failures required before recovery is attempted. */
    int recovery_min_failures = 1;

    /** Try global relocalization at the first valid scan after dropped scans. */
    bool relocalize_after_scan_gap = false;

    /** Minimum correspondences required for a relocalization candidate. */
    int relocalization_min_correspondences = 30;

    /** Minimum inlier ratio required for a relocalization candidate. */
    double relocalization_min_inlier_ratio = 0.10;

    /** Maximum accepted mean nearest-neighbor error for relocalization. */
    double relocalization_max_mean_error = 1.5;

    /** ICP correspondence distance used only during global relocalization. */
    double relocalization_max_correspondence_distance = 2.0;

    /** Number of coarse yaw hypotheses to evaluate around each historical pose. */
    int relocalization_yaw_samples = 24;

    /** Historical pose stride for global relocalization candidates. */
    int relocalization_pose_stride = 10;

    /** Recent historical poses to skip when relocalizing. */
    int relocalization_min_pose_separation = 50;

    /** Maximum ICP iterations for each relocalization hypothesis. */
    int relocalization_max_iterations = 15;

    /** Add photometric terms from LiDAR intensity images to the ICP solve (COIN-LIO style,
     *  see photometric_features.hpp). Needs organized scans matching photometric_model and
     *  per-point intensities; register through the extrinsic overload. Default off. */
    bool photometric = false;

    /** Weight of one photometric residual (per filtered intensity unit) relative to a
     *  point-to-point ICP residual in metres. About three times COIN-LIO's point-to-plane
     *  value: point-to-point correspondences also hold the pose along the weak axis. */
    double photometric_scale = 0.003;

    /** Patches are added along translation directions with fewer than this many aligned
     *  surface normals (COIN-LIO's n_uninformative). */
    double photometric_weak_direction_min_contribution = 25.0;

    /** Grid step (rows, columns) of the surface normals used to find weak directions. */
    int photometric_normal_row_step = 4;
    int photometric_normal_col_step = 8;

    /** Leave translation along the weak directions to the photometric terms: the
     *  point-to-point ICP information along them is projected out whenever patches
     *  contribute (experimental). */
    bool photometric_free_weak_axis = false;

    /** Skip the photometric terms on a scan that follows a gap in the LiDAR stream longer
     *  than this (s): the prediction across the gap is too coarse for 5x5 patches, and a
     *  wrong patch association can pull the pose far off (ENWIDE TunnelD lost 13 deg of
     *  yaw across a 0.28 s gap). Patches are still tracked and refreshed on that scan. */
    double photometric_max_scan_interval_sec = 0.15;

    LidarImageModel photometric_model;
    IntensityImageConfig photometric_image;
    PhotometricFeatureConfig photometric_features;

    /** Register against voxel-wise height ("bump") images instead of point-to-point
     *  (fork addition, default off; see bump_image_map.hpp). The bump map holds every
     *  deskewed point; registration uses a `bump_image_source_voxel_size` downsample of
     *  the scan, keeping all points in the `bump_image_informed_voxels` map voxels with
     *  the most relief. */
    bool bump_image_registration = false;
    BumpImageMapConfig bump_image_map;
    double bump_image_source_voxel_size = 0.1;
    int bump_image_informed_voxels = 300;
    double bump_image_huber_delta = 0.1;
    /** Add the point-to-point system of the ICP keypoints to the bump system with this weight,
     *  each averaged over its own residuals (0 keeps the bump terms alone). The informed voxels
     *  are the map voxels with the most relief; in an open scene they lie mostly on the ground
     *  near the sensor, and the registration slips along it (GrandTour ARC-3 at 307 s, BIEVR-LIO
     *  in the same place), while the keypoints keep the far structure that holds the translation. */
    double bump_image_point_to_point_weight = 0.0;
    /** With point-to-point terms (weight above 0), take the vertical translation from them
     *  alone: the world z translation is marginalised out of the bump system (Schur complement)
     *  before the two are added. On stairs and slopes the bump images pull the height (Oxford Keble 05:
     *  0.10 m off while climbing 4 m, against 0.02 m point-to-point), while they hold the
     *  horizontal and the attitude better. */
    bool bump_image_vertical_from_point_to_point = false;
    /** Add the point-to-point terms only once the registered pose has been this far (m) from the
     *  first pose (0: from the first scan). While a legged robot stands at the start on an open
     *  field, the people walking around it are most of what the keypoints see, and the
     *  point-to-point terms follow them (GrandTour SNOW-3: 0.089 m with them from the start,
     *  0.027 m with the bump terms alone); the bump terms keep the ground's relief. */
    double bump_image_point_to_point_start_distance = 0.0;
    /** Register a scan again without the bump terms when they turn the pose more than
     *  this (deg) away from the IMU prediction (0 disables). In a circular tunnel the roll
     *  about the axis is unobservable and the bump images slip around it by degrees per
     *  scan, while a correctly registered scan stays within about 1.6 deg (GEODE, ENWIDE). */
    double bump_image_max_rotation_correction_deg = 0.0;
    /** Fall back only when at least ``bump_image_rotation_fallback_min_count`` of the last
     *  ``bump_image_rotation_fallback_window`` scans (this one included) exceeded the
     *  correction limit, so an isolated large correction that was right keeps its bump
     *  result. 1 / 1 falls back on every exceeding scan. */
    int bump_image_rotation_fallback_window = 1;
    int bump_image_rotation_fallback_min_count = 1;
    /** Above this correction (deg) a scan falls back at once, whatever the window says
     *  (0 disables). */
    double bump_image_rotation_hard_limit_deg = 0.0;

    /** Close-scene settings. When close_scene_max_median_range is positive, a scan whose
     *  keypoints lie at a median distance below it (m) from the sensor is registered with the
     *  close_scene_* keys below in place of the corresponding settings (a key at 0 keeps the
     *  setting); every other scan uses the configuration as it is. On the development data,
     *  rooms, garages and corridors lie below 9.1 m and open outdoor scenes above 13 m. */
    double close_scene_max_median_range = 0.0;
    double close_scene_max_correspondence_distance = 0.0;
    /** A positive boost turns the localizability weighting on for close scenes. */
    double close_scene_localizability_boost = 0.0;
    bool close_scene_localizability_normal_only = false;
    double close_scene_icp_robust_kernel_scale_factor = 0.0;
    /** icp_tangential_weight for close scenes; a negative value keeps icp_tangential_weight. */
    double close_scene_icp_tangential_weight = -1.0;

    /** When positive, a registered scan is added to the map only if the sensor has moved at
     *  least this far (m) since the last scan that was added. A sensor standing still adds the
     *  same sampling pattern again and again: the voxels fill up (max_points_per_voxel) with the
     *  rings of a spinning LiDAR, and the scans after the start of motion, which sample between
     *  those rings, can no longer add points there. 0 adds every scan. */
    double map_update_min_translation = 0.0;

    /** When positive, every scan interval while the sensor is still within this distance (m)
     *  of where it started, and whose IMU shows no rotation (mean gyro below
     *  initialization_max_gyro_rad_s) and no shaking (standard deviation of the acceleration
     *  magnitude below start_standstill_max_accel_std), re-estimates the IMU biases from that
     *  interval and holds the velocity at zero. A legged robot often settles for a moment and
     *  then stands still before it walks off: the initialization interval sees the settling,
     *  leaves the biases at zero, and in a scene that does not constrain the horizontal (a
     *  flat snow field seen by a narrow-FOV LiDAR) the IMU then drifts while it stands. 0 off. */
    double start_standstill_radius = 0.0;
    double start_standstill_max_accel_std = 0.5;
    /** With start_standstill_radius: a standing-still interval also keeps the position (the scan
     *  is not registered: around a robot standing in an open field, people walking past can be
     *  the only things that constrain the horizontal, and the registration follows them); the
     *  attitude follows the gyro. */
    bool start_standstill_hold = false;
    /** With start_standstill_hold: 0 keeps the position of a standing interval; a positive radius
     *  (m) registers the scan and keeps the position within this distance of where the robot
     *  started standing (a legged robot shifts by a few centimetres while it stands, which a
     *  well-constrained registration follows; people walking past pull it further). Standing scans
     *  are not added to the map either way. */
    double start_standstill_hold_radius = 0.0;
    /** Mean gyro rate (rad/s) below which an interval counts as standing still (0: use
     *  initialization_max_gyro_rad_s). */
    double start_standstill_max_gyro_rad_s = 0.0;
    /** The standstill handling ends this long (s) after the first scan, whatever the IMU says. */
    double start_standstill_max_sec = 60.0;
    /** After a moving start (initialization_max_gyro_rad_s), once the standing-still intervals
     *  near the start (start_standstill_radius) add up to this long (s), level the world frame
     *  with their mean specific force and start the maps again where the sensor stands (0: off).
     *  The first interval of a walking start holds the gait's acceleration and leaves the frame
     *  tilted (a GrandTour mission: 16 deg); a standing robot measures gravity alone. Until then
     *  the standing intervals do not re-estimate the accelerometer bias (it would absorb the
     *  tilt). The sensor has not moved, so the trajectory stays continuous; the output filter
     *  (output_filter_position_sigma) starts once the frame is level. */
    double start_standstill_level_sec = 0.0;
    /** How a standing interval near the start sets the IMU biases: "interval" (each interval
     *  replaces them with its own means; the default), "average" (the mean over every standing
     *  interval so far: a legged robot sways while it stands, and one 0.1-s interval takes the
     *  sway for a gyro bias of several mrad/s, far above a tactical-grade IMU's), or "none"
     *  (the biases are left as they are; the velocity is still held at zero). */
    std::string start_standstill_bias = "interval";
    /** Apply start_standstill_bias only after a static start; after a moving start the standing
     *  intervals leave the biases as they are (its frame is tilted by the walking initialization,
     *  and the bias update would take that tilt for an accelerometer bias in the body frame, wrong
     *  once the robot turns). */
    bool start_standstill_bias_static_only = false;
    /** Let start_standstill_bias set the accelerometer bias too. When false the standing intervals
     *  set only the gyro bias, and the accelerometer bias stays as the initialization left it
     *  (along gravity): a standing interval's specific force against gravity in the registered
     *  attitude takes the attitude's error (0.1 deg: 0.017 m/s^2) for a horizontal bias in the
     *  body frame, which turns with the robot. */
    bool start_standstill_accel_bias = true;
    /** When positive, a close scan whose registration with the close-scene settings leaves
     *  fewer than this fraction of its keypoints within their correspondence cap is
     *  registered again: first with the base settings, then with the close-scene settings
     *  starting from that result. */
    double close_scene_min_inlier_fraction = 0.0;
    /** Record per-scan registration statistics that use no ground truth (see
     *  scan_statistics.hpp): keypoint-to-map distances and the translation information of the
     *  final ICP system. Analysis only; the estimate is unchanged. */
    bool scan_statistics = false;
    /** Distance up to which scan_statistics searches a keypoint's nearest map point (m). */
    double scan_statistics_search_distance = 0.75;
  };

  /** Configuration parameters. */
  Config config;

  /** Local map. */
  VoxelHashMap map;
  BumpImageMap bump_map;

  /** Global sparse map used for kidnap relocalization. This map is never pruned. */
  VoxelHashMap relocalization_map;

  /** Current LiDAR state estimate. */
  State lidar_state;

  /** IMU bias estimates when initialization is enabled. */
  ImuBias imu_bias;

  /** Mean body acceleration estimate. */
  Eigen::Vector3d mean_body_acceleration = Eigen::Vector3d::Zero();

  /** Covariance of body acceleration estimate. */
  Eigen::Matrix3d body_acceleration_covariance = Eigen::Matrix3d::Identity();

  /** IMU measurement statistics since last LiDAR frame. */
  IntervalStats interval_stats;

  /** Scans whose predicted speed came from the gait (gait_speed_prior). */
  std::size_t gait_speed_applied_count = 0;
  std::optional<double> gait_step_length() const { return _gait.step_length(); }

  /** Inertial window solves (inertial_window). */
  std::size_t inertial_window_solve_count = 0;
  const InertialWindow& inertial_window() const { return _inertial_window; }


  explicit LIO(const Config& config_);

  /** Whether a seeded pose has a finalized world orientation for publication. */
  bool has_initialized_pose() const {
    return lidar_state.time > Nsec{0} && (!config.initialization_phase || _initialized);
  }

  /** Add an IMU measurement expressed in the base frame. */
  void add_imu_measurement(const ImuControl& base_imu);

  /**
   * Add an IMU measurement expressed in the IMU frame and transform it
   * to the base frame using the given extrinsic calibration.
   * @param extrinsic_imu2base Extrinsic transform from IMU to base frame.
   * @param raw_imu Raw IMU measurement.
   */
  void add_imu_measurement(const Sophus::SE3d& extrinsic_imu2base, const ImuControl& raw_imu);

  /**
   * Predict world<-base at a timestamp using the same interval-averaged IMU
   * motion model as the LiDAR ICP initial guess.  This is read-only and is
   * used to time-align camera frames before the scan consumes the interval.
   */
  Sophus::SE3d predict_pose_at(const Nsec& time) const;

  /** Set a one-shot world<-base visual pose prior for the next LiDAR scan. */
  void set_visual_pose_prior(const VisualPosePrior& prior) { _visual_pose_prior = prior; }

  /** Remove any pending visual prior. */
  void clear_visual_pose_prior() { _visual_pose_prior.reset(); }

  /** Set a one-shot base-frame radar ego-velocity prior for the next LiDAR scan. */
  void set_radar_velocity_prior(const RadarVelocityPrior& prior) { _radar_velocity_prior = prior; }

  /** Remove any pending radar velocity prior. */
  void clear_radar_velocity_prior() { _radar_velocity_prior.reset(); }

  /**
   * Register a LiDAR scan, applying deskewing based on the initial motion guess
   * and clipping points beyond valid range.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @param intensities Optional per-point reflectivity/intensity, same size and order as
   *   `scan`. Only consumed when config.intensity_constraint is set; nullptr preserves the
   *   exact legacy behavior.
   * @return Deskewed and clipped point cloud.
   */
  Vector3dVector register_scan(const Vector3dVector& scan,
                               const TimestampVector& timestamps,
                               const std::vector<float>* intensities = nullptr);

  /**
   * Register a LiDAR scan for which the extrinsic calibration from lidar to base
   * has already been applied.
   * @param extrinsic_lidar2base Extrinsic from lidar to base frame.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @param intensities Optional per-point reflectivity/intensity, same size and order as
   *   `scan`. See the other overload.
   * @return Deskewed and clipped scan in the original lidar frame.
   */
  Vector3dVector register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                               const Vector3dVector& scan,
                               const TimestampVector& timestamps,
                               const std::vector<float>* intensities = nullptr);

  /** Sequence of registered scan poses with corresponding timestamps. */
  std::vector<std::pair<Nsec, Sophus::SE3d>> poses_with_timestamps;
  /** The output trajectory when output_filter_position_sigma is set (the registered attitude and
   *  the filtered position, one entry per entry of poses_with_timestamps); empty otherwise. */
  std::vector<std::pair<Nsec, Sophus::SE3d>> output_poses_with_timestamps;
  /** One input of the output filter (output_filter_log): 'P' propagate to `time` with world
   *  acceleration `a`; 'U' update with the registered position `a`; 'R' restart at position `a`
   *  with velocity `b`. */
  struct OutputFilterEvent {
    char kind = 'P';
    double time = 0.0;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
  };
  std::vector<OutputFilterEvent> output_filter_events;

  /** Base-frame state propagated from IMU measurements. Runs ahead of `lidar_state` between scans; reset to the
   *  optimized lidar pose after each successful registration. Used for gravity compensation and for publishing
   *  IMU-rate odometry from the ROS wrapper.
   */
  State imu_state;

  /** Opt-in per-scan persistence-gate diagnostics, populated only when enabled. */
  std::vector<DegeneracyPersistenceDiagnosticsSample> degeneracy_persistence_diagnostics;

  /** Aggregate artifact-driven visual-fusion diagnostics. */
  std::size_t visual_prior_attempt_count = 0;
  std::size_t visual_fused_scan_count = 0;
  std::size_t visual_fused_direction_count = 0;
  std::size_t visual_unobservable_direction_count = 0;
  std::vector<VisualObservabilityDiagnosticsSample>
      visual_observability_diagnostics;

  /** Photometric registration diagnostics: scans with an intensity image, scans whose
   *  ICP used photometric patches, and the summed patch count over those scans. */
  std::size_t photometric_frame_count = 0;
  std::size_t photometric_scan_count = 0;
  std::size_t photometric_patch_sum = 0;
  /** Scans whose photometric terms were skipped after a LiDAR gap. */
  std::size_t photometric_gap_skip_count = 0;
  /** Scans registered again without the bump terms (bump_image_max_rotation_correction_deg). */
  std::size_t bump_rotation_fallback_count = 0;
  InertialWindow _inertial_window;
  GaitSpeedEstimator _gait;
  std::optional<Eigen::Vector3d> _gait_direction_body;
  bool _gait_prediction_active = false;
  /** World velocity change from the IMU since the last scan, and per scan: time, world
   *  velocity after the scan, velocity change during the scan, sensor attitude. */
  Eigen::Vector3d _gait_velocity_change = Eigen::Vector3d::Zero();
  std::deque<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d, Sophus::SO3d>> _gait_velocity_history;
  double _gait_last_reversal_time = -1e9;
  /** The last gait speed (m/s) from the learned step. */
  std::optional<double> _gait_last_speed;
  /** Scans where the IMU reversed the direction of travel. */
  std::size_t gait_reversal_count = 0;
  /** Scans whose along-travel correction the gait limited. */
  std::size_t gait_correction_limited_count = 0;
  /** Scans held through a gap in the cadence (gait_hold_without_cadence), and how many of
   *  them stood. */
  std::size_t gait_hold_count = 0;
  std::size_t gait_standing_count = 0;
  ImuPreintegration _window_preintegration;
  Eigen::Vector3d _window_last_angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d _window_last_specific_force = Eigen::Vector3d::Zero();
  void reset_inertial_window_interval() { _window_preintegration = ImuPreintegration{}; }
  /** Per-scan (time [s], bump rotation correction [deg]) while the limit is active. */
  std::vector<std::pair<double, double>> bump_rotation_corrections;
  /** One row per registered scan while Config::scan_statistics is on. */
  std::vector<ScanStatistics> scan_statistics_rows;

  /** Tracked photometric patches. */
  const PhotometricFeatureManager& photometric_features() const { return _photometric_features; }

  /** Aggregate radar-velocity-fusion diagnostics, mirroring the visual counters. */
  std::size_t radar_prior_attempt_count = 0;
  std::size_t radar_fused_scan_count = 0;
  std::size_t radar_disagreement_corrected_scan_count = 0;

  /** Continuous information-weighted radar fusion diagnostics (radar_velocity_continuous_fusion).
   *  attempt_count: a fresh radar prior was available and dt was usable. fused_scan_count: the
   *  information-weighted blend solved successfully and a correction was applied.
   *  correction_magnitude_sum / fused_scan_count gives the mean absolute correction magnitude
   *  (m) applied per fused scan. */
  std::size_t radar_continuous_attempt_count = 0;
  std::size_t radar_continuous_fused_scan_count = 0;
  double radar_continuous_correction_magnitude_sum = 0.0;

  /** Aggregate intensity-constraint diagnostics, mirroring the radar counters.
   *  attempt_count: a confirmed weak direction plus a stored reference profile
   *  were available to correlate against. applied_count: the correlation
   *  cleared the min_correlation/min_filled_bins gates and a prior pose was
   *  actually built (still subject to radar priority and the degeneracy-aware
   *  solve's own blend). */
  std::size_t intensity_prior_attempt_count = 0;
  std::size_t intensity_prior_applied_count = 0;

  /** Scans where the intensity-vs-ICP velocity disagreement gate applied a correction. */
  std::size_t intensity_disagreement_corrected_scan_count = 0;

  /** Sliding-window gravity alignment diagnostics (gravity_window_alignment).
   *  attempt_count: window was full enough to evaluate. applied_count: the
   *  magnitude gate passed and a (possibly identity) correction was applied.
   *  last_tilt_rad: most recent measured window tilt. correction_rad_sum /
   *  applied_count gives the mean correction angle per applied scan. */
  /** Speed hold for a stalled vehicle registration (vehicle_stall_guard). */
  StallGuard stall_guard;
  /** Scans whose roll/pitch correction hit max_tilt_correction_deg. */
  std::size_t tilt_correction_limited_count = 0;
  std::size_t gravity_alignment_attempt_count = 0;
  std::size_t gravity_alignment_applied_count = 0;
  double gravity_alignment_last_tilt_rad = 0.0;
  double gravity_alignment_correction_rad_sum = 0.0;
  /** Angle (deg) between the re-estimated gravity and the initial world z
   *  (moving_start_gravity_window_sec); 0 until it is estimated. */
  double moving_start_gravity_tilt_deg = 0.0;
  /** Tilt (deg) removed by start_standstill_level_sec; 0 until the frame is levelled. */
  double start_level_tilt_deg = 0.0;

  /** Diagnostic-only counters for the intensity-vs-ICP velocity disagreement gate: how many
   *  scans had a stored previous profile to correlate against, how many of those produced a
   *  trustworthy (valid, non-saturated) shift, and how many of those exceeded the disagreement
   *  threshold (streak increments). Cheap to keep always; helps tell "gate never gets a
   *  measurement" apart from "measurements agree with ICP" apart from "streak never sustains".
   */
  std::size_t intensity_disagreement_attempt_count = 0;
  std::size_t intensity_disagreement_valid_shift_count = 0;
  std::size_t intensity_ambiguous_shift_count = 0;
  std::size_t intensity_peak_margin_sample_count = 0;
  double intensity_peak_margin_sum = 0.0;
  double intensity_peak_margin_min = std::numeric_limits<double>::infinity();
  std::vector<IntensityPeakDiagnostic> intensity_peak_diagnostics;
  std::size_t intensity_disagreement_exceeded_threshold_count = 0;

  /** Localizability-weighting diagnostics (localizability_weighting).
   *  attempt_count: weighting was enabled and a trustworthy motion axis existed
   *  this scan. weighted_scan_count: at least one correspondence actually
   *  received a boosted weight from an axis-observing map normal.
   *  boosted_fraction_sum / weighted_scan_count gives the mean fraction of
   *  correspondences boosted per weighted scan (final ICP iteration). */
  std::size_t localizability_attempt_count = 0;
  /** Registered scans that used the close-scene settings (close_scene_max_median_range). */
  std::size_t close_scene_scan_count = 0;
  std::size_t close_scene_fallback_count = 0;
  /** Registered scans not added to the map because the sensor had not moved enough
   *  (map_update_min_translation). */
  std::size_t map_update_skipped_count = 0;
  /** Scan intervals treated as standing still near the start (start_standstill_radius). */
  std::size_t start_standstill_count = 0;
  /** Scans whose keypoints were taken again at the finer voxel (keypoint_shortfall_voxel_multiplier). */
  std::size_t keypoint_shortfall_scan_count = 0;
  std::size_t localizability_weighted_scan_count = 0;
  double localizability_boosted_fraction_sum = 0.0;

  /** Kinematic velocity gate diagnostics (kinematic_velocity_gate). attempt_count: gate was
   *  enabled and IMU data existed for the interval. corrected_scan_count: the clamp fired.
   *  correction_m_sum / corrected_scan_count gives the mean applied correction (m). */
  std::size_t kinematic_gate_attempt_count = 0;
  std::size_t kinematic_gate_corrected_scan_count = 0;
  double kinematic_gate_correction_m_sum = 0.0;

  /** Soft kinematic velocity blend diagnostics. */
  std::size_t kinematic_blend_attempt_count = 0;
  std::size_t kinematic_blend_anchor_refresh_count = 0;
  std::size_t kinematic_blend_corrected_scan_count = 0;
  double kinematic_blend_correction_m_sum = 0.0;
  double kinematic_blend_propagation_weight_sum = 0.0;
  double kinematic_blend_max_propagation_weight = 0.0;
  double kinematic_blend_max_anchor_age_sec = 0.0;
  std::size_t kinematic_blend_scene_evaluation_count = 0;
  std::size_t kinematic_blend_scene_valid_count = 0;
  std::size_t kinematic_blend_scene_rejected_scan_count = 0;
  std::size_t kinematic_blend_scene_cooldown_rejected_scan_count = 0;
  std::size_t kinematic_blend_speed_rejected_scan_count = 0;
  std::size_t kinematic_blend_speed_cooldown_rejected_scan_count = 0;
  std::size_t kinematic_blend_low_speed_rejected_scan_count = 0;
  std::size_t kinematic_blend_yaw_rejected_scan_count = 0;
  std::size_t kinematic_blend_anchor_expiration_count = 0;
  std::size_t kinematic_blend_propagated_speed_clamp_count = 0;
  std::size_t kinematic_blend_invalid_result_count = 0;
  std::size_t kinematic_blend_activity_retained_low_speed_scan_count = 0;
  std::size_t kinematic_blend_inactivity_rejected_scan_count = 0;
  double kinematic_blend_scene_near_fraction_sum = 0.0;
  double kinematic_blend_scene_far_fraction_sum = 0.0;
  double kinematic_blend_max_disagreement_mps = 0.0;
  double kinematic_blend_max_correction_m = 0.0;
  double kinematic_blend_first_correction_time_sec = -1.0;
  double kinematic_blend_last_correction_time_sec = -1.0;
  double kinematic_blend_last_anchor_refresh_time_sec = -1.0;

private:
  GyroDeskewHistory gyro_deskew_history;

  /**
   * Initialize internal odometry state using the given lidar timestamp.
   * @param lidar_time Current lidar timestamp.
   */
  void initialize(const Nsec lidar_time);

  /** First-scan path: stamps state, optionally seeds the map, logs the pose. */
  Vector3dVector bootstrap_first_scan(const Vector3dVector& scan, const Nsec current_lidar_time);

  /** Average body acceleration and angular velocity over the IMU interval, with init-phase and no-IMU fallbacks. */
  std::pair<Eigen::Vector3d, Eigen::Vector3d> motion_priors_from_imu(const Nsec current_lidar_time);

  /** Register a recovery scan at a chosen pose and start a fresh local map. */
  Vector3dVector recover_with_scan(const Vector3dVector& filtered_frame,
                                   const Vector3dVector& map_update_frame,
                                   const Nsec& current_lidar_time,
                                   const Sophus::SE3d& recovery_pose,
                                   const std::string& reason);

  /** Drop an unusable scan while advancing the internal LiDAR timestamp. */
  Vector3dVector drop_failed_scan(const Nsec& current_lidar_time, const std::string& reason);

  /** Try to align the current scan against the unpruned relocalization map. */
  std::optional<Sophus::SE3d> try_global_relocalization(const Vector3dVector& keypoints) const;

  /** Update the sliding local map and, when enabled, the unpruned recovery map. */
  /** `full_frame` is the deskewed scan for the bump map; it is ignored unless bump image
   *  registration is enabled. */
  void update_maps(const Vector3dVector& map_update_frame,
                   const Sophus::SE3d& pose,
                   const Vector3dVector* full_frame = nullptr);
  /** Rotate the local frame (previous pose, local map, gravity window) by `correction`
   *  about `pivot`; see Config::gravity_alignment_relevel_map. */
  void relevel_local_frame(const Sophus::SO3d& correction, const Eigen::Vector3d& pivot);

  /** True if odometry initialization has been completed. */
  bool _initialized = false;

  /** Timestamp of the most recent real IMU measurement. */
  Nsec _last_real_imu_time{0};

  /** Angular velocity of last true IMU measurement expressed in base frame. */
  Eigen::Vector3d _last_real_base_imu_ang_vel = Eigen::Vector3d::Zero();

  /** Consecutive scan registration failures since the last accepted scan. */
  int _consecutive_registration_failures = 0;

  /** Stateful weak-direction confirmation gate for the opt-in degeneracy solve. */
  PersistentWeakDirectionTracker _persistent_weak_direction_tracker;
  std::size_t _adaptive_iteration_hold_remaining = 0;
  /** Persistence gate for bump_image_max_rotation_correction_deg (built on first use). */
  std::optional<BumpRotationGate> _bump_rotation_gate;

  /** One-shot prior consumed by the next scan; never reused after rejection. */
  std::optional<VisualPosePrior> _visual_pose_prior;

  /** One-shot radar velocity prior consumed by the next scan; never reused after rejection. */
  std::optional<RadarVelocityPrior> _radar_velocity_prior;

  /** Consecutive scans with a radar-vs-ICP velocity disagreement. */
  std::size_t _radar_disagreement_streak = 0;

  /** Reference reflectivity profile from the most recent scan that still had a candidate
   *  weak direction, anchored to that scan's own world-frame centroid/axis so the next
   *  scan's profile (built at the same origin/axis) is directly comparable. Reset whenever
   *  the persistence tracker loses its candidate direction. */
  struct StoredIntensityProfile {
    IntensityProfile profile;
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
  };
  std::optional<StoredIntensityProfile> _previous_intensity_profile;

  /** Photometric registration state: patches, and the lidar extrinsic of the scan being
   *  registered (set by the extrinsic overload of register_scan). */
  PhotometricFeatureManager _photometric_features;
  Sophus::SE3d _photometric_base_from_cloud;

  /** Same idea as _previous_intensity_profile but for the Hessian-gate-free intensity
   *  velocity-disagreement gate: anchored to the axis/origin chosen from the *previous*
   *  scan's own ICP motion direction (see unit_axis_from_step), not a persistent weak
   *  direction. Kept separate so the two intensity features never share state. */
  std::optional<StoredIntensityProfile> _previous_intensity_velocity_profile;

  struct StoredOrientedIntensityGrid {
    OrientedIntensityGrid grid;
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d longitudinal_axis = Eigen::Vector3d::UnitX();
    Eigen::Vector3d lateral_axis = Eigen::Vector3d::UnitY();
  };
  std::optional<StoredOrientedIntensityGrid>
      _previous_intensity_velocity_grid;

  /** Consecutive scans with an intensity-vs-ICP velocity disagreement. */
  std::size_t _intensity_disagreement_streak = 0;

  /** Time of the last scan where ICP agreed with inertial propagation. */
  std::optional<Nsec> _kinematic_blend_anchor_time;

  /** Independently propagated world velocity seeded only by an agreement anchor.
   *  Keeping it separate from lidar_state prevents corrected ICP velocity from
   *  feeding back into its own prior. */
  std::optional<Eigen::Vector3d> _kinematic_blend_propagated_velocity_world;

  /** Consecutive scan intervals above kinematic_blend_max_yaw_rate_rad_s. */
  std::size_t _kinematic_blend_turning_streak = 0;

  /** Consecutive scans above kinematic_blend_max_activation_speed_mps. */
  std::size_t _kinematic_blend_speeding_streak = 0;

  /** Consecutive scans below the configured IMU activity threshold. */
  std::size_t _kinematic_blend_inactive_streak = 0;

  /** Most recent range-scene rejection time, in seconds. */
  double _kinematic_blend_scene_last_rejected_time_sec = -1.0;

  /** Most recent persistent activation-speed rejection time, in seconds. */
  double _kinematic_blend_speed_last_rejected_time_sec = -1.0;

  /** Sliding window of (scan time, world-frame raw-accelerometer interval mean) samples for
   *  gravity_window_alignment. Entries older than gravity_window_sec are dropped each scan. */
  std::deque<std::pair<Nsec, Eigen::Vector3d>> _gravity_alignment_window;
  /** The configuration with the close_scene_* keys applied, built on the first close scan. */
  std::optional<Config> _close_scene_config;
  /** Sensor position of the last scan added to the map (map_update_min_translation). */
  std::optional<Eigen::Vector3d> _last_map_update_position;
  /** The sensor has left the start_standstill_radius around its start. */
  bool _left_start = false;
  /** This scan interval stood still near the start (start_standstill_radius). */
  bool _standstill_this_scan = false;
  /** bump_image_point_to_point_start_distance has been reached (latched). */
  bool _bump_point_to_point_started = false;
  /** World-frame gravity (m/s^2): gravity() unless moving_start_gravity_window_sec re-estimated it. */
  Eigen::Vector3d _gravity_world = gravity();
  /** Added to the output filter's world acceleration: the estimated gravity less _gravity_world
   *  (moving_start_gravity_output_only), zero otherwise. */
  Eigen::Vector3d _output_filter_gravity_correction = Eigen::Vector3d::Zero();
  /** Initialization time (s) of a moving start while its gravity window is being collected. */
  std::optional<double> _moving_start_time;
  std::vector<WorldSpecificForceInterval> _moving_start_intervals;
  OutputPositionFilter _output_filter;
  /** The output filter waits for the gravity of a moving start (output_filter_position_sigma). */
  bool _output_filter_held = false;
  /** The gravity has just been estimated again: the filter starts at the next recorded pose. */
  bool _output_filter_resume = false;
  /** The initialization interval was a moving start (initialization_max_gyro_rad_s). */
  bool _moving_start = false;
  /** start_standstill_level_sec: a moving start waits for its standing intervals. */
  bool _start_level_pending = false;
  /** Where the robot started standing (start_standstill_hold_radius); reset when it moves. */
  std::optional<Eigen::Vector3d> _standstill_anchor;
  /** start_standstill_bias "average": sums over the standing intervals near the start. */
  Eigen::Vector3d _standstill_gyro_sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d _standstill_accel_bias_sum = Eigen::Vector3d::Zero();
  double _standstill_duration = 0.0;
  double _standstill_accel_duration = 0.0;
  Eigen::Vector3d _start_level_force_sum = Eigen::Vector3d::Zero();
  double _start_level_duration = 0.0;
  /** Level the world frame with the standing intervals' mean specific force and start the maps
   *  again at the current pose (start_standstill_level_sec). */
  void level_start_frame();
  /** Append the pose just recorded in poses_with_timestamps to output_poses_with_timestamps:
   *  filtered after a registration, otherwise the filter starts again at it. */
  void record_output_pose(bool registered);
  /** Collect the interval ending at current_lidar_time for moving_start_gravity_window_sec and,
   *  once the registered poses cover the window, re-estimate _gravity_world. */
  void update_moving_start_gravity(const Nsec& current_lidar_time, const Sophus::SO3d& current_rotation);
  std::optional<Nsec> _first_scan_time;
};
} // namespace rko_lio::core
