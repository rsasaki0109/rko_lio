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
#include "oriented_intensity_grid.hpp"
#include "persistent_weak_direction.hpp"
#include "selective_visual_fusion.hpp"
#include "voxel_hash_map.hpp"
#include "util.hpp"
#include <array>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {

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

    /** Max distance for correspondences (m). */
    double max_correspondence_distance = 0.5;

    /** Thread count for data association (0 = automatic). */
    int max_num_threads = 0;

    /** Enable initialization phase. */
    bool initialization_phase = false;

    /** Maximum expected jerk (m/s³). */
    double max_expected_jerk = 3;

    /** Enable double downsampling. */
    bool double_downsample = true;

    /** Voxel-size multiplier used only for ICP keypoints in double-downsample mode. */
    double icp_keypoint_voxel_multiplier = 1.5;

    /** Minimum weight for orientation regularization. */
    double min_beta = 200;

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
  };

  /** Configuration parameters. */
  Config config;

  /** Local map. */
  VoxelHashMap map;

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

  explicit LIO(const Config& config_);

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
  std::size_t gravity_alignment_attempt_count = 0;
  std::size_t gravity_alignment_applied_count = 0;
  double gravity_alignment_last_tilt_rad = 0.0;
  double gravity_alignment_correction_rad_sum = 0.0;

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
  void update_maps(const Vector3dVector& map_update_frame, const Sophus::SE3d& pose);

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
};
} // namespace rko_lio::core
