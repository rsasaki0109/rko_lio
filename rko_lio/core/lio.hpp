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
#include "persistent_weak_direction.hpp"
#include "selective_visual_fusion.hpp"
#include "sparse_voxel_grid.hpp"
#include "util.hpp"
#include <array>
#include <optional>
#include <string>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {

struct VisualPosePrior {
  Secondsd time{0.0};
  Sophus::SE3d pose;
  VisualConstraintConfidence confidence;
};

/** One-shot base-frame ego-velocity prior derived from a radar Doppler scan. */
struct RadarVelocityPrior {
  Secondsd time{0.0};
  Eigen::Vector3d velocity_base = Eigen::Vector3d::Zero();
};

struct VisualObservabilityDiagnosticsSample {
  Secondsd time{0.0};
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
    double max_correspondance_distance = 0.5;

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

    /** Minimum populated profile bins required for both scans in a correlation attempt. */
    size_t intensity_min_filled_bins = 40;

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
    double relocalization_max_correspondance_distance = 2.0;

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

  /** Local map as sparse voxel grid (Bonxai). */
  SparseVoxelGrid map;

  /** Global sparse map used for kidnap relocalization. This map is never pruned. */
  SparseVoxelGrid relocalization_map;

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

  explicit LIO(const Config& config_)
      : config(config_),
        map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel),
        relocalization_map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel) {}

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
  Sophus::SE3d predict_pose_at(const Secondsd& time) const;

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
  std::vector<std::pair<Secondsd, Sophus::SE3d>> poses_with_timestamps;

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

  /** Diagnostic-only counters for the intensity-vs-ICP velocity disagreement gate: how many
   *  scans had a stored previous profile to correlate against, how many of those produced a
   *  trustworthy (valid, non-saturated) shift, and how many of those exceeded the disagreement
   *  threshold (streak increments). Cheap to keep always; helps tell "gate never gets a
   *  measurement" apart from "measurements agree with ICP" apart from "streak never sustains".
   */
  std::size_t intensity_disagreement_attempt_count = 0;
  std::size_t intensity_disagreement_valid_shift_count = 0;
  std::size_t intensity_disagreement_exceeded_threshold_count = 0;

private:
  /**
   * Initialize internal odometry state using the given lidar timestamp.
   * @param lidar_time Current lidar timestamp.
   */
  void initialize(const Secondsd lidar_time);

  /** get the convenience struct with accel mag variance and local gravity estimate. */
  std::optional<AccelInfo> get_accel_info(const Sophus::SO3d& rotation_estimate, const Secondsd& time);

  /** Register a recovery scan at a chosen pose and start a fresh local map. */
  Vector3dVector recover_with_scan(const Vector3dVector& filtered_frame,
                                   const Vector3dVector& map_update_frame,
                                   const Secondsd& current_lidar_time,
                                   const Sophus::SE3d& recovery_pose,
                                   const std::string& reason);

  /** Drop an unusable scan while advancing the internal LiDAR timestamp. */
  Vector3dVector drop_failed_scan(const Secondsd& current_lidar_time, const std::string& reason);

  /** Try to align the current scan against the unpruned relocalization map. */
  std::optional<Sophus::SE3d> try_global_relocalization(const Vector3dVector& keypoints) const;

  /** Update the sliding local map and, when enabled, the unpruned recovery map. */
  void update_maps(const Vector3dVector& map_update_frame, const Sophus::SE3d& pose);

  /** True if odometry initialization has been completed. */
  bool _initialized = false;

  /** Latest IMU orientation used for gravity compensation. This is ahead of the rotation in the state. */
  Sophus::SO3d _imu_local_rotation;

  /** Timestamp of the latest IMU orientation. Once a scan is registered, this is reset to the lidar state orientation.
   */
  Secondsd _imu_local_rotation_time = Secondsd{0.0};

  /** Timestamp of the most recent real IMU measurement. */
  Secondsd _last_real_imu_time = Secondsd{0.0};

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

  /** Consecutive scans with an intensity-vs-ICP velocity disagreement. */
  std::size_t _intensity_disagreement_streak = 0;
};
} // namespace rko_lio::core
