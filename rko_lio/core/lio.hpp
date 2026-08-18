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
#include "fixed_lag_pose_optimizer.hpp"
#include "persistent_weak_direction.hpp"
#include "piecewise_gyro_deskew.hpp"
#include "selective_visual_fusion.hpp"
#include "sparse_voxel_grid.hpp"
#include "util.hpp"
#include <array>
#include <deque>
#include <optional>
#include <string>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {

struct VisualPosePrior {
  Secondsd time{0.0};
  Sophus::SE3d pose;
  VisualConstraintConfidence confidence;
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

    /** Integrate per-IMU angular rates for rotational deskew instead of one interval mean. */
    bool piecewise_gyro_deskew = false;

    /** Jointly refine a bounded recent pose window using scan-to-scan constraints. */
    bool fixed_lag_multiscan = false;

    /** Number of recent scans retained as jointly adjustable states. */
    size_t fixed_lag_window_size = 6;

    /** Previous scans matched independently to each newly accepted scan. */
    size_t fixed_lag_neighbor_scans = 3;

    /** ICP iterations used for each scan-to-scan relative constraint. */
    size_t fixed_lag_pairwise_max_iterations = 10;

    /** Weight of scan-to-scan constraints in the fixed-lag graph. */
    double fixed_lag_scan_constraint_weight = 1.0;

    /** Weight of anchor-relative odometry priors in the fixed-lag graph. */
    double fixed_lag_odometry_prior_weight = 0.2;

    /** Maximum pairwise-ICP translation correction accepted as a constraint. */
    double fixed_lag_pairwise_max_translation_m = 0.3;

    /** Maximum pairwise-ICP rotation correction accepted as a constraint. */
    double fixed_lag_pairwise_max_rotation_deg = 5.0;

    /** Minimum current-scan points associated in a pairwise factor. */
    size_t fixed_lag_pairwise_min_correspondences = 100;

    /** Minimum current-scan inlier ratio for a pairwise factor. */
    double fixed_lag_pairwise_min_inlier_ratio = 0.6;

    /** Required relative reduction in pairwise mean nearest-neighbor error. */
    double fixed_lag_pairwise_min_error_reduction = 0.01;

    /** Maximum translation applied to any pose in one window solve. */
    double fixed_lag_max_pose_correction_m = 0.2;

    /** Maximum rotation applied to any pose in one window solve. */
    double fixed_lag_max_pose_correction_deg = 3.0;

    /** Tighter translation cap for the newest, externally published pose. */
    double fixed_lag_max_latest_pose_correction_m = 0.2;

    /** Tighter rotation cap for the newest, externally published pose. */
    double fixed_lag_max_latest_pose_correction_deg = 3.0;

    /** Keep the newest online pose fixed while refining only its history. */
    bool fixed_lag_fix_latest_pose = false;

    /** Robust residual transition for the fixed-lag graph. */
    double fixed_lag_huber_delta_m = 0.25;

    /** Maximum current scan-to-map mean-error regression after a window solve. */
    double fixed_lag_map_max_error_ratio = 1.01;

    /** Minimum retained current scan-to-map correspondences after a window solve. */
    double fixed_lag_map_min_correspondence_ratio = 0.98;

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

  /** Whether both frozen and active local-map layers are empty. */
  bool local_map_empty() const;

  /** Materialize the frozen and active local-map layers for visualization. */
  Vector3dVector local_map_pointcloud() const;

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
        relocalization_map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel),
        _fixed_lag_frozen_map(
            config_.voxel_size, config_.max_range, config_.max_points_per_voxel),
        _fixed_lag_active_map(
            config_.voxel_size, config_.max_range, config_.max_points_per_voxel) {}

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

  /**
   * Register a LiDAR scan, applying deskewing based on the initial motion guess
   * and clipping points beyond valid range.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @return Deskewed and clipped point cloud.
   */
  Vector3dVector register_scan(const Vector3dVector& scan, const TimestampVector& timestamps);

  /**
   * Register a LiDAR scan for which the extrinsic calibration from lidar to base
   * has already been applied.
   * @param extrinsic_lidar2base Extrinsic from lidar to base frame.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @return Deskewed and clipped scan in the original lidar frame.
   */
  Vector3dVector register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                               const Vector3dVector& scan,
                               const TimestampVector& timestamps);

  /** Sequence of registered scan poses with corresponding timestamps. */
  std::vector<std::pair<Secondsd, Sophus::SE3d>> poses_with_timestamps;

  /** Unrefined primary-registration trajectory for fixed-lag attribution. */
  std::vector<std::pair<Secondsd, Sophus::SE3d>> tracking_poses_with_timestamps;

  struct FinalizedFixedLagPose {
    Secondsd time{0.0};
    Sophus::SE3d pose;
  };

  /** Move out poses that can no longer change after fixed-lag marginalization. */
  std::vector<FinalizedFixedLagPose> take_finalized_fixed_lag_poses();

  /** Finalize the remaining active window, used when an offline bag ends. */
  void finalize_fixed_lag_window();

  /** Opt-in per-scan persistence-gate diagnostics, populated only when enabled. */
  std::vector<DegeneracyPersistenceDiagnosticsSample> degeneracy_persistence_diagnostics;

  /** Aggregate artifact-driven visual-fusion diagnostics. */
  std::size_t visual_prior_attempt_count = 0;
  std::size_t visual_fused_scan_count = 0;
  std::size_t visual_fused_direction_count = 0;
  std::size_t visual_unobservable_direction_count = 0;
  std::vector<VisualObservabilityDiagnosticsSample>
      visual_observability_diagnostics;

  /** Fixed-lag gate counters written by the offline ROS wrapper. */
  std::size_t fixed_lag_pairwise_attempt_count = 0;
  std::size_t fixed_lag_pairwise_accept_count = 0;
  std::size_t fixed_lag_window_attempt_count = 0;
  std::size_t fixed_lag_window_accept_count = 0;
  double fixed_lag_max_applied_translation_m = 0.0;
  double fixed_lag_max_applied_rotation_deg = 0.0;
  std::size_t fixed_lag_applied_pose_count = 0;
  double fixed_lag_applied_translation_squared_sum = 0.0;
  double fixed_lag_applied_rotation_deg_squared_sum = 0.0;
  double fixed_lag_max_latest_translation_m = 0.0;
  double fixed_lag_max_latest_rotation_deg = 0.0;

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

  /** Add one accepted scan, solve the active window, and rebuild its map. */
  Sophus::SE3d update_fixed_lag_window(
      const Vector3dVector& map_update_frame,
      const Vector3dVector& keypoints,
      const Secondsd& time,
      const Sophus::SE3d& pose);

  /** Rebuild the adjustable active-map layer from current window poses. */
  void rebuild_fixed_lag_active_map();

  /** Clear all fixed-lag state after recovery or a discontinuity. */
  void reset_fixed_lag_window();

  struct FixedLagFrame {
    Secondsd time{0.0};
    Sophus::SE3d pose;
    Sophus::SE3d odometry_pose;
    Vector3dVector map_update_frame;
    Vector3dVector keypoints;
    std::size_t pose_history_index = 0;
  };

  /** Marginalized scans at their finalized fixed-lag poses. */
  SparseVoxelGrid _fixed_lag_frozen_map;

  /** Recent scans that remain adjustable inside the active window. */
  SparseVoxelGrid _fixed_lag_active_map;

  /** Latest unrefined primary-registration pose, isolated from map refinement. */
  Sophus::SE3d _fixed_lag_tracking_pose;

  std::deque<FixedLagFrame> _fixed_lag_frames;
  std::vector<FixedLagRelativeConstraint> _fixed_lag_scan_constraints;
  std::vector<FinalizedFixedLagPose> _finalized_fixed_lag_poses;

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

  /** Bias-corrected base-frame gyro samples accumulated since the last accepted scan. */
  std::vector<TimedAngularVelocity> _deskew_gyro_samples;

  /** Consecutive scan registration failures since the last accepted scan. */
  int _consecutive_registration_failures = 0;

  /**
   * [instrumentation, additive-only] Scan counter used solely to stride the
   * periodic MapGrowthGauge sampling in update_maps(); never read back into
   * any odometry computation.
   */
  std::size_t _map_growth_scan_counter = 0;

  /** Stateful weak-direction confirmation gate for the opt-in degeneracy solve. */
  PersistentWeakDirectionTracker _persistent_weak_direction_tracker;
  std::size_t _adaptive_iteration_hold_remaining = 0;

  /** One-shot prior consumed by the next scan; never reused after rejection. */
  std::optional<VisualPosePrior> _visual_pose_prior;
};
} // namespace rko_lio::core
