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
#include "sparse_voxel_grid.hpp"
#include "util.hpp"
#include <optional>
#include <string>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {
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

    /** Minimum weight for orientation regularization. */
    double min_beta = 200;

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

  /** Update both the sliding local map and the unpruned relocalization map. */
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
};
} // namespace rko_lio::core
