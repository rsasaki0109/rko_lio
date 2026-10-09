Configuring the odometry
========================

This page is a reference for runtime parameters.
The first group -- the LIO core parameters -- is shared between the Python and ROS interfaces.
The remaining groups (extrinsics, per-point timestamp handling, Python pipeline knobs, ROS launch knobs) cover the wrapper-specific bits.

The defaults are sane and I've used them with success across a number of platforms and datasets.
But your specific application might benefit from tuning -- the descriptions below try to make clear when.

Physical units are always SI.

.. contents::
   :local:
   :depth: 2

LIO core parameters
-------------------

These show up under the top level of a Python config and as ROS launch arguments with the same name.

- **deskew** (`bool`, default ``True``)

  Whether to apply scan deskewing before registration.

  This compensates for the motion that happens during LiDAR scan collection, since your platform is probably moving while the LiDAR is collecting data.

  Unless you have very good reason, always keep this enabled.

.. warning::
  ``deskew=True`` **requires** per-point timestamps in the LiDAR scan, i.e., the scan needs to have ``xyzt`` per-point where ``t`` is time.
  If you cannot provide this, then deskewing should be disabled.

- **double_downsample** (`bool`, default ``True``)

  Useful for dense LiDARs.

  Disabling this for sparse sensors, like a VLP-16 compared to an Ouster-128, can potentially improve results.
  Indoor scenes can also see an improvement by disabling this.

- **legacy_voxel_downsample** (`bool`, default ``False``)

  Uses the unordered-map voxel sampler from before v0.3 for every downsampling
  pass. In double-downsample mode this restores both the map input and the ICP
  keypoint selection.

  Ordering and the representatives selected by the second pass can change the
  accumulated trajectory. Keep the modern default unless a dataset-specific
  profile has measured a regression and records the compatibility setting in
  its benchmark provenance.

- **icp_keypoint_voxel_multiplier** (`float`, default ``1.5``)

  Voxel-size multiplier for the ICP keypoint pass in double-downsample mode.
  It does not change the half-voxel map pass and has no effect when
  ``double_downsample`` is disabled.

  Keep the default unless a dataset-specific benchmark demonstrates a bounded
  improvement and records the setting in its provenance.

- **voxel_size** (`float`, default ``1.0``)

  The voxel resolution of the internal local map (in meters).

  Smaller values create finer maps at higher memory/computation cost.
  Reducing this for sparse sensors or indoor scenes can help improve results.

- **max_range** (`float`, default ``100.0``)

  Maximum usable LiDAR range in meters.

  Points beyond this cutoff are ignored.
  Reducing this is an easy way to reduce compute requirements, since you typically won't need information from 100m away for odometry.
  Nevertheless, this is the default.

- **min_range** (`float`, default ``1.0``)

  Minimum LiDAR range in meters.

  Points closer than this are discarded.
  Useful if your platform shows up in the scan due to occlusions.

- **max_points_per_voxel** (`int`, default ``20``)

  Maximum number of points stored per voxel in the VDB map.

  Affects both memory and ICP data association.

  In case you need more runtime performance, you can reduce this.
  Odometry performance will be a bit affected, but how much depends on the environment.

- **max_correspondence_distance** (`float`, default ``0.5``)

  Maximum distance threshold (meters) for ICP data associations.

- **max_iterations** (`int`, default ``100``)

  Limit on the number of iterations for ICP.

  You can limit this a bit more if runtime is an issue.
  Typically the convergence criterion is satisfied much earlier anyways.

- **convergence_criterion** (`float`, default ``1e-5``)

  Termination criterion for optimization.
  Lower (stricter) values will requires more ICP iterations.

- **max_num_threads** (`int`, default ``0``)

  Only used to parallelize data association for ICP.
  ``0`` means autodetect based on hardware.

  In case compute resources are a constraint, limit this to a few threads and, in order, ``max_points_per_voxel``, ``voxel_size``, ``max_range``, ``max_iterations`` are the parameters you probably care about.

- **initialization_phase** (`bool`, default ``False``)

  Initializes the system orientation (roll and pitch) plus IMU biases using the IMU measurements between the first two LiDAR scans the odometry receives.

  If enabled, the second frame is assumed to be coincident with the first.
  I.e., the assumption is that the system is at rest for that duration and the system is oriented to align with gravity.
  This helps if you start from an inclined surface for example.

  Usually you can leave this enabled. Unless for some reason you need to start the odometry while the system is in motion, then disable this.

  I highly recommend enabling this.

.. warning::
  ``initialization_phase=True`` requires you to ensure that the system starts from rest.
  Otherwise, the system will estimate incorrect biases and the odometry might not work as expected.
  Hence, why this is set to ``False`` by default and is opt-in.

- **max_expected_jerk** (`float`, default ``3.0``)

  This value is used in a Kalman filter to estimate the true body acceleration.

  It should reflect the motion you expect from the platform you will deploy the odometry on.
  A good range is [1-3] m/s³, but it should be fine to leave it at 3 m/s³ as that is a good setting for most platforms.

- **min_beta** (`float`, default ``200.0``)

  The minimum weight applied to an orientation regularization cost during scan alignment.

  Essentially we use the accelerometer readings as an additional observation on the roll and pitch of the system (we need to estimate the true body acceleration using the Kalman filter mentioned above).
  This parameter influences how much importance this additional observation plays in the optimization, as we cannot have a perfect observation of the true body acceleration (the estimate is affected by gravity and the odometry itself).
  The default should be fine for most cases.

- **velocity_window_sec** (`float`, default ``0.0``, fork addition)

  Estimate the velocity from the pose at least this long ago instead of the previous scan's (``0`` keeps the previous scan).
  Over one 0.1 s scan, a single wrong pose correction of a few decimetres becomes a velocity error of metres per second, and the next prediction carries it.
  Where the registration constrains the translation weakly, as on open ground while turning fast, it cannot pull the prediction back and the velocity runs away.
  ``0.3`` stops this on ENWIDE RunwayD, where a handheld sensor is spun at about 190 °/s.

- **skip_registration_after_gap_sec** (`float`, default ``0.0``, fork addition)

  Do not register a scan that follows a LiDAR gap longer than this; its pose is the IMU prediction and it stays out of the map (``0`` disables).
  The first scan after a gap is often partial and its prediction coarse; in a tunnel the registration can then turn it by degrees.
  Use it together with ``velocity_window_sec`` so the skipped scan does not set the velocity.

  You can set it ``-1`` to disable this additional cost.

Extrinsics
----------

The extrinsic transforms between the IMU / LiDAR and the base frame are specified in YAML as ``[qx, qy, qz, qw, x, y, z]`` lists:

.. code-block:: yaml

   extrinsic_imu2base_quat_xyzw_xyz:   [0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0]
   extrinsic_lidar2base_quat_xyzw_xyz: [0.0, 0.0, 0.0, 1.0,  0.1, 0.0, 0.05]

If one of the two keys is set, both must be set; pick identity for whichever frame you want to treat as the base.
The convention is described under :ref:`data-extrinsics-convention`.

When are these required?

- **ROS**: only if the TF tree isn't well defined or topic ``frame_id``\s don't match the TF tree. With a clean TF tree, the extrinsics are looked up automatically.
- **Python rosbag dataloader**: only if the bag has no static TF tree. With a tree, the extrinsics are pulled from it.
- **Python raw / HeLiPR dataloaders**: always required, supplied via the dataloader's own configuration mechanism (``transforms.yaml`` for raw, etc.).

If you specify the extrinsics in a config but the dataloader / TF tree could also provide them, the config values take priority.

.. _config-lidar-timestamps:

LiDAR per-point timestamps
--------------------------

.. automodule:: rko_lio.config
   :no-index:

The same overrides are surfaced as ROS launch arguments under the ``lidar_timestamps.*`` namespace:

.. code-block:: yaml

   lidar_timestamps.multiplier_to_seconds: 0.0   # 0.0 = autodetect; e.g. 1e-6 for microseconds
   lidar_timestamps.force_absolute: false
   lidar_timestamps.force_relative: false

(They can also be written as a nested mapping ``lidar_timestamps:\n  force_absolute: true`` -- ``launch_ros`` flattens nested dicts before passing them as parameter overrides, so both forms are equivalent.)

Photometric registration (ROS, fork addition)
---------------------------------------------

In long, geometrically self-similar tunnels the point-to-point ICP slides along the
tunnel axis. With ``photometric: true`` the ROS node also renders every organized scan
as an intensity image and adds photometric patch residuals to each ICP iteration,
after COIN-LIO (Pfreundschuh et al., ICRA 2024). Patches are chosen where the image
gradient sees motion along the translation directions few surface normals face.
It is off by default and needs an organized cloud (Ouster layout) and its metadata.

- **photometric** (`bool`, default ``False``)
- **photometric_scale** (`float`, default ``0.003``): weight of one photometric residual
  (filtered intensity unit) relative to one ICP residual (m). On ENWIDE TunnelD the
  results are flat between ``0.002`` and ``0.01``.
- **photometric_channel** (`str`, default ``intensity``): point field rendered into the images.
- **photometric_model.\*:** ``altitudes_deg``, ``pixel_shift_by_row``, ``columns``,
  ``beam_offset_mm`` and ``cloud_to_lidar_z_m`` from the Ouster metadata.
- **photometric_image.\*:** line filter kernels, brightness window, static masks, valid range.
- **photometric_features.\*:** patch size, number of patches, lifetime, NCC threshold.

``config/enwide_os0_photometric.yaml`` is a complete example. Where the intensity images
carry little texture along the tunnel (a continuous LED strip, bare concrete) the terms
do not help.

Bump image registration (fork addition)
---------------------------------------

On open, flat ground a point-to-point match sees a plane and cannot fix the in-plane
translation. With ``bump_image_registration: true`` the ICP system is replaced by the
residual of BIEVR-LIO (Pfreundschuh et al., arXiv 2604.14421): every voxel of a second
map stores the height of its surface above a fitted plane on a fine pixel grid, and each
scan point is registered against the height at its pixel. Relief of a few centimetres,
such as grass, then constrains the pose. Photometric terms, when enabled, are added on
top as before. It is off by default.

- **bump_image_registration** (`bool`, default ``False``)
- **bump_image_map.voxel_size** (`float`, default ``0.5``), **bump_image_map.pixel_size**
  (`float`, default ``0.05``): voxel and pixel side length in metres.
- **bump_image_map.weighted** (`bool`, default ``True``): weight pixel updates by inverse range.
- **bump_image_map.smooth** (`bool`, default ``True``): Gaussian-smooth the height images.
- **bump_image_map.normal_tolerance_deg** (`float`, default ``3.0``): reproject a voxel's
  image when its plane normal turns further than this.
- **bump_image_map.max_voxels** (`int`, default ``1500000``): least recently updated voxels
  are dropped beyond this.
- **bump_image_source_voxel_size** (`float`, default ``0.1``): downsampling of the scan
  points used for registration.
- **bump_image_informed_voxels** (`int`, default ``300``): number of map voxels with the
  most relief whose points are all kept; the other voxels keep one point each.
- **bump_image_huber_delta** (`float`, default ``0.1``): Huber threshold in metres.
- **bump_image_max_rotation_correction_deg** (`float`, default ``0.0``): register a scan
  again without the bump terms when they turn the pose more than this many degrees away
  from the IMU prediction (``0`` disables). In a circular tunnel the roll about the axis
  is unobservable, and the bump images slip around it by degrees per scan. A correctly
  registered scan stays within about 1.6 degrees. ``2.0`` takes GEODE Shield_tunnel9
  from 441 m to 67 m ATE and leaves a vehicle-mounted urban tunnel unchanged.

On ENWIDE FieldD (grass) with the photometric terms of
``config/enwide_os0_photometric.yaml`` the ATE drops from 7.2 m to 0.18 m.

Pipeline parameters (Python)
----------------------------

These are passed to the Python ``LIOPipeline`` via the same YAML config as the LIO parameters.

On-disk logging is gated by the CLI flag ``--log`` / ``-l`` (``log_results``, default ``True``).
Without it, ``log_dir``, ``run_name``, and ``dump_deskewed_scans`` have no effect.

- **dump_deskewed_scans** (`bool`, default ``False``)

  Save each deskewed scan to disk under ``log_dir`` as PLY. Useful for debugging the deskewing step or for reuse downstream. Off by default since it generates a lot of data.

- **log_dir** (`Path`, default ``"results"``)

  Where the trajectory file (and dumped scans, if enabled) gets written.

- **run_name** (`str`, default ``"rko_lio_run"``)

  Subdirectory name under ``log_dir``. The run gets an auto-incremented suffix to avoid clobbering previous runs.

Launch parameters (ROS)
-----------------------

These are not part of the LIO algorithm itself; they configure the ROS launch behaviour.
For the full list with descriptions, run ``ros2 launch rko_lio odometry.launch.py -s``.

Mode selection
^^^^^^^^^^^^^^

- **mode** (default ``online``)

  ``online`` subscribes to live topics. ``offline`` drains a rosbag at full speed (also requires ``bag_path``).

- **odom_at_imu_rate** (`bool`, default ``false``)

  When ``true`` and ``mode:=online``, the launch file picks ``online_imu_rate_node``, which additionally publishes IMU-rate odometry. See :doc:`ROS -> IMU-rate odometry <ros>`.

- **bag_path** (offline only, required)

  Path to the bag directory.

- **skip_to_time** (`float`, offline only, default ``0.0``)

  Skip ahead in the bag to this absolute time (seconds) before starting registration.

Topic and frame configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **imu_topic**, **lidar_topic**, **base_frame** (via CLI or config, required with ``autodetect:=false``)

  The IMU topic, the LiDAR topic, and the base frame of the robot.
  Autodetected when left unset. See :ref:`autodetection`.

- **imu_frame**, **lidar_frame** (optional)

  Override these only if the message ``frame_id`` doesn't match the TF tree.
  Also autodetected, from the message ``frame_id``.

- **autodetect** (`bool`, default ``true``, CLI only), **autodetect_timeout** (`float`, default ``10.0``, CLI only)

  Fill in the topics and frames above that you left unset, and how long to wait for the data needed to do so (online only).
  See :ref:`autodetection`.

- **odom_frame** (default ``odom``), **odom_topic** (default ``rko_lio/odom``)

  The odom frame name and the odometry topic name.

- **invert_odom_tf** (`bool`, default ``false``)

  Swap parent / child in the published TF if your topology requires it.

Publishing toggles
^^^^^^^^^^^^^^^^^^

- **publish_local_map** (`bool`, default ``false``), **map_topic** (default ``rko_lio/local_map``), **publish_map_after** (`float`, default ``1.0``)

  Whether to publish the local map, the topic name, and the republish cadence in seconds.

- **publish_deskewed_scan** (`bool`, default ``false``), **deskewed_scan_topic** (default ``rko_lio/frame``)

  Whether to publish the deskewed scan, and the topic name.

- **publish_lidar_acceleration** (`bool`, default ``false``)

  Publish a noisy linear acceleration estimate on ``rko_lio/lidar_acceleration`` (the topic name is fixed, not configurable).

Mode-specific knobs
^^^^^^^^^^^^^^^^^^^

These take effect only in the corresponding mode and otherwise warn that they are being ignored.

- **async.max_lidar_buffer_size** (`int`, default ``50``)

  Threaded path only. Caps the lidar buffer; older frames are dropped past this.

- **async.output_publish_delay_ms** (`int`, default ``0``)

  Threaded path only. Sleeps for this many milliseconds after publishing each
  successfully registered LiDAR frame. Keep the default for live sensing. A
  positive value provides bounded backpressure when an offline producer would
  otherwise outrun a synchronous downstream consumer such as a graph backend.

- **seq.odom_at_imu_rate_topic** (default ``rko_lio/odom_at_imu_rate``)

  IMU-rate output topic name.

- **seq.tf_at_imu_rate** (`bool`, default ``false``)

  Broadcast ``base_frame`` -> ``odom_frame`` TF at IMU rate instead of LiDAR rate.

Disk dumping and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **dump_results** (`bool`, default ``false``)

  On shutdown, dump the resolved configuration and the full trajectory under ``<results_dir>/<run_name>``. The folder name is auto-incremented to avoid overwrites.

- **results_dir** (default ``results``), **run_name** (default ``rko_lio_odometry_run``)

  Where the dump goes and the subdirectory name within.

- **rviz** (`bool`, default ``false``), **rviz_config_file** (default ``config/default.rviz``)

  Launch RViz alongside the odometry. If you leave the rviz config at the default, the launch file patches it with your ``base_frame`` / ``odom_frame`` and enables ``publish_deskewed_scan`` / ``publish_local_map`` so the visualizer has something to show.

- **log_level** (default ``info``)

  ROS log level.

Other
^^^^^

- **config_file**

  Path to a YAML config that supplies any of the above. CLI values override the file.
