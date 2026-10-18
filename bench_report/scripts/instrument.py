"""Add stage profilers to an rko_lio source tree (local measurement only, never committed).

Usage: python3 instrument.py <rko_lio source dir>
Every replacement must match exactly once, otherwise the script fails.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel, pairs):
    path = root / rel
    text = path.read_text()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{rel}: anchor found {count} times: {old[:80]!r}")
        text = text.replace(old, new)
    path.write_text(text)


patch("rko_lio/core/lio.cpp", [
    ("""  auto preproc_result =
      config.deskew ? preprocess_scan(deskew_scan(scan, timestamps, current_lidar_time, relative_pose_at_time), config)
                    : preprocess_scan(scan, config);
""",
     """  Vector3dVector prof_deskewed;
  {
    SCOPED_PROFILER("prof: deskew");
    if (config.deskew) prof_deskewed = deskew_scan(scan, timestamps, current_lidar_time, relative_pose_at_time);
  }
  auto preproc_result = [&]() {
    SCOPED_PROFILER("prof: preprocess");
    return config.deskew ? preprocess_scan(prof_deskewed, config) : preprocess_scan(scan, config);
  }();
"""),
    ("""             const BumpTerms* bump = nullptr) {
""",
     """             const BumpTerms* bump = nullptr) {
  SCOPED_PROFILER("prof: icp()");
"""),
    ("""  _photometric_base_from_cloud = extrinsic_lidar2base;
  if (extrinsic_lidar2base.log().norm() < EPSILON) {""",
     """  SCOPED_PROFILER("prof: register_scan outer");
  _photometric_base_from_cloud = extrinsic_lidar2base;
  if (extrinsic_lidar2base.log().norm() < EPSILON) {"""),
    ("""void LIO::update_maps(const Vector3dVector& map_update_frame,
                      const Sophus::SE3d& pose,
                      const Vector3dVector* full_frame) {
""",
     """void LIO::update_maps(const Vector3dVector& map_update_frame,
                      const Sophus::SE3d& pose,
                      const Vector3dVector* full_frame) {
  SCOPED_PROFILER("prof: update_maps");
"""),
])


lio_path = root / "rko_lio/core/lio.cpp"
lio_text = lio_path.read_text()
alternatives = [
    ("""  transform_points(extrinsic_lidar2base, scan, transformed_scan);
  Vector3dVector frame = register_scan(transformed_scan, timestamps, intensities);
  transform_points(extrinsic_lidar2base.inverse(), frame);
  return frame;""",
     """  {
    SCOPED_PROFILER("prof: extrinsic transform in");
    transform_points(extrinsic_lidar2base, scan, transformed_scan);
  }
  Vector3dVector frame = register_scan(transformed_scan, timestamps, intensities);
  {
    SCOPED_PROFILER("prof: extrinsic transform out");
    transform_points(extrinsic_lidar2base.inverse(), frame);
  }
  return frame;"""),
    ("""  std::transform(scan.cbegin(), scan.cend(), transformed_scan.begin(),
                 [&](const Eigen::Vector3d& p) { return extrinsic_lidar2base * p; });
  Vector3dVector frame = register_scan(transformed_scan, timestamps, intensities);
  transform_points(extrinsic_lidar2base.inverse(), frame);
  return frame;""",
     """  {
    SCOPED_PROFILER("prof: extrinsic transform in");
    std::transform(scan.cbegin(), scan.cend(), transformed_scan.begin(),
                   [&](const Eigen::Vector3d& p) { return extrinsic_lidar2base * p; });
  }
  Vector3dVector frame = register_scan(transformed_scan, timestamps, intensities);
  {
    SCOPED_PROFILER("prof: extrinsic transform out");
    transform_points(extrinsic_lidar2base.inverse(), frame);
  }
  return frame;"""),
]
for old_alt, new_alt in alternatives:
    if lio_text.count(old_alt) == 1:
        lio_text = lio_text.replace(old_alt, new_alt)
        break
else:
    raise SystemExit("outer transform anchor not found")
lio_path.write_text(lio_text)

# build_icp_linear_system: profile the whole call (association + reduce).
lio = (root / "rko_lio/core/lio.cpp").read_text()
anchor = "  const bool weighting_enabled = localizability_axis.has_value() && localizability_boost > 0.0;\n"
assert lio.count(anchor) == 1, "build_icp_linear_system anchor"
lio = lio.replace(anchor, '  SCOPED_PROFILER("prof: build_icp_linear_system");\n' + anchor)
if '#include "profiler.hpp"' not in lio and '#include "rko_lio/core/profiler.hpp"' not in lio:
    lio = lio.replace('#include "lio.hpp"\n', '#include "lio.hpp"\n#include "profiler.hpp"\n', 1)
(root / "rko_lio/core/lio.cpp").write_text(lio)

vhm = root / "rko_lio/core/voxel_hash_map.cpp"
text = vhm.read_text()
text = text.replace('#include "voxel_hash_map.hpp"\n', '#include "voxel_hash_map.hpp"\n#include "profiler.hpp"\n', 1)
old_add = "  add_points(points_transformed);\n  remove_points_far_from_location(origin);\n}"
assert text.count(old_add) == 1, "update anchor"
text = text.replace(old_add, """  {
    SCOPED_PROFILER("prof: map add_points");
    add_points(points_transformed);
  }
  {
    SCOPED_PROFILER("prof: map remove_far");
    remove_points_far_from_location(origin);
  }
}""")
vhm.write_text(text)
print("instrumented", root)

patch("rko_lio/ros/threaded_node.cpp", [
    ("""    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }""",
     """    {
      SCOPED_PROFILER("prof: add_imu_measurements");
      for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
        const core::ImuControl& imu_data = imu_buffer.front();
        lio->add_imu_measurement(extrinsic_imu2base, imu_data);
      }
    }"""),
    ("""      const core::Vector3dVector deskewed_frame = register_scan_locked(scan, time_vector, intensities_ptr);""",
     """      const core::Vector3dVector deskewed_frame = [&]() {
        SCOPED_PROFILER("prof: register_scan_locked");
        return register_scan_locked(scan, time_vector, intensities_ptr);
      }();"""),
    ("""        publish_lidar_outputs(deskewed_frame);
        publish_tf(lio->lidar_state);""",
     """        {
          SCOPED_PROFILER("prof: publish outputs+tf");
          publish_lidar_outputs(deskewed_frame);
          publish_tf(lio->lidar_state);
        }"""),
])
print("instrumented threaded_node", root)

# Keypoint and iteration counters, printed at exit (registration thread only).
lio_path = root / "rko_lio/core/lio.cpp"
lio_text = lio_path.read_text()
counter_decl = """
namespace prof_counters {
struct Counters {
  long icp_calls = 0;
  long systems = 0;
  long keypoints = 0;
  long deskew_points = 0;
  ~Counters() {
    std::cout << "prof counters: icp_calls=" << icp_calls << " systems=" << systems << " keypoints_per_system="
              << (systems ? static_cast<double>(keypoints) / systems : 0.0) << " deskew_points_per_scan="
              << (icp_calls ? static_cast<double>(deskew_points) / icp_calls : 0.0) << "\\n";
  }
};
inline Counters counters;
} // namespace prof_counters
"""
anchor = "using namespace rko_lio::core;\n"
assert lio_text.count(anchor) >= 1
lio_text = lio_text.replace(anchor, anchor + counter_decl, 1)
lio_text = lio_text.replace('  SCOPED_PROFILER("prof: build_icp_linear_system");\n',
                            '  SCOPED_PROFILER("prof: build_icp_linear_system");\n'
                            '  ++prof_counters::counters.systems;\n'
                            '  prof_counters::counters.keypoints += static_cast<long>(frame.size());\n', 1)
lio_text = lio_text.replace('  SCOPED_PROFILER("prof: icp()");\n',
                            '  SCOPED_PROFILER("prof: icp()");\n  ++prof_counters::counters.icp_calls;\n', 1)
lio_text = lio_text.replace('    SCOPED_PROFILER("prof: deskew");\n',
                            '    SCOPED_PROFILER("prof: deskew");\n'
                            '    prof_counters::counters.deskew_points += static_cast<long>(scan.size());\n', 1)
lio_path.write_text(lio_text)
print("counters added", root)
