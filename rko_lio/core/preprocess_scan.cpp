#include "preprocess_scan.hpp"
#include "voxel_down_sample.hpp"
#include <algorithm>

namespace rko_lio::core {

PreprocessingResult preprocess_scan(const Vector3dVector& frame, const LIO::Config& config) {
  Vector3dVector clipped_frame;
  clipped_frame.reserve(frame.size());

  std::for_each(frame.cbegin(), frame.cend(), [&](const auto& point) {
    const double point_range = point.norm();
    if (point_range > config.min_range && point_range < config.max_range) {
      clipped_frame.emplace_back(point);
    }
  });
  clipped_frame.shrink_to_fit();

  const auto downsample_keypoints = [&](const Vector3dVector& points, const double voxel_size) {
    return config.legacy_voxel_downsample ? voxel_down_sample_legacy(points, voxel_size)
                                         : voxel_down_sample(points, voxel_size);
  };

  if (config.double_downsample) {
    // Compatibility mode must cover both passes: changing pass 1 changes both
    // the map input and the representatives available to pass 2.
    Vector3dVector downsampled_frame =
        config.legacy_voxel_downsample ? voxel_down_sample_legacy(clipped_frame, config.voxel_size * 0.5)
                                       : voxel_down_sample_sorted(clipped_frame, config.voxel_size * 0.5);
    // pass 2 feeds icp, so unsorted is fine.
    Vector3dVector keypoints = downsample_keypoints(
        downsampled_frame, config.voxel_size * std::max(0.5, config.icp_keypoint_voxel_multiplier));
    return {.filtered_frame = std::move(clipped_frame),
            .keypoints = std::move(keypoints),
            .map_frame = std::move(downsampled_frame)};
  }
  Vector3dVector keypoints = downsample_keypoints(clipped_frame, config.voxel_size);
  return {.filtered_frame = std::move(clipped_frame), .keypoints = std::move(keypoints), .map_frame = {}};
}

} // namespace rko_lio::core
