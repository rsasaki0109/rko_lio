#pragma once
#include <Eigen/Dense>
#include <vector>

#include "lio.hpp"

namespace rko_lio::core {

struct PreprocessingResult {
  Vector3dVector filtered_frame;
  Vector3dVector keypoints;
  Vector3dVector map_frame; // populated only when config.double_downsample; empty otherwise
};

// clip and downsample the input cloud
PreprocessingResult preprocess_scan(const Vector3dVector& frame, const LIO::Config& config);

// keypoints taken again at config.voxel_size * voxel_multiplier, from the map-resolution
// downsample in double-downsample mode and from the clipped frame otherwise
Vector3dVector refine_keypoints(const PreprocessingResult& result, const LIO::Config& config, double voxel_multiplier);

} // namespace rko_lio::core
