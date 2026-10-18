"""Experiment (local only): add_points with a read-only parallel pass (voxel + existing block)
and the unchanged serial insertion loop, which prefetches the block of the point D ahead.

Usage: python3 exp_addpoints.py <rko_lio source dir> <D>
Apply on top of a tree that already has prefetch()/prefetch_points() (f4a6547 or later).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ahead = int(sys.argv[2])
path = root / "rko_lio/core/voxel_hash_map.cpp"
text = path.read_text()
old = """void VoxelHashMap::add_points(const std::vector<Eigen::Vector3d>& points) {
  const double map_resolution_sq = voxel_size_ * voxel_size_ / max_points_per_voxel_;
  std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d& p) {
    const Voxel voxel = point_to_voxel(p, inv_voxel_size_);
    auto it = map_.try_emplace(voxel).first;"""
new = f"""void VoxelHashMap::add_points(const std::vector<Eigen::Vector3d>& points) {{
  const double map_resolution_sq = voxel_size_ * voxel_size_ / max_points_per_voxel_;
  std::vector<Voxel> voxels(points.size());
  std::vector<const Eigen::Vector3d*> blocks(points.size());
  std::vector<std::uint32_t> sizes(points.size());
  tbb::parallel_for(tbb::blocked_range<std::size_t>{{0, points.size(), 256}}, [&](const tbb::blocked_range<std::size_t>& r) {{
    for (std::size_t i = r.begin(); i != r.end(); ++i) {{
      voxels[i] = point_to_voxel(points[i], inv_voxel_size_);
      const auto search = map_.find(voxels[i]);
      blocks[i] = search != map_.end() ? search.value().data() : nullptr;
      sizes[i] = search != map_.end() ? static_cast<std::uint32_t>(search.value().size()) : 0U;
    }}
  }});
  std::size_t index = 0;
  std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d& p) {{
    const std::size_t i = index++;
    if (i + {ahead} < points.size()) {{
      prefetch_points(blocks[i + {ahead}], sizes[i + {ahead}]);
    }}
    const Voxel& voxel = voxels[i];
    auto it = map_.try_emplace(voxel).first;"""
assert text.count(old) == 1, "add_points anchor"
path.write_text(text.replace(old, new))
print("add_points experiment applied", root, ahead)
