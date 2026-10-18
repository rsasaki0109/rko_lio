"""Self-checks for the two exact ICP speedups (local verification only, never committed).

Usage: python3 verify_exact_patches.py <rko_lio source dir> nn|grain
  nn    : every neighbour search that the cache skips is also run as a full uncached search and the
          two answers are compared bit for bit (point and distance).
  grain : every ICP system is also summed with the old reduction (one task per point, grain size 1)
          and H, b, chi, weight sum and correspondence count are compared bit for bit.
The counts are printed to stderr at exit as EXACT_CHECK lines. Every replacement must match exactly
once, otherwise the script fails.
"""
import sys
from pathlib import Path

root, mode = Path(sys.argv[1]), sys.argv[2]


def patch(rel, pairs):
    path = root / rel
    text = path.read_text()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{rel}: anchor found {count} times: {old[:80]!r}")
        text = text.replace(old, new)
    path.write_text(text)


if mode == 'nn':
    patch("rko_lio/core/voxel_hash_map.cpp", [
        ("""namespace rko_lio::core {

VoxelHashMap::VoxelHashMap(""",
         """#include <atomic>
#include <cstdio>
#include <cstring>
namespace rko_lio::core {
struct ExactCheck {
  std::atomic<unsigned long long> checked{0}, mismatched{0};
  ~ExactCheck() {
    std::fprintf(stderr, "EXACT_CHECK nn skipped_searches_checked=%llu mismatched=%llu\\n", checked.load(),
                 mismatched.load());
  }
};
ExactCheck exact_check;

VoxelHashMap::VoxelHashMap("""),
        ("""      if (distance + moved + reuse_margin(query) < cache.reference_others_distance) {
        return std::make_tuple(cache.reference_neighbor, distance);""",
         """      if (distance + moved + reuse_margin(query) < cache.reference_others_distance) {
        const auto [full_point, full_distance] = get_closest_neighbor(query);
        exact_check.checked.fetch_add(1, std::memory_order_relaxed);
        if (std::memcmp(full_point.data(), cache.reference_neighbor.data(), 3 * sizeof(double)) != 0 ||
            std::memcmp(&full_distance, &distance, sizeof(double)) != 0) {
          exact_check.mismatched.fetch_add(1, std::memory_order_relaxed);
        }
        return std::make_tuple(cache.reference_neighbor, distance);"""),
    ])
elif mode == 'grain':
    patch("rko_lio/core/lio.cpp", [
        ("""#include "lio.hpp"
""",
         """#include "lio.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
namespace {
struct ExactCheck {
  std::atomic<unsigned long long> checked{0}, mismatched{0};
  ~ExactCheck() {
    std::fprintf(stderr, "EXACT_CHECK grain systems_checked=%llu mismatched=%llu\\n", checked.load(),
                 mismatched.load());
  }
};
ExactCheck exact_check;
} // namespace
"""),
        ("""  SystemSumBody body(add_point_system);
  tbb::parallel_deterministic_reduce(
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend(), kLeafPoints}, body);""",
         """  SystemSumBody body(add_point_system);
  tbb::parallel_deterministic_reduce(
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend(), kLeafPoints}, body);
  // The old reduction: one task per point (grain size 1), each range summed point by point.
  struct OldBody {
    decltype(add_point_system)& add_point;
    SystemSum sum;
    explicit OldBody(decltype(add_point_system)& add_point_) : add_point(add_point_) {}
    OldBody(OldBody& other, tbb::split) : add_point(other.add_point) {}
    void operator()(const tbb::blocked_range<points_iterator>& r) {
      for (auto point = r.begin(); point != r.end(); ++point) {
        add_point(*point, sum);
      }
    }
    void join(OldBody& rhs) { sum.add(rhs.sum.H, rhs.sum.b, rhs.sum.chi, rhs.sum.weight, rhs.sum.correspondences); }
  };
  OldBody old_body(add_point_system);
  tbb::parallel_deterministic_reduce(tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()}, old_body);
  exact_check.checked.fetch_add(1, std::memory_order_relaxed);
  if (std::memcmp(old_body.sum.H.data(), body.sum.H.data(), 36 * sizeof(double)) != 0 ||
      std::memcmp(old_body.sum.b.data(), body.sum.b.data(), 6 * sizeof(double)) != 0 ||
      std::memcmp(&old_body.sum.chi, &body.sum.chi, sizeof(double)) != 0 ||
      std::memcmp(&old_body.sum.weight, &body.sum.weight, sizeof(double)) != 0 ||
      old_body.sum.correspondences != body.sum.correspondences) {
    exact_check.mismatched.fetch_add(1, std::memory_order_relaxed);
  }"""),
    ])
else:
    raise SystemExit(f"unknown mode {mode}")
