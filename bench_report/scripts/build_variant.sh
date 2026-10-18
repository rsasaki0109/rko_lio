#!/usr/bin/env bash
# Usage: build_variant.sh <name> <commit> <release|test>
# Builds rko_lio at <commit> in its own colcon workspace /home/user/ws/<name>.
set -eo pipefail
name="$1"; commit="$2"; kind="${3:-release}"
ws="/home/user/ws/${name}"
src="${ws}/src/rko_lio"
mkdir -p "${ws}/src"
if [[ ! -d "${src}" ]]; then
  git -C /home/user/rko_lio worktree add --detach "${src}" "${commit}" >/dev/null
fi
test "$(git -C "${src}" rev-parse HEAD)" = "$(git -C /home/user/rko_lio rev-parse "${commit}^{commit}")"
set +u; source /opt/ros/jazzy/setup.bash; set -u
export CCACHE_DIR=/home/user/.ccache CCACHE_BASEDIR=/home/user/ws CCACHE_MAXSIZE=4G CCACHE_NOHASHDIR=1
deps=(
  -DFETCHCONTENT_SOURCE_DIR_TSL-ROBIN-MAP=/home/user/tessil/robin-map
  -DFETCHCONTENT_SOURCE_DIR_EIGEN3=/home/user/deps_src/eigen-5.0.1
  -DFETCHCONTENT_SOURCE_DIR_SOPHUS=/home/user/strasdat/sophus
  -DFETCHCONTENT_SOURCE_DIR_TBB=/home/user/uxlfoundation/onetbb
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/home/user/nlohmann/json
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/user/catchorg/catch2
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
)
if [[ "${kind}" == "test" ]]; then
  testflags=(-DBUILD_TESTING=ON -DRKO_LIO_BUILD_TESTS=ON)
else
  testflags=(-DBUILD_TESTING=OFF)
fi
cd "${ws}"
start=$(date +%s)
colcon build --packages-select rko_lio --event-handlers console_direct- --cmake-args \
  -DCMAKE_BUILD_TYPE=Release "${testflags[@]}" -DRKO_LIO_BUILD_ROS=ON -DRKO_LIO_FETCH_CONTENT_DEPS=ON "${deps[@]}"
echo "BUILD_OK ${name} ${commit} ${kind} $(( $(date +%s) - start ))s"
