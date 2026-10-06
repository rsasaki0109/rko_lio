/*
 * MIT License
 *
 * Copyright (c) 2026 Sasaki
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


// Unit of the incoming IMU linear acceleration.
//
// The LIO expects m/s^2, but Livox drivers (MID-360, the Unitree Go2's LiDAR)
// publish acceleration in units of g. Read as m/s^2, gravity then has
// magnitude 1: initialization absorbs the missing ~8.8 m/s^2 into the
// accelerometer bias, and once the sensor tilts (a Go2 standing up) that bias
// becomes a spurious acceleration that moves the odometry.
#pragma once

#include <stdexcept>
#include <string>

namespace rko_lio::core {

inline constexpr double STANDARD_GRAVITY = 9.80665;

// m/s^2 per unit of the incoming acceleration: "mps2", "g", or "auto", which
// decides from the first sample. Below half of standard gravity, a reading in
// m/s^2 is implausible for a sensor that feels gravity, so it is taken as g.
inline double acceleration_scale(const std::string& unit, double first_magnitude) {
  if (unit == "mps2") {
    return 1.0;
  }
  if (unit == "g") {
    return STANDARD_GRAVITY;
  }
  if (unit == "auto") {
    return first_magnitude < 0.5 * STANDARD_GRAVITY ? STANDARD_GRAVITY : 1.0;
  }
  throw std::invalid_argument("imu_acceleration_unit must be mps2, g or auto, got: " + unit);
}

} // namespace rko_lio::core
