/*
 * MIT License
 *
 * Copyright (c) 2026 rsasaki0109.
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

#include <rko_lio/core/bump_rotation_gate.hpp>

#include <catch2/catch_test_macros.hpp>

using rko_lio::core::BumpRotationGate;

TEST_CASE("Window and count of one fall back on every exceeding scan", "[bump_rotation_gate]") {
  BumpRotationGate gate(2.0, 1, 1, 0.0);
  REQUIRE_FALSE(gate.fall_back(1.9));
  REQUIRE(gate.fall_back(2.1));
  REQUIRE_FALSE(gate.fall_back(0.5));
  REQUIRE(gate.fall_back(3.0));
}

TEST_CASE("An isolated large correction keeps its bump result", "[bump_rotation_gate]") {
  BumpRotationGate gate(2.0, 5, 2, 0.0);
  REQUIRE_FALSE(gate.fall_back(2.5));  // first exceedance in the window
  for (int i = 0; i < 4; ++i) {
    REQUIRE_FALSE(gate.fall_back(0.3));
  }
  REQUIRE_FALSE(gate.fall_back(2.5));  // the earlier one has left the window
  REQUIRE(gate.fall_back(2.5));        // two within five scans
  REQUIRE_FALSE(gate.fall_back(0.3));  // below the limit never falls back
}

TEST_CASE("The hard limit falls back at once", "[bump_rotation_gate]") {
  BumpRotationGate gate(2.0, 5, 3, 6.0);
  REQUIRE_FALSE(gate.fall_back(5.0));
  REQUIRE(gate.fall_back(7.0));
}

TEST_CASE("A count above the window can never be met", "[bump_rotation_gate]") {
  BumpRotationGate gate(2.0, 1, 2, 0.0);
  for (int i = 0; i < 10; ++i) {
    REQUIRE_FALSE(gate.fall_back(10.0));
  }
}
