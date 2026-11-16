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

// ROS-free parameter access for the RKO-LIO drivers. The ROS 2 node implements it with
// rclcpp::Node::declare_parameter; the ROS-free driver with a plain name -> value table. The
// parameter names, types and defaults are those of the ROS 2 node (declare_driver_config).

#pragma once
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace rko_lio::driver {

class Parameters {
public:
  virtual ~Parameters() = default;

  // Each returns the configured value of `name`, or `default_value` when it is not set.
  virtual bool get_bool(const std::string& name, bool default_value) = 0;
  virtual int64_t get_int(const std::string& name, int64_t default_value) = 0;
  virtual double get_double(const std::string& name, double default_value) = 0;
  virtual std::string get_string(const std::string& name, const std::string& default_value) = 0;
  virtual std::vector<double> get_double_array(const std::string& name, const std::vector<double>& default_value) = 0;
  virtual std::vector<int64_t> get_int_array(const std::string& name,
                                             const std::vector<int64_t>& default_value) = 0;
  // A parameter without a default; throws when it is not set.
  virtual std::string get_required_string(const std::string& name) = 0;

  virtual void info(const std::string& message) = 0;
  virtual void warn(const std::string& message) = 0;

  // Same call shape as rclcpp::Node::declare_parameter<T>(name, default).
  template <typename T>
  T declare(const std::string& name, const T& default_value) {
    if constexpr (std::is_same_v<T, bool>) {
      return get_bool(name, default_value);
    } else if constexpr (std::is_same_v<T, int>) {
      return static_cast<int>(get_int(name, default_value));
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return get_int(name, default_value);
    } else if constexpr (std::is_same_v<T, double>) {
      return get_double(name, default_value);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return get_string(name, default_value);
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      return get_double_array(name, default_value);
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
      return get_int_array(name, default_value);
    } else {
      static_assert(!sizeof(T), "unsupported parameter type");
    }
  }
};

} // namespace rko_lio::driver
