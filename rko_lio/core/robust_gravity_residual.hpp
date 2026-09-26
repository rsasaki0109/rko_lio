#pragma once

#include "util.hpp"
#include <cmath>

namespace rko_lio::core {
struct GravityResidualLoss {
  double weight;
  double cost;
};

// Experimental fixed-scale Tukey loss on the gravity-vector innovation.
// One g is a hypothesis to evaluate, not a calibrated sensor-noise bound.
// No history or dataset-specific switches; the IRLS weight is recomputed
// at each ICP iterate. The input is a finite squared residual norm.
inline GravityResidualLoss robust_gravity_residual(double squared_error) {
  const double scale_squared = GRAVITY_MAG * GRAVITY_MAG;
  if (squared_error >= scale_squared) return {0.0, scale_squared / 6.0};
  const double u = squared_error / scale_squared;
  const double remaining = 1.0 - u;
  // Polynomial form avoids cancellation near zero.
  return {remaining * remaining, squared_error * (3.0 - 3.0 * u + u * u) / 6.0};
}

struct GravityDirectionResidual {
  Eigen::Vector3d residual;
  GravityResidualLoss loss;
};

// Gravity constrains direction; transient specific-force magnitude must not
// by itself reject a direction-consistent reference or strengthen its torque.
inline GravityDirectionResidual robust_gravity_direction(const Eigen::Vector3d& predicted_gravity,
                                                         const Eigen::Vector3d& measured_gravity) {
  const double magnitude = measured_gravity.norm();
  if (!std::isfinite(magnitude) || magnitude <= 1e-8)
    return {Eigen::Vector3d::Zero(), {0.0, 0.0}};
  const Eigen::Vector3d residual = predicted_gravity - (GRAVITY_MAG / magnitude) * measured_gravity;
  return {residual, robust_gravity_residual(residual.squaredNorm())};
}
}  // namespace rko_lio::core
