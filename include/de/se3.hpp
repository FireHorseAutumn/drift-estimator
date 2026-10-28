#pragma once

// Minimal SO(3)/SE(3) for the reference EKF. Self-contained: A2 is standalone
// (no sibling repos), so the few group operations the filter needs live here.
//
// Conventions:
//   Pose T = (R, t) maps points p -> R p + t; composition a * b applies b
//   first (matrix order). Perturbations are 6-vectors d = [omega; v]
//   (rotation first), applied on the right: T (+) d = T * Exp(d).
//   Adj_T = [[R, 0], [t]_x R, R] for that ordering, so
//   T * Exp(d) = Exp(Adj_T * d) * T.

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace de {

inline Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
  Eigen::Matrix3d m;
  m << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(), w.x(), 0.0;
  return m;
}

inline Eigen::Matrix3d exp3(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d W = skew(w);
  // Series branch below 1e-4: in the closed forms 1 - cos(th) and th - sin(th)
  // cancel catastrophically (both round to exactly 0 below ~1.5e-8, zeroing
  // the coefficients), while the truncated series terms contribute < 1e-22
  // at the switch.
  if (th < 1e-4) {
    const double th2 = th * th;
    return Eigen::Matrix3d::Identity() + (1.0 - th2 / 6.0) * W +
           (0.5 - th2 / 24.0) * W * W;
  }
  const double a = std::sin(th) / th;
  const double b = (1.0 - std::cos(th)) / (th * th);
  return Eigen::Matrix3d::Identity() + a * W + b * W * W;
}

inline Eigen::Vector3d log3(const Eigen::Matrix3d& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

struct Pose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Pose operator*(const Pose& o) const { return {R * o.R, R * o.t + t}; }
  Pose inverse() const { return {R.transpose(), R.transpose() * (-t)}; }
};

// Exp([omega; v]) with the left-Jacobian V = I + b [w]_x + c [w]_x^2.
inline Pose expSE3(const Eigen::Matrix<double, 6, 1>& d) {
  const Eigen::Vector3d w = d.head<3>(), v = d.tail<3>();
  const Eigen::Matrix3d R = exp3(w);
  const double th = w.norm();
  const Eigen::Matrix3d W = skew(w);
  Eigen::Matrix3d V;
  if (th < 1e-4) {
    const double th2 = th * th;
    V = Eigen::Matrix3d::Identity() + (0.5 - th2 / 24.0) * W +
        (1.0 / 6.0 - th2 / 120.0) * W * W;
  } else {
    const double b = (1.0 - std::cos(th)) / (th * th);
    const double c = (th - std::sin(th)) / (th * th * th);
    V = Eigen::Matrix3d::Identity() + b * W + c * W * W;
  }
  return {R, V * v};
}

// Log(T) = [omega; V^-1 t].
inline Eigen::Matrix<double, 6, 1> logSE3(const Pose& T) {
  const Eigen::Vector3d w = log3(T.R);
  const double th = w.norm();
  const Eigen::Matrix3d W = skew(w);
  Eigen::Matrix3d Vinv;
  if (th < 1e-4) {
    const double th2 = th * th;
    Vinv = Eigen::Matrix3d::Identity() - 0.5 * W +
           (1.0 / 12.0 + th2 / 720.0) * W * W;
  } else {
    const double half = 0.5 * th;
    const double c =
        (1.0 - 0.5 * th * std::cos(half) / std::sin(half)) / (th * th);
    Vinv = Eigen::Matrix3d::Identity() - 0.5 * W + c * W * W;
  }
  Eigen::Matrix<double, 6, 1> d;
  d.head<3>() = w;
  d.tail<3>() = Vinv * T.t;
  return d;
}

inline Eigen::Matrix<double, 6, 6> adjoint(const Pose& T) {
  Eigen::Matrix<double, 6, 6> A;
  A.topLeftCorner<3, 3>() = T.R;
  A.topRightCorner<3, 3>().setZero();
  A.bottomLeftCorner<3, 3>() = skew(T.t) * T.R;
  A.bottomRightCorner<3, 3>() = T.R;
  return A;
}

// ad_[w;v] = [[w_x, 0], [v_x, w_x]], the SE(3) Lie-bracket matrix.
inline Eigen::Matrix<double, 6, 6> adSE3(const Eigen::Matrix<double, 6, 1>& d) {
  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  A.topLeftCorner<3, 3>() = skew(d.head<3>());
  A.bottomLeftCorner<3, 3>() = skew(d.tail<3>());
  A.bottomRightCorner<3, 3>() = skew(d.head<3>());
  return A;
}

// Inverse right Jacobian of SE(3) at r: d/dv Log(Exp(r) Exp(v)) at v = 0
// equals jrInvSE3(r) * v (BCH: ln(Exp(r) Exp(v)) ~= r + Jr(r)^-1 v for small
// v). Bernoulli series I + ad/2 + ad^2/12 - ad^4/720 + ... (group-general);
// truncated at ad^2 — the next term is O(|r|^4), below 1e-11 for the
// innovation magnitudes this filter sees (<= ~1e-2).
inline Eigen::Matrix<double, 6, 6> jrInvSE3(
    const Eigen::Matrix<double, 6, 1>& r) {
  const Eigen::Matrix<double, 6, 6> A = adSE3(r);
  return Eigen::Matrix<double, 6, 6>::Identity() + 0.5 * A +
         (1.0 / 12.0) * A * A;
}

}  // namespace de
