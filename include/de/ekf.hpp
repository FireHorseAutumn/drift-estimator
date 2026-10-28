#pragma once

// Reference error-state Kalman filter over stacked rig extrinsics — the CPU
// correctness anchor. The GPU kernel (src/ekf.cu) is gated against this.
//
// State: error 6-vector per camera, dx = [omega; v] stacked 6N, defined
// against nominal poses nom_i: the live estimate is nom_i * Exp(dx_i).
// Process model is a random walk (extrinsic drift is slow: vibration,
// thermal), Q = spectral density * dt per axis, separate for rotation and
// translation.
//
// Observations are matcher-agnostic relative poses: z ~ T_a^-1 T_b with a
// 6x6 noise covariance. The error state is retracted into the nominals
// after EVERY accepted observation, so at update time dx = 0 and the
// residual r = Log(z^-1 * (nom_a^-1 * nom_b)) has Jacobians
// H = JrInv(r) * [-Adj(T_ab^-1) | I ], with T_ab = nom_a^-1 * nom_b and
// JrInv the inverse right Jacobian of SE(3) (I + ad/2 + ad^2/12 - ad^4/720
// + ... — the filter converges without it, but the gate's chi-square is
// only exact with the exact linearization).
//
// State convention: dx estimates the nominals' ERROR, nom_i ≈ true_i *
// Exp(dx_i); retraction therefore applies the NEGATIVE correction,
// nom_i <- nom_i * Exp(-dx_i). (Applying +dx doubles the error instead of
// cancelling it — caught by the synthetic drift gate.)
//
// Two decisions carried over from the archived pipeline's failure review:
//   1. The innovation gate is a proper chi-square on the 6-dof Mahalanobis
//      distance of the residual (default 95%). The archived slip applied a
//      threshold to the raw residual norm, which biased corrections toward
//      large-innovation directions; the gate here is on r^T S^-1 r and is
//      applied BEFORE the update touches the state.
//   2. Joseph form for every covariance write: P <- P - K(HP) - (HP)^T K^T
//      + K S K^T keeps P symmetric positive semidefinite under roundoff.
//      The reference materializes the dense blocks (rig N is single-digit);
//      the GPU kernel owns the fast version.
//
// Gauge note: relative-pose observations cannot see a rigid motion of the
// whole rig, so the absolute (gauge) mode is anchored only by the
// random-walk prior and wanders at that rate. What is observable — and what
// the tests gate on — is the rig-internal relative geometry, which is also
// the quantity that matters for calibration drift.
//
// Detect-before-correct: with correct_enabled = false the filter runs the
// predict/gate machinery and accumulates misfit statistics but never
// applies or retracts a correction — the nominals are left bitwise
// untouched. detection_stat() is the mean Mahalanobis distance of the last
// update() call measured against the OBSERVATION noise alone
// (r^T R^-1 r): the P-aware innovation distance cannot detect drift in this
// mode because the random-walk prior inflates P until the innovations look
// consistent — that is the filter being honest about its uncertainty, not
// evidence the rig is stable. The R-only misfit instead measures the
// rig against the matcher noise floor: ~6 (the 6-dof chi-square mean) on a
// consistent rig, growing quadratically with drift.
//
// Rejection paths, all explicit and all before any state touch:
//   - observations with camera indices outside [0, n_cams), non-finite
//     measurements, or a noise covariance that is not positive definite
//     (a NaN residual would PASS the chi-square gate: NaN > threshold is
//     false — it must never get there). These count neither toward the
//     return value nor the detection statistic.
//   - the chi-square gate itself.
//   - predict() with a negative or non-finite dt (never subtract process
//     noise); negative or NaN process densities clamp to zero at
//     construction.
//   - setNominal() with a wrongly sized vector (the rig keeps its previous
//     value — identity poses until the first valid call, so
//     update-before-setNominal is defined, not an out-of-bounds read).
//   - n_cams <= 0 builds an empty filter that rejects every observation.

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "se3.hpp"

namespace de {

struct RelPoseObs {
  int cam_a = 0, cam_b = 0;  // observes T_a^-1 T_b; indices into the rig
  Pose t_ab;                 // measured
  // Noise covariance; must be positive definite (report a noise floor, not
  // zero) — the gate distance and detection statistic are undefined
  // otherwise, and the observation is rejected.
  Eigen::Matrix<double, 6, 6> r =
      Eigen::Matrix<double, 6, 6>::Identity();
};

class DriftEkf {
 public:
  // q_rot / q_trans: random-walk spectral densities per axis (rad^2/s,
  // m^2/s); negative or NaN values clamp to zero. n_cams <= 0 builds an
  // empty filter (every observation is rejected).
  DriftEkf(int n_cams, double q_rot, double q_trans)
      : n_(n_cams > 0 ? n_cams : 0),
        q_rot_(q_rot > 0.0 ? q_rot : 0.0),
        q_trans_(q_trans > 0.0 ? q_trans : 0.0),
        nom_(n_),  // identity rig until setNominal
        dx_(Eigen::VectorXd::Zero(6 * n_)),
        p_(Eigen::MatrixXd::Zero(6 * n_, 6 * n_)) {}

  bool correct_enabled = true;

  // Wrong-sized rigs are rejected; the previous value is kept.
  void setNominal(const std::vector<Pose>& nom) {
    if (static_cast<int>(nom.size()) != n_) return;
    nom_ = nom;
  }
  const std::vector<Pose>& nominal() const { return nom_; }
  const Eigen::MatrixXd& covariance() const { return p_; }
  double detection_stat() const { return last_mean_d2_; }

  // Negative or non-finite dt is rejected: never subtract process noise.
  void predict(double dt) {
    if (!std::isfinite(dt) || dt < 0.0) return;
    for (int i = 0; i < n_; ++i) {
      p_.block<3, 3>(6 * i, 6 * i).diagonal().array() += q_rot_ * dt;
      p_.block<3, 3>(6 * i + 3, 6 * i + 3).diagonal().array() +=
          q_trans_ * dt;
    }
  }

  // Innovation-gated update over one frame's observations; returns the
  // number of valid observations that passed the gate (applied only when
  // correct_enabled). gate_chi2 = 11.07 is the 95% point on 6 dof.
  int update(const std::vector<RelPoseObs>& obs, double gate_chi2 = 11.07) {
    int accepted = 0;
    int valid = 0;
    double d2_sum = 0.0;
    for (const auto& o : obs) {
      // Explicit rejections, all before any state touch (see header).
      if (o.cam_a < 0 || o.cam_a >= n_ || o.cam_b < 0 || o.cam_b >= n_)
        continue;
      const Pose t_ab = nom_[o.cam_a].inverse() * nom_[o.cam_b];
      const Eigen::Matrix<double, 6, 1> r = logSE3(o.t_ab.inverse() * t_ab);
      if (!r.allFinite()) continue;
      // R must be positive definite: it keeps S > 0 (so the LDLT below is
      // sound) and makes the gate distance and detection statistic defined.
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> r_ldlt(o.r);
      if (r_ldlt.info() != Eigen::Success ||
          !(r_ldlt.vectorD().array() > 0.0).all())
        continue;
      ++valid;
      // H = JrInv(r) * [-Adj(t_ab^-1) | I].
      const Eigen::Matrix<double, 6, 6> jri = jrInvSE3(r);
      Eigen::Matrix<double, 6, 6> ha;
      ha.topLeftCorner<3, 3>() = -t_ab.R.transpose();
      ha.topRightCorner<3, 3>().setZero();
      ha.bottomLeftCorner<3, 3>() = t_ab.R.transpose() * skew(t_ab.t);
      ha.bottomRightCorner<3, 3>() = -t_ab.R.transpose();
      const Eigen::Matrix<double, 6, 6> ha_eff = jri * ha;
      const Eigen::Matrix<double, 6, 6>& hb_eff = jri;

      const auto Paa = p_.block<6, 6>(6 * o.cam_a, 6 * o.cam_a);
      const auto Pab = p_.block<6, 6>(6 * o.cam_a, 6 * o.cam_b);
      const auto Pbb = p_.block<6, 6>(6 * o.cam_b, 6 * o.cam_b);
      const Eigen::Matrix<double, 6, 6> S =
          ha_eff * Paa * ha_eff.transpose() + ha_eff * Pab * hb_eff.transpose() +
          hb_eff * Pab.transpose() * ha_eff.transpose() +
          hb_eff * Pbb * hb_eff.transpose() + o.r;
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> s_ldlt(S);
      const double d2 = r.dot(s_ldlt.solve(r));
      // Detection statistic: misfit against the observation noise alone,
      // accumulated over valid observations, gate-rejected ones included
      // (drift beyond the gate must still be detected).
      d2_sum += r.dot(r_ldlt.solve(r));
      if (d2 > gate_chi2) continue;  // gate BEFORE the state is touched
      ++accepted;
      if (!correct_enabled) continue;

      // K = P H^T S^-1 (dense 6N blocks; rig N is single-digit, the GPU
      // kernel owns the fast version).
      const Eigen::Matrix<double, 6, 6> s_inv =
          s_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      Eigen::MatrixXd hp = Eigen::MatrixXd::Zero(6, 6 * n_);
      for (int j = 0; j < n_; ++j)
        hp.block<6, 6>(0, 6 * j) =
            ha_eff * p_.block<6, 6>(6 * o.cam_a, 6 * j) +
            hb_eff * p_.block<6, 6>(6 * o.cam_b, 6 * j);
      Eigen::MatrixXd k = Eigen::MatrixXd::Zero(6 * n_, 6);
      k.block<6, 6>(6 * o.cam_a, 0) =
          (Paa * ha_eff.transpose() + Pab * hb_eff.transpose()) * s_inv;
      k.block<6, 6>(6 * o.cam_b, 0) =
          (Pab.transpose() * ha_eff.transpose() + Pbb * hb_eff.transpose()) *
          s_inv;

      dx_ += k * r;
      const Eigen::MatrixXd khp = k * hp;
      p_ += -khp - khp.transpose() + k * (S * k.transpose());
      const Eigen::MatrixXd pt = p_.transpose();  // alias-safe symmetrization
      p_ = 0.5 * (p_ + pt);
      retract();  // dx = 0 again before the next observation's Jacobians
    }
    last_mean_d2_ = valid > 0 ? d2_sum / static_cast<double>(valid) : 0.0;
    return accepted;
  }

  // Fold the NEGATIVE of the error state into the nominals and zero it:
  // dx estimates the error (nom ≈ true * Exp(dx)), so the correction is
  // nom <- nom * Exp(-dx). The exact reset also reparameterizes the error
  // coordinates by a Jacobian differing from I at FIRST order in |dx|;
  // carrying P unchanged is the standard ESKF approximation, accurate to
  // O(|dx|) — at the correction magnitudes here (<< 1e-3 rad per update)
  // the neglected relative change in P is << 1e-3, and the suite's
  // PSD/symmetry gates cover the accumulated effect.
  void retract() {
    for (int i = 0; i < n_; ++i)
      nom_[i] = nom_[i] * expSE3(-dx_.segment<6>(6 * i));
    dx_.setZero();
  }

 private:
  int n_;
  double q_rot_, q_trans_;
  std::vector<Pose> nom_;
  Eigen::VectorXd dx_;
  Eigen::MatrixXd p_;
  double last_mean_d2_ = 0.0;
};

}  // namespace de
