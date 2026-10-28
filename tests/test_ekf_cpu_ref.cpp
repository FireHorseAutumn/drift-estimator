// CPU reference EKF gate (CV-7 ordering: reference first, GPU kernel later).
//
//   1. se3 sanity: Exp/Log round-trip, adjoint identity.
//   2. Analytic observation Jacobians vs numerical differentiation.
//   3. Synthetic drift: slow random walk on all true extrinsics, noisy
//      relative-pose observations; the EKF must hold the rig-internal
//      (gauge-free) error far below the no-correction baseline.
//   4. Innovation gate: 10% gross outliers must not contaminate the estimate
//      (the archived bias-slip regression); an ungated control run must be
//      visibly worse, so the gate is doing the work.
//   5. P stays symmetric PSD over a long run (Joseph form).
//   6. Detect-before-correct: nominals bitwise untouched; the detection
//      statistic separates a drifting rig from a consistent one.
//   7. Input validation: out-of-range indices, degenerate noise, NaN
//      measurements, bad dt/densities, wrong-sized nominals, degenerate
//      rig sizes — every rejection path explicit, state bitwise untouched.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "de/ekf.hpp"

using namespace de;

namespace {

int failures = 0;

void check(bool ok, const std::string& name) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

double uniform01(uint64_t& s) {
  return static_cast<double>(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

double gauss(uint64_t& s) {
  const double u1 = 1e-300 + (1.0 - 1e-300) * uniform01(s);
  return std::sqrt(-2.0 * std::log(u1)) *
         std::cos(6.28318530717958647692 * uniform01(s));
}

Eigen::Matrix<double, 6, 1> gauss6(uint64_t& s) {
  Eigen::Matrix<double, 6, 1> v;
  for (int i = 0; i < 6; ++i) v(i) = gauss(s);
  return v;
}

// The test rig: three cameras, ~25 cm baselines.
std::vector<Pose> rigNominals() {
  Pose c0;
  Pose c1{exp3({0.02, -0.03, 0.01}), {0.25, 0.02, 0.03}};
  Pose c2{exp3({-0.015, 0.02, -0.025}), {0.10, -0.18, 0.06}};
  return {c0, c1, c2};
}

struct SimCfg {
  int steps = 2000;
  double dt = 0.1;
  double q_rot = 5e-7;    // true random-walk density, rad^2/s
  double q_trans = 5e-7;  // m^2/s
  double s_rot = 2e-3;    // observation noise sigma, rad
  double s_trans = 2e-3;  // m
  double outlier_frac = 0.0;
  double outlier_rad = 0.5;
};

// Gauge-free rig error: mean |Log(est_ab^-1 * true_ab)| over the pairs.
double rigError(const std::vector<Pose>& est, const std::vector<Pose>& tru) {
  double acc = 0.0;
  int cnt = 0;
  for (const auto& [a, b] : {std::pair{0, 1}, {std::pair{0, 2}}, {1, 2}}) {
    const Pose e = est[a].inverse() * est[b];
    const Pose t = tru[a].inverse() * tru[b];
    acc += logSE3(e.inverse() * t).norm();
    ++cnt;
  }
  return acc / cnt;
}

struct RunOut {
  double tail_rig_ekf = 0.0;  // mean rig error over the last 100 steps
  double tail_rig_base = 0.0;
  double tail_d2 = 0.0;  // mean pre-gate Mahalanobis d2 over the tail
  int accepted = 0, total = 0;
};

RunOut runSim(const SimCfg& cfg, bool correct, bool gate_on, uint64_t seed) {
  const std::vector<Pose> nom0 = rigNominals();
  std::vector<Pose> tru = nom0;
  DriftEkf ekf(3, cfg.q_rot, cfg.q_trans);
  ekf.setNominal(nom0);
  ekf.correct_enabled = correct;
  uint64_t rng = seed;
  Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
  R.topLeftCorner<3, 3>() *= cfg.s_rot * cfg.s_rot;
  R.bottomRightCorner<3, 3>() *= cfg.s_trans * cfg.s_trans;

  RunOut out;
  const double sqr_qr = std::sqrt(cfg.q_rot * cfg.dt);
  const double sqr_qt = std::sqrt(cfg.q_trans * cfg.dt);
  const int tail = std::min(100, cfg.steps);
  for (int k = 0; k < cfg.steps; ++k) {
    for (int i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 6, 1> w = gauss6(rng);
      w.head<3>() *= sqr_qr;
      w.tail<3>() *= sqr_qt;
      tru[i] = tru[i] * expSE3(w);
    }
    std::vector<RelPoseObs> obs;
    for (const auto& [a, b] : {std::pair{0, 1}, {std::pair{0, 2}}, {1, 2}}) {
      RelPoseObs o;
      o.cam_a = a;
      o.cam_b = b;
      o.r = R;
      Eigen::Matrix<double, 6, 1> n = gauss6(rng);
      n.head<3>() *= cfg.s_rot;
      n.tail<3>() *= cfg.s_trans;
      o.t_ab = tru[a].inverse() * tru[b] * expSE3(n);
      if (uniform01(rng) < cfg.outlier_frac) {
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        g(0) = cfg.outlier_rad;
        o.t_ab = o.t_ab * expSE3(g);
      }
      obs.push_back(o);
    }
    ekf.predict(cfg.dt);
    out.accepted += ekf.update(obs, gate_on ? 11.07 : 1e300);
    out.total += static_cast<int>(obs.size());
    if (k >= cfg.steps - tail) {
      out.tail_rig_ekf += rigError(ekf.nominal(), tru);
      out.tail_rig_base += rigError(nom0, tru);
      out.tail_d2 += ekf.detection_stat();
    }
  }
  out.tail_rig_ekf /= tail;
  out.tail_rig_base /= tail;
  out.tail_d2 /= tail;
  return out;
}

}  // namespace

int main() {
  // --- 1. se3 sanity ---
  {
    uint64_t s = 42;
    const Eigen::Matrix<double, 6, 1> d = gauss6(s) * 0.01;
    check((logSE3(expSE3(d)) - d).norm() < 1e-12, "se3: Exp/Log round-trip");
    Pose t{exp3({0.1, -0.2, 0.3}), {0.4, -0.5, 0.6}};
    const Pose lhs = t * expSE3(d);
    const Pose rhs = expSE3(adjoint(t) * d) * t;
    check((lhs.R - rhs.R).norm() + (lhs.t - rhs.t).norm() < 1e-12,
          "se3: T Exp(d) == Exp(Adj_T d) T");
    // Branch-cut sweep: Exp(Log(T)) == T must hold on the whole group,
    // including the antipodal cut, both sides of the 1e-4 series threshold,
    // and the cancellation region of the closed-form coefficients.
    double cut = 0.0;
    for (const double th : {3.141592653589793, 3.141592652589793,
                            3.140592653589793, 0.5, 1.1e-4, 9e-5, 1e-8,
                            1e-13}) {
      const Eigen::Vector3d w = Eigen::Vector3d(0.3, -0.5, 0.8).normalized() * th;
      Eigen::Matrix<double, 6, 1> d6;
      d6.head<3>() = w;
      d6.tail<3>() = Eigen::Vector3d(0.1, -0.2, 0.3);
      const Pose tp = expSE3(d6);
      const Pose rt = expSE3(logSE3(tp));
      cut = std::max(cut, (rt.R - tp.R).norm() + (rt.t - tp.t).norm());
    }
    std::printf("  se3 branch-cut round-trip worst = %.3e\n", cut);
    check(cut < 1e-12, "se3: Exp(Log(T)) == T across the branch cuts");
  }

  // --- 2. Jacobians vs numerical differentiation ---
  {
    uint64_t s = 7;
    const std::vector<Pose> nom = rigNominals();
    const Pose z = nom[0].inverse() * nom[1] * expSE3(gauss6(s) * 1e-3);
    auto residual = [&](const std::vector<Pose>& nm) {
      return logSE3(z.inverse() * (nm[0].inverse() * nm[1]));
    };
    const Eigen::Matrix<double, 6, 1> r0 = residual(nom);
    const Pose t_ab = nom[0].inverse() * nom[1];
    Eigen::Matrix<double, 6, 6> ha;
    ha.topLeftCorner<3, 3>() = -t_ab.R.transpose();
    ha.topRightCorner<3, 3>().setZero();
    ha.bottomLeftCorner<3, 3>() = t_ab.R.transpose() * skew(t_ab.t);
    ha.bottomRightCorner<3, 3>() = -t_ab.R.transpose();
    const Eigen::Matrix<double, 6, 6> jri = jrInvSE3(r0);
    const Eigen::Matrix<double, 6, 6> ha_eff = jri * ha;
    const Eigen::Matrix<double, 6, 6> hb_eff = jri;
    double worst = 0.0;
    const double eps = 1e-7;
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> e = Eigen::Matrix<double, 6, 1>::Zero();
      e(j) = eps;
      std::vector<Pose> nm = nom;
      nm[0] = nm[0] * expSE3(e);  // perturbs dx_a component j
      worst =
          std::max(worst, ((residual(nm) - r0) / eps - ha_eff.col(j)).norm());
      nm = nom;
      nm[1] = nm[1] * expSE3(e);  // perturbs dx_b component j
      worst =
          std::max(worst, ((residual(nm) - r0) / eps - hb_eff.col(j)).norm());
    }
    std::printf("  jacobian worst |num - analytic| = %.3e\n", worst);
    check(worst < 1e-5,
          "observation Jacobians match numerical differentiation");
  }

  // --- 3. synthetic drift reduction ---
  {
    const RunOut o = runSim(SimCfg{}, true, true, 0xDE17);
    std::printf("  drift: tail rig-error ekf %.4e vs baseline %.4e (ratio %.3f), "
                "accepted %d/%d\n",
                o.tail_rig_ekf, o.tail_rig_base,
                o.tail_rig_ekf / o.tail_rig_base, o.accepted, o.total);
    check(o.tail_rig_ekf < o.tail_rig_base / 5.0,
          "EKF reduces synthetic drift >= 5x vs no correction");
    check(o.tail_rig_ekf < 5e-3, "EKF tail rig error below 5e-3");
  }

  // --- 4. innovation gate under 10% gross outliers ---
  {
    SimCfg cfg;
    cfg.outlier_frac = 0.10;
    const RunOut gated = runSim(cfg, true, true, 0x6A7E);
    const RunOut raw = runSim(cfg, true, false, 0x6A7E);
    std::printf("  gate: tail rig-error gated %.4e vs ungated %.4e\n",
                gated.tail_rig_ekf, raw.tail_rig_ekf);
    check(gated.tail_rig_ekf < 5e-3,
          "gated run stays on the rig under outliers");
    check(raw.tail_rig_ekf > 3.0 * gated.tail_rig_ekf,
          "ungated control is contaminated (the gate does the work)");
  }

  // --- 5. P symmetric PSD over a long run ---
  {
    SimCfg cfg;
    cfg.steps = 5000;
    const std::vector<Pose> nom0 = rigNominals();
    std::vector<Pose> tru = nom0;
    DriftEkf ekf(3, cfg.q_rot, cfg.q_trans);
    ekf.setNominal(nom0);
    uint64_t rng = 0x9C0D;
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
    R.topLeftCorner<3, 3>() *= cfg.s_rot * cfg.s_rot;
    R.bottomRightCorner<3, 3>() *= cfg.s_trans * cfg.s_trans;
    const double sqr_qr = std::sqrt(cfg.q_rot * cfg.dt);
    const double sqr_qt = std::sqrt(cfg.q_trans * cfg.dt);
    for (int k = 0; k < cfg.steps; ++k) {
      for (int i = 0; i < 3; ++i) {
        Eigen::Matrix<double, 6, 1> w = gauss6(rng);
        w.head<3>() *= sqr_qr;
        w.tail<3>() *= sqr_qt;
        tru[i] = tru[i] * expSE3(w);
      }
      std::vector<RelPoseObs> obs;
      for (const auto& [a, b] : {std::pair{0, 1}, {std::pair{0, 2}}, {1, 2}}) {
        RelPoseObs o;
        o.cam_a = a;
        o.cam_b = b;
        o.r = R;
        Eigen::Matrix<double, 6, 1> n = gauss6(rng);
        n.head<3>() *= cfg.s_rot;
        n.tail<3>() *= cfg.s_trans;
        o.t_ab = tru[a].inverse() * tru[b] * expSE3(n);
        obs.push_back(o);
      }
      ekf.predict(cfg.dt);
      ekf.update(obs);
    }
    const Eigen::MatrixXd& P = ekf.covariance();
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
    const double min_ev = es.eigenvalues().minCoeff();
    std::printf("  psd: max asym %.3e, min eigenvalue %.3e\n", asym, min_ev);
    check(asym == 0.0, "P exactly symmetric (re-symmetrized every write)");
    check(min_ev > -1e-12, "P positive semidefinite to roundoff");
  }

  // --- 6. detect-before-correct ---
  {
    const std::vector<Pose> nom0 = rigNominals();
    SimCfg clean_cfg;
    clean_cfg.steps = 400;
    clean_cfg.q_rot = 0.0;  // consistent rig: no drift, matched filter prior
    clean_cfg.q_trans = 0.0;
    SimCfg drift_cfg;
    drift_cfg.steps = 400;
    drift_cfg.q_rot = 5e-5;  // 100x the default drift for a loud signal
    drift_cfg.q_trans = 5e-5;
    const RunOut clean = runSim(clean_cfg, false, true, 0xC1EA);
    const RunOut drifted = runSim(drift_cfg, false, true, 0xD21F);

    DriftEkf probe(3, 1e-7, 1e-7);
    probe.correct_enabled = false;
    probe.setNominal(nom0);
    RelPoseObs o;
    o.cam_a = 0;
    o.cam_b = 1;
    o.t_ab = nom0[0].inverse() * nom0[1];
    o.r *= 1e-6;
    probe.predict(0.1);
    probe.update({o});
    bool bitwise_same = true;
    for (int i = 0; i < 3; ++i)
      if ((probe.nominal()[i].R - nom0[i].R).norm() != 0.0 ||
          (probe.nominal()[i].t - nom0[i].t).norm() != 0.0)
        bitwise_same = false;
    check(bitwise_same, "detect mode leaves nominals bitwise untouched");
    std::printf("  detect: tail mean d2 clean %.3f vs drifted %.3f\n",
                clean.tail_d2, drifted.tail_d2);
    check(clean.tail_d2 > 2.0 && clean.tail_d2 < 14.0,
          "clean rig detection stat near the 6-dof chi-square mean");
    check(drifted.tail_d2 > 4.0 * clean.tail_d2,
          "drifted rig elevates the detection stat");
  }

  // --- 7. invalid inputs: every rejection path is explicit ---
  {
    const std::vector<Pose> nom = rigNominals();
    RelPoseObs good;
    good.cam_a = 0;
    good.cam_b = 1;
    good.t_ab = nom[0].inverse() * nom[1];  // exactly consistent: r = 0
    RelPoseObs oob = good;
    oob.cam_b = 5;
    RelPoseObs neg_idx = good;
    neg_idx.cam_a = -1;
    RelPoseObs zero_r = good;
    zero_r.r.setZero();
    RelPoseObs neg_r = good;
    neg_r.r(0, 0) = -1.0;
    RelPoseObs nan_z = good;
    nan_z.t_ab.t.x() = std::numeric_limits<double>::quiet_NaN();

    DriftEkf ekf(3, 1e-6, 1e-6);
    ekf.setNominal(nom);
    ekf.predict(0.1);  // nonzero P, so "untouched" below is observable
    const Eigen::MatrixXd p_snap = ekf.covariance();
    const std::vector<Pose> nom_snap = ekf.nominal();
    const int acc = ekf.update({oob, neg_idx, zero_r, neg_r, nan_z});
    double touched = (ekf.covariance() - p_snap).norm();
    for (int i = 0; i < 3; ++i)
      touched += (ekf.nominal()[i].R - nom_snap[i].R).norm() +
                 (ekf.nominal()[i].t - nom_snap[i].t).norm();
    check(acc == 0,
          "invalid observations rejected (oob index, zero/indefinite R, NaN z)");
    check(touched == 0.0,
          "rejected observations leave the state bitwise untouched");
    check(ekf.detection_stat() == 0.0,
          "rejected observations excluded from the detection stat");

    // A valid observation in the same frame is still processed; the NaN one
    // must not drag it down. (The NaN residual would PASS the chi-square
    // gate if it ever reached it: NaN > threshold is false.)
    check(ekf.update({good, nan_z}) == 1 && ekf.detection_stat() < 1e-24,
          "valid observation in a mixed frame still accepted");

    DriftEkf ekf2(3, 1e-6, 1e-6);
    ekf2.setNominal(nom);
    ekf2.predict(0.1);
    const Eigen::MatrixXd p0 = ekf2.covariance();
    ekf2.predict(-0.05);
    ekf2.predict(std::numeric_limits<double>::quiet_NaN());
    check((ekf2.covariance() - p0).norm() == 0.0,
          "negative/NaN dt rejected by predict");

    DriftEkf ekf3(3, -1e-6, std::numeric_limits<double>::quiet_NaN());
    ekf3.setNominal(nom);
    ekf3.predict(0.1);
    check(ekf3.covariance().diagonal().sum() == 0.0,
          "negative/NaN process densities clamp to zero");

    const std::vector<Pose> two{nom[0], nom[1]};
    ekf3.setNominal(two);
    check(ekf3.nominal().size() == 3, "wrong-sized setNominal rejected");

    DriftEkf bare(3, 1e-6, 1e-6);  // setNominal never called: identity rig
    bare.predict(0.1);
    bare.update({good});
    check(bare.covariance().allFinite(), "update before setNominal is defined");

    DriftEkf zero_cam(0, 1e-6, 1e-6);
    check(zero_cam.update({good}) == 0,
          "zero-camera rig rejects all observations");
    DriftEkf neg_cam(-2, 1e-6, 1e-6);
    check(neg_cam.update({good}) == 0,
          "negative camera count builds an empty filter");
  }

  std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
