# drift-estimator (A2 — the novel one)

Matcher-agnostic pose observations -> innovation-gated EKF over 6N stacked
extrinsics -> sliding-window pose-graph re-solve -> atomic extrinsics swap.
Detect-before-correct mode supported.

## Claims
- Persistent EKF: 18x18 covariance P resident in shared memory across the whole
  sequence. No library call sequence can keep state resident across frames —
  this is the "no library can express this" claim.
- WMMA Joseph form on 18x18 batched symmetric updates.
- Hamming-kNN as 1-bit GEMM with fused top-2 + Lowe-ratio epilogue (CPU matcher
  bar: 30-80 ms/pair).

## Kill criterion
- If the EKF does not reduce injected drift vs the no-correction baseline on
  EuRoC + synthetic drift, say so with numbers.
- If the 1-bit GEMM matcher loses to plain CPU AKAZE matching, publish that.

## Layout
- `include/de/`  — EKF math, Joseph update, matcher interface (headers)
- `src/`         — kernels (.cu), window optimizer, drift injection
- `tests/`       — correctness vs CPU reference EKF
- `benchmarks/`  — end-to-end drift reduction + matcher timing; JSON out
- `data/`        — EuRoC download script, synthetic-drift generator

## Standalone
No sibling repos, no archive code. All kernels are self-contained here
(EKF, Joseph form, 1-bit matcher). Datasets: EuRoC (public) plus synthetic
drift injection; SuperPoint/LightGlue ONNX weights are public.

## TODO (in order)
1. [ ] `src/ekf.cu` — persistent-EKF kernel (P in shared memory)
2. [ ] `include/de/joseph.cuh` — WMMA Joseph form
3. [ ] `src/matcher_1bit.cu` — bit-packed descriptors, XOR-popcount GEMM + fused epilogue
4. [ ] `data/gen_synthetic_drift.py` — inject drift into EuRoC GT poses
   (the C++ synthetic-drift harness lives in tests/test_ekf_cpu_ref.cpp;
   this script is only the EuRoC-side injection)
5. [x] `tests/test_ekf_cpu_ref.cpp` — CPU reference EKF landed FIRST (gate
   order): error-state filter over the stacked 6N extrinsics (random-walk
   drift model, chi-square innovation gate applied before any state touch,
   Joseph-form covariance writes), self-contained SE(3) in
   `include/de/se3.hpp`. Suite: analytic Jacobians vs numerical
   differentiation (7e-10), synthetic random-walk drift held at 1.8e-3 vs
   2.6e-2 uncorrected (14.7x), 10% gross outliers rejected (ungated control
   29x worse — the archived innovation-gate bias-slip regression, covered),
   P exactly symmetric / PSD over 5000 steps, detect-before-correct mode
   leaves nominals bitwise untouched with the R-only misfit stat separating
   a drifting rig from a consistent one at >1000x. Max-pass sweep same day:
   SE(3) series thresholds raised 1e-12 -> 1e-4 (closed-form coefficients
   collapse to exactly 0 below ~1.5e-8; translation round-trip was 8.7e-11,
   now 7.7e-15 worst across the antipode and both sides of the switch), and
   every rejection path made explicit — out-of-range camera indices,
   non-finite measurements, non-positive-definite observation noise (a NaN
   residual previously PASSED the gate: NaN > threshold is false),
   negative/NaN dt and process densities, wrong-sized nominals,
   update-before-setNominal, degenerate rig sizes. 25 checks. ctest 1/1.
6. [ ] `benchmarks/bench_drift.cpp` — drift reduction vs baseline; JSON out
7. [ ] README: EuRoC results table + kill-criterion outcome

Data: `data/fetch_euroc.sh` (ASL format, default V1_01). robotics.ethz.ch
was unreachable on 2026-10-28 (both 80/443 time out), so the script is
committed untested — verify the URL before relying on it.
