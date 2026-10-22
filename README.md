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
5. [ ] `tests/test_ekf_cpu_ref.cpp` — CPU reference, gate GPU against it
6. [ ] `benchmarks/bench_drift.cpp` — drift reduction vs baseline; JSON out
7. [ ] README: EuRoC results table + kill-criterion outcome
