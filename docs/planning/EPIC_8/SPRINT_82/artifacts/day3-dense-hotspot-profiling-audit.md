# Sprint 82 Day 3: Dense Hotspot Profiling Audit

## Purpose

Re-rank the live dense-helper and backend-ceiling costs against the current
solver, benchmark, and proof-owner surfaces so Sprint 82 moves from one real
contradiction center instead of another generic “performance improvement”
bucket.

## Main Result

Sprint 82's broad backend problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - Cholesky CSC dense-kernel descriptor and supernodal consumer lane
- strongest second target:
  - LDL^T backend/runtime parity and supernodal dense-kernel follow-through
- strongest third target:
  - QR and SVD dense-workspace ceiling
- strongest support-only but real target:
  - benchmark/runtime measurability and package/runtime interpretation

## Strongest Current Contradiction

The strongest current contradiction center is now explicit:

- `src/sparse_dense.c` still owns only a builtin scalar dense-kernel surface
  for the highest-value Cholesky inner kernels
- the current backend-aware seam is narrow:
  - one `chol_dense_kernels_t` builtin descriptor
  - one test-only override path
  - no maintained optional accelerated runtime path yet
- `src/sparse_chol_csc_supernodal.c` is already the clearest direct-family
  consumer because it runs:
  - dense diagonal factor
  - batched panel solve
  - backend-contract failure handling

That makes the highest-value Sprint 82 first move the dense-kernel
descriptor/runtime-selection seam centered on the Cholesky CSC supernodal lane,
not package wording or broader solver-family adoption first.

## Secondary Backend Ceilings

### LDL^T Backend / Runtime Parity
The strongest second contradiction remains the LDL^T direct-family lane:

- `src/sparse_ldlt.c` already has explicit solver-level backend selection
- `AUTO`, `LINKED_LIST`, and `CSC` dispatch semantics are already caller-visible
- that makes LDL^T the strongest second adoption lane after the first
  dense-kernel descriptor landing

### QR / SVD Dense-Workspace Ceiling
The strongest third contradiction remains the dense-workspace-heavy QR and SVD
surfaces:

- `src/sparse_qr.c` still carries large local dense workspace and column
  extraction/update cost
- `src/sparse_svd.c` still carries dense-intermediate and outer-product
  tradeoff logic

These are real Epic 8 targets, but they do not read like the best first
bounded accelerated-backend landing center compared with the supernodal
Cholesky lane.

## Support-Only But Real Follow-Through

The strongest support-tier backend surfaces are now explicit:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_svd.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`

These are real Sprint 82 surfaces, but they should remain support-only unless
the first backend landing truly forces them to move.

## Interpretation

The useful Day 3 clarification is now fixed:

- Sprint 82 should begin with the dense-kernel descriptor and Cholesky CSC
  supernodal consumer lane
- LDL^T backend/runtime parity should remain the strongest second
  implementation lane
- QR/SVD dense-workspace follow-through remains real, but it is not the first
  contradiction center

## Exit State

- Sprint 82 now has one ranked live backend contradiction map grounded in the
  tree.
- The first implementation center is fixed to the Cholesky dense-kernel
  descriptor/runtime lane before Day 4 boundary work.
