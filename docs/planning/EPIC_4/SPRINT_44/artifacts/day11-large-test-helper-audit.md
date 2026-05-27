# Sprint 44 Day 11 Artifact: Large-Test Helper Audit

## Purpose

Audit the four largest current test binaries so Sprint 44's maintainability
batch is driven by real helper/fixture duplication, not by file size alone,
and lock Day 12 to the smallest useful consolidation batch.

## 1. Current Large-Test Landscape

The current hotspot set remains:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_svd.c` = `3746`
- `tests/test_ldlt_csc.c` = `3637`
- `tests/test_qr.c` = `3291`

These files are all large enough to justify an audit, but they do not present
the same kind of maintainability problem.

The audit split is:

- clean first-helper target:
  - `tests/test_qr.c`
- strong later focused candidates:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_svd.c`

## 2. `tests/test_qr.c`: Best First Target

### Existing helper seam

`tests/test_qr.c` already has visible helper structure:

- `qr_reconstruction_error(...)`
- `compare_dense_sparse_qr(...)`

### Repeated patterns

The file repeats the same broad validation shapes across many scenarios:

- factor QR
- inspect rank / permutation properties
- compute reconstruction error
- compare dense and sparse paths
- compare QR and LU solve behavior
- assert residuals under variant matrix shapes:
  - square
  - tall
  - wide
  - nearly singular
  - larger fixture-driven cases

### Why this is the best Day 12 landing

- the seam is already partially factored
- the repeated validation shape is broad but coherent
- the likely cleanup can stay inside one file
- it does not require redesigning fixture ownership, corpus loading, or
  binary-level test organization

### Best bounded extraction shape

Day 12 should most likely extract:

- one shared QR result-validation helper
- one shared reconstruction / residual assertion helper

It should avoid broad file restructuring.

## 3. `tests/test_chol_csc.c`: Large but Highly Specialized

### Real duplication

The file contains visible repeated seams:

- SPD fixture builders
- scalar-vs-batched and scalar-vs-supernodal cross-check harnesses
- repeated solve/residual comparisons
- repeated supernode extract/writeback/eliminate-panel setup

### Why it is not the best first landing

Much of the repetition is specialized to subsystem-specific claims:

- supernodal panel operations
- CSC threshold dispatch
- SuiteSparse fixture spot-checks
- factor writeback / representation parity

### Audit conclusion

- this is a real maintainability target
- it is better suited to a later focused cleanup than to Sprint 44's first
  small helper batch

## 4. `tests/test_ldlt_csc.c`: Strong Fixture-Builder Signal, but Domain-Tied

### Real helper seams

The file already exposes strong reusable-looking helpers:

- `build_dense_ldlt_with_pivots(...)`
- `build_dense_spd(...)`
- `build_random_symmetric(...)`
- `build_kkt_5x5(...)`
- `build_kkt_10x10(...)`
- `build_symmetric(...)`
- `build_ldlt_from_triples(...)`
- `ldlt_csc_factor_state_matches(...)`
- `s20_solve_residual(...)`

### Repeated patterns

- indefinite fixture construction
- two-pass factor comparison
- solve residual validation
- dense-oracle comparison

### Why it is not the first Sprint 44 landing

The same file is tightly tied to specialized LDLT/CSC correctness contracts:

- pivot-size behavior
- supernode extract/writeback
- KKT regressions
- linked-list vs CSC parity
- permutation/composition details

### Audit conclusion

- the maintainability signal is real
- but the file is better handled in a later focused domain batch than in the
  first Sprint 44 helper pass

## 5. `tests/test_svd.c`: Broad Validation Repetition, Better as a Later Batch

### Existing helper seam

The file already contains useful helpers:

- `gk_reconstruction_error(...)`
- `orthogonality_error(...)`
- `validate_gk(...)`

### Repeated patterns

- UV reconstruction checks
- orthogonality checks
- full vs economy option shapes
- low-rank and rank-deficient families
- recurring `sparse_svd_compute(...)` plus sigma-order assertions

### Why it is not the first landing

The remaining repetition is spread across several SVD regimes rather than one
especially clean consolidation seam.

### Audit conclusion

- this is a good later maintainability target
- it is weaker than `tests/test_qr.c` as the first Sprint 44 extraction batch

## 6. Concrete Day 12 Target Set

Primary target:

- `tests/test_qr.c`

Best bounded extraction classes:

- one shared QR result-validation helper
- one shared reconstruction / residual assertion helper
- optionally one small setup wrapper if it removes obvious repeated setup

Optional second target only if the QR batch stays obviously small:

- one narrow helper seam from `tests/test_chol_csc.c`

## 7. Explicit Non-Goals

Day 12 should not become:

- file splitting
- a cross-file helper-library project
- a broad test-framework redesign
- a behavior-change batch
- a four-file cleanup wave

## Bottom Line

Day 11 confirms that Sprint 44 has a real large-test maintainability target,
but it should stay tightly bounded:

- `tests/test_qr.c` is the clearest first helper-consolidation landing
- `tests/test_chol_csc.c`, `tests/test_ldlt_csc.c`, and `tests/test_svd.c`
  remain legitimate later candidates
- Day 12 should extract one or two high-signal QR helpers, with any second
  file only participating if the batch remains obviously small
