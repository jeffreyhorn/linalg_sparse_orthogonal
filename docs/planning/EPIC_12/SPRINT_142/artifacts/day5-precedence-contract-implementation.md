# Day 5 Precedence Contract Implementation

## Purpose

Day 5 lands the mechanical precedence contract surface selected from the
Day 4 design. The implementation result is intentionally conservative: the
audited behavior already matches the Day 4 contract for public typed backend
dispatch, typed analysis-vs-env precedence, OpenMP/report context, and
sentinel row boundaries. No source behavior or public ABI needed to change.

The Day 5 implementation batch therefore consists of:

- a validation-backed contract ledger tying each precedence rule to an
  executable owner;
- focused local validation of the selected owner tests;
- explicit non-implementation decisions for controls whose promotion belongs
  to Day 6.

## Implemented Contract Ledger

| Contract rule | Mechanical owner | Day 5 implementation result | Validation |
| --- | --- | --- | --- |
| Explicit Cholesky backend beats AUTO threshold. | `sparse_cholesky_opts_t::backend`, `src/sparse_cholesky.c`, `tests/test_chol_csc.c` | Existing implementation retained. Forced linked-list and forced CSC paths are already executable through public options. | `build/test_chol_csc` passed. |
| Cholesky AUTO uses `SPARSE_CSC_THRESHOLD`. | `s63_cholesky_dispatch_select_backend`, `tests/test_chol_csc.c`, `tests/test_direct_csc_dispatch.c` | Existing implementation retained. Below-threshold and at/above-threshold paths remain validated by `used_csc_path`. | `build/test_chol_csc` passed. |
| Invalid Cholesky backend enum is rejected. | `sparse_cholesky_factor_opts`, integration retry tests | Existing implementation retained. Invalid enum returns `SPARSE_ERR_BADARG`; retry behavior remains covered. | `build/test_chol_csc` passed. |
| Explicit LDLT backend beats AUTO threshold. | `sparse_ldlt_opts_t::backend`, `src/sparse_ldlt.c`, `tests/test_ldlt_backend_dispatch.c` | Existing implementation retained. Forced linked-list and forced CSC paths are already executable through public options. | `build/test_ldlt_backend_dispatch` passed. |
| LDLT AUTO uses `SPARSE_CSC_THRESHOLD`. | `ldlt_dispatch_select_backend`, `tests/test_ldlt_backend_dispatch.c` | Existing implementation retained. Below-threshold and at/above-threshold paths remain validated by `used_csc_path`. | `build/test_ldlt_backend_dispatch` passed. |
| LDLT forced-CSC empty-matrix exception remains documented but not hardened with a new public fixture. | `ldlt_dispatch_select_backend`, `include/sparse_ldlt.h` | No new test added because `sparse_create(0,0)` is not a public matrix fixture in this library. Avoided fabricating an internal `SparseMatrix` shell only to reach private state. | Existing LDLT dispatch validation passed; Day 6/Day 10 may clarify wording if this exception should remain public documentation. |
| Eigensolver explicit backend beats AUTO. | `sparse_eigs_opts_t::backend`, `src/sparse_eigs.c`, eigensolver tests | Existing implementation retained. Explicit grow-m, thick-restart, and LOBPCG selections are already validated. | `build/test_eigs_thick_restart` and `build/test_eigs_lobpcg` passed. |
| Eigensolver AUTO priority is deterministic. | `s46_select_backend`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Existing implementation retained. AUTO LOBPCG priority, thick-restart threshold, and grow-m fallback are already executable. | `build/test_eigs_thick_restart` and `build/test_eigs_lobpcg` passed. |
| Typed analysis fields override compatibility env vars. | `sparse_analysis_opts_t::reorder_opts`, `src/sparse_analysis.c`, `tests/test_reorder_nd.c` | Existing implementation retained. Tests cover typed root bisection, max-n, coarsening, coarsest bisection, separator lift, and supernodal postorder overrides. | `build/test_reorder_nd` passed with one known skip unrelated to precedence failure. |
| Compatibility env values remain scoped and defaultable. | ND/FM parsers, dense helper parsers, SVD low-rank parser | No promotion or behavior change on Day 5. Env-only controls remain compatibility/maintainer controls until Day 6 selection. | Covered by existing focused tests where present; deferred controls listed below. |
| OpenMP and `OMP_NUM_THREADS` remain build/report context. | `Makefile`, CMake, `src/sparse_matrix.c`, `src/sparse_eigs.c`, report scripts | No source or build metadata changes. The contract records that OpenMP is compile-time and `OMP_NUM_THREADS` is caller/runtime-owned. | Existing OpenMP test owner remains `tests/test_omp.c`; not rerun because no OpenMP code changed. |
| Sentinel row boundaries remain local. | `tests/corpus/manifests/report_families.tsv`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | Existing report-index behavior retained. Hard-gate and advisory sentinel rows remain distinguishable. | `python3 tests/test_normalize_report_index.py` passed. |

## Focused Validation Run

| Command | Result | Contract coverage |
| --- | --- | --- |
| `make build/test_chol_csc build/test_ldlt_backend_dispatch build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_reorder_nd` | Passed | Built focused direct-solver, eigensolver, and typed analysis precedence owners. |
| `build/test_chol_csc` | Passed: 92 tests, 0 failed, 0 skipped. | Cholesky AUTO/forced dispatch, invalid backend rejection, `used_csc_path` telemetry, CSC solve/writeback behavior. |
| `build/test_ldlt_backend_dispatch` | Passed: 22 tests, 0 failed, 0 skipped. | LDLT AUTO/forced dispatch, eigensolver API validation coverage co-owned by this historical test file. |
| `build/test_eigs_thick_restart` | Passed: 23 tests, 0 failed, 0 skipped. | Thick-restart explicit and AUTO dispatch, threshold behavior, `backend_used`, `peak_basis_size`. |
| `build/test_eigs_lobpcg` | Passed: 29 tests, 0 failed, 0 skipped. | LOBPCG explicit dispatch, AUTO LOBPCG priority, no-preconditioner fallback to Lanczos-family route. |
| `build/test_reorder_nd` | Passed: 35 tests, 0 failed, 1 skipped. | Typed analysis controls overriding compatibility env vars and ND/FM default behavior. The skip is the existing separator-lift-weight differentiation case, not a failed precedence assertion. |
| `python3 tests/test_normalize_report_index.py` | Passed. | Sentinel/report-index hard-gate and advisory row boundaries. |

## Deferred Implementation Decisions

| Control | Day 5 decision | Owner day |
| --- | --- | --- |
| Cholesky dense helper typed promotion | Not promoted on Day 5. Current env selector remains maintainer/compatibility. | Day 6 typed-control selection. |
| LDLT dense helper invalid-env parity | Not changed on Day 5. Existing report fields expose request/selected/fallback; focused invalid-env parity can be selected if Day 6 promotes or surfaces the control. | Day 6 typed-control selection. |
| `SPARSE_SVD_LOWRANK_OUTER` | Not promoted on Day 5. Classified as maintainer/runtime env selector. | Day 6 typed-control selection or explicit deferral. |
| FM strategy/debug/profile env vars | Not promoted on Day 5. Kept maintainer-only. | Day 6 explicit deferral list. |
| Dispatch-only sentinel expansion | Not implemented on Day 5. Candidate list remains available for Day 8/Day 9 sentinel work. | Days 8-9 sentinel expansion. |
| Package/link metadata changes | Not touched. Runtime precedence wording does not justify package or ABI changes. | Sprint 143 handoff if needed. |

## Day 5 Contract Status

| Completion criterion | Status | Evidence |
| --- | --- | --- |
| Selected precedence behavior is mechanically testable. | Complete | Focused owner tests built and passed. |
| Unsupported or maintainer-only controls remain clearly scoped. | Complete | Deferred implementation decisions table. |
| Any C/header change has focused validation recorded. | Not applicable | No C/header files were modified on Day 5. |
