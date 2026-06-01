# Sprint 52 Day 12: Post-Landing Compatibility Audit

## Purpose

Day 12 verifies that the landed Sprint 52 Phase 2 branch still matches the
Sprint 50-51 compatibility contract after the deeper integration,
benchmark-proof, adoption, and regression batches.

The goal is not to reopen implementation. The goal is to confirm that the live
branch still reads and behaves like a bounded analysis/factor/refactor
strengthening rather than an accidental redesign.

## Main Day 12 Conclusion

Sprint 52 still matches the intended compatibility fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- repeated direct runs remain centered on `sparse_analysis_t` and
  `sparse_factors_t`
- reuse/refactor semantics remain honestly bounded
- README / example / benchmark claims remain tied to measured or explicitly
  bounded behavior
- no blocker-level residual drift surfaced before the validation sweep

## Audited Surfaces

### Shared repeated-run direct contract

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`

These still present the repeated-run direct workflow as:

1. zero/init analysis and factors
2. analyze once
3. factor / solve
4. refactor / solve many
5. free explicitly

The public wording and the shared implementation still agree that:

- reuse preserves symbolic/permutation setup
- numeric factor contents are rebuilt from new values
- refactor is a same-pattern numeric refresh path
- gross-structure rejection is cheap (`nnz` drift + basic state checks), not a
  full structural-pattern verifier

### Family-local one-shot direct surfaces

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

These still preserve the intended split:

- LU remains the simple/default copied-matrix one-shot path
- Cholesky remains the one-shot SPD surface
- LDL^T remains the family-local owned-factor surface
- all three headers point stable-pattern repeated runs back to the shared
  `sparse_analysis.h` contract rather than trying to duplicate or replace it

That means Sprint 52 strengthened the shared repeated-run story without
demoting or hiding the one-shot family APIs.

### Caller-facing adoption surfaces

- `README.md`
- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/README.md`

These surfaces still align cleanly:

- `README.md` presents the repeated-run direct workflow explicitly while still
  calling the one-shot direct APIs first-class peer entry points
- `examples/example_analysis.c` teaches the same boundary in code comments and
  runtime output
- `examples/README.md` still keeps small examples one-shot-first while
  correctly calling out `example_analysis` as the strongest repeated-run direct
  example
- `benchmarks/README.md` still describes `bench_refactor` and
  `bench_refactor_csc` as proof of the same public caller story rather than as
  a separate benchmark-only abstraction

### Public proof surface

- `tests/test_integration.c`

The direct repeated-run public proof now covers the expected high-signal
boundaries:

- explicit lifecycle parity against one-shot family defaults
- zeroed/unfactored solve rejection
- zero-init first-factorization support
- family/dimension mismatch rejection on refactor and solve
- old-factor preservation on failure
- cheap gross-structure drift rejection

This is enough coverage to support the Sprint 52 Phase 2 claim without turning
the test suite into a broad internal-detail mirror.

## Residual-Risk Notes

No blocker-level residual drift surfaced.

The remaining residual risks are the expected bounded ones:

- LU remains the strongest intentionally family-local special-case seam
- structural compatibility is still enforced only via cheap boundary checks and
  `nnz` drift rejection, not a full sparsity-pattern verifier
- benchmark evidence remains representative measured proof, not a guarantee of
  uniform speedup across every matrix family

These are follow-up boundaries, not Sprint 52 closeout blockers.

## Day 13 Pre-Validation Checklist

### Required full gate

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

### Truthfulness anchors

- `ctest -N --test-dir build/quality-review-cmake`
- Makefile/CMake parity check
- full reviewed CMake `ctest`

### Targeted Sprint 52 follow-ons

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_integration`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

## Operational Result

Sprint 52 is now positioned for a clean Day 13 validation sweep:

1. the live branch still matches the Sprint 50-51 compatibility fence
2. the deeper integration still reads as bounded Phase 2 work rather than
   redesign
3. the final validation checklist is explicit and complete

That leaves the remaining sprint work in validation and closeout, not in
unexpected compatibility repair.
