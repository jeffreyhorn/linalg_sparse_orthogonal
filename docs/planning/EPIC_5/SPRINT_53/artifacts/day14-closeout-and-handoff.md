# Sprint 53 Day 14 - Closeout and handoff

Date: 2026-06-01
Branch: `sprint-53`

## Summary

Sprint 53 closes the CSC direct-solver completion and dispatch follow-through
work from a measured validated baseline rather than from partial CSC depth or
docs intent.

The sprint started from the Sprint 52 validated Phase 2 direct-lifecycle
package and ended with deeper analysis-aware indefinite LDL^T CSC completion,
tighter LDL^T dispatch ownership, real indefinite factor-many benchmark proof,
clearer Cholesky/LDL^T CSC contract wording, and stronger repeated-run CSC
regression evidence.

## Delivered package

Sprint 53 leaves behind one coherent CSC follow-through package:

- deeper shared LDL^T CSC repeated-run integration in:
  - `src/sparse_analysis.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc_internal.h`
- tighter CSC dispatch ownership in:
  - `src/sparse_ldlt.c`
  - `include/sparse_ldlt.h`
- stronger bounded CSC proof in:
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_sprint20_integration.c`
- real indefinite factor-many benchmark proof in:
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/README.md`
- reconciled top-level CSC contract wording in:
  - `README.md`

## Delivered CSC state

Sprint 53 closes the main Sprint 53 seams in a bounded way:

- the shared repeated-run LDL^T CSC path and the one-shot CSC dispatch path now
  share:
  - resolved-analysis preparation
  - CSC completion orchestration
  - supernodal-attempt / scalar-fallback ownership
- LDL^T backend selection is now centralized and more explicit
- forced CSC telemetry stays truthful about the selected numeric path
- invalid shared-helper configuration no longer disappears behind unrelated
  scalar fallback
- the repo now has both:
  - SPD CSC repeated-run proof
  - indefinite CSC repeated-run proof

## Preserved contract

Sprint 53 preserved the Sprint 50-52 compatibility fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- repeated direct runs remain analysis/factors-centric around:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation setup, not stale numeric factor
  contents
- repeated-run structure validation remains a cheap bounded guard rather than a
  full structural-pattern verifier
- Cholesky CSC dispatch remains intentionally simpler than LDL^T CSC dispatch
- no raw internal CSC/native storage layout was exposed
- no generic public direct handle or public CSC container redesign was
  introduced

## Validation close state

Sprint 53 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- reviewed CMake parity = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `make quality-review-full` reviewed CMake total time = `124.22 sec`

Targeted Sprint 53 follow-ons also passed:

- `./build/test_integration`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/example_analysis`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`

Representative direct results:

- `example_analysis` kept residuals at `4.44e-16`
- `bench_refactor_csc nos4` kept the SPD CSC repeated-run path ahead:
  - `speedup_refactor = 1.64x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_refactor_csc --indefinite-kkt` kept the indefinite repeated-run path
  ahead:
  - `speedup_refactor = 1.36x`
  - `res_public = 2.96e-16`
  - `res_csc = 2.96e-16`

## Handoff to Sprint 54

Sprint 54 no longer needs to prove that CSC follow-through is real on the
highest-value LDL^T repeated-run paths.

The next bounded queue can therefore focus on real post-Sprint-53 work such
as:

- CSC/dispatch depth beyond the bounded Sprint 53 completion seams
- later public or internal cleanup where LU or other family-local differences
  remain intentionally special-case
- broader benchmark or caller-surface evolution that builds on the now-
  validated CSC repeated-run package
- any later structural-pattern validation deepening if a future sprint decides
  to pay that complexity cost

## Project-plan impact

Sprint 53 does not require a `PROJECT_PLAN.md` update.

Reason:

- the sprint closed from the planned Day 13 validated baseline
- the delivered package still matches the Epic 5 Sprint 53 intent
- no blocker or replanning queue surfaced during closeout

## Conclusion

Sprint 53 is complete. It hands off a validated CSC follow-through package
with deeper analysis-aware indefinite LDL^T completion, tighter dispatch
ownership, preserved first-class one-shot family entries, honestly bounded
repeated-run semantics, real SPD and indefinite factor-many benchmark proof,
and stable reviewed-baseline truthfulness anchors.
