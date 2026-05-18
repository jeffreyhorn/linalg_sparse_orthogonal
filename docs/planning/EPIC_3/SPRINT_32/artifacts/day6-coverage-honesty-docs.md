# Sprint 32 Day 6 Coverage-Honesty Docs

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Document the post-Day-5 test truthfulness model in maintainer-facing docs, verify that the docs match the actual executed `tests/test_reorder_nd.c` surface, and record the small residual queue that remains outside the Day 5 structural cleanup.

## Documentation Added

Day 6 updates `README.md` so the project-level testing docs now state:

- default `make test` / `ctest` coverage comes from ordinary `RUN_TEST(...)`
- `RUN_TEST_SLOW(...)` is enabled with `SPARSE_TEST_SLOW=1`
- `RUN_TEST_EXPERIMENTAL(...)` is enabled with `SPARSE_TEST_EXPERIMENTAL=1`
- historical or retired target evidence belongs in `docs/planning/` artifacts, not as commented-out `RUN_TEST(...)` lines in normal suite files

That closes the gap between the Day 3 policy decision, the Day 4 harness support, and the public maintainer guidance.

## Before / After Truthfulness Model

### Before Sprint 32 Day 5

`tests/test_reorder_nd.c` mixed live coverage with dormant historical scaffold:

- `26` `static void test_*` functions
- `23` active `RUN_TEST(...)` calls
- `3` commented-out `RUN_TEST(...)` calls
- `3` `-Wunused-function` warnings, all explained by the dormant trio

The dormant trio encoded historical close-to-target claims for `Pres_Poisson` that Sprint 27 and Sprint 28 had already missed or retired.

### After Sprint 32 Day 5

`tests/test_reorder_nd.c` now matches what the suite actually executes:

- `23` active tests
- `0` commented-out `RUN_TEST(...)` lines
- `0` dormant compiled Pres_Poisson close-to-target helpers
- `0` local `-Wunused-function` warnings

The remaining active advisory tests still cover the live non-default ND axes truthfully:

- HCC Kuu-safe parity
- annealing differs-from-baseline
- root-spectral differs-from-multilevel
- thick-restart differs-from-baseline
- fixed-K weight-scheme differentiation
- Sprint 28 supernodal-postorder safety checks

## Category Rules

Sprint 32's category model is now:

### Active

- registered with `RUN_TEST(...)`
- runs in ordinary `make test` and `ctest`
- must stay green in the default path

### Slow opt-in

- registered with `RUN_TEST_SLOW(...)`
- enabled explicitly with `SPARSE_TEST_SLOW=1`
- still represents a current supported contract, not a stale expectation

### Experimental opt-in

- registered with `RUN_TEST_EXPERIMENTAL(...)`
- enabled explicitly with `SPARSE_TEST_EXPERIMENTAL=1`
- still represents a current live contract on a non-default path

### Historical evidence

- preserved in `docs/planning/` notes, decisions, and sprint artifacts
- not compiled as dormant suite code
- not represented as commented-out `RUN_TEST(...)`

## Contributor Rule

Future contributors should not merge long-lived dormant scaffold in normal suite files.

Use this decision rule instead:

1. If the contract is current and cheap enough, keep it active with `RUN_TEST(...)`.
2. If the contract is current but too expensive for the default path, use `RUN_TEST_SLOW(...)`.
3. If the contract is current but intentionally non-default, use `RUN_TEST_EXPERIMENTAL(...)`.
4. If the target is historical, missed, or retired, move the evidence to docs instead of keeping compiled dead test code.

## Residual Out-of-Scope Queue

A repo-wide search after Day 5 still finds older "failing-as-expected" narrative comments in some test files, including:

- `tests/test_eigs.c`
- `tests/test_svd.c`
- `tests/test_chol_csc.c`

Day 6 sampled those hits and found a narrower issue than the old `test_reorder_nd.c` problem:

- they are prose/history comments
- they do not currently imply active commented-out `RUN_TEST(...)` registrations in the sampled regions
- they are better treated as follow-up comment cleanup than as structural truthfulness drift

That distinction matters. Sprint 32 Day 5 closed an actual executed-vs-dormant mismatch. The residual queue is mostly wording cleanup.

## Conclusion

Sprint 32 now has an explicit and documented truthfulness model:

- active tests are the default regression surface
- slow and experimental tests are opt-in but still real
- historical evidence lives in docs

The README now reflects the Day 4 and Day 5 implementation state, and the remaining truthfulness debt is small enough to treat as follow-up comment hygiene rather than hidden dormant coverage.
