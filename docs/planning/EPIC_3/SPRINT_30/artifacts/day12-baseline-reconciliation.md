# Sprint 30 Day 12 Baseline Reconciliation

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Reconcile the Sprint 30 warning baseline with the post-core-fix state, confirm that the compile-hygiene playbook matches what the sprint actually observed, and consolidate the queued follow-up work into a reviewable handoff state for Days 13 and 14.

## Reconciled Before/After Counts

### Authoritative full-tree path: Apple Clang CMake build

Day 1 baseline:

- full-tree warnings: `123`
- `src`: `11`
- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

Post-core-fix state (reproduced on Day 8 and again after the Day 9 strict-pass fix):

- full-tree warnings: `112`
- `src`: `0`
- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

Net delta from Day 1 to the post-core-fix state:

- full-tree: `-11`
- `src`: `-11`
- `tests`: `0`
- `benchmarks`: `0`
- `examples`: `0`

Interpretation:

- Sprint 30’s implemented code cleanup changed exactly one warning area: the core library.
- No auxiliary-area warning volume changed during Days 4-11 because those days were triage, workflow, and policy work rather than auxiliary cleanup.

### Warning-class reconciliation

Day 1 baseline:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `45`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Post-core-fix state:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `34`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Net class delta:

- `-Wdouble-promotion`: `-11`
- all other measured classes: `0`

Interpretation:

- The Day 4-5 cleanup removed only the intended `src/` `-Wdouble-promotion` cluster.
- No class-shifting or warning substitution occurred.

### Secondary path: Makefile `all`

Day 1:

- warnings: `0`

Day 8 post-core-fix replay:

- warnings: `0`

Interpretation:

- the Makefile `all` path stayed clean throughout Sprint 30
- that result remains library-only in scope and does not contradict the CMake full-tree auxiliary-warning backlog

## Strict-Pass Reconciliation

Day 9 added a stricter compile-only pass with:

- `-Wstrict-prototypes`
- `-Wmissing-prototypes`

That pass surfaced exactly one new `src/` warning:

- `src/sparse_types.c`
- `-Wmissing-prototypes`
- helper: `sparse_set_errno_`

Same-day outcome:

- the helper received a local prior declaration
- the strict-tree warning profile collapsed back to the Day 8 post-core-fix counts exactly

Result:

- Sprint 30 ended Day 12 with no unresolved strict-only `src/` warning debt
- there is no separate stricter-pass backlog to carry into Sprint 31

## Playbook Reconciliation

The Sprint 30 compile-hygiene playbook now matches observed sprint behavior:

1. The Apple Clang CMake build remained the authoritative full-tree baseline.
2. The Makefile `all` path remained a useful but narrower library-only cross-check.
3. `src/` warnings were treated as Sprint-blocking:
   - Day 4-5 removed the baseline `src/` warnings.
   - Day 9 fixed the one strict-only `src/` warning immediately.
4. Pre-existing auxiliary warnings were allowed to remain only after measurement and explicit queueing:
   - Day 10 queued the `tests/` warning backlog.
   - Day 11 queued the `benchmarks/` and `examples/` warning backlog.
5. Warning-closure claims were backed by before/after counts, named build paths, and regression validation.

No Day 12 policy correction was needed. The draft Day 6 playbook was directionally right; Day 12 simply upgrades it from draft to finalized-for-Sprint-30 status because the later sprint evidence now exists.

## Finalized Follow-Up Queues

### Sprint 31 input queue

Benchmark/example first-fix subset:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`
- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`

Themes:

- stale reorder CLI and usage drift
- `_POSIX_C_SOURCE` and `snprintf` portability debt
- designated-initializer conversion for public-facing options structs
- one residual low-priority double-promotion cleanup in benchmark code

### Sprint 32 input queue

Test first-fix subset:

- `tests/test_reorder_nd.c`

Themes:

- dormant or non-executed scaffolding
- `-Wunused-function` debt
- designated-initializer cleanup in the same file while touched

### Later auxiliary cleanup queue

Higher-volume test initializer files:

- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_colamd.c`
- `tests/test_cholesky.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_reorder.c`

Residual mechanical double-promotion files:

- `tests/test_sprint6_integration.c`
- `tests/test_svd.c`
- `tests/test_bidiag.c`
- `benchmarks/bench_convergence.c`
- remaining smaller one- and two-warning sites already enumerated in Day 10 and Day 11 artifacts

### Stricter-pass queue

- none open after Day 9

## Reviewable Sprint 30 State After Day 12

Sprint 30 has produced a coherent engineering package rather than a loose set of observations:

- Day 1 established a reproducible full-tree warning baseline.
- Day 2 classified the warning debt by area and warning class.
- Days 3-5 removed the targeted core-library warning cluster.
- Day 6 defined the compile-hygiene decision rules.
- Day 7 made baseline reproduction repeatable.
- Day 8 proved the cross-path scope difference cleanly.
- Day 9 checked the cleanup under stricter compile settings and closed the only new `src/` issue.
- Day 10 converted the test-warning backlog into an explicit queue.
- Day 11 converted the benchmark/example warning backlog into an explicit queue.
- Day 12 reconciled the counts, policies, and follow-up inputs.

## Day 12 Conclusion

Sprint 30’s measured state is now internally consistent:

- the authoritative full-tree warning count moved from `123` to `112`
- the entire reduction came from removing the intended `src/` warning cluster
- the remaining debt is auxiliary, classified, and queued
- the playbook now reflects not just intent but observed sprint behavior

## Validation

End-of-day verification passed:

- `make format`
- `make lint`
- `make test`
