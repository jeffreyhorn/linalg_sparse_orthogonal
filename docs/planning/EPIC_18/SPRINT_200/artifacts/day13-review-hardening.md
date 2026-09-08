# Sprint 200 Day 13 Review Hardening

## Scope

Day 13 audits the Sprint 200 selected-owner proof surface for unnecessary
breadth, invariant coverage, gate naming, and evidence consistency before final
closeout. No new implementation behavior is added.

## Diff-Surface Audit

| Surface | Review finding |
| --- | --- |
| `src/sparse_etree.c` | Changes are limited to `sparse_symbolic_lu()` output clearing and allocation-hook reachability for selected owner allocations. |
| `tests/test_etree.c` | New tests are limited to selected symbolic LU allocation failure, cleanup, stale-output suppression, input preservation, and retry. |
| `Makefile` | Adds one selected-owner gate: `symbolic-lu-allocation-failure-gate`. |
| `tests/test_symbolic_lu_allocation_failure_gate_registration.py` | Guards the selected gate name, environment selector, test registration, failure-site table, cleanup assertions, hook reset, and retry assertion. |
| Public docs | README, INSTALL, and maintainer guide describe selected symbolic LU proof only and retain broad reliability non-claims. |
| Planning docs | Project plan, residual queue, Epic retrospective, working notes, and day artifacts now distinguish in-progress Sprint 200 evidence from final Day 14 closeout. |

## Invariant-To-Test Traceability

| Invariant | Test or guard evidence | Day 13 assessment |
| --- | --- | --- |
| S200-LU-CLEAN-01 | `test_symbolic_lu_allocation_failures_clear_outputs`; `test_symbolic_lu_allocation_failures_cleanup_sweep` | Covered by all selected failure sites and repeated cleanup sweep. |
| S200-LU-CLEAN-02 | `test_symbolic_lu_allocation_failures_clear_outputs`; `test_symbolic_lu_allocation_failures_recover_on_retry` | Covered for propagated symbolic Cholesky failures and U-building failures through free-safe output assertions. |
| S200-LU-PUB-01 | `assert_symbolic_failure_free_safe(&sym_L)` in failure tests; `assert_symbolic_lu_retry_output_fresh()` in retry tests | Covered for failed L publication and later fresh success. |
| S200-LU-PUB-02 | `assert_symbolic_failure_free_safe(&sym_U)` in failure tests; `assert_symbolic_lu_retry_output_fresh()` in retry tests | Covered for failed U publication and later fresh success. |
| S200-LU-STALE-01 | Stale sentinel outputs in `expect_symbolic_lu_allocation_failure()` and retry helper | Covered by pre-filled metadata sentinels that must be cleared on failure. |
| S200-LU-RETRY-01 | `test_symbolic_lu_allocation_failures_recover_on_retry` | Covered for each selected failure site after resetting allocation injection. |
| S200-LU-INPUT-01 | `assert_unsym_3x3_symbolic_lu_input_intact()` | Covered after failure and retry. |
| S200-LU-INPUT-02 | `assert_identity_perm_intact()` | Covered for selected valid-permutation failure sites and retry. |
| S200-LU-HOOK-01 | Allocation reset before assertions in helpers; `assert_allocation_hook_probe_after_reset()` | Covered by cleanup sweep and registration guard text checks. |
| S200-LU-HOOK-02 | Wrapper allocation changes in `sparse_symbolic_lu()`; selected failure-site table | Covered by the focused gate and registration guard. |
| S200-LU-SCOPE-01 | `make symbolic-lu-allocation-failure-gate`; README/INSTALL/maintainer wording; planning status rows | Covered after Day 13 status hardening. |

## Claim Vocabulary Review

Accepted vocabulary:

- selected `sparse_symbolic_lu()` allocation-failure owner;
- focused local deterministic allocation-failure proof;
- requested-output cleanup;
- stale-output suppression;
- caller-owned matrix/permutation preservation;
- retry-after-reset behavior;
- bounded known fixtures.

Rejected vocabulary remains:

- broad allocation-failure coverage;
- operating-system OOM guarantee;
- platform parity or hosted proof;
- package/install reliability;
- generated-tooling reliability;
- direct solver, eigensolver, graph, SVD, matrix-construction, or analysis
  allocation-failure proof;
- state-of-the-art reliability support.

## Closeout Checklist Draft

| Item | Closeout check |
| --- | --- |
| 200.1 Owner selection | Confirm `sparse_symbolic_lu()` remains the only Sprint 200 selected owner. |
| 200.2 Invariant record | Confirm Day 4 invariants have test or guard evidence in the traceability table above. |
| 200.3 Harness reachability | Confirm selected direct allocations use existing private wrapper hooks and reset behavior is tested. |
| 200.4 Regression proof | Confirm failed-allocation, cleanup, stale-output, caller-input, and retry tests are linked. |
| 200.5 Focused gate | Confirm `make symbolic-lu-allocation-failure-gate` and the registration guard still pass. |
| 200.6 Docs and validation | Confirm Day 11 docs, Day 12 validation, and Day 13 review-hardening evidence are reconciled before retrospective preparation. |

## Day 13 Validation

Day 13 changed documentation and planning artifacts only. The full C quality
gate already passed on Day 12 for the branch's C and test changes.

Commands run after the Day 13 documentation updates:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
make symbolic-lu-allocation-failure-gate
make docs-check
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py` | PASS | `symbolic-lu-allocation-failure-gate-registration: passed`. |
| `make symbolic-lu-allocation-failure-gate` | PASS | Selected gate reported 3 tests, 0 failures, 0 skipped, and 3054 assertions. |
| `make docs-check` | PASS | Doxygen and API docs coverage completed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| `git diff --check` | PASS | Whitespace validation completed after Day 13 updates. |
