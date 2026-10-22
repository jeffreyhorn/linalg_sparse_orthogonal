# Sprint 217 Day 14: Validation And Claim Closeout

**Scope:** [Day 14 plan](../PLAN.md), [Day 13 focused/instrumented ledger](day13-focused-instrumented-validation.md), and the selected Sprint 217 norm, ILUT, and iterative diagnostic owner. Validation ran locally on 2026-10-20 on `sprint-217` (Darwin 24.6.0, x86_64, Apple Clang 17.0.0 for the normal build). These are local results, not cross-platform acceptance.

## Required Final Gate

On 2026-10-20, the required sequence `make format && make lint && make test` exited **0** on the normal, non-sanitized build. A `make -s format-check` passed both before and after the sequence; formatting introduced no new source change requiring a separate focused rerun. `make lint` completed its strict `-Werror` compile, `clang-tidy`, and `cppcheck` phases without a blocking diagnostic. The ordinary `make test` run completed 59 test binaries on that day's tree, before the later PR #241 review fixes:

| Measure | Result |
| --- | ---: |
| Tests run | 2001 |
| Tests failed | 0 |
| Framework-counted skips | 4 |
| Assertions | 1,177,540 |

The four counted skips were the framework self-test, the opt-in slow and experimental wrappers, and `test_analysis_typed_nd_sep_lift_weight_overrides_env` (its `bcsstk04` environment balance weight did not differ from the typed hybrid baseline). Two additional test-internal `[SKIP]` messages, outside the framework total, concern opt-in large matrices and a mutex-only concurrent-insert case. These cases are not represented as passes. The full test output is retained locally in `/tmp/sprint217-day14-quality.log`.

The final `west0067` run again reported an ILUT solve 2-norm of `1.915e+274`, `finite_z=67/67`, `max_abs_z=1.914849e+274`, and an independent factor-system backward error of `8.249302e-80` against `1e-12`. The right-preconditioned GMRES diagnostic still reported `converged=0` after 500 iterations. This is a finite, checked **factor-system** result, not proof of `A*z=r` or GMRES convergence.

## Documentation And Hygiene

| Command | Result |
| --- | --- |
| `make -s docs-check` | Pass; Doxygen generated 18 public-header reference pages and 18 source pages |
| `make -s api-docs-validate` | Pass; coverage, local-only publication guard and regressions, and seven-document routing check |
| `make -s source-list-check` | Pass; 49 library sources |
| `python3 tests/test_external_baseline_claim_docs.py` | Pass; five tests |
| `make -s format-check` | Pass after the final Make chain |
| `git diff --check` | Pass on tracked changes; new Markdown also checked for trailing whitespace |

The [Day 13 ledger](day13-focused-instrumented-validation.md) separately records full UBSan and ASan runs, each with 2001 tests, zero failures, four counted skips, and no sanitizer runtime diagnostic, plus 12/12 selected CTest registrations. ASan's Make target disables leak detection. Its isolated `/tmp` build did not replace the normal build used above. The selected CMake build emitted nonfatal Apple Clang `-Wdouble-promotion` warnings for `NAN`/`INFINITY` macro conversions; it was not warning-free, but the selected targets and CTest checks passed.

## Item Reconciliation

The hours below are the **plan estimates**, not a claim of measured elapsed labor. Every planned item has a corresponding artifact and a passing final gate for its selected scope.

| Item | Planned hours | Evidence and disposition |
| --- | ---: | --- |
| 217.1 Contract And Call-Site Audit | 20 | [Day 1 inventory](day1-norm-consumer-inventory.md), [Day 2 behavior contract](day2-numerical-behavior-contract.md); selected consumers and statuses defined |
| 217.2 Scaled Norm Implementation | 28 | [Day 6 scaled norm](day6-scaled-norm-implementation.md), [Day 7 infinity norm](day7-infinity-norm-integration.md), [Day 8 finalization](day8-norm-finalization-ilut-gate.md); both public norm helpers and compatibility cases pass |
| 217.3 ILUT And Iterative Hardening | 28 | [Day 9 factor validity](day9-ilut-validity-failure-behavior.md), [Day 10 solver statuses](day10-iterative-convergence-hardening.md); selected false-success paths reject invalid data |
| 217.4 Extreme-Scale Regression | 24 | [Day 3 finite extremes](day3-finite-extreme-norm-fixtures.md), [Day 4 nonfinite/solver fixtures](day4-nonfinite-solver-ilut-fixtures.md), [Day 5 fixture closure](day5-fixture-closure-scaled-norm-design.md); originally red cases are green in the final suite |
| 217.5 Public Contract Update | 20 | [Day 11 API traceability](day11-public-vector-solver-contract.md), [Day 12 claim boundary](day12-user-guidance-claim-boundary.md); headers and user guidance retain the scoped non-claims |
| 217.6 Validation And Evidence | 20 | [Day 13 ledger](day13-focused-instrumented-validation.md) and this Day 14 record; focused, instrumented, CMake/CTest, full Make, documentation, and hygiene gates pass |
| **Total planned** | **140** | **14 days at 10 planned hours each** |

## Claim Boundary And Residuals

The selected owner is complete: finite representable vector norms avoid intermediate square/sum overflow or whole-norm underflow in the covered cases; NaN takes precedence over infinity; the selected CG/GMRES/MINRES/BiCGSTAB and block-CG paths do not publish convergence for the covered nonfinite caller or computed diagnostics; and `west0067` has a finite-entry, independent factor-system gate rather than `norm > 0` alone. NULL/nonpositive-length vector compatibility remains unchanged.

This does **not** establish all-solver numerical conformance, accurate `A*z=r` behavior for ILUT, useful preconditioned convergence on `west0067`, leak freedom, cross-platform parity, performance improvement, external-baseline parity, or state-of-the-art status. The skipped modes are not accepted as evidence for their own coverage; the required selected norm and ILUT checks ran and passed. The CMake warning scope and local-host-only validation also remain visible above. No residual issue blocks the selected Sprint 217 completion criteria; broader numerical and performance work stays with later Epic 20 sprints.

## Post-Fix Validation (2026-10-21)

After the PR #241 review fixes, including the near-maximum norm boundary
regressions, an exact fallback for ambiguous overflow classification with
both input orderings covered, and ordered ILUT backward-error assertions,
`make format && make lint && make test` exited 0 on the normal build.
`make test` passed all 59 Make test binaries. The affected binaries were
also run directly on the updated tree:

| Binary | Tests run | Failed | Skipped | Assertions |
| --- | ---: | ---: | ---: | ---: |
| `test_sparse_vector` | 44 | 0 | 0 | 92 |
| `test_ilu` | 37 | 0 | 0 | 520 |
| `test_bicgstab` | 53 | 0 | 0 | 395 |
| `test_block_solvers` | 18 | 0 | 0 | 184 |
| `test_stagnation` | 47 | 0 | 0 | 322 |

The October 20 full-suite, sanitizer, and selected CTest counts above and in
the Day 13 ledger remain dated historical evidence, not a claim that those
instrumented or CMake gates ran again after the review fixes. The post-fix
Make gate is the validation of the updated source tree.

## CG Initialization Follow-Up Validation (2026-10-21)

After moving initial residual checks ahead of unused CG recurrence products
in assembled, matrix-free, and block paths, `make format && make lint && make test`
exited 0. The full Make suite passed. The directly rerun affected binaries were:

| Binary | Tests run | Failed | Skipped | Assertions |
| --- | ---: | ---: | ---: | ---: |
| `test_iterative` | 95 | 0 | 0 | 795 |
| `test_block_solvers` | 19 | 0 | 0 | 191 |

The new cases cover a near-converged large RHS with an otherwise overflowing
preconditioned dot product in all three CG paths, including a mixed block where
the unconverged column still iterates. The earlier instrumented and CMake gates
were not rerun for this follow-up.

## Recurrence Follow-Up Validation (2026-10-22)

After moving CG post-update convergence ahead of next-step recurrence work and
BiCGSTAB initial convergence ahead of `rho`, `make format && make lint && make test`
exited 0. The full Make suite passed. Direct runs of the affected binaries gave:

| Binary | Tests run | Failed | Skipped | Assertions |
| --- | ---: | ---: | ---: | ---: |
| `test_iterative` | 97 | 0 | 0 | 807 |
| `test_block_solvers` | 20 | 0 | 0 | 201 |
| `test_bicgstab` | 55 | 0 | 0 | 409 |

These additions cover the three CG post-update paths and both BiCGSTAB entry
points with large finite initial residuals, including zero-iteration-budget
behavior. The earlier sanitizer and CMake gates were not rerun for this change.

## MINRES Follow-Up Validation (2026-10-22)

After handling zero iteration budgets before Lanczos initialization and
clarifying the zero-inner-product breakdown contract,
`make format && make lint && make test` exited 0. The full Make suite passed.
A direct `test_minres` run recorded 48 tests, 0 failures, 0 skips, and 743
assertions. The new cases cover zero-budget residual reporting without a
preconditioner call and both zero-output and underflowed positive initial
preconditioned inner products. Sanitizer and CMake gates were not rerun for
this follow-up.
