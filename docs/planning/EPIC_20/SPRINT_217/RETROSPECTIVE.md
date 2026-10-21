# Sprint 217 Retrospective

**Sprint:** 217 - Reliable Norms and ILUT Numerical Diagnostics

**Duration:** 14 planned days on branch `sprint-217`

**Status:** Six selected items closed locally; review and merge pending

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) assigns six items and 140
estimated hours. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md),
and [Day 14 closeout](artifacts/day14-validation-claim-closeout.md) link the
daily evidence. Estimates are not measured hours. Here, complete means the
selected vector-norm, iterative-status, ILUT diagnostic, documentation, and
validation contracts passed their stated gates, not that every solver is
numerically conformant or that ILUT makes `west0067` converge.

- [x] Inventoried public norm consumers and defined finite, nonfinite,
      NULL/empty, and selected convergence behavior before changing kernels.
- [x] Implemented scaled `vec_norm2` and NaN-aware `vec_norminf` without
      changing their signatures or historical NULL/nonpositive-length result.
- [x] Added extreme-scale, mixed NaN/infinity, and selected solver failure
      fixtures with independent numerical expectations.
- [x] Hardened the selected CG, GMRES, MINRES, BiCGSTAB, matrix-free, and
      block-CG status paths against covered false convergence, and made the
      `west0067` ILUT factor solve pass a finite-entry/backward-error gate.
- [x] Updated public headers and user guidance with error semantics and
      precise limits on what the factor-system test proves.
- [x] Passed the full normal Make quality chain, focused and instrumented
      tests, selected CMake/CTest registrations, and affected docs/API gates.
- [ ] Established all-solver numerical conformance, `A*z=r` accuracy or
      useful ILUT-GMRES convergence for `west0067`, external parity, portable
      speed, or state-of-the-art status. None was a selected deliverable.

## What Went Well

1. **Contract-first fixtures exposed real ambiguity.** Day 1 recorded
   `west0067: ILUT solve norm=inf` without calling it a nonfinite solution.
   Days 2-5 defined the expected result matrix and introduced deliberately
   red extreme-scale and solver-status cases. That separated arithmetic
   overflow in the old square/sum norm from actual invalid solution data.
2. **The norm fix remained compatible and testable.** The scaled 2-norm
   avoids false whole-norm infinity or zero in the covered representable
   cases, while the infinity norm now propagates NaN. The NULL/nonpositive
   return of `+0.0` stayed intact. The vector suite grew from 24 Day 1 tests
   to 40 final tests, all passing.
3. **The ILUT gate became an independent numerical check.** The final
   `west0067` solve has 67/67 finite entries, `||z||_2=1.915e+274`, and a
   scaled, permutation-aware factor-system backward error of
   `8.249302e-80` against a `1e-12` limit. Negative controls reject zero,
   nonfinite, invalid-permutation, and extreme-product cases. The gate does
   not silently promote an incomplete-factor solve into an `A*z=r` claim.
4. **Selected solver failures stopped looking like convergence.** Caller
   nonfinite vectors now return `SPARSE_ERR_BADARG`; covered computed
   numeric failures return `SPARSE_ERR_NUMERIC`, `converged=0`, and a NaN
   residual diagnostic. A full-suite MINRES stagnation fixture was updated
   only after a subnormal ratio was shown to round to zero, preserving its
   finite nonconvergence expectations.
5. **The final gate matched the claim boundary.** Day 13 passed full UBSan
   and ASan suites and 12/12 selected CTest registrations. Day 14 passed
   `make format && make lint && make test` plus docs/API and source-list
   checks on the normal build. The exact skips, platform, and sanitizer
   limitations are recorded rather than folded into an all-green claim.

## What Didn't Go Well

1. **The old ILUT assertion was too weak.** `norm > 0` accepted infinity,
   leaving the original `west0067` output uninterpretable. The new factor
   check resolves that ambiguity but still does not show that the
   preconditioner helps GMRES: both selected variants remained
   nonconverged after 500 iterations.
2. **Several existing status paths were inconsistent.** Initial nonfinite
   guesses, nonfinite RHS values, and computed overflow could previously
   return success or ordinary nonconvergence on selected paths. Fixing them
   required cross-file result handling and a MINRES stagnation-test
   adjustment, rather than a vector-helper-only change.
3. **Repository-wide checks are costly.** An intermediate Day 9 full lint
   attempt was interrupted in `cppcheck` and was not counted as a pass.
   Day 14 reran the full chain to completion; the API local-only fixture
   suite also took several minutes. Focused checks helped iteration, but
   they did not replace the final full gate.
4. **Evidence remains platform- and mode-bounded.** The selected CMake
   build passed with nonfatal Apple Clang `-Wdouble-promotion` warnings.
   The ASan Make target disables leak detection. The full Make test run
   recorded four framework skips and two further test-internal skip
   messages. Those modes were not treated as passing coverage, and no
   hosted Windows or independent external-library result was collected.

## Final Metrics And Claim

| Metric | Sprint 217 close state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Daily evidence | 14 primary day artifacts, one Day 8 solution snapshot, working notes, and this retrospective |
| Focused Make binaries | 8/8 passed; 341 tests, 0 failures, 0 skips, 3022 assertions |
| Full normal Make chain | `make format && make lint && make test` passed; 59 binaries, 2001 tests, 0 failures, 4 framework skips, 1,177,540 assertions |
| Instrumented Make runs | UBSan and ASan each passed 2001 tests with 0 failures and 4 framework skips; no runtime diagnostic; ASan leak detection disabled |
| Selected CMake/CTest | 12/12 tests passed on local Apple Clang build; nonfatal macro-conversion warnings |
| Generated API checks | 18 public-header reference pages and 18 source pages; routing/local-only guards passed |
| `west0067` ILUT gate | 67/67 finite entries; factor-system error `8.249302e-80 <= 1e-12`; both ILUT-GMRES variants nonconverged at 500 iterations |

These metrics are the October 20 close state, before the PR #241 review
regressions. The [post-fix validation](artifacts/day14-validation-claim-closeout.md#post-fix-validation-2026-10-21)
records a later passing full Make chain and updated affected-binary counts;
the instrumented and selected CTest runs above were not repeated on that tree.

The earned result is the **selected shared norm and solver-diagnostic owner**,
including a meaningful ILUT factor-system regression. It is not broad
solver-family numerical conformance, accurate `A*z=r` for incomplete
factors, useful preconditioning of `west0067`, leak freedom, platform parity,
external-baseline parity, portable timing, or state-of-the-art performance.

## Daily Evidence

- [Day 1 consumer inventory](artifacts/day1-norm-consumer-inventory.md)
- [Day 2 numerical behavior contract](artifacts/day2-numerical-behavior-contract.md)
- [Day 3 finite extreme fixtures](artifacts/day3-finite-extreme-norm-fixtures.md)
- [Day 4 nonfinite and solver fixtures](artifacts/day4-nonfinite-solver-ilut-fixtures.md)
- [Day 5 fixture closure and norm design](artifacts/day5-fixture-closure-scaled-norm-design.md)
- [Day 6 scaled 2-norm implementation](artifacts/day6-scaled-norm-implementation.md)
- [Day 7 infinity-norm integration](artifacts/day7-infinity-norm-integration.md)
- [Day 8 norm and ILUT gate](artifacts/day8-norm-finalization-ilut-gate.md)
- [Day 9 ILUT validity and failure behavior](artifacts/day9-ilut-validity-failure-behavior.md)
- [Day 10 iterative convergence hardening](artifacts/day10-iterative-convergence-hardening.md)
- [Day 11 public contract traceability](artifacts/day11-public-vector-solver-contract.md)
- [Day 12 claim boundary](artifacts/day12-user-guidance-claim-boundary.md)
- [Day 13 focused/instrumented validation](artifacts/day13-focused-instrumented-validation.md)
- [Day 14 final gate and closeout](artifacts/day14-validation-claim-closeout.md)

## Residual Handoff

Sprint 218 can take the planned `sparse_matmul` construction work on its own
acceptance gate; this sprint's numerical result is a prerequisite, not a
matmul performance claim. Broad solver-family conformance and additional
ILUT/preconditioning quality evidence remain separate follow-up work.
Epic 20's later external validator/correctness, Windows freshness, timing,
and source-release sprints each retain their own promotion gates. The
skipped test modes, CMake macro warnings, and disabled ASan leak detection
also remain explicit limitations of this local closeout.
