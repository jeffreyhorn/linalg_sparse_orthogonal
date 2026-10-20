# Sprint 217 Working Notes

**State:** Selected Sprint 217 scope complete after Day 14 validation; broader
numerical conformance remains outside this sprint. **Plan:**
[Sprint 217 plan](PLAN.md). **Epic scope:** [project plan](../PROJECT_PLAN.md).

## Day 1 Consumer Map

The [Day 1 inventory](artifacts/day1-norm-consumer-inventory.md) records each
public-helper call site, its use, and its numerical risk. Production callers of
`vec_norm2` are confined to scalar iterative solvers, MINRES, and block CG.
There are no production callers of `vec_norminf`, and no public norm-helper
calls in direct or eigensolver source files. Direct-solver tests do use the
public helper for residual checks. Several tests and one example define local
functions with the same names; they are not consumers of `sparse_vector.c`.

The current `vec_norm2` accumulates `v[i] * v[i]` before taking `sqrt`; this
can overflow or underflow for finite inputs. `vec_norminf` skips NaNs because
`fabs(NaN) > mx` is false. Both return `0.0` for NULL or nonpositive length.
These are observed implementation behaviors, not yet approved contracts.

## Decision Log

| ID | Decision or open question | State |
| --- | --- | --- |
| D1-1 | Preserve the observed NULL and nonpositive-length return of `0.0` until Day 2 resolves the compatibility contract. | Resolved by D2-1 |
| D1-2 | Define mixed NaN/infinity behavior and the representable-result rule before changing either norm. | Resolved by D2-2 |
| D1-3 | Determine whether zero/nonfinite RHS and residual norms require solver status changes under existing error codes. | Resolved by D2-3 |
| D1-4 | Do not treat `west0067` `norm=inf` as proof of nonfinite solution entries; first separate norm-arithmetic overflow from solver output. | Recorded |
| D1-5 | Day 1 is audit and baseline only; no C source or test changes. | Recorded |

## Day 1 Baseline Test Matrix

| Area | Existing focused coverage and Day 1 result | Planned gap |
| --- | --- | --- |
| Vector norms | `test_sparse_vector`: 24 tests pass; ordinary, zero, NULL, and `n=0` cases | Finite extremes; NaN/infinity/mixed inputs; negative length |
| ILUT | `test_ilu`: 34 tests pass; `west0067` prints `ILUT solve norm=inf` | Finite-entry check and independent factor-product residual gate |
| Scalar iterative | `test_iterative`: 85 tests pass | Tiny nonzero RHS, extreme residuals, nonfinite status paths |
| MINRES | `test_minres`: 43 tests pass | Extreme scales and direct square/sum residual calculations |
| Block iterative | `test_block_solvers`: 15 tests pass | Per-column underflow and nonfinite reporting |

## Evidence Ledger

Baseline: 2026-10-20, branch `sprint-217`, commit
`bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7`; Darwin 24.6.0 x86_64,
Apple clang 17.0.0. `make -s build/test_sparse_vector build/test_ilu
build/test_iterative` and `make -s build/test_minres build/test_block_solvers`
both exited 0. All five binaries exited 0; counts and diagnostic output are
in the [Day 1 inventory](artifacts/day1-norm-consumer-inventory.md). These are
local baselines, not evidence that the extreme-scale contract is satisfied.

## Day 2 Handoff

The [Day 2 contract](artifacts/day2-numerical-behavior-contract.md) records
the exact result/status matrix, scaled-sum-of-squares choice, accuracy gates,
and independent `west0067` factor-product acceptance calculation. Decisions:

| ID | Decision | Implementation owner |
| --- | --- | --- |
| D2-1 | Keep `0.0` for NULL or `n <= 0` in both norm helpers; no elements are inspected. | Days 3-7 |
| D2-2 | NaN dominates infinity; finite representable nonzero 2-norms must not become zero or infinity through intermediate arithmetic. Infinity norm propagates NaN. | Days 3-7 |
| D2-3 | Selected solvers reject nonfinite caller-supplied RHS/initial guess with `SPARSE_ERR_BADARG`, and nonfinite computed norms/ratios with `SPARSE_ERR_NUMERIC`; neither may be reported converged. | Days 8-10 |
| D2-4 | Retain the exact zero-RHS fast path; use finite relative-residual checks and define zero-beta breakdown separately. | Days 8-10 |
| D2-5 | Require `SPARSE_OK`, finite ILUT solution/factors, and independently computed, scaled `L(Uz) - Pr` backward error for `west0067`; do not demand exact `Az = r`. | Days 8-10 |
| D2-6 | Document public norm and selected solver contracts once implemented; these notes are design intent, not current API behavior. | Days 11-12 |

MINRES has separate square/sum true-residual code that a shared norm fix alone
will not repair. Day 3 should add finite extreme-scale norm fixtures against
the current implementation before changing kernels. Preserve the distinction
between design decisions and passing evidence when updating this ledger.

## Day 3 Finite Extreme-Scale Fixtures

The [Day 3 before-fix record](artifacts/day3-finite-extreme-norm-fixtures.md)
documents six new vector tests in `tests/test_sparse_vector.c`: four failing
2-norm cases and two passing controls. Expected values use `hypot` for paired
inputs or exact finite magnitudes; the checks require finite results and
relative error at most `8*DBL_EPSILON` for nonzero normal results. New control
checks use explicit finiteness/equality, because the existing `ASSERT_NEAR`
macro alone does not reject NaN.

Evidence at commit `bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7` before any
production norm change: `clang-format --dry-run --Werror
tests/test_sparse_vector.c` exit 0; `make build/test_sparse_vector` exit 0;
`./build/test_sparse_vector` exit 1, 30 run, 4 failed, 0 skipped, 61
assertions. The four failures are expected and **not waived**: old `vec_norm2`
returns `inf` for representable large/mixed/near-equal signed norms and `0`
for a representable tiny nonzero norm. Full `make test` cannot be called green
on this branch until the planned kernel fix and regressions pass. Day 4 owns
nonfinite and solver fixtures; Days 5-7 own the scaled norm implementation.

## Day 4 Nonfinite And Solver Fixtures

The [Day 4 fixture record](artifacts/day4-nonfinite-solver-ilut-fixtures.md)
adds nonfinite vector cases; fixed small-system controls and tiny-RHS cases
for CG, GMRES, MINRES, and BiCGSTAB; nonfinite-RHS and computed-residual
status cases; and an observational `west0067` factor-system diagnostic.
The new tests are registered with `RUN_TEST` in five existing test binaries,
which are already included in Make and CTest. No new test target is needed.

`make format-check`, `make lint`, and the five-binary build exited 0. The pre-fix test
exits were: vector 1 (36 run, 7 failed), iterative 1 (89 run, 4 failed),
MINRES 1 (45 run, 2 failed), BiCGSTAB 1 (51 run, 1 failed), ILU 0 (34 run,
0 failed). Four solver tiny-RHS cases returned `SPARSE_OK`, `converged=1`,
but independently measured `true_rel=1`; these are demonstrated false
successes, not merely static-analysis risks. GMRES and MINRES nonfinite RHS
returned `SPARSE_ERR_NOT_CONVERGED` rather than the Day 2 target
`SPARSE_ERR_BADARG`. CG's overflowed computed residual also returned
`SPARSE_ERR_NOT_CONVERGED` rather than `SPARSE_ERR_NUMERIC`; BiCGSTAB already
returned `SPARSE_ERR_NUMERIC` for the equivalent case.

For `west0067`, the selected ILUT solve had 67/67 finite entries and
`max|z|=1.914849e+274`. Because `||z||_2 <= sqrt(67)*max|z| < 1.6e275`,
the true norm is representable as `double`; the printed `norm=inf` is caused
by the old square/sum norm arithmetic. The independently computed factor
backward error was `8.242351e-80` on this macOS run. This is diagnostic
evidence only: the old weak assertion remains and the factor-error threshold
is not yet enforced. It does not establish `A*z=r` or useful preconditioning.

Day 5 should review the fixture cross-product and preserve all expected
failures; no full-suite green claim is possible until the later kernel and
solver changes pass these tests.

## Day 5 Fixture Closure And Norm Design

The [Day 5 fixture and design record](artifacts/day5-fixture-closure-scaled-norm-design.md)
closes the boundary cases missing from Days 3-4: `DBL_MAX` singleton versus
genuine final overflow, smallest-subnormal singleton/pair, signed-zero and
empty infinity-norm controls, CG nonfinite initial guess with zero RHS, and
an invalid second block-CG column. It maps each planned norm branch to a
test and explains which selected-solver branches still require Days 8-10
fixtures. No production code or public API was changed, and no failing
expectation was relaxed.

At base commit `bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7` with these
uncommitted fixtures, `make -s format-check`, `make lint`, and the focused
six-binary build exited 0. The focused binary results are:

| Binary | Exit | Run | Failed | Skipped | Assertions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test_sparse_vector` | 1 | 39 | 9 | 0 | 83 |
| `test_iterative` | 1 | 90 | 5 | 0 | 765 |
| `test_block_solvers` | 1 | 16 | 1 | 0 | 173 |
| `test_minres` | 1 | 45 | 2 | 0 | 710 |
| `test_bicgstab` | 1 | 51 | 1 | 0 | 381 |
| `test_ilu` | 0 | 34 | 0 | 0 | 328 |

The additional Day 5 red cases are the two norm2 boundary tests, CG
invalid-guess/zero-RHS precedence, and block-CG invalid-column handling.
The old kernel computes `inf` for the representable `DBL_MAX` singleton and
zero for the smallest subnormal singleton. These are regressions the later
implementation must satisfy. `make test` cannot pass with the intentionally
red fixtures and is deferred until the kernel/solver changes; the recorded
quality checks are not a claim that the full suite passes.

## Day 6 Scaled 2-Norm

The [Day 6 implementation record](artifacts/day6-scaled-norm-implementation.md)
describes the `vec_norm2` scale/sum-of-squares change and its focused
evidence. The public signature and NULL/nonpositive-length `+0` behavior
remain unchanged. NaN takes precedence over infinity even when it appears
later in the vector; `vec_norminf` is deliberately untouched until Day 7.

After rebuilding against the changed library, the vector suite has 39 run,
3 failed, 0 skipped, and 83 assertions. All `vec_norm2` fixtures pass;
the three remaining failures are the Day 7 infinity-norm NaN cases. The
iterative, block, MINRES, BiCGSTAB, and ILUT suites respectively have
90/3, 16/1, 45/1, 51/0, and 34/0 run/failed. `west0067` now prints a
finite 2-norm of `1.915e+274` with all 67 entries finite and independent
factor backward error `8.242351e-80`; the numeric acceptance gate is not
yet enforced. `make -s format-check`, `make -s lint`, and the focused builds
passed. An isolated UBSan vector build emitted no runtime diagnostic and
reproduced only the three expected infinity-norm failures. The
remaining suite failures are retained for Days 7-10 rather than skipped.

## Day 7 Infinity Norm And Integration

The [Day 7 norm integration record](artifacts/day7-infinity-norm-integration.md)
documents explicit NaN propagation in `vec_norminf`, the passing 40-test
vector suite, and a finite near-`DBL_MAX` pair checked against `hypot`.
Every Day 2 public-norm input class now has a passing fixture. The six
rebuilt call-site smoke binaries show no new failures; `test_iterative`,
`test_block_solvers`, and `test_minres` still contain five selected-solver
status failures in total, while BiCGSTAB, ILU, and integration pass. The
artifact lists exact guards and source branches for Days 8-10. `make -s
format-check` and `make -s lint` passed. The isolated UBSan vector binary
passed all 40 tests with no runtime diagnostic. Solver and ILUT acceptance
work remains open; norm integration is not a claim of full-suite green status.

## Day 8 Norm Finalization And Initial ILUT Gate

The [Day 8 gate record](artifacts/day8-norm-finalization-ilut-gate.md) closes
item 217.2's planned norm work with 40/40 vector tests passing and no new
precision warning on this host. An opt-in run captured all 67 actual
`west0067` solution entries in the [solution snapshot](artifacts/day8-west0067-z-values.md).
The old `norm=inf` was intermediate square/sum overflow: all entries are
finite and `sqrt(67)*max|z|=1.567371e+275 < DBL_MAX`.

`test_ilut_west0067` now requires factor and solve success before reading
output, nonnull factors, 67 finite entries, a finite positive norm, and a
finite independent factor-system backward error no larger than the Day 2
limit. Locally, `eta=8.242351e-80` against `1e-12`; `test_ilu` passes
34/34 with 334 assertions. Day 9 must add negative controls and review the
oracle's scale portability. This remains an ILUT factor-solve validation,
not an exact `A*z=r` or preconditioned-GMRES convergence claim.

After the final test cleanup edit, `make -s format-check`, `make -s lint`,
and the focused vector/ILU builds passed. Both focused binaries exited 0
with no skips. The full test suite remains intentionally red on the selected
solver-status fixtures until Days 9-10.

## Day 9 ILUT Validity And Failure Behavior

The [Day 9 record](artifacts/day9-ilut-validity-failure-behavior.md) closes
the selected factor-system gate with a scale-aware independent `L(Uz)`
backward-error oracle. Unit controls cover identity and nonidentity row
permutations, zero and nonfinite solutions, a nonfinite factor, a duplicate
permutation, and an extreme product that overflows before normalization if
computed directly. The selected `west0067` solve remains finite with
`eta=8.249302e-80` against `1e-12` and a representable 2-norm.

`sparse_ilu_solve` now reports `SPARSE_ERR_NUMERIC` for nonfinite factors,
RHS, intermediates, or output, including a finite-input overflow control.
Output may be partial on failure and is not inspected as a solution then.
The focused ILU binary passes 37/37 tests, 520 assertions, and no skips.
Both selected ILUT-GMRES variants return finite results after 500 iterations
without convergence; neither is described as solving `west0067` well.
Day 10 still owns the remaining iterative solver-status fixtures.
Formatting and touched-file static checks passed. Repository-wide lint was
interrupted in `cppcheck` after file 106/109, so no full lint pass is claimed.

## Day 10 Iterative Convergence Hardening

The [Day 10 record](artifacts/day10-iterative-convergence-hardening.md)
closes item 217.3's selected status ledger. CG, GMRES, MINRES, BiCGSTAB,
their covered matrix-free variants, and block CG validate finite caller
vectors before the nonempty zero-RHS fast path and reject invalid computed
norms/ratios with `SPARSE_ERR_NUMERIC`, a NaN result diagnostic, and no
convergence claim. CG and block CG now use final true residuals; MINRES uses
the scaled vector norm for true-residual evaluation. The five previously
failing selected fixtures are green, with new overflow, invalid
preconditioner, and matrix-free entry controls also green.

The full suite exposed a legacy MINRES stagnation expectation that assumed
the solver would reach 5000 iterations. A nonzero subnormal recurrence
estimate divided by a finite RHS norm rounded to zero at iteration 3295;
the new `SPARSE_ERR_NUMERIC` status is required by the Day 2 contract. The
fixture now verifies that status without weakening its finite
nonconvergence cases. After this adjustment, `make -s format-check` and
`make -s test` passed. Focused iterative, MINRES, BiCGSTAB, and block
binaries passed 93/93, 46/46, 52/52, and 17/17 tests respectively.
Targeted `clang-tidy` and `cppcheck` passed; full `make lint` was not rerun.
Day 11 owns public-header contract wording for these statuses.

## Day 11 Public Vector And Solver Contract

The [Day 11 traceability record](artifacts/day11-public-vector-solver-contract.md)
maps the revised `sparse_vector.h`, `sparse_iterative.h`, and `sparse_ilu.h`
contracts to the Day 3-10 regression fixtures. The norm helpers now describe
scaled accumulation, final-result range, NaN precedence, and historical
NULL/nonpositive-length behavior. The selected iterative declarations now
distinguish nonfinite caller input (`SPARSE_ERR_BADARG`) from computed
numeric failure (`SPARSE_ERR_NUMERIC`, no convergence, NaN diagnostic),
including the result-field limits on other hard errors. ILUT documentation
states that a successful factor-system solve does not prove accuracy for
`A*z=r` or useful preconditioning.

`make -s api-docs-validate` passed with local-only/routing guards green;
`make -s docs-check` passed again after the final callback wording edit, with
all 18 public-header reference and source pages covered. The three touched
generated pages were inspected for the new
text; a Doxygen attachment issue in the first draft was corrected and
regenerated. `make -s format-check` and `git diff --check` passed. No Day 11
code or signature change required a new solver test run; the Day 10 full
`make test` pass remains the current suite evidence. Day 12 owns user-facing
guidance beyond the API boundary.

## Day 12 User Guidance And Claim Calibration

The [Day 12 claim matrix](artifacts/day12-user-guidance-claim-boundary.md)
freezes item 217.5's 20 planned hours across Days 11-12. README, solver
selection, and the source-controlled API index now point to the selected
norm, solver-status, and ILUT factor-system contracts without promoting the
`west0067` diagnostic into an `A*z=r`, convergence, or state-of-the-art
claim. The Day 11 vector header was tightened to say tiny contributions may
round away even though scaling avoids false whole-norm zero/infinity from
direct squaring.

`make -s docs-check` passed with 18 public-header pages and 18 source pages;
the regenerated vector page was inspected. API routing, its regression
suite, the existing external-baseline claim guard (5 tests), new local link
targets, `make -s format-check`, and `git diff --check` passed. No new
phrase-sensitive guard was introduced: routing has a stable check, while
the existing external-baseline guard is limited to the selected pilot.
The broader Sprint 217 non-claim was reviewed against the claim matrix.
Days 13-14 retain the focused/instrumented and full C-quality validation work.

## Day 13 Focused And Instrumented Validation

The [Day 13 ledger](artifacts/day13-focused-instrumented-validation.md)
records eight directly run focused Make binaries: 341 tests, zero failures
or skips, and 3022 assertions. The selected `west0067` factor solve has
67/67 finite output entries, a finite scaled norm (`1.915e+274`), and a
factor-system backward error of `8.249302e-80` against `1e-12`. Its two
ILUT-GMRES variants remain nonconverged after 500 iterations; the Day 1
`norm=inf` observation was a naive-norm overflow, not proof that the
preconditioned solve is useful.

Full `make sanitize` (UBSan) and `make asan` runs each passed 2001 framework
tests with zero failures and four counted skips; two further test-internal
skip messages are called out in the ledger. No sanitizer runtime diagnostic
appeared. ASan leak detection was disabled by the Make target. The 12
selected CMake/CTest registrations passed. Apple Clang's CMake build had
nonfatal float-to-double `NAN`/`INFINITY` macro warnings, not a failed
build. Sanitizer builds used isolated `/tmp` output directories because
their targets run `clean`; the existing normal `build/` remained intact for
Day 14's required full quality sequence.

## Day 14 Final Gate And Selected-Scope Closeout

The [Day 14 validation and claim record](artifacts/day14-validation-claim-closeout.md)
reconciles all six planned items to their artifacts and 140 planned hours.
The required `make format && make lint && make test` chain passed on the
normal Apple Clang build. The full Make suite ran 2001 tests across 59
binaries with zero failures, four counted skips, and 1,177,540 assertions.
The final `west0067` factor-system error remained `8.249302e-80` against
`1e-12`, with all 67 output values finite; the preconditioned GMRES case
remained nonconverged after 500 iterations.

`make -s docs-check`, `make -s api-docs-validate`,
`make -s source-list-check`, the five external-baseline claim-doc tests,
and a post-format `make -s format-check` passed. The Day 13 UBSan, ASan,
and 12/12 selected CTest results are separate supporting evidence. Skips,
ASan's disabled leak detection, nonfatal CMake warnings, and the local-host
scope are stated in the closeout record. The selected norm and solver-status
owner is complete; no all-solver or ILUT convergence claim is made.
