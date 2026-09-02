# Sprint 193 Retrospective

**Sprint:** 193 - Selected Large Review-Surface Reduction
**Duration:** 14 days (Days 1-14 landed on branch `sprint-193`)
**Status:** Complete; one selected QR external-reference review surface reduced
with guard coverage and residuals documented

## Source Artifact Note

Sprint 193 was executed from the Epic 17 project-plan section for Sprint 193
and lives under `docs/planning/EPIC_17/SPRINT_193/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 193 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited large test/review surfaces and selected exactly one bounded
      extraction target: the QR external-reference rank/nullspace/threshold
      cluster.
- [x] Defined the selected helper/proof-owner boundary before moving code.
- [x] Extracted the selected cluster into
      `tests/test_qr_external_ref_helpers.h`.
- [x] Preserved `tests/test_qr.c` as the QR proof-owner executable with `main`,
      `RUN_TEST(...)` registration, and the scoped economy test body.
- [x] Added reader failure-path coverage for invalid arguments and unsupported
      fixtures.
- [x] Added `make qr-external-ref-helper-guard` and Python fixture tests for
      helper boundary drift.
- [x] Documented the helper/proof-owner split, source-list absence, and focused
      forced-rebuild caveat in `docs/maintainer_guide.md`.
- [x] Confirmed no public headers, production sources, QR tolerance policy,
      rank policy, solver behavior, or generated report surface were changed.
- [x] Ran the final required C quality gate because `.c` and `.h` files
      changed.

## What Went Well

1. **The sprint stayed focused.** The extraction selected one QR
   external-reference cluster instead of attempting to split every large QR
   test family in one pass.

2. **The review-surface reduction is measurable.** `tests/test_qr.c` dropped
   from 3970 lines to 3040 lines while the selected cluster became a named
   1003-line family-local helper.

3. **The proof-owner boundary stayed explicit.** `tests/test_qr.c` still owns
   executable registration and the selected economy test, while the helper owns
   only the selected rank/nullspace/threshold external-reference readers and
   tests.

4. **Boundary drift is now guarded.** The new guard checks required files,
   proof-owner registration, selected test ownership, header-only registration,
   source-list absence, and maintainer documentation markers.

5. **The formatter-stability issue was caught before closeout.** Day 12 exposed
   that clang-format could reorder helper includes in a way that hid external
   reference declarations; the helper now owns its `test_solver_helpers.h`
   dependency after defining `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`.

6. **The final validation was broad enough for the changed surface.** The
   sprint ran source-list checks, guard tests, CMake compile/parity, formatting,
   lint, and the full Makefile test suite.

## What Didn't Go Well

1. **Header-only helper edits still have a focused rebuild trap.** A plain
   `make build/test_qr && ./build/test_qr` can miss header-only changes if the
   binary is considered up to date. The maintainer guide now records the forced
   rebuild requirement.

2. **The selected extraction still leaves large QR surfaces.** The economy,
   sparse-mode, and refinement clusters remain in `tests/test_qr.c` because
   moving them would have broadened the sprint beyond one selected claim.

3. **Guard coverage adds review volume.** The shell guard and Python negative
   tests are useful, but they make future boundary changes intentionally
   coupled to test and documentation updates.

4. **The helper remains a large header.** The selected extraction reduced the
   main proof owner but did not create a smaller reusable test library. That was
   intentional to avoid new build-system or source-manifest ownership.

5. **CMake compile produced an existing warning in an unrelated test.** The
   reviewed CMake gate still passed, but the output included an unrelated
   `test_svd_partial_corpus.c` double-promotion warning during clean build.

## Final Metrics

### Validation

| Metric | Sprint 193 close state |
| --- | --- |
| source-list check | passed with 49 library sources |
| QR helper guard regression tests | passed |
| QR helper Make guard | passed |
| reviewed CMake compile/parity path | passed configure, clean rebuild, `ctest -N`, and test-count parity |
| CMake test count | 59 |
| Makefile test count | 59 |
| final `make format` | passed |
| final `make lint` | passed strict warnings, clang-tidy, and cppcheck |
| final `make test` | passed with final `All tests passed.` |
| `test_qr` | passed, 79 tests, 0 failures, 0 skips, 976 assertions |
| `test_reorder_nd` | passed, 35 tests, 0 failures, 1 skip |
| `test_reorder_amd_qg` | passed, 7 tests, 0 failures, 0 skips, 2068 assertions |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 193 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint closeout artifacts added | 1 |
| Sprint retrospective files added | 1 |
| Makefile guard targets added | 1 |
| Shell guard scripts added | 1 |
| Python guard tests added | 1 |
| Maintainer docs changed | 1 |
| C test files changed | 1 |
| C test helper headers added | 1 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Production source files changed | 0 |

### Review-Surface Metrics

| Metric | Sprint 193 close state |
| --- | ---: |
| `tests/test_qr.c` baseline lines | 3970 |
| `tests/test_qr.c` final lines | 3040 |
| Main QR proof-owner reduction | 930 lines |
| `tests/test_qr_external_ref_helpers.h` lines | 1003 |
| `test_qr` baseline registered tests | 77 |
| `test_qr` final registered tests | 79 |
| Reader failure-path tests added | 2 |
| Selected extraction claims closed | 1 |
| Broad QR refactor claims added | 0 |
| QR tolerance-policy claims added | 0 |
| Public API/ABI claims added | 0 |
| Production solver behavior claims added | 0 |
| Performance claims added | 0 |
| State-of-the-art claims added | 0 |

## Closed Claim

Sprint 193 closes this bounded implementation claim:

One selected QR external-reference rank/nullspace/threshold review surface is
now extracted from `tests/test_qr.c` into
`tests/test_qr_external_ref_helpers.h`, with `tests/test_qr.c` preserved as the
proof-owner executable, two reader failure-path tests added, helper/source-list
ownership mechanically guarded, maintainer documentation aligned, and final
format/lint/test/CMake/source-list validation passed.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-review-surface-intake.md](./artifacts/day1-review-surface-intake.md);
- [day2-candidate-ranking.md](./artifacts/day2-candidate-ranking.md);
- [day3-selected-cluster-contract.md](./artifacts/day3-selected-cluster-contract.md);
- [day4-extraction-boundary-design.md](./artifacts/day4-extraction-boundary-design.md);
- [day5-mechanical-extraction-scaffold.md](./artifacts/day5-mechanical-extraction-scaffold.md);
- [day6-helper-movement.md](./artifacts/day6-helper-movement.md);
- [day7-cleanup-ownership.md](./artifacts/day7-cleanup-ownership.md);
- [day8-source-list-guards.md](./artifacts/day8-source-list-guards.md);
- [day9-behavior-coverage.md](./artifacts/day9-behavior-coverage.md);
- [day10-boundary-documentation.md](./artifacts/day10-boundary-documentation.md);
- [day11-integrated-validation.md](./artifacts/day11-integrated-validation.md);
- [day12-full-quality-gate.md](./artifacts/day12-full-quality-gate.md);
- [day13-review-surface-audit.md](./artifacts/day13-review-surface-audit.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

No public API/ABI, production solver behavior, QR tolerance policy, rank
policy, generated report, benchmark, portability, performance, release, or
state-of-the-art claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Economy QR external-reference test remains in `tests/test_qr.c` | Future QR review-surface owner | Select the economy cluster explicitly, define its helper boundary, preserve registration, and rerun focused/full validation. |
| Sparse-mode and refinement QR clusters remain in `tests/test_qr.c` | Future QR review-surface owner | Audit each cluster separately and extract only if the move stays behavior-preserving and guardable. |
| Header-only focused rebuild caveat | Test/build owner | Add dependency tracking for helper headers or continue forcing rebuild before focused QR execution. |
| Large helper size | Future test-structure owner | Consider a second split only if it reduces review burden without creating source-list or build-system ownership ambiguity. |
| Existing unrelated CMake warning | Future warning-hygiene owner | Review `tests/test_svd_partial_corpus.c` double-promotion warning separately; Sprint 193 did not touch that file. |

## Next-Sprint Readiness

Sprint 194 can start from a completed selected review-surface reduction pattern
rather than another broad QR refactor.

| Future need | Sprint 193 handoff |
| --- | --- |
| Additional QR extraction | Repeat the selected-cluster sequence: candidate audit, boundary contract, mechanical move, registration preservation, guard, docs, focused validation, full C gate. |
| Focused QR validation | Force-rebuild `build/test_qr` after helper header edits before running `./build/test_qr`. |
| Guard changes | Update shell guard, Python guard fixtures, maintainer docs, and selected `RUN_TEST(...)` ownership together. |
| Build-system changes | Keep helper headers out of library source manifests unless a future sprint intentionally changes the ownership model. |
| Claim wording | Continue describing this as review-surface reduction only, not solver, tolerance, performance, or state-of-the-art evidence. |

## Validation Retrospective

Sprint 193 changed a C test file and added a C test helper header, so the full
C quality gate was required and run.

The final Day 14 validation command was:

```sh
make source-list-check && \
python3 tests/test_qr_external_ref_helper_guard.py && \
make qr-external-ref-helper-guard && \
make quality-review-cmake-compile && \
make format && \
make lint && \
make test
```

It passed. A final `git diff --check` also passed after the closeout docs were
added.
