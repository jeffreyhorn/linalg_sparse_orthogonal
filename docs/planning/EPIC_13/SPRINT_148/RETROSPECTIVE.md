# Sprint 148 Retrospective

**Sprint:** 148 - Windows Staged Test Portability Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-148`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 148 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Audited the Windows-staged test sources and CMake/CI registration policy
      for `test_threads`, `test_sprint4_integration`, and `test_fuzz`.
- [x] Selected per-test portability dispositions with explicit rollback rules
      and expected Windows CTest count deltas.
- [x] Added `tests/test_thread_helpers.h` as a test-only portable thread helper
      preserving POSIX pthread behavior and adding a Win32 backend.
- [x] Promoted `test_threads` into the reviewed Windows CMake subset.
- [x] Promoted `test_sprint4_integration` into the reviewed Windows CMake
      subset while preserving its concurrent SuiteSparse Cholesky lane.
- [x] Promoted `test_fuzz` into the reviewed Windows CMake subset through a
      portable `.mtx` temp-file helper.
- [x] Updated CMake registration and Windows CI policy so
      `EXPECTED_WINDOWS_CTEST_COUNT` moves from `56` to `59`.
- [x] Updated README, INSTALL, and maintainer-guide support wording so the
      three promoted tests are no longer described as Windows-staged.
- [x] Preserved Windows non-claims for Makefile parity, `pkg-config` parity,
      separate reviewed install-validation parity, package-manager support,
      shared-library support, runtime-loader behavior, dynamic ABI support, and
      broad Windows parity.
- [x] Ran focused local CMake validation for all promoted targets.
- [x] Ran required full C gate: `make format && make lint && make test`.
- [x] Recorded hosted Windows proof as pending PR CI because no PR existed at
      Day 13/Day 14 closeout.

## What Went Well

1. **The sprint closed the exact staged-test gap it selected.** The three named
   Windows-excluded CMake tests were all audited, ported, registered, documented,
   and validated locally. No staged-test surface remained silently deferred.

2. **The helper boundary was small and reusable.** `tests/test_thread_helpers.h`
   let `test_threads` and `test_sprint4_integration` keep their existing
   `void *(*)(void *)` worker shape, POSIX pthread behavior, stress counts, and
   diagnostics while isolating the Windows-specific thread lifecycle.

3. **The fuzz lane stayed intact.** The portable temp-file helper allowed
   `test_fuzz` to keep file-backed Matrix Market malformed-input coverage,
   deterministic property seeds, and large CSC lifecycle/reorder properties in
   the same target instead of splitting the lane.

4. **CTest count policy changed only after implementation evidence existed.**
   The workflow stayed at `56` while individual ports landed, then moved to
   `59` after all three registrations were complete and local enumeration
   matched the expected final surface.

5. **Support wording stayed bounded.** Day 12 updated current public/support
   docs to match the promoted CMake surface without implying Windows Makefile,
   `pkg-config`, install-validation, package-manager, shared-library, dynamic
   ABI, or broad platform parity.

## What Didn't Go Well

1. **Hosted Windows evidence remains pending.** Local macOS and CMake checks
   caught syntax, registration, and `_WIN32` static-analysis issues, but only
   PR CI can prove MSVC configure/build/full CTest execution.

2. **The sprint carried a long uncommitted branch.** Days 6, 8, and 10 each
   changed C surfaces before the final retrospective commit. The Day 13 full
   gate mitigated this, but the review diff is larger than a single small
   portability patch.

3. **The public docs needed a separate cleanup pass.** The implementation and
   workflow promotion were correct by Day 11, but README/INSTALL/maintainer
   wording still described the old staged state until Day 12.

4. **Count-sensitive Windows CI remains brittle by design.** The explicit
   `EXPECTED_WINDOWS_CTEST_COUNT=59` guard is useful, but future test additions
   must update the count deliberately or CI will fail.

5. **Sprint 149 still has a distinct platform decision.** Sprint 148 closed
   staged CMake test portability, not Windows install-validation parity. That
   separation is correct, but it leaves another Windows support boundary for
   the next sprint.

## Final Metrics

### Validation

| Metric | Sprint 148 close state |
| --- | --- |
| tracked `.c` changes | yes |
| tracked `.h` changes | yes |
| full C quality gate required | yes |
| focused CMake configure/build for promoted targets | passed |
| local CTest enumeration | `Total Tests: 59` |
| focused promoted-target CTest | passed: 3 tests, 0 failures |
| full C quality gate | passed: `make format && make lint && make test` |
| `cppcheck` `_WIN32` coverage | covered `test_threads`, `test_sprint4_integration`, and `test_fuzz` |
| documentation whitespace validation | passed |
| stale public/support staged-wording search | passed |
| `git diff --check` | passed |
| hosted Windows CI | pending PR CI; no PR existed during Day 13/Day 14 closeout |

### Artifact Package

| Metric | Sprint 148 close state |
| --- | ---: |
| daily artifacts under `SPRINT_148/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 3 |
| test helper headers added | 1 |
| workflow files changed | 1 |
| public/support docs changed | 3 |

## Closed Claim

Sprint 148 closes this claim:

The previously staged Windows CMake test surfaces `test_threads`,
`test_sprint4_integration`, and `test_fuzz` have been ported or adapted for the
reviewed Windows CMake subset, and the reviewed Windows expected CTest count is
now `59` with explicit hosted MSVC proof requirements.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-intake.md](./artifacts/day1-windows-intake.md);
- [day2-staged-test-source-audit.md](./artifacts/day2-staged-test-source-audit.md);
- [day3-cmake-ci-registration-audit.md](./artifacts/day3-cmake-ci-registration-audit.md);
- [day4-portability-decision-matrix.md](./artifacts/day4-portability-decision-matrix.md);
- [day5-thread-test-port-design.md](./artifacts/day5-thread-test-port-design.md);
- [day6-thread-test-port-implementation.md](./artifacts/day6-thread-test-port-implementation.md);
- [day7-sprint4-integration-port-design.md](./artifacts/day7-sprint4-integration-port-design.md);
- [day8-sprint4-integration-port-implementation.md](./artifacts/day8-sprint4-integration-port-implementation.md);
- [day9-fuzz-property-port-design.md](./artifacts/day9-fuzz-property-port-design.md);
- [day10-fuzz-property-port-implementation.md](./artifacts/day10-fuzz-property-port-implementation.md);
- [day11-cmake-ci-promotion-batch.md](./artifacts/day11-cmake-ci-promotion-batch.md);
- [day12-docs-alignment.md](./artifacts/day12-docs-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md).

## Next-Sprint Readiness

Sprint 149 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Windows staged CMake tests | Treat `test_threads`, `test_sprint4_integration`, and `test_fuzz` as promoted, subject to PR CI confirmation. |
| Windows CTest count | Baseline is `EXPECTED_WINDOWS_CTEST_COUNT=59`; update only with intentional before/after enumeration. |
| Hosted evidence | Verify the PR Windows CI reports configure/build success, `Total Tests: 59`, and full CTest success. |
| Windows install/downstream confidence | Still supplemental; do not treat it as reviewed install-validation parity without a new Sprint 149 decision and evidence. |
| Windows Makefile and `pkg-config` parity | Still non-claims unless separately implemented and proven. |
| Package/ABI support | Static-first package posture remains; shared-library and dynamic ABI support remain out of scope. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 148 close:

- hosted Windows proof for the promoted tests until PR CI runs;
- Windows install-validation parity until Sprint 149 makes a product decision;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- package-manager support;
- shared-library support;
- runtime-loader behavior;
- dynamic ABI support;
- broad Windows platform parity beyond the reviewed MSVC CMake lane.

Still consciously constrained rather than silently solved:

- no broad Windows ecosystem parity claim;
- no Windows package-manager or installer claim;
- no generated report pass evidence from source-controlled rows alone;
- no portable performance claim from this sprint;
- no external-library comparison claim;
- no unqualified state-of-the-art sparse linear algebra claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-windows-intake.md](./artifacts/day1-windows-intake.md)
- [day2-staged-test-source-audit.md](./artifacts/day2-staged-test-source-audit.md)
- [day3-cmake-ci-registration-audit.md](./artifacts/day3-cmake-ci-registration-audit.md)
- [day4-portability-decision-matrix.md](./artifacts/day4-portability-decision-matrix.md)
- [day5-thread-test-port-design.md](./artifacts/day5-thread-test-port-design.md)
- [day6-thread-test-port-implementation.md](./artifacts/day6-thread-test-port-implementation.md)
- [day7-sprint4-integration-port-design.md](./artifacts/day7-sprint4-integration-port-design.md)
- [day8-sprint4-integration-port-implementation.md](./artifacts/day8-sprint4-integration-port-implementation.md)
- [day9-fuzz-property-port-design.md](./artifacts/day9-fuzz-property-port-design.md)
- [day10-fuzz-property-port-implementation.md](./artifacts/day10-fuzz-property-port-implementation.md)
- [day11-cmake-ci-promotion-batch.md](./artifacts/day11-cmake-ci-promotion-batch.md)
- [day12-docs-alignment.md](./artifacts/day12-docs-alignment.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md)
