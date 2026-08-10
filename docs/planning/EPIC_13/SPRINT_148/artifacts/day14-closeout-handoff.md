# Sprint 148 Day 14: Closeout Handoff

## Purpose

Close Sprint 148 by publishing the final Windows staged-test portability
outcome, tying validation evidence to the promoted CMake surface, and handing
Windows install-validation parity forward to Sprint 149 as a separate decision.

## Final Staged-Test Outcome

Sprint 148 closes the staged Windows CMake test gap for the three named test
surfaces.

| Former Staged Surface | Final Outcome | Evidence |
| --- | --- | --- |
| `test_threads` | Promoted into the reviewed Windows CMake subset through `tests/test_thread_helpers.h`. | Day 6 implementation, Day 11 workflow count update, Day 13 focused/full validation. |
| `test_sprint4_integration` | Promoted into the reviewed Windows CMake subset through the same portable thread helper. | Day 8 implementation, Day 11 workflow count update, Day 13 focused/full validation. |
| `test_fuzz` | Promoted into the reviewed Windows CMake subset through portable `.mtx` temp-file creation and cleanup. | Day 10 implementation, Day 11 workflow count update, Day 13 focused/full validation. |

No Sprint 148 staged-test surface remains intentionally excluded from the
reviewed Windows CMake subset.

## Support Boundary After Sprint 148

The reviewed Windows support statement is:

- MSVC 2022 via CMake configure/build;
- `ctest -N` enumeration with `EXPECTED_WINDOWS_CTEST_COUNT=59`;
- full hosted Windows CTest execution;
- included promoted targets: `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`.

The following remain non-claims:

- Windows Makefile parity;
- Windows `pkg-config` parity;
- separate reviewed Windows install-validation parity;
- package-manager support;
- shared-library support;
- runtime-loader behavior;
- dynamic ABI support;
- broad Windows parity beyond the hosted MSVC CMake lane.

## Validation Summary

Day 13 completed the integrated local validation pass:

- `cmake -S . -B build`: passed;
- `cmake --build build --target test_threads test_sprint4_integration test_fuzz`:
  passed;
- `ctest --test-dir build -N`: passed with `Total Tests: 59`;
- `ctest --test-dir build -R '^(test_threads|test_sprint4_integration|test_fuzz)$'
  --output-on-failure`: passed;
- `make format && make lint && make test`: passed;
- final full test output ended with `All tests passed.`

Day 14 documentation hygiene:

- stale public/support staged-exclusion wording search: passed;
- trailing-whitespace check over Sprint 148 artifacts and touched public docs:
  passed;
- `git diff --check`: passed.

## Hosted Windows Evidence

No pull request exists for `sprint-148` yet, so hosted Windows CI is pending.
The PR must provide the final reviewed Windows evidence:

- Windows CMake configure passes;
- Windows CMake build passes;
- Windows `ctest -N` reports `Total Tests: 59`;
- Windows full CTest passes.

If hosted Windows reports a different count or a promoted-target failure, treat
that as a Sprint 148 PR fix, not a Sprint 149 install-parity issue.

## Sprint 149 Handoff

Sprint 149 can start from this baseline:

- staged-test portability for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz` is closed for the reviewed Windows CMake lane;
- Windows install/downstream confidence remains supplemental;
- Windows install-validation parity must be evaluated independently from the
  Sprint 148 test promotion;
- package and ABI non-claims remain intact unless Sprint 149 adds evidence and
  explicit policy changes.

Suggested Sprint 149 opening checks:

1. Inspect the merged Sprint 148 hosted Windows CI result.
2. Confirm `EXPECTED_WINDOWS_CTEST_COUNT=59` remains stable after merge.
3. Decide whether Windows install-validation parity should remain supplemental,
   be promoted, or be explicitly deferred with blockers.
4. Keep Windows Makefile and `pkg-config` parity separate from CMake package
   confidence unless direct evidence is added.

## Retrospective Input Notes

- What worked: deferring the workflow count change until all three source ports
  landed avoided partial expected-count churn.
- What worked: the small test-only thread helper closed two pthread blockers
  without weakening POSIX coverage.
- What worked: the fuzz temp-file helper kept the existing file-backed parser
  coverage in one target rather than splitting the lane.
- Watch item: hosted Windows proof is still pending because the branch has no
  PR yet.
- Watch item: Sprint 149 should not infer install/package parity from CTest
  promotion.
- Follow-through: PR review should verify that public docs, workflow comments,
  and hosted Windows evidence all agree on the `59`-test reviewed CMake surface.
