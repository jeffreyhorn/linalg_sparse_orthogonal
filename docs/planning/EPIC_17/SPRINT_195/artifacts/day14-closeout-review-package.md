# Sprint 195 Day 14: Closeout and Review Package

## Purpose

Package Sprint 195 for review by tying the selected reliability owner, harness
choice, regression tests, focused gate, documentation boundaries, and
validation evidence back to Epic 17 items 195.1 through 195.6.

## Selected Proof Summary

Sprint 195 selected `sparse_symbolic_cholesky()` in `src/sparse_etree.c` as the
bounded reliability owner. The implementation converts the selected non-empty
symbolic column-pointer allocation to the repository allocation wrapper so the
existing deterministic allocation-failure harness can prove cleanup and retry
behavior across the selected symbolic output construction path.

The proof covers selected `sparse_symbolic_cholesky()` output allocation,
partial-state cleanup, stale-output suppression, repeated cleanup after
failure, and retry-after-reset behavior on bounded fixtures. It does not claim
broad allocation-failure coverage for symbolic LU, analysis, direct solvers,
matrix construction, generated tooling, package/install paths, OS OOM behavior,
concurrent allocation-hook use, platform parity, or state-of-the-art
reliability.

## Item-to-Evidence Traceability

| Item | Evidence |
| --- | --- |
| 195.1 Owner Selection | `artifacts/day1-reliability-intake.md`, `artifacts/day2-owner-selection-scoring.md`, and `WORKING_NOTES.md` select `sparse_symbolic_cholesky()` by allocation density, cleanup complexity, user impact, and testability. |
| 195.2 Invariant Record | `artifacts/day3-selected-owner-invariant-record.md` records publication, cleanup, stale-output, retry, caller-owned input, and unsupported-breadth invariants before implementation. |
| 195.3 Harness Extension | `src/sparse_etree.c` uses wrapper-controlled allocation for the selected non-empty `sym->col_ptr` allocation; `tests/test_etree.c` uses `sparse_alloc_test_fail_after(...)` and reset helpers; `artifacts/day4-harness-design.md`, `day5-harness-scaffold.md`, and `day6-selected-owner-harness-integration.md` record the harness decision and integration. |
| 195.4 Regression Tests | `tests/test_etree.c` adds deterministic allocation hook reachability, partial-state cleanup, stale-output suppression, double-free-safe cleanup, caller-input preservation, and retry-after-reset tests; `artifacts/day7-failed-allocation-regression-tests.md`, `day8-cleanup-stale-output-proof.md`, and `day9-successful-retry-proof.md` record the coverage. |
| 195.5 Focused Gate And Docs | `Makefile` adds `make symbolic-allocation-failure-gate`; `CMakeLists.txt` labels `test_etree` with `allocation_failure`; `tests/test_symbolic_allocation_failure_gate_registration.py` guards focused coverage; `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` publish exact selected-owner claims and retained non-claims; `artifacts/day10-focused-gate-definition.md` and `day11-claim-boundaries.md` record the gate and wording. |
| 195.6 Validation | `artifacts/day12-focused-validation-source-ownership.md` and `artifacts/day13-full-quality-gate.md` record source-list, focused gate, CMake selector, docs grep, formatting, lint, and full test results. |

## Review Checklist

- Review `src/sparse_etree.c` to confirm the selected `sym->col_ptr`
  allocation is wrapper-controlled before publication and still preserves
  normal success semantics.
- Review `tests/test_etree.c` to confirm failure injection resets before
  assertions, failed calls leave `sparse_symbolic_t` empty, repeated cleanup is
  safe, caller-owned fixture data remains intact, and retry output matches the
  known 5x5 oracle.
- Review `tests/test_symbolic_allocation_failure_gate_registration.py` to
  confirm the focused gate cannot silently drop the selected test names, CMake
  label, Make target, or `col_ptr` fail-after case.
- Review `Makefile` and `CMakeLists.txt` to confirm focused local execution is
  reproducible through both Make and CTest selection.
- Review `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` to confirm
  they state the selected symbolic proof without widening it into broad
  reliability, platform, package, generated-tooling, OS OOM, concurrency, or
  state-of-the-art claims.
- Review `artifacts/day13-full-quality-gate.md` for the final full-quality
  gate results before merge.

## Closeout Checks

```sh
git status --short --branch
git diff --stat
git diff -- CMakeLists.txt Makefile src/sparse_etree.c tests/test_etree.c tests/test_symbolic_allocation_failure_gate_registration.py | sed -n '1,260p'
git diff -- README.md INSTALL.md docs/maintainer_guide.md | sed -n '1,260p'
make source-list-check
python3 tests/test_symbolic_allocation_failure_gate_registration.py
git ls-files --others --exclude-standard
rg -n "195\.1|195\.2|195\.3|195\.4|195\.5|195\.6|Owner Selection|Invariant Record|Harness|Regression Tests|Focused Gate|Validation" docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md docs/planning/EPIC_17/SPRINT_195/artifacts
```

## Closeout Results

| Check | Result |
| --- | --- |
| Scope diff | Code changes are limited to the selected symbolic Cholesky allocation proof, focused gate wiring, and claim-boundary documentation. |
| Source-list ownership | `make source-list-check` passed with 49 library sources. |
| Focused registration guard | `python3 tests/test_symbolic_allocation_failure_gate_registration.py` passed. |
| Untracked files | `git ls-files --others --exclude-standard` listed only Sprint 195 day artifacts that should be included with the sprint package. |
| Item traceability | Targeted grep found item 195.1 through 195.6 coverage in working notes and artifacts. |

## Final Residuals

- The proof remains local to selected `sparse_symbolic_cholesky()` output
  allocation behavior on bounded fixtures.
- No broad allocation-failure, OS OOM, concurrent allocation-hook, platform,
  package/install, generated-tooling, performance, release, or state-of-the-art
  reliability claim is made.
- Future reliability work should select a new owner and repeat the same
  invariant-first, focused-gate, claim-boundary process rather than extending
  Sprint 195 wording by implication.
