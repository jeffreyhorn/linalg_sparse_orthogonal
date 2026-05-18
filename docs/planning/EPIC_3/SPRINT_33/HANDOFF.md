# Sprint 33 Handoff

**Source sprint:** 33  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 33's dead-code tooling, policy, and first cleanup
pass into concrete starting constraints for Sprint 34 and later Epic 3 quality
work.

## Starting State For Sprint 34

Sprint 33 does **not** hand off residual warning debt, dormant-scaffold debt, or
a remaining definitely-unused internal cleanup queue.

Authoritative validated quality state at Sprint 33 close:

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `ctest -N --test-dir build/sprint33-day1-cmake`: `53` registered tests
- `ctest --test-dir build/sprint33-day1-cmake --output-on-failure`: `53 / 53`
  passed

Validated dead-code workflow state at Sprint 33 close:

- `make deadcode-report`: passed
- `make deadcode-check`: passed

Current dead-code report buckets:

- `coverage-gap`: `7`
- `public-surface-review`: `4`
- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`
- `definitely-unused-internal-candidate`: `0`

## Dead-Code Workflow Contract Now In Force

Sprint 33 delivered these maintainer-facing targets:

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`

Generated artifact paths:

- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/report.md`
- `build/deadcode/report.tsv`

Interpretation contract now in force:

- the workflow is conservative evidence, not full reachability proof
- active code, opt-in test code, historical evidence, public-surface findings,
  and secondary static-analysis signals are distinct classes
- exported installed-header findings are review items, not automatic deletion
  candidates
- `deadcode-check` enforces report completeness, not zero findings

## Cleanup Result From Sprint 33

Sprint 33 removed the only approved definitely-unused internal candidate:

- `chol_csc_dump_supernodes`

Location removed from:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_internal.h`

That closes the first cleanup-ready internal queue produced by the current
Sprint 33 evidence model.

## Residual Deferred Queue

Sprint 33 does hand off a **tooling/reporting** queue, not a code-removal queue.

### Priority A: compile-db coverage gap

The current `xunused` compilation database still misses:

- benchmark:
  - `bench_svd`
- examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- scanner silence on those paths is not evidence of absence
- later CI/dead-code enforcement work should either broaden this coverage or
  explicitly preserve the exclusion list as a documented limitation

### Priority B: dead-code workflow execution model

The `deadcode*` targets currently share:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- they should be treated as serial/operator-invoked targets
- later CI integration should preserve serialization or isolate the worktree/build
  paths so concurrent invocations cannot race

### Priority C: residual reviewed / deferred evidence

Audited public keeps:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

These are not Sprint 34 cleanup targets unless the public API contract itself is
changed intentionally.

Residual evidence still deferred to later review:

- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`

These remain supporting analysis buckets, not direct deletion instructions.

## Sprint 34 First-Fix Queue

Sprint 34 should start from quality enforcement and tooling-hardening, not from
another speculative cleanup pass.

### Priority A: make the dead-code workflow enforcement-safe

Suggested work:

- preserve the Sprint 33 report/check targets
- harden them for CI or other repeated invocation
- decide whether serialization alone is acceptable or whether the build/artifact
  paths should be isolated automatically

### Priority B: improve analysis coverage before stronger enforcement

Suggested work:

- expand the dead-code compilation database toward the full Makefile tooling
  surface
- explicitly re-check `bench_svd` and the six missing examples once they are
  brought under coverage

### Priority C: preserve Sprint 32/Sprint 33 invariants

Later Epic 3 work should preserve all of these:

- warning-clean validated quality flows
- `53` registered CTest tests
- live opt-in coverage through `tests/test_framework_optin.c`
- no return of commented-out dormant test registrations
- no relabeling of audited public keeps as cleanup-ready code without a new API
  decision

## Reproduction Commands

Use these commands before and after later Epic 3 quality work:

1. `make format`
2. `make lint`
3. `make test`
4. `cmake --build build/sprint33-day1-cmake --parallel 1 --clean-first`
5. `ctest -N --test-dir build/sprint33-day1-cmake`
6. `ctest --test-dir build/sprint33-day1-cmake --output-on-failure`
7. `make deadcode-report`
8. `make deadcode-check`

Expected stable comparison targets at Sprint 33 close:

- `make lint`: passing
- `make test`: passing
- CTest registered tests: `53`
- full CTest run: `53 / 53` passing
- `definitely-unused-internal-candidate`: `0`
- compile-db coverage gap list: unchanged unless intentionally broadened

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day7-report-wiring-and-first-report.md](./artifacts/day7-report-wiring-and-first-report.md)
- [day8-public-surface-audit.md](./artifacts/day8-public-surface-audit.md)
- [day10-cleanup-batch1.md](./artifacts/day10-cleanup-batch1.md)
- [day11-reconciliation.md](./artifacts/day11-reconciliation.md)
- [day12-documentation-refresh.md](./artifacts/day12-documentation-refresh.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
