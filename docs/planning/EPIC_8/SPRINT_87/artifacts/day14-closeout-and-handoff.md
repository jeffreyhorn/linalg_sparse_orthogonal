# Sprint 87 Day 14: Closeout and Handoff

## Purpose

Close Sprint 87 from the validated Day 13 baseline and leave one explicit
handoff queue for Sprint 88 and the later Epic 8 implementation sprints.

## Closeout State

Sprint 87 now closes as one coherent Epic 8 packaging, ABI, and
cross-platform quality-convergence package across:

- package / ABI / consumer gap rerank
- bounded product-matrix and package-contract architecture
- Day 6 bounded packaging batch
- Day 9 bounded consumer-proof expansion
- Day 11 bounded workflow/platform follow-through
- validated Day 13 close baseline

The preserved fence stayed intact:

- Sprint 87 sharpened the static-first package/export contract instead of
  reopening a broader shared-library product lane
- the first packaging landing stayed product-owned inside `CMakeLists.txt`
- the consumer-proof expansion stayed script-owned inside
  `tests/test_install.sh` and did not redistribute correctness ownership into
  reviewed numerical proof surfaces
- the retained CMake install/export proof stayed owned by
  `tests/test_cmake_install.sh`
- the workflow follow-through stayed supplemental and macOS-local inside
  `.github/workflows/macos-ci.yml`
- Windows scope remained the narrower maintained CMake-first consumer subset
  and did not claim reviewed install/export parity
- canonical maintained reporting stayed unchanged under
  `make bench-canonical-report`

## Project-Plan Recheck

`docs/planning/EPIC_8/PROJECT_PLAN.md` does not need a Sprint 87 correction.

The landed Sprint 87 package still supports the intended Epic 8 execution
order:

1. Sprint 88 front-door usability and workflow simplification after the
   package/platform contract is stable enough for public-surface cleanup
2. Sprint 89 final integration, external comparison, and Epic 8 closeout
   after the front-door guidance layers are reconciled
3. later package/platform widening only where bounded evidence still justifies
   it beyond the maintained static-first contract

## Validated Baseline

Sprint 87 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 299.15 sec`
- `./build/quality-review-cmake/test_reorder` -> `38 / 38`
- `./build/quality-review-cmake/test_reorder_nd` -> `35 / 35` with `1` skip
- `./build/quality-review-cmake/test_reorder_amd_qg` -> `7 / 7`
- `./build/quality-review-cmake/test_graph` -> `61 / 61`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `bash tests/test_install.sh` -> `13` passed, `0` failed
- `bash tests/test_cmake_install.sh` -> `15` passed, `0` failed
- `make bench-canonical-report`

This means Sprint 87 hands off from one measured packaging/consumer/workflow
baseline rather than from package-contract design intent alone.

## Handoff Queue

The ranked carry-forward queue from Sprint 87 is now fixed explicitly:

1. front-door usability and workflow simplification on top of the now-stable
   static-first package and consumer contract
2. final integration, external comparison, and Epic 8 closeout after the
   usability surfaces are simplified
3. later package/platform widening only where bounded evidence justifies more
   than the maintained static-first and narrower-platform contract

## Bottom Line

Sprint 87 achieved its purpose: the project now has one sharper static-first
package/export contract, one stronger maintained local consumer proof story,
one better-aligned supplemental macOS package lane, and one smaller gap
between the install story and the reviewed story without widening ABI or
platform claims beyond the surfaces the repo can realistically maintain.
Sprint 88 can now simplify the front door on top of a more truthful package
and workflow contract instead of re-deciding install/export semantics first.
