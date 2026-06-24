# Sprint 87 Retrospective

**Sprint:** 87 — Packaging, ABI & Cross-Platform Quality Convergence  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 87 fixed the package/consumer/workflow baseline, proof split,
      and implementation-day validation contract before landing packaging work
- [x] the strongest live package / ABI / install-export contradiction map was
      reranked from the current tree rather than inherited generically from
      Sprint 86
- [x] Sprint 87 fixed one explicit first implementation fence centered on:
  - `CMakeLists.txt`
- [x] Sprint 87 landed one bounded packaging/export batch:
  - the generated `SparseConfigVersion.cmake` now uses `ExactVersion`
  - the installed CMake package no longer advertises same-major-version
    compatibility the repo does not maintain
- [x] Sprint 87 landed one bounded consumer-proof expansion batch:
  - `tests/test_install.sh` still proves the basic installed compile/link/run
    lane
  - it now also proves the maintained `examples/cmake_example/main.c` source
    compiles and runs through installed `pkg-config` metadata
- [x] Sprint 87 landed one bounded workflow/platform follow-through batch:
  - `.github/workflows/macos-ci.yml` now reuses the maintained local
    `tests/test_install.sh` proof owner directly
  - the supplemental macOS package lane is closer to the maintained local
    proof than it was at sprint start
- [x] Sprint 87 used bounded follow-through correctly:
  - `tests/test_cmake_install.sh` remained the retained CMake install/export
    proof owner
  - `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` moved only where
    the package/export contract actually changed
  - `.github/workflows/windows-ci.yml` correctly remained the narrower
    CMake-first consumer subset and did not widen into reviewed install/export
    parity
- [x] Sprint 87 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- [x] Sprint 87 closed with one explicit Sprint 88-first handoff queue instead
      of another generic Epic 8 packaging summary

## What Went Well

1. **Sprint 87 chose the right first package lane.**
   The sprint did not start by widening Windows scope, opening a speculative
   shared-library lane, or rewriting workflows first. It first tightened the
   package/export contract at the strongest product owner:
   `CMakeLists.txt`.

2. **The Day 6 packaging landing made the export story more truthful.**
   Moving the generated CMake package version contract from
   `SameMajorVersion` to `ExactVersion` removed the strongest remaining
   installed ABI overclaim without widening the product surface.

3. **The consumer-proof batch strengthened real downstream evidence.**
   Day 9 did not settle for another tiny ad hoc consumer. It made the
   maintained example source prove the installed Make/pkg-config lane, which
   is a materially better consumer story.

4. **The workflow follow-through reused the maintained proof owner instead of
   drifting into a parallel script.**
   Day 11 improved the macOS supplemental lane by making it call
   `tests/test_install.sh` directly rather than preserving a thinner,
   hand-rolled subset.

5. **Scope discipline stayed intact.**
   Sprint 87 kept the repo static-first, did not widen ABI guarantees, did not
   claim reviewed install/export parity on macOS or Windows, and did not
   redistribute correctness ownership out of the existing proof surfaces.

6. **Sprint 88 now starts from a more truthful package and workflow contract.**
   The front-door simplification sprint can work on top of a sharper
   install/export story instead of first resolving package-version and
   consumer-proof ambiguity.

## What Didn't Go Well

1. **Sprint 87 did not open a shared-library product lane.**
   That was the correct bounded decision, but it means the repo still leaves
   shared/static widening as later work rather than closing that question here.

2. **Platform convergence remains intentionally asymmetric.**
   macOS improved as supplemental package evidence, but Windows still remains
   the narrower reviewed CMake-first consumer subset and does not yet carry a
   separate install/export validation lane.

3. **The strongest package improvements remained mostly outside `*.c` / `*.h`
   implementation surfaces.**
   That was right for the sprint, but it means Sprint 87 improved truthfulness
   and consumer confidence more than it changed runtime or numerical internals.

4. **The reviewed runtime long pole was inherited rather than attacked here.**
   Sprint 87 closed from a clean Day 13 baseline, but the reviewed path still
   carries:
   - reviewed CMake total = `299.15 sec`
   - reviewed `test_reorder_nd` = `142.76 sec`

5. **The canonical reporting face stayed unchanged by design.**
   That was the correct bounded choice, but it also means Sprint 87 did not
   widen benchmark or reporting ownership as part of the package story.

## Final Metrics

### Validation and package-proof anchors

| Metric | Sprint 87 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `299.15 sec` |
| reviewed `test_reorder_nd` time | `142.76 sec` |
| focused `test_reorder` follow-on | `38 / 38` |
| focused `test_reorder_nd` follow-on | `35 / 35` with `1` skip |
| focused `test_reorder_amd_qg` follow-on | `7 / 7` |
| focused `test_graph` follow-on | `61 / 61` |
| Make/pkg-config install proof | `bash tests/test_install.sh` passed |
| Make/pkg-config install proof totals | `13` passed, `0` failed |
| CMake install/export proof | `bash tests/test_cmake_install.sh` passed |
| CMake install/export proof totals | `15` passed, `0` failed |
| canonical reporting follow-on | `make bench-canonical-report` passed |

### Product-contract headline

| Metric | Sprint 87 close state |
|---|---:|
| maintained release shape | static-first |
| generated CMake package version contract | `ExactVersion` |
| same-major-version CMake compatibility claim | removed |
| local Make/pkg-config consumer proof | strengthened |
| local CMake install/export consumer proof | retained and widened for exact-version checks |
| supplemental macOS package lane | aligned to `tests/test_install.sh` |
| Windows install/export parity claim | not widened |

### Sprint 87 artifact package

| Metric | Sprint 87 close state |
|---|---:|
| total artifact files under `SPRINT_87/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-packaging-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-maintained-consumer-surface-recheck.md`
  - `day3-release-package-gap-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-support-surface-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-packaging-abi-boundary.md`
  - `day5-product-matrix-design.md`
  - `day6-packaging-batch.md`
  - `day8-consumer-proof-expansion-design.md`
  - `day9-consumer-proof-expansion-batch.md`
  - `day10-workflow-platform-follow-through-design.md`
  - `day11-workflow-platform-follow-through-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 87 close state |
|---|---:|
| build/package owner files touched | `1` |
| install/export proof scripts touched | `2` |
| workflow files touched | `1` |
| repo-wide support docs requiring follow-through | `3` |
| C/C++ implementation files touched | `0` |
| public header files touched | `0` |
| export template files touched | `0` |

Notes:

- build/package owner files touched:
  - `CMakeLists.txt`
- install/export proof scripts touched:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- workflow files touched:
  - `.github/workflows/macos-ci.yml`
- repo-wide support docs requiring follow-through:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- support surfaces intentionally left untouched after recheck:
  - `.github/workflows/ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`

## Residual Deferred Debt

Sprint 87 deliberately stopped after the highest-value package / consumer /
workflow package. The main open work it hands forward is:

- front-door usability and workflow simplification
- final integration, external comparison, and Epic 8 closeout
- later package/platform widening only where bounded evidence still justifies
  more than the maintained static-first contract

Still consciously constrained rather than silently “solved”:

- no shared-library product guarantee
- no broad ABI-compatibility guarantee
- no reviewed install/export parity claim on macOS or Windows
- no benchmark/reporting ownership widening
- no proof-owner redistribution out of
  `tests/test_install.sh` / `tests/test_cmake_install.sh`

Not carried forward as unresolved Sprint 87 debt:

- the package / ABI / consumer baseline recheck
- the live package-gap rerank
- the bounded product-matrix architecture contract
- the Day 6 packaging/export landing
- the Day 9 consumer-proof expansion
- the Day 11 workflow/platform follow-through
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 88-first handoff queue

## Key Deliverables

1. **One bounded packaging/export contract tightening landed at the product owner.**
   `CMakeLists.txt` now emits exact-version CMake package semantics that match
   the maintained static-first and no-broad-ABI contract.

2. **One bounded consumer-proof expansion landed on the maintained local Unix lane.**
   `tests/test_install.sh` now proves the installed `pkg-config` metadata can
   build and run the maintained example source, not just a tiny synthetic
   consumer.

3. **One bounded CMake install/export proof widening stayed retained and explicit.**
   `tests/test_cmake_install.sh` now proves exact-version success and
   mismatched-version rejection on the installed CMake package surface.

4. **One bounded workflow/platform follow-through improved fidelity without widening scope.**
   `.github/workflows/macos-ci.yml` now reuses the maintained local proof
   owner directly while still remaining supplemental package evidence rather
   than reviewed install/export parity.

5. **The support surfaces now read more truthfully about the package contract.**
   `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` now align with
   the exact-version CMake export behavior and the retained static-first
   package story.

## Bottom Line

Sprint 87 achieved its purpose. The project now has one sharper static-first
package/export contract, one stronger local consumer proof story, one more
truthful supplemental macOS package lane, and one smaller gap between the
install story and the reviewed story, all without widening ABI or platform
claims beyond the surfaces the repo can realistically maintain. Sprint 88 can
now simplify the front door on top of a steadier package and workflow
contract instead of reopening install/export semantics first.
