# Sprint 218 Day 14: Quality And Claim Closeout

**State:** The selected Sprint 218 matmul owner meets the sprint completion
gate on 2026-10-23. This is local validation on `sprint-218` at baseline
HEAD `271bd3b704326296332cb397385ed458436ca0b7` plus the uncommitted
Sprint 218 worktree changes. No commit, push, or PR is implied by this
record.

## Full Quality Gates

The host was macOS 15.7.9 (Darwin 24.6.0, x86_64) with Apple Clang
17.0.0 (`clang-1700.6.4.2`). The normal Make build used C11, the project
warning flags, and `-O2`. Its fresh directory was
`build/s218-day14-normal`; the reviewed CMake clean build used
`build/s218-day14-cmake`. Neither reused the Day 11 comparison binaries nor
an instrumented archive. A default `make clean` was intentionally not run:
it would remove the detached baseline worktree and frozen comparison
executables nested under `build/`.

| Exact command, in execution order | Result | Evidence boundary |
| --- | --- | --- |
| `make format` | PASS, exit 0 | `clang-format -i` did not change the measured candidate source or benchmark driver SHA-256. |
| `make BUILDDIR=build/s218-day14-normal lint` | PASS, exit 0 | Fresh normal build; benchmark/example tooling, strict `-Werror` compile, clang-tidy, and cppcheck completed. |
| `make BUILDDIR=build/s218-day14-normal test` | PASS, exit 0; final `All tests passed.` | Full Make suite, 59 registered test binaries. Opt-in/framework fixture skips remain skips, not passes. |
| `make QUALITY_REVIEW_CMAKE_DIR=build/s218-day14-cmake quality-review-cmake` | PASS, exit 0 | Fresh configure, serial clean rebuild, CTest inventory/parity, and all 60 CTests passed. The count is 59 Make tests plus one CTest-only focused selector. |
| `ctest --test-dir build/s218-day14-cmake -L '^matmul$' --output-on-failure` | PASS, exit 0; 1/1 | Separately repeated labeled matmul CTest. |
| `make BUILDDIR=build/s218-day14-normal matmul-allocation-failure-gate` | PASS, exit 0; registration passed, 41 tests, 0 failed/skipped, 372262 assertions | Final normal-build allocation failure, ownership, and retry gate. |
| `make format-check`; `make source-list-check`; `git diff --check` | PASS, exits 0; 49 library sources | Post-run formatting, source manifest, and tracked whitespace hygiene. |
| `shasum -a 256 -c docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw-sha256.txt` | PASS, exit 0; 180/180 files | Raw before/after sample integrity after all builds. |

The CMake suite reported 100% of 60 CTests passing. Its default build
emitted nonfatal `-Wdouble-promotion` warnings for `NAN`/`INFINITY` macro
conversions in existing sources and tests; this is not a warning-free claim.
The Make output includes opt-in and fixture-dependent skips, including the
framework self-check/slow/experimental cases and a nested-dissection
balance-weight fixture. Those modes were not promoted to covered behavior.
No required matmul case was skipped.

The [Day 13 instrumented ledger](day13-measurement-freeze-quality.md)
separately records passing default-slab and one-node-slab `SANITIZE=all`
(ASan/UBSan) focused CTests, plus native macOS `leaks` reporting zero leaked
bytes on the normal focused binary. macOS ASan leak detection was disabled;
the native leak check is the leak observation. Those were focused tests,
not a full sanitizer run of the repository.

## Item Reconciliation

Hours are the sprint's planned estimates, not measured elapsed labor.

| Item | Planned hours | Completion evidence |
| --- | ---: | --- |
| 218.1 Baseline And Semantics | 20 | [Day 1 contract/cost inventory](day1-matmul-contract-cost-inventory.md) and [Day 2 pinned baseline](day2-matmul-baseline.md) fix the physical-index, ordering, threshold, shape, and failure expectations and three comparable shapes. |
| 218.2 Output Builder | 32 | [Day 3 design](day3-builder-ownership-design.md), [Day 4 row append](day4-row-append-implementation.md), [Day 5 column tails](day5-column-tail-integration.md), and [Day 6 builder review](day6-builder-review-dense-oracle.md) replace per-emitted-entry `sparse_insert` scans with private ordered row/column-tail linking. |
| 218.3 Correctness Matrix | 28 | [Day 7 cancellation/determinism](day7-cancellation-determinism.md) and [Day 8 seeded/consumer checks](day8-seeded-shape-consumer-matrix.md) add independent dense products, ordering, empty, threshold, duplicate, seeded, and downstream-consumer coverage. |
| 218.4 Failure Lifecycle | 24 | [Day 9 allocation map](day9-allocation-site-failure-map.md), [Day 10 rollback/instrumentation](day10-rollback-retry-sanitizer.md), and [Day 11 signoff](day11-failure-closure-measurement-protocol.md) cover explicit shell, workspace, first/later slab failures, NULL output, unchanged inputs, stale-output ownership, cleanup, and immediate retry. |
| 218.5 Controlled Measurement | 18 | [Day 11 protocol](day11-failure-closure-measurement-protocol.md), [Day 12 profile](day12-matmul-profile.md), and [Day 13 freeze](day13-measurement-freeze-quality.md) retain 60 validated processes, 180 raw files, integrity manifest, sample CSV, timing spread, and whole-process memory observations. |
| 218.6 Quality Closure | 18 | [Day 13 focused/instrumented results](day13-measurement-freeze-quality.md) and this Day 14 full Make/CMake record pass the required selected gates. |
| **Total planned** | **140** | **14 days at 10 planned hours each; all six items closed for the selected scope.** |

## Claim Boundary And Residuals

`sparse_matmul` still accumulates contributions and sorts touched columns,
but no longer performs repeated row-head or column-head scans for each
retained output entry. Linking after the sort is O(1) per retained node;
this is **not** a claim that the entire multiplication is linear-time.
Numerical values, ordering, threshold behavior, output ownership, and
explicit allocation-failure recovery are checked by the final gates.
No public header, library source-list, or unrelated storage-format change
appeared in the final diff.

The [frozen local profile](day12-matmul-profile.md) supports a selected-host
timing observation only: the candidate's median call was faster for the
`wide-row` and `rectangular` fixtures with nonoverlapping ten-sample ranges.
The `banded` ranges overlapped and three candidate pairs were slower, so no
robust narrow-row gain is claimed. Whole-process RSS and footprint are not
kernel allocation measurements and do not establish memory improvement.
The added column-tail workspace is an explicit cost. These tests do not
establish arbitrary-shape, cross-platform, all-kernel, library-wide, or
state-of-the-art performance. External review and future platforms may
still expose issues; no known failed or unavailable *required selected*
gate blocks Sprint 218 closeout on this host.

The raw manifest, derived CSV, driver, candidate source, and baseline and
candidate executable hashes still match the [Day 13 freeze](day13-measurement-freeze-quality.md).
The baseline detached worktree remains clean; the sprint worktree remains
uncommitted for the subsequent retrospective and PR step.

## Post-PR Portability Follow-Up (2026-10-24)

The original Day 14 result predates PR #242's Linux reviewed CMake failure:
the new benchmark compiled on macOS, but strict C11 on Linux did not expose
`clock_gettime` or `CLOCK_MONOTONIC`. A pre-include POSIX feature declaration
was added to `benchmarks/bench_matmul.c`; no fixture, timer, validation, or
library logic was changed. The benchmark source hash after that fix was
`61d5bf2cd0896dc4b9f6b5507195781f5a72be20f4891370e3d683dc3806aa61`.
The later PR #242 review fix adds a no-argument mode for `make bench`; its
source hash is
`11e2321d73bc662f5ba5b81b9ed05662dd0cd34541fbecb74f12496147508877`.
The original benchmark source and two frozen binaries retain the hashes in
the [Day 13 table](day13-measurement-freeze-quality.md). All 180 raw Day 12
files still verify against their manifest; no measurement was resampled or
attributed to the revised source. Local post-fix checks passed: `make format`,
`make BUILDDIR=build/s218-postci-normal lint`,
`make BUILDDIR=build/s218-postci-normal test`, and
`make QUALITY_REVIEW_CMAKE_DIR=build/s218-postci-cmake quality-review-cmake`
(60/60 CTests, count parity). `make format-check`,
`make source-list-check`, and `git diff --check` also passed. Both frozen
benchmark binaries retain their Day 13 hashes. These macOS checks do not
replace a new Linux CI run for the originally failing platform.

The follow-up Linux reviewed CMake job passed on commit `8236f10f`, but the
Linux reviewed Make lint job then failed a different check: clang-tidy
reported `bugprone-sizeof-expression` for `sizeof(*col_tails)` in the matmul
workspace allocation. That element is a `Node *`; spelling its size as
`sizeof(Node *)`, as the existing row/column header allocations do, leaves
the allocation unchanged. The current `src/sparse_matrix.c` SHA-256 is
`7cd4c6cf461db03557b6b680961b1eb8f5d94d782ce90f09b40103345b0f2ff1`.
The Day 12 profile and its frozen binaries still refer to the earlier
source hash in the Day 13 table. Local `make format`, normal-build
`make lint` and `make test`, and a rebuilt CMake `test_matmul` CTest all
passed after this second change. A new Linux lint result is required to
confirm the diagnostic is cleared by CI's clang-tidy version.
