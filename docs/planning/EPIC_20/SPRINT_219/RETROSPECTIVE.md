# Sprint 219 Retrospective

**Sprint:** 219 - Enforced Installed Index-Width Contract

**Duration:** 14 planned days on branch `sprint-219`

**Status:** Four items closed locally; selected hosted and final acceptance gates open

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) assigns six items and 140
estimated hours. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md),
and [Day 14 closeout](artifacts/day14-quality-and-contract-closeout.md)
retain the daily evidence. Estimates are allocations, not measured effort.
The intended result is a static package whose supported consumers select
the archive's 32- or 64-bit index width, or reject a mismatch before an
index-bearing call. Local verification is not a substitute for the required
hosted Linux, macOS, and Windows package lanes.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 219.1 ABI Width Inventory | 20 h | Closed: public ABI inventory, mismatch reproduction, and failure design. |
| 219.2 Build Configuration | 30 h | Closed locally: validated Make/CMake selectors, generated width identity, separate trees, and stale-tree rejection. |
| 219.3 Consumer Propagation | 26 h | Closed locally: generated header, imported target, pkg-config flag, static validator, and pre-use consumer check. |
| 219.4 Positive/Negative Consumers | 24 h | Closed locally: clean-prefix 32/64 consumers and wrong-width/mixed-prefix negatives. |
| 219.5 Selected 64-Bit Lane | 20 h | Open: selected local functional and UBSan tests passed; registered hosted jobs have not run. |
| 219.6 Contract Validation | 20 h | Open: local full-quality and installed-contract checks passed; required hosted acceptance and full-sprint signoff remain. |

The first four items account for 100 planned hours of locally closed work.
The remaining 40 hours are **not** marked complete, and the 140-hour sprint
plan is **not** closed. This is a status of evidence, not an assertion about
actual hours worked.

## What Went Well

1. **The ABI inventory found a reproducible unsafe boundary.** A 64-bit
   consumer could previously compile and link against a 32-bit package.
   The Day 1 query-only probe stopped short of calling an index-bearing API,
   and Day 2 defined separate compile-time, package, and runtime checks.
2. **Build identity became explicit without changing the default.** Make
   and CMake still default to 32 bits. Both accept only 32 or 64, generate
   matching `SPARSE_BUILT_IDX_BITS` headers, and refuse opposite-width
   reuse of a build tree. The same-build test compares typedef, header, and
   linked archive widths.
3. **Both installed discovery routes carry the same contract.** The CMake
   imported target and `sparse.pc` publish a package-width definition
   distinct from an explicit consumer `SPARSE_IDX_BITS`. The public header
   rejects disagreement; the static validator checks header, metadata,
   and the real archive's `sparse_idx_bits()` result.
4. **Negative consumers test failure before unsafe use.** Relocated CMake
   imports, isolated `PKG_CONFIG_LIBDIR` consumers, swapped archives,
   foreign include/library order, stale caches, and wrong explicit widths
   have named rejection phases. The maintained example checks the linked
   archive before its first index-bearing matrix operation.
5. **Selected 64-bit behavior has executable evidence.** A fresh 64-bit
   lane checks header/archive identity, insertion and LU, Matrix Market,
   CSR/CSC conversion, and text indices above `INT32_MAX` without a huge
   allocation. Focused 64-bit UBSan CTests also passed for the functional,
   SuiteSparse, and QR cases after width-sensitive test diagnostics were
   repaired.

## What Didn't Go Well

1. **Hosted evidence was unavailable at closeout.** The branch had no
   upstream, PR, or workflow runs when Day 14 was recorded. The Linux and
   macOS selected 64-bit jobs, installed-consumer jobs, and Windows
   default-width package route are registered but unproven. Earlier
   `master` runs cannot close this branch's gate.
2. **A broad 64-bit sanitizer run was too slow for the local sweep.** The
   UBSan Make target passed earlier binaries without a reported sanitizer
   error, but `test_reorder_nd` remained CPU-active for over ten minutes and
   was interrupted. The target exited nonzero. Only the bounded 3/3 focused
   UBSan result is counted as a sanitizer pass.
3. **Compiler warnings exposed width-sensitive tests.** The broad 64-bit
   attempt found `%d` diagnostics for `idx_t` and an `abs()` conversion in
   QR. These were fixed with `SPARSE_PRIDX` and a non-narrowing difference,
   then retested. Apple Clang's nonfatal `NAN`/`INFINITY` double-promotion
   warnings and duplicate `-lm` link warnings remain separate from the
   installed-width contract.
4. **The safety check is a maintained-consumer convention, not an ABI
   firewall.** A manually linked caller can omit the pre-use query, and
   duplicate same-name compiler definitions can mask an earlier conflicting
   flag. The installed metadata and fixture are precise about this limit;
   they do not make mixed-width calls safe.
5. **Early progress labels drifted from the plan.** Day 6-7 artifacts
   described item 219.3 as 30 hours even though its allocation is 26.
   Those labels and the notes header were corrected during retrospective
   preparation; day-by-day evidence and the 140-hour plan were unchanged.

## Final Metrics And Claim

| Metric | Sprint 219 local state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked; 100 h locally closed, 40 h gated |
| Daily evidence | 14 primary day artifacts, working notes, this retrospective |
| Make width and static package | Width fixture 26/26; `test_install.sh` 26/26; static validator 26 tests |
| Installed consumer matrices | CMake width 18/18; pkg-config width 40/40; pre-use archive check 12/12; `test_cmake_install.sh` 38/38 |
| Full default-width quality | `make format`, `make lint`, and `make test` passed in an isolated Make tree; source-list check: 49 library sources |
| Reviewed CMake/CTest | Clean rebuild, Make/CMake test-count parity, 62/62 CTests passed |
| Selected 64-bit functional | Five named cases and focused CTest 1/1 passed; typedef, header, archive all report 64 |
| Selected 64-bit sanitizer | Focused UBSan CTests 3/3 passed; broad `make sanitize` was interrupted and did not pass |
| Hosted acceptance | No `sprint-219` runs or PR at Day 14 closeout; required lane conclusions pending |

The earned local claim is an **enforced width contract for the maintained
static-package consumer routes**, with default 32-bit preservation and
selected 64-bit behavior. It is not a shared-library, mixed-width,
cross-version, arbitrary manual-link, all-algorithm 64-bit, or
state-of-the-art performance claim.

## Daily Evidence

- [Day 1 public ABI inventory](artifacts/day1-public-abi-width-inventory.md)
- [Day 2 width and failure design](artifacts/day2-width-contract-failure-design.md)
- [Day 3 CMake selector](artifacts/day3-cmake-width-selection.md)
- [Day 4 Make selector and stamp](artifacts/day4-make-width-selection.md)
- [Day 5 build-system parity](artifacts/day5-build-system-parity.md)
- [Day 6 CMake installed width](artifacts/day6-cmake-installed-width.md)
- [Day 7 pkg-config and validator](artifacts/day7-pkg-config-width.md)
- [Day 8 pre-use archive check](artifacts/day8-executable-width-check.md)
- [Day 9 CMake consumer matrix](artifacts/day9-cmake-consumer-matrix.md)
- [Day 10 pkg-config mixed-prefix matrix](artifacts/day10-pkgconfig-mixed-prefix-matrix.md)
- [Day 11 selected 64-bit functional lane](artifacts/day11-selected-64-bit-lane.md)
- [Day 12 hosted registration and open gate](artifacts/day12-hosted-64-bit-registration.md)
- [Day 13 public/install contract](artifacts/day13-public-install-width-contract.md)
- [Day 14 quality and contract closeout](artifacts/day14-quality-and-contract-closeout.md)

## Residual Handoff

After publication, retain the exact commit SHA, workflow URLs, commands,
archive identities, failures, and rerun conclusions for the Linux/macOS
selected 64-bit jobs and installed-package jobs, plus the Windows
default-width route. Do not close items 219.5 or 219.6 until those required
jobs pass. A bounded or suitably resourced full 64-bit sanitizer run would
broaden confidence, but the current record makes no full-suite claim.
Keep separate build/install prefixes per width and the pre-use archive check
in maintained consumers; do not promote this static-package result into a
shared-library or universal ABI promise.
