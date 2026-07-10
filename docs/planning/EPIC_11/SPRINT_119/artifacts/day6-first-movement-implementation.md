# Sprint 119 Day 6 First Movement Batch Implementation

## Purpose

Day 6 implements the first movement batch selected on Day 3 and designed on
Days 4-5: move the lowest-risk private eigensolver selection and lifting
helpers out of `src/sparse_eigs.c` into a focused private source owner.

The movement preserves public API, public headers, public documentation,
package/ABI surfaces, benchmark surfaces, and eigensolver support claims.

## Implementation Scope

| Field | Value |
|---|---|
| Movement type | Private source-boundary extraction. |
| Functions moved | `s20_select_indices`, `s20_lift_ritz_vectors`. |
| New source owner | `src/sparse_eigs_selection_internal.c`. |
| Original source owner | `src/sparse_eigs.c`. |
| Private declaration owner | Unchanged: `src/sparse_eigs_internal.h`. |
| Build metadata touched | `Makefile`, `CMakeLists.txt`, `build-metadata/library_sources.txt`. |
| Public API impact | None. |
| Public claim impact | None. |
| Test membership impact | None expected; POSIX CMake `ctest -N` remained at `54`. |

## Code Changes

| File | Change |
|---|---|
| `src/sparse_eigs_selection_internal.c` | Added as the private implementation owner for selection ordering and Ritz-vector lifting helpers. |
| `src/sparse_eigs.c` | Removed the moved helper bodies; existing callers continue through unchanged private declarations. |
| `Makefile` | Added `$(SRCDIR)/sparse_eigs_selection_internal.c` to library sources. |
| `CMakeLists.txt` | Added `src/sparse_eigs_selection_internal.c` to CMake library sources. |
| `build-metadata/library_sources.txt` | Added the new source so source-list verification matches the build metadata. |

## Behavior Boundary Preserved

| Behavior | Day 6 result |
|---|---|
| Largest/smallest ordering | Preserved by moving the existing implementation without semantic changes. |
| Nearest-sigma transformed ordering | Preserved by moving the existing two-pointer selection logic unchanged. |
| Column-major vector publication | Preserved by moving the existing lift loop unchanged. |
| Shift-invert vector publication | Preserved; shift-invert continues to share the unchanged lifting helper behavior. |
| Thick-restart selection/lifting | Preserved through unchanged private function signatures and focused test coverage. |
| LOBPCG selection adjacency | Preserved; LOBPCG continues to consume `s20_select_indices` through the private header. |
| Public eigensolver claims | Unchanged; no ARPACK, SciPy, LAPACK, broad nonsymmetric, state-of-the-art, or portable performance claim was added. |

## Validation Results

| Command | Result |
|---|---|
| `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg` | Pass. |
| `./build/test_eigs` | Pass: 43 tests, 0 failed, 955 assertions. |
| `./build/test_eigs_thick_restart` | Pass: 23 tests, 0 failed, 384 assertions. |
| `./build/test_eigs_lobpcg` | Pass: 29 tests, 0 failed, 287 assertions. |
| `make source-list-check` | Initially failed because the new source was missing from `build-metadata/library_sources.txt`; after updating the manifest, rerun passed with 49 library sources. |
| `cmake -S . -B build-cmake-review && cmake --build build-cmake-review && ctest --test-dir build-cmake-review -N` | Pass; CMake built the new source and `ctest -N` reported `Total Tests: 54`. |
| `make format && make lint && make test` | Pass; all tests passed. |

## Build-System Evidence

The Day 6 movement touched both Makefile and CMake source membership. The
local source-list check caught the missing metadata entry before finalizing the
change, and the follow-up run passed after adding
`src/sparse_eigs_selection_internal.c` to
`build-metadata/library_sources.txt`.

The CMake proof built the library with the new source file and confirmed that
the reviewed POSIX CTest registration count stayed at `54`, matching the Day 5
expected count.

## Residual Movement List

| Residual | Status after Day 6 | Next owner |
|---|---|---|
| `lanczos_iterate_op` movement | Deferred; still requires recurrence-specific proof before movement. | Later Sprint 119 residual or future sprint handoff. |
| Shift-invert setup/conversion movement | Deferred to the Day 11 boundary decision. | Sprint 119 Day 11-12. |
| Selection/lifting proof audit | Still needed as a post-movement audit despite successful focused validation. | Sprint 119 Day 8-10. |
| Broad eigensolver private-owner bucket | Deferred; not part of the first safe movement batch. | Sprint 119 closeout residuals unless separately proven. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| First movement batch code change is complete. | Complete. |
| Build metadata matches moved files. | Complete. |
| Immediate focused compile and source-list checks were run. | Complete. |
| Public API and claim boundaries are unchanged. | Complete. |
| Required full C quality chain passed after `.c` changes. | Complete. |
| Residual movement list is recorded. | Complete. |
