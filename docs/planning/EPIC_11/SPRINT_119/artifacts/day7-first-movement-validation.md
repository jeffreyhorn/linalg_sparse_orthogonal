# Sprint 119 Day 7 First Movement Batch Focused Validation

## Purpose

Day 7 validates the Day 6 private eigensolver movement batch against the
focused consumer proof defined on Day 5. The validation confirms that moving
`s20_select_indices` and `s20_lift_ritz_vectors` into
`src/sparse_eigs_selection_internal.c` preserved eigensolver behavior, build
membership, CMake registration, and public claim boundaries.

## Validation Scope

| Surface | Day 7 scope |
|---|---|
| Focused eigensolver behavior | Grow-m, shift-invert, repeated-handle, thick-restart, and LOBPCG-adjacent consumers. |
| Build membership | Makefile library source list and source-list metadata. |
| CMake membership | CMake configure/build with `src/sparse_eigs_selection_internal.c` included. |
| CTest registration | Reviewed POSIX CMake test count remains `54`. |
| Full C quality chain | Required because the branch contains Day 6 `.c` and build metadata changes. |
| Public claims | No public API, docs, package, ABI, benchmark, or support-claim changes. |

## Focused Consumer Proof Results

| Command | Result |
|---|---|
| `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg` | Pass; all three focused binaries were already up to date. |
| `./build/test_eigs` | Pass: 43 tests, 0 failed, 955 assertions. |
| `./build/test_eigs_thick_restart` | Pass: 23 tests, 0 failed, 384 assertions. |
| `./build/test_eigs_lobpcg` | Pass: 29 tests, 0 failed, 287 assertions. |

## Source-List And CMake Evidence

| Check | Result |
|---|---|
| `make source-list-check` | Pass: 49 library sources. |
| `cmake -S . -B build-cmake-review && cmake --build build-cmake-review && ctest --test-dir build-cmake-review -N` | Pass; CMake compiled `src/sparse_eigs_selection_internal.c` and `ctest -N` reported `Total Tests: 54`. |

The temporary `build-cmake-review` directory was removed after the CMake proof.

## Full Quality Gate

| Command | Result |
|---|---|
| `make format && make lint && make test` | Pass. |

The full chain formatted the new source, ran strict compile warnings,
clang-tidy, cppcheck, and the full Makefile test suite. The final test output
ended with `All tests passed.`

## Behavior Invariant Confirmation

| Invariant | Evidence |
|---|---|
| Largest/smallest ordering remains stable. | `test_eigs`, `test_eigs_lobpcg`, and `test_ldlt_backend_dispatch` selection tests passed. |
| Nearest-sigma selection remains stable. | Shift-invert, thick-restart KKT nearest-sigma, and LOBPCG nearest-sigma tests passed. |
| Column-major vector publication remains stable. | Grow-m, shift-invert, thick-restart, and LOBPCG vector-publication boundary tests passed. |
| Repeated-handle consumers remain stable. | `test_public_handle_*_prepare_reuse_and_growth` tests passed in `test_eigs`. |
| CMake/CTest membership remains stable. | CMake build passed and `ctest -N` stayed at `54`. |
| Public claim boundary remains stable. | No public docs, headers, package, ABI, benchmark, or support wording changed. |

## Updated Movement Evidence

Day 7 upgrades the Day 6 implementation evidence from immediate proof to
complete focused validation:

- the moved helpers compile and link through Makefile and CMake consumers;
- focused grow-m, shift-invert, thick-restart, LOBPCG, and repeated-handle
  behavior remains unchanged;
- source-list metadata and CMake library membership agree;
- CTest registration did not drift;
- the required full C quality chain passed.

## Residuals

| Residual | Status after Day 7 |
|---|---|
| Selection/lifting post-movement proof audit | Still scheduled for Day 8 to document dependency and invariant proof after the successful movement. |
| Day 9 selection/lifting movement slot | Should become an explicit no-op or evidence consolidation unless Day 8 finds a corrective follow-up, because the helpers already moved safely on Day 6. |
| Shift-invert setup/conversion | Still deferred to Day 11 boundary decision. |
| `lanczos_iterate_op` | Still deferred pending recurrence-specific proof. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Focused consumer proof results exist. | Complete. |
| Source-list and CMake parity evidence exists. | Complete. |
| CTest count evidence exists. | Complete: `Total Tests: 54`. |
| Required quality-check output summary exists. | Complete. |
| Movement evidence is updated with observed results. | Complete. |
| Failures are fixed or treated as blockers. | Complete; no Day 7 failures remained. |
