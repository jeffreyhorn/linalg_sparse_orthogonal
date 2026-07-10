# Sprint 119 Day 10 Selection and Lifting Validation

## Purpose

Day 10 performs the final focused validation refresh for the
`s20_select_indices` and `s20_lift_ritz_vectors` private-source movement
before Sprint 119 shifts to the shift-invert boundary decision. The helpers
already moved together on Day 6, were validated on Day 7, audited on Day 8,
and consolidated on Day 9. Day 10 confirms that the movement remains clean
against focused eigensolver consumers, source-list metadata, CMake/CTest
registration, and the full C quality gate.

## Focused Validation Results

| Command | Result |
|---|---|
| `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg` | Pass; all focused eigensolver binaries were up to date. |
| `./build/test_eigs` | Pass: 43 tests, 0 failed, 955 assertions. |
| `./build/test_eigs_thick_restart` | Pass: 23 tests, 0 failed, 384 assertions. |
| `./build/test_eigs_lobpcg` | Pass: 29 tests, 0 failed, 287 assertions. |

## Source-List, CMake, And CTest Evidence

| Command | Result |
|---|---|
| `make source-list-check` | Pass: 49 library sources. |
| `cmake -S . -B build-cmake-review && cmake --build build-cmake-review && ctest --test-dir build-cmake-review -N` | Pass; CMake compiled `src/sparse_eigs_selection_internal.c` and `ctest -N` reported `Total Tests: 54`. |

The temporary `build-cmake-review` directory was removed before the full
quality chain.

## Required Full Quality Gate

| Command | Result |
|---|---|
| `make format && make lint && make test` | Pass; final output ended with `All tests passed.` |

The quality chain reran clang-format, strict compile warnings, clang-tidy,
cppcheck, and the full Makefile test suite. The moved
`src/sparse_eigs_selection_internal.c` source passed clang-format,
clang-tidy, cppcheck, focused eigensolver tests, and full test coverage.

## Consumer Proof Matrix

| Consumer | Protected behavior | Day 10 evidence |
|---|---|---|
| Grow-m Lanczos | Largest/smallest/nearest-sigma selection, vector lift, partial vector publication, repeated-handle behavior. | `./build/test_eigs` passed. |
| Shift-invert through grow-m | Largest transformed `|theta|` selection, original-space eigenvalue conversion, and original-space vector publication. | `./build/test_eigs` passed shift-invert coverage. |
| Thick-restart Lanczos | Arrowhead selection, locked-vector lift, final result-vector lift, nearest-sigma parity. | `./build/test_eigs_thick_restart` passed. |
| LOBPCG adjacency | Rayleigh-Ritz selection and adjacent public-result parity; LOBPCG keeps its own vector publication path. | `./build/test_eigs_lobpcg` passed. |
| Direct selector contract | Repeated largest/smallest and nearest-sigma tie behavior. | Full `make test` reran `test_ldlt_backend_dispatch` successfully. |

## Public Claim Boundary

Day 10 made no public API, public header, package, ABI, workflow, benchmark,
documentation, or support-claim changes. The movement remains an internal
source-boundary improvement only.

No broad eigensolver parity claim was introduced. The existing non-claims
remain intact:

- no ARPACK parity claim;
- no SciPy or LAPACK eigensolver parity claim;
- no broad nonsymmetric eigensolver support claim;
- no state-of-the-art eigensolver replacement claim;
- no portable performance claim from this private movement.

## Shift-Invert Readiness

Selection/lifting Item 4 is complete after Day 10:

- both helpers moved together;
- no selection/lifting residual movement remains;
- source-list, Makefile, CMake, and CTest membership are stable;
- focused eigensolver consumers passed;
- full C quality gate passed.

Sprint 119 can proceed to Day 11 shift-invert boundary analysis with no
outstanding blocker from the selection/lifting movement.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Grow-m proof results exist. | Complete. |
| Thick-restart proof results exist. | Complete. |
| LOBPCG-adjacent proof results exist. | Complete. |
| CTest count and CMake evidence exists. | Complete: `Total Tests: 54`. |
| Updated source-movement evidence exists. | Complete. |
| Item 4 validation is complete. | Complete. |
| Required checks passed before shift-invert work begins. | Complete. |
| No broad eigensolver parity claim was introduced. | Complete. |
