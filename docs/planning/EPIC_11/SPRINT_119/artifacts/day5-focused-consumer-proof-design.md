# Sprint 119 Day 5 Focused Consumer Proof Design

## Purpose

Day 5 defines the proof package required before moving the first Sprint 119
source-boundary batch. The planned movement remains the paired private helper
extraction from `src/sparse_eigs.c` into
`src/sparse_eigs_selection_internal.c`:

- `s20_select_indices`
- `s20_lift_ritz_vectors`

This artifact records focused tests, behavior invariants, expected CTest
membership, validation commands, and a filled source-movement evidence draft.
It does not move code.

## Focused Consumer Test List

| Consumer family | Focused test binary | Proof coverage |
|---|---|---|
| Grow-m value selection | `build/test_eigs` | Largest/smallest public results, grow-m retry, and selection through public `sparse_eigs_sym`. |
| Grow-m vector publication | `build/test_eigs` | `test_s114_growm_vector_lift_public_boundary` and partial vector publication boundary. |
| Shift-invert selection and lift | `build/test_eigs` | `test_shift_invert_*`, `test_s114_shift_invert_vector_publication_boundary`, and `test_s114_shift_invert_growm_conversion_nearest_sigma`. |
| Repeated-handle grow-m surface | `build/test_eigs` | Public handle prepare/reuse/growth and on-demand workspace behavior remains unchanged. |
| Thick-restart selection and lift | `build/test_eigs_thick_restart` | Thick-restart parity, KKT nearest-sigma parity, and `test_s114_thick_restart_vector_publication_boundary`. |
| LOBPCG selection adjacency | `build/test_eigs_lobpcg` | LOBPCG RR selection, nearest-sigma tests, Lanczos parity, AUTO dispatch, and adjacent public-result parity. |

## Behavior Invariant Table

| Invariant | Protected behavior | Consumer proof |
|---|---|---|
| Selection helper keeps `LARGEST` order | `sel_idx[0]` maps to the largest Ritz value in an ascending `theta` array. | Grow-m and LOBPCG largest tests in `test_eigs` and `test_eigs_lobpcg`. |
| Selection helper keeps `SMALLEST` order | `sel_idx[0]` maps to the smallest Ritz value in an ascending `theta` array. | Grow-m, thick-restart, and LOBPCG smallest tests. |
| `NEAREST_SIGMA` still selects largest transformed magnitude | Two-pointer scan over ascending transformed `theta` still feeds original-space nearest-sigma conversion. | `test_shift_invert_*`, `test_thick_restart_kkt_nearest_sigma_parity`, `test_lobpcg_nearest_sigma_*`. |
| Selection remains bounded for `take < m` | `take = min(k_want, m)` and no underflow at the center of the two-pointer scan. | Public partial-result and nearest-sigma tests. |
| Lift preserves column-major result layout | Output column `j` remains `eigenvectors_out + j * n`. | Grow-m and thick-restart vector-publication tests. |
| Lift uses selected projected vector column | Full-space Ritz vector remains `V * Y[:, sel_idx[j]]`. | Grow-m, shift-invert, and thick-restart residual/orthogonality checks. |
| Shift-invert vector publication remains original-space | Eigenvectors are lifted unchanged because shift-invert shares eigenspaces with the original operator. | `test_s114_shift_invert_vector_publication_boundary` and nearest-sigma tests. |
| LOBPCG selection remains backend-adjacent only | LOBPCG continues to consume `s20_select_indices`; vector publication remains owned by LOBPCG. | LOBPCG nearest-sigma, dispatch, and public-result parity tests. |
| Public headers remain stable | No caller-visible API, ABI, options, or result fields change. | Compile/link proof plus public header diff review. |
| CTest membership remains stable | Movement creates no new test binary and removes none. | `ctest -N` count remains `54` on the reviewed POSIX CMake parity path. |

## Expected CTest Count

| Surface | Expected count | Source |
|---|---:|---|
| Reviewed POSIX CMake parity path | 54 | Sprint 118 Day 3 baseline and Day 14 closeout. |
| Windows reviewed CMake subset | No Day 5 change | Movement design does not change CTest membership; any implementation day should use platform-specific expected count already owned by CI if touched. |

The first movement batch should not add or remove test executables. Any CTest
count change is a blocker until explained.

## Focused Rerun Commands For Implementation Day

These commands are expected after Day 6 code/build metadata changes, assuming
the usual Makefile build path is used:

| Command | Required because | Classification |
|---|---|---|
| `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg` | Proves moved helper bodies compile and link through grow-m, thick-restart, and LOBPCG consumers. | focused local/review-supporting |
| `./build/test_eigs` | Proves grow-m, shift-invert, repeated-handle, and vector-publication behavior. | focused local/review-supporting |
| `./build/test_eigs_thick_restart` | Proves thick-restart selection, lift, parity, and nearest-sigma behavior. | focused local/review-supporting |
| `./build/test_eigs_lobpcg` | Proves LOBPCG selection adjacency, nearest-sigma, and backend dispatch parity. | focused local/review-supporting |
| `make source-list-check` or equivalent reviewed wrapper | Required because a new source file is expected in Makefile/CMake source membership. | reviewed if available |
| `cmake -S . -B build-cmake-review && cmake --build build-cmake-review && ctest --test-dir build-cmake-review -N` | Required if implementation touches CMake source membership; proves CMake build and CTest registration. | reviewed/supplemental depending on local lane |
| `make format && make lint && make test` | Required because Day 6 is expected to modify `.c` files and build metadata. | reviewed |

If the implementation touches only docs unexpectedly, fall back to docs hygiene
only. If it touches `.c` or `.h`, the full C quality chain is required.

## Source-Movement Evidence Draft

### Scope

| Field | Value |
|---|---|
| Sprint/day | Sprint 119 Day 5 draft for Day 6 implementation |
| Artifact owner | Sprint 119 |
| Work type | private-owner extraction / source movement |
| Touched surfaces | Expected: `src/sparse_eigs.c`, new `src/sparse_eigs_selection_internal.c`, `Makefile`, `CMakeLists.txt`; no tests or public headers expected. |
| Explicitly out of scope | Public eigensolver API, shift-invert lifecycle, Lanczos recurrence, LOBPCG vector publication, workspace allocation, docs/claims, package/ABI, benchmark/performance. |

### Baseline

| Baseline item | Current value |
|---|---|
| Starting files | `src/sparse_eigs.c`, `src/sparse_eigs_internal.h`, `src/sparse_eigs_thick_restart.c`, `src/sparse_eigs_lobpcg.c` |
| Starting line counts | `src/sparse_eigs.c`: 1412; `src/sparse_eigs_internal.h`: 631; `src/sparse_eigs_thick_restart.c`: 915; `src/sparse_eigs_lobpcg.c`: 401 |
| Starting CTest count | 54 on Sprint 118 reviewed POSIX CMake parity baseline |
| Current product truth references | Sprint 118 Day 8 product truth map and Sprint 118 Day 14 closeout |
| Current non-claims | No ARPACK, SciPy, LAPACK, broad nonsymmetric eigensolver, state-of-the-art, or portable performance claim |

### Proof Values

| Proof value | Protected behavior or invariant | Evidence before change |
|---|---|---|
| Selection order | Largest, smallest, and nearest-sigma ordering remain stable. | Day 2 inventory, Day 3 ranking, focused tests listed above. |
| Vector publication | Lifted Ritz vectors preserve column-major output and selected projected columns. | Day 4 design and focused vector-publication tests. |
| Build membership | Makefile and CMake both include the new source file. | Day 4 source-list plan; Day 6 validation commands. |
| CTest registration | Test executable membership remains unchanged. | Day 5 expected CTest count and Day 6 `ctest -N`. |
| Public claim boundary | Internal movement does not create new public support claims. | Sprint 118 non-claims and Day 4 public impact note. |

### Behavior Boundary

- Boundary being moved: `s20_select_indices` and `s20_lift_ritz_vectors`
  implementation bodies and their attached behavior comments.
- Boundary not being moved: `lanczos_iterate_op`, `lanczos_iterate`,
  `s20_op_shift_invert`, shift-invert setup/cleanup/conversion, grow-m retry,
  thick-restart state management, LOBPCG RR implementation, refinement,
  workspace allocation, public validation, public docs.
- Consumer paths affected: grow-m backend, thick-restart backend, LOBPCG
  selection path, shift-invert publication path, repeated-handle public tests
  through unchanged backend calls.
- Unsupported or expected-failure behavior that must remain visible:
  singular shift-invert, invalid options, nonconvergence/partial publication,
  backend AUTO dispatch boundaries, and current non-claims.

### Old/New File Plan

| Current file | Proposed file | Ownership after change | Notes |
|---|---|---|---|
| `src/sparse_eigs.c` | `src/sparse_eigs_selection_internal.c` | Selection ordering and vector-publication helper implementation. | Move bodies without changing signatures or semantics. |
| `src/sparse_eigs_internal.h` | unchanged | Private declarations remain shared. | No public header change. |

### Internal Header And Private API Contract

| Contract item | Decision |
|---|---|
| Internal headers added or changed | No new header planned; `src/sparse_eigs_internal.h` remains declaration owner. |
| Private functions moved | `s20_select_indices`, `s20_lift_ritz_vectors`. |
| Public API impact | None. |
| ABI/package impact | None. |

### Source-List, Makefile, And CMake Impact

| Surface | Expected update | Validation |
|---|---|---|
| Makefile/source list | Add `$(SRCDIR)/sparse_eigs_selection_internal.c` near eigensolver sources. | `make source-list-check` or reviewed equivalent, plus full `make format && make lint && make test`. |
| CMake | Add `src/sparse_eigs_selection_internal.c` near eigensolver sources. | CMake configure/build and `ctest -N`. |
| CTest membership | Same, expected 54 on POSIX reviewed parity baseline. | `ctest -N` count proof. |

### Change Plan For Day 6

1. Create `src/sparse_eigs_selection_internal.c`.
2. Move the comments and bodies for `s20_select_indices` and
   `s20_lift_ritz_vectors` out of `src/sparse_eigs.c`.
3. Include `sparse_eigs_internal.h` and required standard headers in the new
   source file.
4. Keep declarations unchanged in `src/sparse_eigs_internal.h`.
5. Add the new source to `Makefile` `LIB_SRCS`.
6. Add the new source to `CMakeLists.txt` library sources.
7. Build and run focused eigensolver tests.
8. Run required source-list/CMake/CTest/full C quality checks.
9. Record observed results in the Day 6/Day 7 implementation evidence.

### Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | none | No update. |
| INSTALL | none | No update. |
| Solver/docs/examples | none | No update. |
| Benchmark/performance wording | none | No update. |

### Rollback Or Defer Plan

- Rollback path: restore both helper bodies to `src/sparse_eigs.c`, delete
  `src/sparse_eigs_selection_internal.c`, remove the new source from
  `Makefile` and CMake, rerun focused tests and required quality.
- Defer condition: any compile/link, CTest count, public-result, vector-layout,
  LOBPCG selection, or shift-invert parity failure that is not trivially an
  implementation typo.
- Partial-move handling: not allowed. Both helpers move together or both stay
  in `src/sparse_eigs.c`.

### Non-Claims Preserved

- No ARPACK parity claim.
- No SciPy/LAPACK eigensolver parity claim.
- No broad nonsymmetric eigensolver support claim.
- No state-of-the-art eigensolver replacement claim.
- No portable performance claim from this movement.

## Day 6 Implementation Checklist

| Check | Required before Day 6 code edit |
|---|---|
| Behavior boundary is explicit. | Complete. |
| Focused consumer tests are named. | Complete. |
| Expected CTest count is recorded. | Complete. |
| Makefile and CMake impact is recorded. | Complete. |
| Full C quality trigger is recorded. | Complete. |
| Rollback/defer plan is recorded. | Complete. |
| Public API and non-claim boundaries are recorded. | Complete. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Focused consumer test list exists. | Complete. |
| Behavior invariant table exists. | Complete. |
| Expected CTest count and focused rerun commands are defined. | Complete. |
| Source-movement evidence draft is filled for the planned movement. | Complete. |
| Day 6 implementation checklist exists. | Complete. |
| Movement has focused proof before code changes. | Complete. |
| Required quality gates are known. | Complete. |
| No consumer path depends on implicit or undocumented behavior. | Complete. |
