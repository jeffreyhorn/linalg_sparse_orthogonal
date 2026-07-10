# Sprint 119 Day 8 Selection and Lifting Proof Audit

## Purpose

Day 8 audits the selection and lifting helper movement after the successful
Day 6 implementation and Day 7 validation. The planned Day 8 question was
whether `s20_select_indices` and `s20_lift_ritz_vectors` can move safely. The
branch now has the evidence to answer that question: both helpers moved
together into `src/sparse_eigs_selection_internal.c`, and focused consumer,
source-list, CMake, CTest, and full quality gates passed.

This artifact records the dependency map, affected behavior, invariants,
compile-unit proof, move-together decision, defer conditions, and Day 9
implementation checklist.

## Dependency Map

| Helper | Private declaration | Implementation owner | Direct consumers | Test evidence |
|---|---|---|---|---|
| `s20_select_indices` | `src/sparse_eigs_internal.h` | `src/sparse_eigs_selection_internal.c` | Grow-m backend in `src/sparse_eigs.c`; thick-restart backend in `src/sparse_eigs_thick_restart.c`; LOBPCG Rayleigh-Ritz step in `src/sparse_eigs_lobpcg.c`. | `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, direct selector tests in `test_ldlt_backend_dispatch`. |
| `s20_lift_ritz_vectors` | `src/sparse_eigs_internal.h` | `src/sparse_eigs_selection_internal.c` | Grow-m vector publication and partial publication in `src/sparse_eigs.c`; thick-restart locked-block and result-vector publication in `src/sparse_eigs_thick_restart.c`. | `test_eigs`, `test_eigs_thick_restart`; no direct LOBPCG lift use because LOBPCG owns its own vector publication path. |

## Consumer Behavior Matrix

| Consumer | `s20_select_indices` behavior | `s20_lift_ritz_vectors` behavior | Proof status |
|---|---|---|---|
| Grow-m Lanczos | Selects converged and partial Ritz values for largest, smallest, and nearest-sigma modes. | Publishes converged and partial Ritz vectors in column-major public result buffers. | Passed Day 7 `./build/test_eigs` and full `make test`. |
| Shift-invert through grow-m | Uses largest transformed `|theta|` ordering before original-space `sigma + 1 / theta` conversion. | Publishes original-space vectors because shift-invert shares eigenspaces with the original operator. | Passed Day 7 shift-invert coverage in `test_eigs`. |
| Repeated-handle grow-m | Reuses the same private helpers through prepared workspace paths. | Preserves vector layout across prepare/reuse/growth workflows. | Passed Day 7 repeated-handle coverage in `test_eigs`. |
| Thick-restart Lanczos | Selects locked/published Ritz pairs from the arrowhead basis. | Publishes locked restart vectors and final result vectors from `V * Y_arrow`. | Passed Day 7 `./build/test_eigs_thick_restart` and full `make test`. |
| LOBPCG | Selects Rayleigh-Ritz values from the block Gram matrix. | Not a direct consumer; LOBPCG has its own vector-publication path. | Passed Day 7 `./build/test_eigs_lobpcg` and full `make test`. |

## Public-Result Invariants

| Invariant | Required behavior | Evidence |
|---|---|---|
| Largest ordering | `sel_idx[0]` maps to the largest Ritz value in ascending `theta`. | Direct selector test plus grow-m and LOBPCG largest tests passed. |
| Smallest ordering | `sel_idx[0]` maps to the smallest Ritz value in ascending `theta`. | Direct selector test plus grow-m, thick-restart, and LOBPCG smallest tests passed. |
| Nearest-sigma ordering | Selection remains largest transformed `|theta|`, with existing right-end tie behavior preserved. | Direct selector tie test, shift-invert tests, thick-restart KKT nearest-sigma, and LOBPCG nearest-sigma tests passed. |
| Bounded partial publication | `take = min(k_want, m)` keeps partial result emission bounded and stable. | Grow-m partial-publication and m-cap exhaustion tests passed. |
| Column-major result vectors | Output column `j` remains `eigenvectors_out + j * n`. | Grow-m and thick-restart vector-publication boundary tests passed. |
| Selected projected vector column | Lift remains `V * Y[:, sel_idx[j]]`. | Ritz residual and vector orthogonality checks passed in focused eigensolver tests. |
| Public claim boundary | Internal movement does not widen eigensolver support statements. | No public docs, headers, package, ABI, benchmark, or support wording changed. |

## Compile-Unit And Build Proof

| Surface | Required proof | Observed proof |
|---|---|---|
| Makefile build | New source is in `LIB_SRCS` and focused binaries link. | Day 7 focused build passed. |
| Source-list metadata | New source appears in `build-metadata/library_sources.txt`. | Day 7 `make source-list-check` passed with 49 library sources. |
| CMake build | New source appears in CMake library sources and builds. | Day 7 CMake proof compiled `src/sparse_eigs_selection_internal.c`. |
| CTest registration | Test membership does not drift. | Day 7 `ctest -N` reported `Total Tests: 54`. |
| Full quality | Required because the branch contains `.c` movement. | Day 7 `make format && make lint && make test` passed. |

## Move-Together Decision

| Question | Decision |
|---|---|
| Can `s20_select_indices` move safely? | Yes; it already moved safely as part of the Day 6 paired movement. |
| Can `s20_lift_ritz_vectors` move safely? | Yes; it already moved safely as part of the Day 6 paired movement. |
| Should they move separately? | No. Their behavior contracts are adjacent enough that separate movement would create needless proof split and rollback ambiguity. |
| Should either helper defer? | No current deferral is needed for these two helpers. |
| Should Day 9 perform more code movement? | No by default. Day 9 should record the movement as already complete and consolidate evidence unless Day 8 review uncovers a corrective follow-up. |

## Explicit Defer Conditions

The paired movement would need to be reverted or deferred if any of these
conditions appeared in later validation:

- largest, smallest, or nearest-sigma ordering drift;
- partial-result count or publication drift;
- column-major vector publication drift;
- grow-m, thick-restart, or LOBPCG focused consumer regression;
- Makefile/CMake source membership mismatch;
- CTest registration drift without a reviewed count update;
- public API, ABI, docs, benchmark, or claim wording drift caused by the
  private movement.

No such condition is present after Day 7 validation.

## Day 9 Implementation Checklist

| Check | Day 9 action |
|---|---|
| Confirm helper bodies remain in `src/sparse_eigs_selection_internal.c`. | Required. |
| Confirm declarations remain private in `src/sparse_eigs_internal.h`. | Required. |
| Confirm no public headers or public docs changed for this movement. | Required. |
| Confirm Makefile, CMake, and source-list metadata still include the new source. | Required. |
| Avoid duplicate movement. | Required; helpers have already moved. |
| Record explicit no-op/evidence consolidation unless corrective work is found. | Required. |
| If corrective code work is needed, rerun focused eigensolver, source-list, CMake/CTest, and full C quality gates. | Conditional. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Selection/lifting dependency map exists. | Complete. |
| Grow-m/thick-restart/LOBPCG proof matrix exists. | Complete. |
| Move-together or split decision is documented. | Complete: move together, already complete. |
| Explicit defer conditions are documented. | Complete. |
| Day 9 implementation checklist exists. | Complete. |
| Public-result invariants are documented. | Complete. |
