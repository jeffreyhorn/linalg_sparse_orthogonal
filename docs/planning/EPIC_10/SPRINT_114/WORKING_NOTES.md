# Sprint 114 Working Notes

## Sprint Goal

Sprint 114 converts Sprint 113's residual proof-owner debt into
dependency-ordered follow-through batches. The sprint proves eigensolver
movement prerequisites before any source split, cleans up bounded
direct/iterative exact-RHS setup, and performs bounded SVD proof-owner cleanup
without introducing broad abstractions or unsupported public-support claims.

## Starting Constraints

- Do not repeat Sprint 113 grow-m behavior selection, proof design, or proof
  implementation.
- Do not repeat Sprint 113 eigensolver movement/no-move decision or no-move
  contract publication.
- Do not repeat Sprint 113 LDLT CSC external dense-reference oracle cleanup.
- Do not repeat Sprint 113 partial-SVD vector residual cleanup.
- Do not infer public API, install-header, helper-target, source-list, Make,
  CMake, reviewed CTest, package/platform, or ABI support changes from proof
  cleanup.
- Keep matrices, tolerances, expected values, residuals, dimensions, and
  solver-specific proof values visible unless a boundary artifact explicitly
  proves a helper can hide setup without hiding behavior.
- If `.c`, `.h`, build-system, source-list, or test registration files change,
  run `make format && make lint && make test` before proceeding.

## Completed Work Excluded From Sprint 114 Scope

| Completed work | Source evidence | Sprint 114 handling |
|---|---|---|
| Sprint 113 residual intake and duplicate-work fence | Sprint 113 Day 1 artifact and retrospective | Use as evidence; do not repeat as unresolved debt. |
| Grow-m behavior owner selection and proof design | Sprint 113 Days 2-3 artifacts | Use as prior proof pattern only. |
| Grow-m behavior test batch | Sprint 113 Day 4 artifact and `tests/test_eigs.c` | Exclude; Sprint 114 adds different eigensolver proofs. |
| Eigensolver movement/no-move decision | Sprint 113 Day 5 artifact | Revisit only after new Sprint 114 prerequisites land. |
| Eigensolver no-move contract publication | Sprint 113 Day 6 artifact | Preserve unless Day 10 proves one narrow movement is safe. |
| LDLT CSC external dense-reference oracle cleanup | Sprint 113 Day 8 artifact and `tests/test_ldlt_csc.c` | Exclude; use as direct/iterative cleanup example. |
| SVD partial-vector residual cleanup | Sprint 113 Day 10 artifact and `tests/test_svd_partial_helpers.h` | Exclude; use as SVD helper strictness example. |
| Proof-owner metrics and non-claims artifact | Sprint 113 Day 11 artifact | Carry forward metrics boundaries. |
| Integrated validation plan and execution | Sprint 113 Days 12-13 artifacts | Use as validation pattern. |
| Sprint 113 closeout and handoff | Sprint 113 Day 14 artifact and retrospective | Use as residual source; do not repeat completed work. |

## Residual Eigensolver Proof Owners

| Residual owner | Primary area | Proof needed before movement | Planned day |
|---|---|---|---:|
| `lanczos_iterate_op` behavior across dispatch paths | `src/sparse_eigs.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Basic, thick-restart, and LOBPCG-adjacent observable behavior. | 2-3 |
| Repeated/clustered Ritz selection | Ritz value selection and eigensolver tests | Repeated and clustered spectrum fixtures with visible expected ordering and tolerances. | 4-5 |
| Ritz vector lifting and publication boundary | Ritz vector publication in public result structs | Residual/normalization checks plus requested/converged result-shape assertions. | 6-7 |
| Partial-result publication after `m_cap` exhaustion | bounded Lanczos/grow-m exits | Converged count, result shape, iteration, and non-overrun proof. | 8 |
| Shift-invert grow-m conversion | nearest-sigma / shift-invert path | Public-result proof for conversion, backend reporting, residuals, and basis behavior. | 9 |
| Eigensolver source movement decision | source ownership and build metadata | One safe owner must be proven before any movement; otherwise continue no-move contract. | 10 |

## Residual Direct/Iterative Proof Owners

| Residual owner | Primary file | Proof values at risk | Planned day |
|---|---|---|---:|
| QR sequential RHS setup | `tests/test_qr.c` | least-squares residuals, refinement before/after residuals, literal RHS values. | 11-12 |
| CG preconditioner-specific exact-RHS setup | `tests/test_iterative.c` | preconditioner setup, residual norms, iteration comparisons. | 11-12 |
| GMRES exact-RHS setup | `tests/test_iterative.c` | restart settings, convergence status, residual norms, lucky-breakdown behavior. | 11-12 |
| BiCGSTAB exact-RHS setup | `tests/test_iterative.c` | breakdown behavior, convergence status, residual norms. | 11-12 |
| MINRES exact-RHS setup | `tests/test_iterative.c` | symmetry assumptions, preconditioner behavior, residual norms. | 11-12 |
| Broad direct/iterative oracle abstraction | multiple solver tests | Blocked until more solver-specific cleanup lanes prove common ownership. | Deferred |

## Residual SVD Proof Owners

| Residual owner | Primary file | Proof values at risk | Planned day |
|---|---|---|---:|
| Reconstruction helper movement by storage contract | `tests/test_svd.c` | reconstruction residuals, shape, and storage layout. | 13 |
| U/Vt orthogonality helper movement by leading-dimension convention | `tests/test_svd.c` | economy/full leading dimensions and dot-product thresholds. | 13 |
| Moore-Penrose product helper extraction | `tests/test_svd.c` | product dimensions and Moore-Penrose identities. | 13 |
| Dense low-rank proof-loop cleanup | `tests/test_svd.c` | retained singular-value error bounds and Frobenius residuals. | 13 |
| Sparse low-rank proof-loop cleanup | `tests/test_svd.c` | dense-vs-sparse residuals, drop tolerance, corpus fixture names. | 13 |
| Condition-number proof logic cleanup | `tests/test_svd.c` | finite/infinite condition values and rectangular interpretation. | 13 |
| Broad SVD proof abstraction | SVD helper layer | Blocked until storage, leading-dimension, product, low-rank, and condition proof owners all agree. | Deferred |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Residual intake, duplicate fence, implementation order, working notes baseline. | Item 1 |
| 2 | `lanczos_iterate_op` behavior proof design. | Item 2 |
| 3 | `lanczos_iterate_op` behavior proof implementation. | Item 2 |
| 4 | Repeated/clustered Ritz selection proof design. | Item 3 |
| 5 | Repeated/clustered Ritz selection proof implementation. | Item 3 |
| 6 | Ritz vector lifting and publication-boundary proof design. | Item 4 |
| 7 | Ritz vector lifting and publication-boundary proof implementation. | Item 4 |
| 8 | Partial-result publication after `m_cap` exhaustion. | Item 5 |
| 9 | Shift-invert grow-m conversion proof. | Item 6 |
| 10 | Eigensolver source movement decision or one narrow movement. | Item 7 |
| 11 | Direct/iterative exact-RHS cleanup design. | Item 8 |
| 12 | Direct/iterative exact-RHS cleanup implementation. | Item 8 |
| 13 | SVD proof-owner cleanup batch. | Item 9 |
| 14 | Validation, metrics, non-claim handoff, and residual queue. | Item 10 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs; local relative Markdown link check when links change. |
| Test `.c` or `.h` files | Focused affected test binary build/run; `make format && make lint && make test`; `git diff --check`. |
| Source `.c` or private `.h` files | Focused affected tests; source-list/build checks as applicable; `make format && make lint && make test`. |
| Public headers or install headers | Public/install-header drift review; `make format && make lint && make test`; install/package checks if package semantics change. |
| Make/CMake/source-list metadata | `make source-list-check`; focused Make/CMake build or CTest checks; full quality gate if code changed. |
| CTest registration | `ctest -N` through the relevant CMake build path; document expected count changes explicitly. |

## Day 1 Notes

- Created Sprint 114 working notes and artifact directory.
- Re-read the Sprint 114 section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 113 retrospective residual deferred debt and closeout notes.
- Explicitly excluded completed Sprint 113 work:
  - residual intake and duplicate-work fence;
  - grow-m behavior owner selection, proof design, and test batch;
  - eigensolver movement/no-move decision and no-move contract;
  - LDLT CSC external dense-reference oracle cleanup;
  - SVD partial-vector residual cleanup;
  - proof-owner metrics and non-claims artifact;
  - integrated validation plan and execution;
  - Sprint 113 closeout and handoff.
- Built dependency order:
  - eigensolver proofs for Lanczos behavior, Ritz selection, vector lifting,
    partial-result publication, and shift-invert grow-m conversion;
  - eigensolver movement/no-move decision after proof prerequisites;
  - direct/iterative exact-RHS cleanup after eigensolver movement decision;
  - SVD proof-owner cleanup after direct/iterative cleanup;
  - final validation, metrics, and non-claim handoff.
- Added Day 1 artifact:
  - `artifacts/day1-residual-intake-and-boundary.md`.

## Day 2 Notes

- Designed the `lanczos_iterate_op` behavior proof across the three required
  surfaces:
  - basic grow-m Lanczos through `s46_run_growm_backend`;
  - thick-restart empty-state and backend parity through
    `tests/test_eigs_thick_restart.c`;
  - LOBPCG-adjacent parity through `tests/test_eigs_lobpcg.c`, without
    claiming LOBPCG directly calls `lanczos_iterate_op`.
- Identified existing adjacent tests to preserve:
  - `test_growm_explicit_capacity_pins_peak_basis_size`;
  - `test_growm_retry_progress_steps_accumulate_iterations`;
  - `test_thick_restart_iterate_empty_state_matches_lanczos`;
  - `test_thick_restart_single_phase_matches_grow_m`;
  - `test_lobpcg_vs_lanczos_laplacian`.
- Chose deterministic fixtures for Day 3:
  - diagonal SPD `diag(1..6)` for direct recurrence parity;
  - shifted tridiagonal SPD with `n = 64`, `k = 2`, and explicit
    `max_iterations = 64` for grow-m public behavior;
  - Laplacian tridiagonal with `n = 30`, `k = 4`, and `max_iterations = 200`
    for LOBPCG-adjacent parity.
- Set Day 3 proof visibility rules:
  - keep matrix values, tolerances, iteration budgets, expected observables,
    and backend selections visible at test call sites;
  - avoid public API, install-header, source-list, helper-target, and reviewed
    CTest membership drift;
  - avoid broad cross-file helper extraction.
- Added Day 2 artifact:
  - `artifacts/day2-lanczos-iterate-behavior-design.md`.

## Day 3 Notes

- Implemented the Day 2 `lanczos_iterate_op` behavior proof targets:
  - added a grow-m public behavior proof in `tests/test_eigs.c` that forces
    `SPARSE_EIGS_BACKEND_LANCZOS`, checks convergence/public result fields,
    verifies returned Ritz residuals on the shifted tridiagonal fixture, and
    treats `max_iterations` as a per-run basis cap rather than a cumulative
    iteration cap;
  - added a thick-restart empty-state recurrence parity proof in
    `tests/test_eigs_thick_restart.c` on a non-diagonal tridiagonal fixture,
    comparing `m_actual`, `V`, `alpha`, and `beta` against
    `lanczos_iterate`;
  - added a LOBPCG-adjacent public parity proof in `tests/test_eigs_lobpcg.c`
    that compares LOBPCG and grow-m Lanczos eigenvalues, backend identities,
    convergence counts, iteration visibility, and Ritz residuals.
- Kept all proof values visible at call sites:
  - matrix dimensions;
  - `k`;
  - backend selections;
  - tolerances;
  - iteration budgets;
  - expected public result assertions.
- Preserved the Day 2 non-claim boundaries:
  - no public API or install-header changes;
  - no source-list, helper-target, Make, CMake, or CTest registration changes;
  - LOBPCG remains adjacent parity evidence, not direct
    `lanczos_iterate_op` ownership.

## Day 4 Notes

- Designed the repeated/clustered Ritz selection proof for Day 5.
- Separated exact repeated-value proof from public Lanczos multiplicity claims:
  - exact repeated Ritz values should be tested directly at
    `s20_select_indices`;
  - clustered-but-distinct spectra can be tested through the public
    eigensolver surface.
- Identified Day 5 selector targets:
  - repeated sorted `theta` arrays for `LARGEST` and `SMALLEST`;
  - equal-magnitude `NEAREST_SIGMA` tie behavior, preserving the current
    right-endpoint-first selector contract.
- Identified Day 5 public solver target:
  - a small clustered diagonal spectrum with visible `1e-5` gaps,
    `tol = 1e-12`, `reorthogonalize = 1`, and value/order assertions only.
- Preserved movement blockers:
  - no Ritz selection source movement until Day 5 proof lands;
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest membership drift for this proof lane.
- Added Day 4 artifact:
  - `artifacts/day4-ritz-selection-proof-design.md`.

## Day 5 Notes

- Implemented repeated/clustered Ritz selection proof:
  - added direct `s20_select_indices` repeated-value assertions for
    `LARGEST` and `SMALLEST` in `tests/test_ldlt_backend_dispatch.c`;
  - added direct `NEAREST_SIGMA` equal-magnitude tie assertions preserving the
    right-endpoint-first selector contract;
  - added a public clustered diagonal grow-m Lanczos test in
    `tests/test_eigs.c` for top-cluster values `{10.0, 9.99999, 9.99998}`.
- Kept exact repeated-value proof separate from public scalar-Lanczos
  multiplicity claims.
- Kept public clustered proof to eigenvalue values/order only; no eigenvector
  uniqueness claim was added.
- Preserved non-claim boundaries:
  - no Ritz selector source movement;
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest registration changes.
- Added Day 5 artifact:
  - `artifacts/day5-ritz-selection-proof.md`.

## Day 6 Notes

- Designed the Ritz vector lifting and publication-boundary proof for Day 7.
- Inventoried the current publication paths:
  - grow-m Lanczos converged and `m_cap` exhaustion branches in
    `src/sparse_eigs.c`;
  - shift-invert grow-m publication through the same lift path with
    `lambda = sigma + 1 / theta`;
  - thick-restart converged and budget/restart-cap fallthrough branches in
    `src/sparse_eigs_thick_restart.c`;
  - LOBPCG final publication from `X[:, j]` in
    `src/sparse_eigs_lobpcg.c`.
- Defined public result invariants for:
  - `eigenvalues[0..n_converged)`;
  - `eigenvectors` column-major layout;
  - `n_requested`, `n_converged`, `iterations`, `residual_norm`, and
    `backend_used`.
- Defined Day 7 test targets:
  - grow-m vector lift on a non-diagonal SPD fixture;
  - shift-invert original-space vector publication;
  - thick-restart vector lift boundary;
  - LOBPCG publication boundary with `block_size > k`;
  - partial-result sentinel preservation beyond `n_converged`.
- Kept vector-publication helper movement blocked until Day 7 implementation
  evidence exists.
- Preserved non-claim boundaries:
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest registration changes;
  - no eigensolver source movement.
- Added Day 6 artifact:
  - `artifacts/day6-ritz-vector-publication-design.md`.

## Day 7 Notes

- Implemented the vector lifting and publication-boundary proof:
  - added grow-m vector lift assertions on a non-diagonal Laplacian fixture in
    `tests/test_eigs.c`;
  - added shift-invert original-space vector publication assertions in
    `tests/test_eigs.c`;
  - added grow-m partial-publication sentinel assertions in
    `tests/test_eigs.c`;
  - added thick-restart vector lift boundary assertions in
    `tests/test_eigs_thick_restart.c`;
  - added LOBPCG `block_size > k` vector-publication assertions in
    `tests/test_eigs_lobpcg.c`.
- Kept proof values visible at test call sites:
  - matrix dimensions and generators;
  - `k` and `block_size`;
  - `sigma`;
  - backend selection;
  - tolerances and iteration budgets;
  - residual, orthogonality, and sentinel thresholds.
- Helper movement assessment:
  - grow-m and thick-restart share `s20_lift_ritz_vectors`, but partial-state
    publication still differs enough to keep movement blocked;
  - LOBPCG publishes from `X[:, j]`, so it is not a direct candidate for
    Lanczos lift-helper reuse;
  - Day 8 partial-result proof is still required before any shared
    partial-publication helper can safely hide control flow.
- Preserved non-claim boundaries:
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest registration changes;
  - no eigensolver source movement.
- Added Day 7 artifact:
  - `artifacts/day7-ritz-vector-publication-proof.md`.

## Day 8 Notes

- Implemented explicit `m_cap` exhaustion proof in `tests/test_eigs.c`:
  - forced `SPARSE_EIGS_BACKEND_LANCZOS`;
  - used `n = 80`, `k = 3`, and `max_iterations = 16` so
    `m_cap = m_init = 16`;
  - used `tol = 1e-18` to force `SPARSE_ERR_NOT_CONVERGED`;
  - asserted `n_requested`, `n_converged`, `backend_used`,
    `peak_basis_size`, `iterations`, finite residual, and progress callback
    shape;
  - asserted finite descending published values and nonzero finite vector
    columns;
  - asserted sentinel values and vector columns beyond `k` remain untouched.
- Kept helper movement blocked:
  - grow-m partial publication is now pinned;
  - thick-restart partial publication still has separate restart-state
    fallthrough;
  - shift-invert grow-m conversion still needs Day 9 proof.
- Preserved non-claim boundaries:
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest registration changes;
  - no eigensolver source movement.
- Added Day 8 artifact:
  - `artifacts/day8-partial-result-publication-proof.md`.

## Day 9 Notes

- Implemented shift-invert grow-m conversion proof in `tests/test_eigs.c`:
  - forced `SPARSE_EIGS_BACKEND_LANCZOS`;
  - used a Laplacian tridiagonal fixture with `n = 24`, `k = 4`,
    `sigma = 1.37`, and `max_iterations = 24`;
  - kept the closed-form expected eigenvalues visible through
    `lambda_p = 2 - 2 cos(p*pi/(n + 1))`;
  - asserted the expected nearest-sigma order for `p = 10, 9, 11, 8`;
  - asserted the transformed-theta magnitude contract through
    `abs(1 / (lambda - sigma))`;
  - asserted `n_requested`, `n_converged`, `backend_used`,
    `peak_basis_size`, `iterations`, residual, and progress callback shape;
  - asserted original-space Ritz residuals and vector orthonormality.
- Preserved movement boundaries:
  - no eigensolver source movement was performed;
  - no public API, install-header, source-list, helper-target, Make, CMake, or
    reviewed CTest registration changes were introduced;
  - Day 10 still owns the evidence-backed movement/no-move decision.
- Focused validation passed:
  - `make build/test_eigs && ./build/test_eigs`
  - `43` tests, `0` failures, `956` assertions.
- Added Day 9 artifact:
  - `artifacts/day9-shift-invert-growm-conversion-proof.md`.

## Day 10 Notes

- Reviewed the Sprint 114 eigensolver proof stack:
  - Day 2-3 Lanczos behavior design and implementation;
  - Day 4-5 repeated/clustered Ritz selection design and implementation;
  - Day 6-7 vector lifting and publication-boundary design and proof;
  - Day 8 bounded grow-m partial-result publication proof;
  - Day 9 shift-invert grow-m conversion proof.
- Considered narrow movement candidates:
  - moving `s20_select_indices`;
  - moving `s20_lift_ritz_vectors`;
  - moving shift-invert setup/conversion;
  - moving `lanczos_iterate_op`;
  - moving the whole grow-m backend.
- Published a continued no-move decision:
  - `s20_select_indices` movement would cross grow-m, thick-restart, LOBPCG,
    source-list, and build metadata boundaries;
  - `s20_lift_ritz_vectors` movement remains blocked by different grow-m and
    thick-restart partial-publication states;
  - shift-invert movement remains coupled to LDLT factor lifecycle,
    `used_csc_path_ldlt`, operator selection, public error propagation, and
    cleanup ownership;
  - `lanczos_iterate_op` movement needs explicit source-list/compile-unit
    proof for all consumers;
  - whole grow-m backend movement is too broad for Sprint 114.
- Preserved non-claim boundaries:
  - no eigensolver source movement;
  - no public API, install-header, helper-target, Make, CMake, source-list, or
    reviewed CTest registration changes;
  - no package, platform, Windows, ABI, or CMake parity claim.
- Day 10 is documentation-only. Day 9 already ran the full required C gate
  after the latest `.c` edits.
- Added Day 10 artifact:
  - `artifacts/day10-eigensolver-movement-decision.md`.

## Day 11 Notes

- Designed the direct/iterative exact-RHS cleanup batch for Day 12.
- Inspected QR sequential RHS setup in `tests/test_qr.c`:
  - `make_qr_exact_rhs` already owns sequential `x_exact[i] = i + 1`
    allocation and `b = A*x_exact`;
  - Day 12 should reuse it only for square/sequential solve cases and keep
    least-squares dimensions, residual thresholds, and refinement before/after
    values visible.
- Inspected CG and GMRES setup in `tests/test_iterative.c`:
  - the file already has `make_iterative_exact_rhs` plus sequential/sine
    pattern callbacks;
  - many later CG/GMRES tests still hand-roll exact-vector allocation,
    filling, and `compute_rhs`;
  - Day 12 should clean generated-RHS cases while leaving small analytical
    literal vectors inline.
- Inspected BiCGSTAB setup in `tests/test_bicgstab.c`:
  - SuiteSparse and cross-solver tests repeat sequential exact-RHS setup;
  - Day 12 may add a file-local sequential exact-RHS helper, but ILU/ILUT
    options, accepted nonconvergence behavior, and residual thresholds must
    remain at call sites.
- Inspected MINRES setup in `tests/test_minres.c`:
  - SPD, KKT, preconditioner, and direct-comparison tests repeat
    sequential/sine/cosine exact-RHS setup;
  - Day 12 may add file-local pattern helpers without hiding SPD/KKT fixture
    assumptions or preconditioner/comparison assertions.
- Blocked broad abstraction:
  - no cross-solver exact-RHS oracle;
  - no shared direct/iterative helper target;
  - no public API, install-header, source-list, Make, CMake, or reviewed CTest
    registration changes.
- Defined Day 12 focused validation:
  - `make build/test_qr build/test_iterative build/test_bicgstab build/test_minres`
  - `./build/test_qr`
  - `./build/test_iterative`
  - `./build/test_bicgstab`
  - `./build/test_minres`
  - `make format && make lint && make test`
- Day 11 is documentation-only. The latest C gate remains Day 9's
  `make format && make lint && make test`.
- Added Day 11 artifact:
  - `artifacts/day11-direct-iterative-exact-rhs-cleanup-design.md`.

## Day 12 Notes

- Implemented bounded direct/iterative exact-RHS cleanup.
- QR disposition:
  - no code movement was needed because `tests/test_qr.c` already has
    `make_qr_exact_rhs`;
  - focused QR validation remained part of the Day 12 proof scope.
- CG cleanup in `tests/test_iterative.c`:
  - reused existing `require_iterative_exact_rhs` for sequential RHS setup in
    `test_cg_diagonal_preconditioner` and `test_cg_precond_laplacian`;
  - kept diagonal preconditioner arrays, Laplacian fixture details, iteration
    comparisons, and residual thresholds visible.
- GMRES cleanup in `tests/test_iterative.c`:
  - reused existing generated-RHS helper for
    `test_gmres_large_unsymmetric`,
    `test_gmres_max_iter_exceeded`,
    `test_gmres_restart_comparison`, and
    `test_gmres_diagonal_preconditioner`;
  - kept restart sizes, tolerance values, convergence/nonconvergence
    expectations, and true residual checks at call sites.
- BiCGSTAB cleanup in `tests/test_bicgstab.c`:
  - added file-local sequential exact-RHS helper functions;
  - used them in `test_bicgstab_west0067`, `test_bicgstab_steam1`,
    `test_bicgstab_orsirr_1`, and
    `test_s103_bicgstab_steam1_ilu_vs_gmres30_reference`;
  - kept ILU/ILUT options and accepted nonconvergence behavior visible.
- MINRES cleanup in `tests/test_minres.c`:
  - added file-local exact-RHS helpers for sequential, sine, and scaled
    sequential patterns;
  - used them in selected SPD, KKT, preconditioner, and LDLT comparison
    tests;
  - kept SPD/KKT fixture assumptions, preconditioner construction,
    comparison solver calls, iteration expectations, and residual gates
    inline.
- Preserved non-claim boundaries:
  - no cross-solver exact-RHS oracle;
  - no helper target, public API, install-header, source-list, Make, CMake, or
    reviewed CTest registration changes.
- Focused validation passed:
  - `make build/test_qr && ./build/test_qr && ./build/test_iterative && ./build/test_bicgstab && ./build/test_minres`
  - `test_qr`: `73` tests, `0` failures, `654` assertions;
  - `test_iterative`: `80` tests, `0` failures, `713` assertions;
  - `test_bicgstab`: `61` tests, `0` failures, `464` assertions;
  - `test_minres`: `43` tests, `0` failures, `702` assertions.
- Added Day 12 artifact:
  - `artifacts/day12-direct-iterative-exact-rhs-cleanup.md`.

## Day 13 Notes

- Implemented bounded SVD proof-owner cleanup in `tests/test_svd.c`.
- Reconstruction cleanup:
  - added storage-explicit reconstruction helpers for max error and relative
    Frobenius residual;
  - used them in economy and full-mode SVD reconstruction tests while keeping
    `U` and `Vt` leading dimensions visible at call sites.
- U/Vt orthogonality cleanup:
  - added a Vt row-orthogonality helper that requires explicit row count,
    column count, and leading dimension;
  - retained the existing `orthogonality_error` helper for U columns.
- Moore-Penrose cleanup:
  - added `svd_pinv_first_moore_penrose_error` for the first identity
    `A * A+ * A ≈ A`;
  - preserved the product-dimension proof comments in tall and rectangular
    pseudoinverse tests.
- Low-rank cleanup:
  - added dense low-rank Frobenius residual helper;
  - added sparse-vs-dense and sparse-vs-sparse Frobenius comparison helpers;
  - kept ranks, drop tolerances, fixture names, and residual thresholds at the
    test sites.
- Condition-number cleanup:
  - added finite and infinite condition-number assertion helpers;
  - kept expected values, singular cases, and rectangular interpretation
    visible in each test.
- Preserved non-claim boundaries:
  - no public SVD API, install-header, helper-target, source-list, Make,
    CMake, or reviewed CTest registration changes;
  - no helper moved out of `tests/test_svd.c`;
  - no broad SVD proof abstraction was claimed.
- Focused validation passed:
  - `make build/test_svd && ./build/test_svd`
  - `test_svd`: `98` tests, `0` failures, `1580` assertions.
- Added Day 13 artifact:
  - `artifacts/day13-svd-proof-owner-cleanup.md`.

## Day 14 Notes

- Closed Sprint 114 with validation, metrics, non-claim, and residual handoff
  evidence.
- Touched-surface review:
  - eigensolver proof tests changed in `tests/test_eigs.c`,
    `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c`, and
    `tests/test_ldlt_backend_dispatch.c`;
  - direct/iterative exact-RHS cleanup changed in `tests/test_iterative.c`,
    `tests/test_bicgstab.c`, and `tests/test_minres.c`;
  - SVD proof-owner cleanup changed in `tests/test_svd.c`;
  - no `src`, `include`, Make, CMake, CTest registration, package, CI, or
    install metadata files changed.
- Proof-owner metrics:
  - `14` Sprint 114 artifacts;
  - `8` touched C test files;
  - `9` explicit `test_s114...` proof tests;
  - `0` eigensolver source movements;
  - `0` public API/header, source-list/build metadata, helper-target, or
    reviewed CTest membership changes.
- Final Day 14 validation passed:
  - `make source-list-check`: `PASS (48 library sources)`;
  - `make format && make lint && make test`;
  - `git diff --check`;
  - trailing-whitespace scan across Sprint 114 docs and touched test files.
- Non-claims preserved:
  - no public API, install-header, package, ABI, Windows, CMake parity, or
    reviewed CTest membership claim;
  - no broad direct/iterative oracle;
  - no broad SVD proof abstraction;
  - no helper target outside file-local test helpers.
- Residual deferred debt remains dependency-ordered:
  - eigensolver source-boundary follow-through;
  - direct/iterative cross-solver oracle decision;
  - SVD shared proof-helper decision;
  - package, ABI, platform, and adoption validation when those surfaces change.
- Added Day 14 artifact:
  - `artifacts/day14-validation-metrics-and-handoff.md`.
