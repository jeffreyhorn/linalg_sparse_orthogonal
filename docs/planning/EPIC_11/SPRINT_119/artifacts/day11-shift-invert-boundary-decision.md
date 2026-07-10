# Sprint 119 Day 11 Shift-Invert Boundary Decision

## Purpose

Day 11 decides whether Sprint 119 should split shift-invert setup/conversion
out of `src/sparse_eigs.c` or explicitly defer that movement. The decision is
based on current ownership, LDLT lifecycle dependencies, backend telemetry,
operator selection, transformed eigenvalue conversion, error propagation, and
cleanup obligations.

## Decision

Do not split shift-invert setup/conversion in Sprint 119. Defer the source
movement and use Day 12 to validate the explicit deferral.

The shift-invert path is currently too coupled to the public eigensolver entry
flow to move safely after the completed selection/lifting movement without a
dedicated lifecycle extraction design. The current code has correct behavior
and broad focused coverage; moving it now would create more risk than value.

## Current Ownership Map

| Concern | Current owner | Notes |
|---|---|---|
| Shift-invert operator callback | `s20_op_shift_invert` in `src/sparse_eigs.c` | Thin wrapper over `sparse_ldlt_solve`; `ctx` is a precomputed `sparse_ldlt_t`. |
| Shifted matrix construction | `s46_sparse_eigs_sym_impl` in `src/sparse_eigs.c` | Copies `A`, subtracts `sigma` from the diagonal, and owns early allocation/error exits. |
| LDLT factor lifecycle | `s46_sparse_eigs_sym_impl` in `src/sparse_eigs.c` | Zero-initializes `ldlt_shift`, factors `A - sigma*I`, and frees it on every post-setup return path. |
| Backend telemetry | `s46_sparse_eigs_sym_impl` in `src/sparse_eigs.c` | Publishes `result->used_csc_path_ldlt` from `sparse_ldlt_opts_t::used_csc_path`. |
| Operator dispatch | `s46_sparse_eigs_sym_impl` and `s46_run_backend` | Switches from `s20_op_matvec/A` to `s20_op_shift_invert/&ldlt_shift` before backend dispatch. |
| Original-space eigenvalue conversion | Grow-m and thick-restart result publication | Converts transformed Ritz values with `sigma + 1.0 / theta` for converged and partial results. |
| LOBPCG nearest-sigma behavior | `src/sparse_eigs_lobpcg.c` through shared operator callback and selector | Consumes the shift-invert operator through `op_fn/op_ctx`; vector publication stays LOBPCG-owned. |
| Reusable handle/workspace entry | `sparse_eigs_sym_with_handle` and `sparse_eigs_sym_with_workspace_internal` | Both flow through `s46_sparse_eigs_sym_impl`, so any split must preserve one-shot and reusable-workspace behavior. |

## LDLT Lifecycle Proof Notes

| Lifecycle point | Required behavior | Current evidence |
|---|---|---|
| Zeroed factor state | `sparse_ldlt_free(&ldlt_shift)` must be safe before and after factoring. | `ldlt_shift` is initialized as `{0}` before optional setup. |
| Shifted matrix allocation | Allocation failure returns `SPARSE_ERR_ALLOC` without leaking an LDLT factor. | `A_shifted = sparse_copy(A)` failure returns immediately before factor ownership begins. |
| Diagonal shift mutation | Any `sparse_set` error frees `A_shifted` and returns the exact error. | Setup loop frees `A_shifted` before returning on mutation failure. |
| Factor failure | Factor error frees both `ldlt_shift` and `A_shifted`, publishes telemetry, and returns the exact error. | `sparse_ldlt_factor_opts` failure path frees both owners and propagates `err`. |
| Success path | `A_shifted` and `ldlt_shift` remain live for the backend call, then are freed before refinement. | `op_ctx` points at stack-owned `ldlt_shift` until `s46_run_backend` returns; cleanup follows immediately. |
| Explicit max-iteration badarg | If validation after setup rejects explicit iteration budget, both owners are freed. | The `max_iterations` badarg path frees `ldlt_shift` and `A_shifted`. |
| Refinement boundary | Shift-invert factor ownership must end before optional Rayleigh-quotient refinement begins. | Cleanup occurs before `s29_maybe_refine(...)`. |

## Operator, Error, And Cleanup Proof Notes

| Area | Proof |
|---|---|
| Operator selection | `NEAREST_SIGMA` is the only mode that swaps `op_fn` from `s20_op_matvec` to `s20_op_shift_invert`; all other modes keep `op_ctx = A`. |
| Error propagation | `s20_op_shift_invert` directly returns `sparse_ldlt_solve`; `lanczos_iterate_op` and backend callers propagate operator errors. |
| Singular sigma behavior | If `A - sigma*I` is singular during setup, `sparse_ldlt_factor_opts` returns `SPARSE_ERR_SINGULAR`, and the public API propagates that error. |
| Backend telemetry | `result->used_csc_path_ldlt` is reset during result initialization and set from `used_csc_path` immediately after the LDLT factor attempt. |
| Partial-result conversion | Grow-m converts transformed values in both converged and m-cap partial result paths. |
| Thick-restart conversion | Thick-restart converts transformed values in both converged and partial result paths. |
| LOBPCG adjacency | LOBPCG consumes the same operator callback and selection helper; no shift-invert setup split can ignore LOBPCG nearest-sigma coverage. |
| Cleanup ownership | Cleanup is currently local and visible. A split would need a private context struct and one cleanup helper to avoid duplicated frees or stale stack pointers. |

## Focused Shift-Invert Coverage

| Test surface | Covered behavior |
|---|---|
| `test_shift_invert_diagonal_k3` | Interior nearest-sigma value selection and deterministic tie behavior. |
| `test_shift_invert_indefinite_small` | Symmetric-indefinite LDLT factor path through shift-invert. |
| `test_shift_invert_singular_sigma` | `SPARSE_ERR_SINGULAR` propagation when `sigma` is exactly an eigenvalue. |
| `test_shift_invert_eigenvectors` | Original-space vector correctness after transformed operator solve. |
| `test_shift_invert_wide_spectrum_middle` | Interior convergence on wide-spectrum diagonal input. |
| `test_s114_shift_invert_vector_publication_boundary` | Original-space vector publication boundary. |
| `test_s114_shift_invert_growm_conversion_nearest_sigma` | Transformed-theta ordering and original eigenvalue conversion proof. |
| `test_indefinite_shift_invert_uses_csc_above_threshold` | AUTO LDLT routes large indefinite shift-invert setup through CSC and publishes telemetry. |
| `test_indefinite_shift_invert_uses_linked_list_below_threshold` | AUTO LDLT routes small indefinite shift-invert setup through linked-list path and publishes telemetry. |
| `test_s20_eigs_nearest_sigma_day12` | Public nearest-sigma sanity coverage in backend dispatch tests. |
| `test_lobpcg_nearest_sigma_*` | LOBPCG-adjacent nearest-sigma behavior through the shared operator/selection path. |

## Split/Defer Decision

| Option | Decision | Reason |
|---|---|---|
| Move `s20_op_shift_invert` alone | Defer | The function is tiny; moving it alone adds build/source churn without reducing the real lifecycle coupling. |
| Move shifted-matrix construction alone | Defer | It must return a live matrix/factor pair and preserve all failure cleanup paths; a split needs a context owner first. |
| Move LDLT setup and cleanup into a private owner | Defer | This is the right future direction, but it needs a private context type, init/cleanup contract, and one-shot/handle proof. |
| Move transformed eigenvalue conversion | Defer | Conversion currently lives at each result-publication site; moving it separately risks drifting converged and partial-result paths. |
| Keep current code and validate deferral | Choose | Preserves correct behavior and lets Sprint 119 close with explicit residual ownership. |

## Future Movement Design Requirement

A future shift-invert movement should be a dedicated private-owner extraction,
not an opportunistic helper move. It should define:

1. a private `sparse_eigs_shift_invert_context` or equivalent;
2. a setup helper that owns shifted matrix construction, LDLT factorization,
   telemetry publication, and exact error propagation;
3. a cleanup helper that is safe on zeroed and partially initialized state;
4. explicit lifetime rules for one-shot, handle, workspace, grow-m,
   thick-restart, and LOBPCG paths;
5. transformed-value conversion helpers or a documented decision to keep
   conversion at backend publication sites;
6. focused tests for singular sigma, LDLT linked-list/CSC telemetry,
   grow-m partial results, thick-restart partial results, LOBPCG nearest-sigma,
   vector publication, and reusable handle/workspace behavior.

## Day 12 Validation Checklist

Day 12 should validate the explicit deferral rather than move code:

| Check | Required command or review |
|---|---|
| Confirm no Day 12 source/build movement is made for shift-invert. | `git diff --name-only` review. |
| Confirm focused shift-invert behavior remains green. | `./build/test_eigs`; optionally `./build/test_eigs_lobpcg` for LOBPCG nearest-sigma adjacency. |
| Confirm source-list metadata remains stable. | `make source-list-check`. |
| Confirm CMake/CTest count remains stable if any build metadata changed. | Not required for documentation-only Day 12 unless build files change; otherwise CMake build and `ctest -N`. |
| Confirm full quality if `.c` or `.h` changes happen unexpectedly. | `make format && make lint && make test`. |
| Record residual owner. | Sprint 119 Day 12/14 artifacts. |

## Residual Owner

| Residual | Owner | Handoff condition |
|---|---|---|
| Shift-invert setup/conversion private-owner extraction | Future sprint after Sprint 119, unless Day 12 uncovers a small safe corrective change. | Requires the dedicated context/lifecycle design above plus focused one-shot, handle, CMake, CTest, and full quality evidence. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Shift-invert ownership map exists. | Complete. |
| LDLT lifecycle dependency proof notes exist. | Complete. |
| Operator/error/cleanup proof notes exist. | Complete. |
| Split or defer decision exists. | Complete: defer source movement. |
| Validation checklist exists. | Complete. |
| Shift-invert movement was not attempted without cleanup and error proof. | Complete. |
| Deferral has explicit reason and future owner. | Complete. |
