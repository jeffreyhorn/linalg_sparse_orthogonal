# Sprint 32 Day 2 `test_reorder_nd.c` Audit

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Define exactly which parts of `tests/test_reorder_nd.c` are active, dormant, historical, or misleading so the Sprint 32 structural cleanup can remove honesty drift without damaging the real current protection surface.

## Structure Summary

Current structure of `tests/test_reorder_nd.c`:

- `26` `static void test_*` functions
- `23` active `RUN_TEST(...)` calls
- `3` commented-out `RUN_TEST(...)` calls
- `0` unreferenced `static void test_*` functions outside that split

The dormant trio is:

1. `test_finest_fm_annealing_pres_poisson_close_to_target`
2. `test_nd_root_spectral_pres_poisson_close_to_target`
3. `test_non_pipeline_pres_poisson_close_to_target`

This matters because the file’s `3` `-Wunused-function` warnings are not random dead-code drift. They map exactly to the three commented-out tests and therefore to the file’s truthfulness problem.

## What Is Actually Active Today

The active suite already covers four distinct categories of behavior:

1. Core ND correctness and argument handling
   - grid/path validity
   - singleton / NULL / non-square validation
2. Public API fill and residual contracts
   - `bcsstk14` fill bound vs AMD
   - `Pres_Poisson` default-path fill bound vs AMD
   - determinism
   - Cholesky, LU, and LDLT ND dispatch/residual checks
3. Advisory-axis smoke or parity checks
   - HCC Kuu-safe parity
   - annealing differs-from-baseline
   - root-spectral differs-from-multilevel
   - thick-restart differs-from-baseline
   - fixed-K weight-scheme differentiation
4. Sprint 28 supernodal-postorder contracts
   - etree composition contract
   - corpus fill invariance
   - skip/no-reorder behavior
   - determinism
   - `n = 1`

That active surface is honest: every active test is run in `main()`, and every active assertion corresponds to a current supported behavior or measured bound.

## Dormant Trio Classification

### `test_finest_fm_annealing_pres_poisson_close_to_target`

Observed role:

- historical Sprint 27 Day 12 scaffold
- encoded contract: `Pres_Poisson` under annealing must land within `2pp` of the literal `0.85x` target, expressed as `<= 0.87x`
- comment explicitly says it was failing-as-expected and left commented out

Why it is dormant:

- Sprint 27 measured annealing at `0.943x`
- Sprint 27 Day 13 then concluded the Sprint 27 default was already the best Pres_Poisson result across the full combination matrix
- annealing shipped advisory-only, not as a path expected to close the headline target

Audit verdict:

- not a good candidate for future opt-in execution in the main suite
- it tests a known-missed historical target, not a live current contract
- best end state is delete from the suite file and preserve evidence in sprint docs

### `test_nd_root_spectral_pres_poisson_close_to_target`

Observed role:

- historical Sprint 27 Day 12 scaffold
- encoded contract: `Pres_Poisson` under root-spectral must land within `2pp` of the literal `0.85x` target, also expressed as `<= 0.87x`
- comment explicitly says it was failing-as-expected and left commented out

Why it is dormant:

- Sprint 27 measured root-spectral at `0.944x`
- Sprint 27 Day 13 concluded root-spectral remained advisory-only and regressed the headline metric relative to default

Audit verdict:

- same outcome as the annealing close-to-target stub
- this is historical evidence, not truthful active or opt-in coverage
- best end state is delete from suite code and retain the measurement trail in docs

### `test_non_pipeline_pres_poisson_close_to_target`

Observed role:

- Sprint 28 Day 10 scaffold for the non-pipeline supernodal-postorder pivot
- encoded contract: `Pres_Poisson` under `SPARSE_SUPERNODAL_POSTORDER=on` must land within `2pp` of the literal `0.85x` target
- comment explicitly says the target was formally retired and the `RUN_TEST(...)` stays commented out unless fundamentally different machinery appears

Why it is dormant:

- Sprint 28 measured no fill change at all: `0.9226x` both with env off and env on
- Sprint 28 `non_pipeline_decision.md` explains that this path is fill-invariant by construction and formally retires the literal target after six sprint misses

Audit verdict:

- strongest delete/docs-only candidate in the file
- unlike the Sprint 27 pair, this is not merely an unmet target that might later close under the same mechanism
- the cited decision document already says the relevant machinery would need to be fundamentally different

## Mechanical Debt Coupled To The File

`tests/test_reorder_nd.c` still carries `4` `-Wmissing-field-initializers` warnings at active option-struct sites, including:

- `sparse_analysis_opts_t`
- `sparse_cholesky_opts_t`
- `sparse_lu_opts_t`
- `sparse_ldlt_opts_t`

Those warnings are real cleanup work, but they are separate from the dormant-scaffold issue. The key structural point is:

- dormant scaffold explains the file’s `3` `-Wunused-function` warnings
- positional initialization explains the file’s `4` initializer warnings

Sprint 32 should close both while the file is open, but they should not be conflated.

## Recommended End State

### Keep active

- all `23` currently active tests
- especially the advisory-axis smoke/parity tests, because they are the truthful current contracts for those non-default behaviors

### Remove from suite code or move to docs

- `test_finest_fm_annealing_pres_poisson_close_to_target`
- `test_nd_root_spectral_pres_poisson_close_to_target`
- `test_non_pipeline_pres_poisson_close_to_target`

Reason:

- each one encodes a historical target that the project explicitly missed or retired
- leaving them compiled but not run misstates what CI protects
- the supporting evidence already exists in Sprint 27 and Sprint 28 decision docs

### Day 3 design implication

If Sprint 32 introduces an explicit slow or experimental mechanism, it should target live non-default checks whose semantics remain current.

This dormant trio is a poor fit for that mechanism because:

- the asserted success criteria are stale
- the supporting sprint docs already carry the historical evidence
- keeping them in-tree would preserve misleading target-oriented scaffolding rather than clarify the active suite

## Conclusion

`tests/test_reorder_nd.c` does not have a broad dead-code problem. It has a very specific honesty problem:

- three historical Pres_Poisson close-to-target stubs remain compiled
- none is part of the active suite
- all three are tied to explicitly missed or retired target claims

That makes the preferred Sprint 32 cleanup direction clear: preserve the history in docs, but stop compiling dormant failing-as-expected target stubs as if they were part of the current protection surface.
