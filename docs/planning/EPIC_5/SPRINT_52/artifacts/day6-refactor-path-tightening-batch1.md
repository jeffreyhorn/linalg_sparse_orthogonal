# Sprint 52 Day 6: Refactor Path Tightening Batch I

## Purpose

Day 6 tightens the public repeated-run direct refactor contract without
broadening Sprint 52 into a larger API redesign. The target is the main
remaining shared-path weakness left after Days 4 and 5:
`sparse_refactor_numeric(...)` was still a safe wrapper, but it was too
permissive and too close to being just a second spelling of
`sparse_factor_numeric(...)`.

## Main Day 6 Conclusion

The shared refactor path is now materially more truthful than it was at Sprint
52 start:

- `sparse_refactor_numeric(...)` still preserves the Sprint 51 zero-init
  first-factorization path
- once a factors object is no longer zeroed, it must now match the analysis
  family and dimension before refactor proceeds
- malformed family payload is rejected up front
- replacement factorization is still built into a temporary object first, so
  old factors survive any later refactor failure unchanged

This stays inside the Sprint 52 scope fence:

- no public API redesign
- no raw internal storage exposure
- no one-shot API demotion
- no LU routing expansion
- no claim of incremental numeric updates

## Touched Code

### `src/sparse_analysis.c`

Day 6 adds two small shared helpers:

- `sparse_factors_is_zeroed(...)`
- `sparse_refactor_validate_existing_factors(...)`

The refactor flow now splits cleanly:

1. reject NULL inputs
2. reject matrix/analysis shape mismatch
3. accept a fully zeroed `sparse_factors_t`
4. otherwise require:
   - `factors->F` present
   - `factors->n == analysis->n`
   - `factors->type == analysis->type`
   - for LDL^T:
     - `D`
     - `D_offdiag`
     - `pivot_size`
     - `ldlt_perm`
   - for non-LDL^T:
     - none of those LDL^T-only payload fields may be populated
5. factor into a temporary `new_factors`
6. replace the old factors only on success

The important behavior change is not a new algorithm. It is a tighter contract
around when refactor is allowed to replace an existing factorization.

### `include/sparse_analysis.h`

The public doc contract for `sparse_refactor_numeric(...)` now matches the
live implementation more closely:

- `factors` may be zeroed for the first numeric factorization
- a non-zeroed factors object must already match the analysis family and
  dimension
- malformed preexisting factor payload is a caller-visible bad-argument case
- reuse still preserves symbolic/permutation setup, not old numeric factor
  contents

### `tests/test_integration.c`

Day 6 adds the strongest direct public-lifecycle proof for the new contract:

- `test_public_lifecycle_refactor_rejects_mismatched_existing_factors(...)`
  - build valid LU factors
  - attempt refactor through a Cholesky analysis
  - verify `SPARSE_ERR_BADARG`
  - verify the old LU factors still solve the original LU matrix afterward
- `test_public_lifecycle_refactor_preserves_old_factors_on_failure(...)`
  - build valid Cholesky factors
  - attempt refactor on a matrix intentionally made non-symmetric
  - verify `SPARSE_ERR_NOT_SPD`
  - verify the old Cholesky factors still solve the original system afterward

The preexisting zero-init refactor test remains in place, so the batch proves
both the preserved permissive entry case and the tighter replacement rules.

## Important Contract Detail

Day 6 preserves the public repeated-run ownership model from Sprint 50/51:

- `sparse_analysis_t` still owns symbolic/permutation setup
- `sparse_factors_t` still owns numeric factor state
- `sparse_refactor_numeric(...)` still rebuilds numeric factors from new values
  rather than incrementally updating them

What changed is that the function is now stricter about replacing an existing
factors object that does not actually correspond to the provided analysis.

## Explicit Non-Landings

Day 6 intentionally does **not** do these yet:

- redesign `sparse_factors_t`
- add public family-specific refactor entry points
- reopen LU routing
- add incremental numeric-update machinery
- broaden into tutorial/README/example rewriting
- expose raw CSC/native factor layout

## Validation

Because `*.c` / `*.h` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

## Truthfulness Anchors Preserved

The maintained reviewed baseline stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 229.89 sec`

## Focused Follow-Ons

The highest-value repeated-run direct proof stayed clean:

- `./build/test_integration`
  - `31 / 31` passed
  - `Assertions: 2015`

## Day 6 Operational Result

Sprint 52 now has a tighter public repeated-run refactor contract in code, not
just in design notes:

1. first-factorization through a zeroed `sparse_factors_t` still works
2. stale or mismatched existing factors are now rejected before replacement
3. old factors survive refactor failure intact

That makes the shared analysis/factors lifecycle more credible for repeated
direct runs without pretending Sprint 52 has already become an incremental
direct refactor system.
