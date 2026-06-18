# Sprint 78 Day 6 - Source Decomposition Batch

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Land the first bounded LDL^T CSC implementation ownership cleanup defined on Day 5 without widening into a broader subsystem rewrite or proof-surface spill.

## Main Result
Sprint 78 now has one landed first source batch inside the Day 5 fence:
- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_internal.h`

The batch stayed local and behavior-preserving while reducing the strongest mixed-role review seam in the LDL^T CSC implementation.

## Landed Ownership Cleanup
The writeback path in `src/sparse_ldlt_csc.c` now has explicit local helper ownership:
- `ldlt_csc_writeback_build_public_l(...)`
  - owns public `SparseMatrix` materialization for the factored `L`
- `ldlt_csc_writeback_copy_public_aux(...)`
  - owns auxiliary-array publication into `sparse_ldlt_t`

The linked-list fallback wrapper path now has explicit local helper ownership:
- `ldlt_csc_wrapper_validate_input(...)`
  - owns preflight validation for the wrapper entry
- `ldlt_csc_wrapper_copy_input_perm(...)`
  - owns preserved fill-reducing permutation capture
- `ldlt_csc_wrapper_publish_factor_payload(...)`
  - owns D / pivot / perm publication back into `LdltCsc`
- `ldlt_csc_wrapper_rebuild_csc_factor(...)`
  - owns rebuilt CSC-factor transplant after linked-list factorization

The internal header comments in `src/sparse_ldlt_csc_internal.h` now name the same private helper-cluster split directly.

## Preserved Fence
The Day 6 batch preserved:
- current public LDL^T behavior and error surface
- current scalar/native versus linked-list-wrapper execution behavior
- current proof ownership
- current support-only status for:
  - `src/sparse_ldlt_csc_supernodal.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`

The batch explicitly did not widen into:
- public-header or API edits
- giant-test cleanup
- broader chronology scrub
- unrelated direct-solver, benchmark, packaging, or platform work

## Validation
Because `*.c` and `*.h` changed, ran:
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 317.04 sec`

## Exit State
- The first Sprint 78 source decomposition batch is landed.
- The strongest LDL^T CSC mixed-role seam is now split into bounded helper clusters instead of one large body.
- Sprint 78 can now rerank the next maintainability pressure from a cleaner LDL^T CSC baseline.
