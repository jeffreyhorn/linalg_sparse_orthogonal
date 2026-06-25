# Sprint 89 Day 9: External Comparison Sweep

## Purpose

Execute the bounded external comparison protocol fixed on Day 8 and determine
whether Sprint 89 still needs any real final cross-surface fix batch before
Epic 8 closeout.

## Main Result

Sprint 89 now has one real bounded external comparison package across the
retained correctness, package-shape, and touched-runtime lanes:

- maintained correctness comparison owner:
  - `./build/quality-review-cmake/test_chol_csc`
- maintained package-shape proof owners:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- bounded runtime-reference support owner:
  - `make bench-reorder-sprint86`

The strongest result is explicit:

- correctness agrees strongly
- package/install/export shape agrees strongly
- runtime evidence remains mixed but bounded and interpretable
- no final implementation contradiction was exposed

## Correctness Comparison Outcome

The maintained SPD correctness lane stayed clean:

- `./build/quality-review-cmake/test_chol_csc`
  - `151 / 151` tests passed
- retained external dense reference readings:
  - `nos4`
    - `max|x-x_ref| = 4.690e-13`
    - `rel_residual = 3.907e-15`
  - `bcsstk04`
    - `max|x-x_ref| = 3.224e-11`
    - `rel_residual = 3.010e-16`

Interpretation:

- the bounded external SPD comparison lane agrees strongly on both retained
  fixtures
- the in-repo direct-family Cholesky path remains numerically aligned with the
  retained external dense reference helper
- no correctness mismatch was exposed

## Package-Shape Comparison Outcome

The maintained package-shape lane stayed clean:

- `bash tests/test_install.sh`
  - `13` passed
  - `0` failed
- `bash tests/test_cmake_install.sh`
  - `15` passed
  - `0` failed
  - `0` skipped

Interpretation:

- the maintained static-first Make/pkg-config install surface still matches
  the shipped contract
- the maintained CMake install/export surface still matches the exact-version
  and bounded package-shape contract
- no local install/export or consumer-shape contradiction was exposed

## Runtime-Reference Outcome

The bounded touched-runtime lane stayed mixed but interpretable:

- `make bench-reorder-sprint86`
  - `bcsstk14`
    - `amd`: `nnz_L=116071`, `reorder_ms=108.3`
    - `nd`: `nnz_L=132634`, `reorder_ms=401.2`
  - `Pres_Poisson`
    - `amd`: `nnz_L=2668793`, `reorder_ms=7035.0`
    - `nd`: `nnz_L=2474435`, `reorder_ms=5687.8`

Interpretation:

- the retained Sprint 86 runtime reading remains truthful:
  - `bcsstk14` favors AMD on both fill and reorder time
  - `Pres_Poisson` favors ND on both fill and reorder time
- this remains bounded branch-local comparison evidence
- it is not a broad timing gate and not a broad superiority claim
- no new touched contradiction was exposed large enough, by itself, to justify
  one last implementation batch

## Agreement / Difference Split

The strongest Day 9 agreement/difference split is now explicit:

- agrees strongly:
  - maintained SPD external differential correctness
  - maintained install/export and consumer-shape contract
- differs acceptably:
  - touched reorder/ND runtime behavior remains fixture-dependent
  - the repo remains intentionally bounded rather than broadly best-in-class
    on ordering/runtime behavior
- needs calibration rather than implementation:
  - final closeout wording should preserve the bounded mixed-runtime reading
    and avoid overclaiming uniform ND wins

## Final-Fix Decision Input

The strongest final-fix decision input is now explicit:

- no correctness mismatch was exposed
- no package/install/export contradiction was exposed
- no touched runtime contradiction was exposed that clearly justifies one last
  bounded source or proof-owner batch

The likely Day 11 outcome is therefore:

- explicit no-op final fix batch
- or, at most, bounded wording calibration if Day 10 identifies a necessary
  support-surface clarification

## Validation

The Day 9 comparison package was executed with:

- `./build/quality-review-cmake/test_chol_csc`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make build/include/sparse_version.h`
- `make bench-reorder-sprint86`

## Exit State

- Sprint 89 now has a real bounded external comparison package rather than
  only internal validation or inferred confidence.
- The comparison result is strong enough to retire the expected final
  implementation batch unless later design or validation surfaces a new
  contradiction.
- Day 10 can design the last landing from evidence, not from generic endgame
  pressure.
