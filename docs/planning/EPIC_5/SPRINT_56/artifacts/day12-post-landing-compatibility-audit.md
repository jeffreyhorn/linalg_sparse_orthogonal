# Sprint 56 Day 12 - post-landing compatibility audit

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Audit the landed Sprint 56 branch against the preserved decomposition fences:

- no public API redesign
- no solver-family support-boundary drift
- no behavior-visible repeated-run lifecycle drift
- real and measurable ownership reduction in the CSC/SVD hotspot files
- aligned build-surface ownership after the new extraction batches

## Preserved public and implementation fences

The strongest compatibility fact is structural:

- `git diff --name-only master...HEAD` shows no `include/` changes

Interpretation:

- Sprint 56 did not change the public direct-solver or SVD surface area
- the sprint stayed decomposition-first rather than slipping into API work

The landed Sprint 56 branch still matches the preserved fences:

- no public header/API redesign
- no solver-family support-boundary drift
- no behavior-visible repeated-run lifecycle drift
- no build-system divergence between the Makefile and CMake paths

## Ownership reductions remain real and measurable

Live post-Day-11 line counts:

- `src/sparse_ldlt_csc.c` = `2127`
- `src/sparse_ldlt_csc_supernodal.c` = `392`
- `src/sparse_chol_csc.c` = `1532`
- `src/sparse_chol_csc_supernodal.c` = `544`
- `src/sparse_svd.c` = `1319`
- `src/sparse_svd_partial.c` = `402`

Compared with the Sprint 56 Day 1 baseline:

- `src/sparse_ldlt_csc.c`: `2723 -> 2127`
- `src/sparse_chol_csc.c`: `2194 -> 1532`
- `src/sparse_svd.c`: `1728 -> 1319`

Interpretation:

- the retained LDLT CSC main file is smaller by `596` lines
- the retained Cholesky CSC main file is smaller by `662` lines
- the retained SVD main file is smaller by `409` lines
- all three new extracted files are large enough to represent true owned seams,
  not cosmetic spill files

## Build-surface alignment remains exact

The Sprint 56 extracted files are named consistently in both build systems:

- `Makefile`
  - `src/sparse_svd_partial.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`
- `CMakeLists.txt`
  - `src/sparse_svd_partial.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`

Interpretation:

- the Makefile/CMake ownership surfaces remain aligned
- no latent build-path drift surfaced after the Sprint 56 splits

## Residual drift audit

No blocker-level residual drift surfaced before final validation.

Non-blocking residual queue remains bounded to future maintainability work:

- deeper CSC legacy-comment cleanup beyond the bounded Day 11 sweep
- later CSC decomposition phases if the remaining retained files still justify
  another ownership split
- later SVD/private-header cleanup only if it clearly improves maintainability
  without reopening public/API scope

Interpretation:

- the remaining queue is future-facing rather than a hidden Sprint 56 defect
- Sprint 56 is ready for Day 13 validation from the current landed state

## Day 13 validation checklist

Required full validation gate:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Truthfulness anchors:

- `ctest -N --test-dir build/quality-review-cmake`
- Makefile/CMake parity
- full reviewed CMake `ctest`

Targeted Sprint 56 follow-ons:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_svd`
- `./build/test_integration`
- `./build/bench_refactor_csc`
- `./build/example_analysis`

## Conclusion

Sprint 56 Day 12 confirms that the landed branch still matches the preserved
Sprint 56 fences:

- the sprint remained decomposition-first and did not touch public headers
- the ownership reductions are real and measurable in the three hotspot areas
- Makefile and CMake still agree on the extracted source inventory
- no blocker-level drift remains before final validation

Sprint 56 can move into Day 13 from the current landed state without reopening
its design boundary.
