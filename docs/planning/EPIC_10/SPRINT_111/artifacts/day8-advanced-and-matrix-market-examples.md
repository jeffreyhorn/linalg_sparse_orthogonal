# Day 8 Advanced and Matrix Market Examples

## Purpose

Day 8 closes the Sprint 111 advanced-example gap without broadening public
claims. Eigensolver and SVD examples already exist and remain bounded
one-shot teaching workflows. The missing user-facing route was a dedicated
Matrix Market load/use example that is not primarily an eigensolver fixture
loader and does not imply a public Matrix I/O module or builder API.

## Touched Files

- `examples/example_matrix_market.c`
- `examples/README.md`
- `docs/solver_selection.md`
- `CMakeLists.txt`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day8-advanced-and-matrix-market-examples.md`

## Implementation Summary

Added `examples/example_matrix_market.c`, which:

- loads `tests/data/tridiagonal_20.mtx` through `sparse_load_mm(...)`;
- reports parse/I/O failures with `sparse_err_t`, and reports
  `sparse_errno()` for `SPARSE_ERR_IO`;
- builds `b = A * 1` through `sparse_matvec(...)`;
- solves through the normal one-shot LU workflow on a `sparse_copy(...)`;
- reports residual and solution error;
- frees all dynamic vectors and the loaded matrix.

The example must be run from the project root so the test-data path resolves:

```bash
./build/example_matrix_market
```

## Build Registration

| Build Surface | Change |
|---|---|
| Makefile | No direct edit required; `examples/*.c` is picked up by the existing wildcard. |
| CMake | Added explicit `example_matrix_market` target linked against `sparse_lu_ortho`. |

## Advanced Example Boundary

| Area | Decision |
|---|---|
| Eigensolver | Keep `example_eigs` as the bounded one-shot symmetric eigensolver example. Do not broaden into nonsymmetric eigensolver or portable performance claims. |
| SVD | Keep `example_svd_lowrank` as the concise SVD/rank/condition/low-rank example. Do not claim external dense oracle completeness. |
| Matrix Market | Add a dedicated load/use example based on public load/save function contracts, not private source ownership. |

## Public API Guardrails

- The Matrix Market example includes only public library headers plus the local
  example allocation helper.
- It uses `sparse_load_mm(...)` as a public function, not a public module.
- It does not expose the private Matrix builder or Matrix I/O source owner.
- It keeps error handling visible at the example level.
- It uses the same normal solver-selection rules as any loaded public matrix.

## Validation Plan

Required validation for this example-source day:

- `make examples`
- `./build/example_matrix_market`
- `git diff --check`
- trailing-whitespace scan over touched docs and example source

## Completion Criteria Status

- Advanced examples avoid portability and performance overclaims.
- Matrix Market load/use now has a dedicated public-entry example.
- Matrix Market error and cleanup behavior are visible.
- Build registration is updated for both Makefile behavior and CMake behavior.
