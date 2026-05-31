# Sprint 49 Day 6 Artifact: Public Lifecycle API Integration Batch II

## Purpose

Back the Day 5 public repeated-run handle declarations with real implementation
and wrapper routing, while preserving the existing one-shot caller model and
avoiding any leak of raw internal workspace layout.

## Main Day 6 Conclusion

Sprint 49 now has a real public repeated-run implementation path for the first
bounded iterative and eigensolver families.

This batch stayed intentionally narrow:

- implementation / wrapper integration:
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
  - `src/sparse_eigs_internal.h`
- unchanged public contract anchor:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- unchanged compatibility one-shot entries remain first-class:
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - `sparse_eigs_sym(...)`

The batch stayed within the Day 4 boundary:

- it implemented the Day 5 public lifecycle declarations
- it routed one-shot wrappers through that new public lifecycle path
- it did not widen into README/tutorial/example migration yet
- it did not expose raw internal workspace owners, typed internal views, or
  storage-layout promises

## Landed Public Lifecycle Implementation

### Iterative repeated-run implementation

`src/sparse_iterative.c` now implements:

- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_free(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

Interpretation:

- `sparse_iter_handle_t` is now a real caller-owned reusable anchor
- the handle owns a `sparse_iter_workspace_t` through `internal_state`
- prepare calls allocate or grow capacity through the existing checked internal
  workspace prepare helpers
- run calls delegate to the already-landed workspace-backed CG/GMRES solve
  implementations

### One-shot iterative wrappers now reuse the public path

The one-shot iterative entries now behave as compatibility wrappers over the
new public lifecycle seam:

- `sparse_solve_cg(...)`
- `sparse_solve_gmres(...)`

They now:

1. initialize a temporary public handle
2. run the corresponding `*_with_handle(...)` path
3. free the temporary handle

Why this matters:

- repeated-run and one-shot execution now share one implementation path
- public compatibility is preserved without duplicated solve logic
- Sprint 49 does not create a second-class public repeated-run API

### Eigensolver repeated-run implementation

`src/sparse_eigs.c` and `src/sparse_eigs_internal.h` now implement:

- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_free(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

Interpretation:

- `sparse_eigs_handle_t` is now a real caller-owned reusable anchor
- the handle owns a `sparse_eigs_workspace_t` through `internal_state`
- prepare calls pre-allocate backend-shaped reusable capacity for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG
- the with-handle run path delegates to the shared backend implementation with
  caller-owned reusable workspace

### One-shot eigensolver wrapper now reuses the public path

`sparse_eigs_sym(...)` now follows the same compatibility pattern:

1. initialize a temporary public handle
2. run `sparse_eigs_sym_with_handle(...)`
3. free the temporary handle

That is the right Day 6 outcome:

- the existing public eigensolver entry remains fully supported
- the repeated-run lifecycle surface is now real, not parallel-documentary
- callers that do not care about reuse are unaffected

## Important Internal Integration Detail

To keep the public eigensolver lifecycle path coherent across the main backend
families, Day 6 also widened one internal seam:

- `s21_lobpcg_solve(...)` now accepts an optional reusable workspace pointer

Why that was necessary:

- Day 5’s public eigensolver handle contract was intentionally backend-agnostic
- Day 6 could not leave LOBPCG as a hidden per-call heap exception without
  weakening the meaning of the public repeated-run path

Why this stayed safe:

- the change remained internal-only
- no raw LOBPCG workspace layout became public API
- the public contract still talks in initialize / prepare / run / free terms

## Important Boundary Decisions

This batch deliberately did **not** yet land:

- public matrix-free repeated-run entries
- public block/minres/BiCGSTAB repeated-run entries
- benchmark/example migration
- README/tutorial migration guidance
- public exposure of internal storage layout or typed internal views

That was the correct fence:

- Day 6 needed to make Day 5’s public declarations true
- it did not need to turn Sprint 49 into a broad public solver API redesign

## Validation

Because `*.c` and `*.h` changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Because this was a substantial public-API integration batch, the stronger
reviewed baseline also ran:

```bash
make quality-review-full
```

That also passed, including:

- reviewed CMake parity still at `53`
- Makefile/CMake parity still `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 422.68 sec`

Targeted touched-family follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_iterative`
- `./build/example_eigs`

Representative direct results:

- `test_iterative`: all `76` tests passed
- `test_eigs`: all `25` tests passed
- `test_eigs_lobpcg`: all `26` tests passed
- `example_iterative`: GMRES converged in `25` iterations unpreconditioned and
  `9` with ILU(0)
- `example_eigs`: all three shipped demos converged, including the LOBPCG
  `bcsstk04` section at `3 / 3`

## Sprint 49 Position After Day 6

The public lifecycle story is now materially stronger:

1. Day 5 declared the bounded public repeated-run contract
2. Day 6 wired that contract to the real internal reuse seams
3. existing one-shot entries now preserve compatibility by delegating through
   the same handle-backed implementation path

That makes the next queue much clearer:

- document the migration path
- prove cross-surface agreement in tests/examples/benchmarks/docs
- finish the final residual review and Epic 4 closeout sweep

## Bottom Line

Day 6 delivered:

- real public iterative repeated-run implementation for CG/GMRES
- real public eigensolver repeated-run implementation for symmetric eigensolve
- coherent internal LOBPCG reuse under the same public lifecycle contract
- compatibility-preserving wrapper routing for one-shot entries
- a fully green required gate, reviewed baseline, and focused follow-on sweep

That is the right bounded implementation landing for Sprint 49 Day 6.
