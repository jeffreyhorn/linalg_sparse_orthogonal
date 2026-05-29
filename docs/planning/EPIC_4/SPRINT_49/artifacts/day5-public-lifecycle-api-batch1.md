# Sprint 49 Day 5 Artifact: Public Lifecycle API Landing Batch I

## Purpose

Land the first bounded public lifecycle/workspace API exposure after the
internal groundwork sprints, while preserving the existing one-shot caller
surface and avoiding any leak of raw internal workspace layout.

## Main Day 5 Conclusion

Sprint 49 now has a real public repeated-run lifecycle surface, not just an
internal reuse seam.

That first landing is intentionally narrow:

- public iterative repeated-run declarations:
  - `include/sparse_iterative.h`
- public eigensolver repeated-run declarations:
  - `include/sparse_eigs.h`
- unchanged compatibility one-shot entries remain the public behavior anchor:
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - `sparse_eigs_sym(...)`

The batch stayed within the Day 4 boundary:

- it exposed the minimal intended public handle/type/function surface
- it did not widen into implementation integration yet
- it did not widen into example, benchmark, or README migration work yet
- it did not publish internal workspace owners, typed views, or storage-layout
  promises

## Landed Public Lifecycle Surface

### Iterative repeated-run handle

`include/sparse_iterative.h` now exposes:

- `sparse_iter_handle_t`
- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_free(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

Interpretation:

- the public repeated-run iterative surface is lifecycle-centric
- the handle is the caller-owned reusable anchor
- the current public repeated-run landing is deliberately limited to the
  already-migrated direct CG/GMRES paths

### Eigensolver repeated-run handle

`include/sparse_eigs.h` now exposes:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_free(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

Interpretation:

- the public repeated-run eigensolver surface now has the same lifecycle shape
  as the iterative one
- the public contract talks in prepare/run/free terms rather than private
  grow-m / thick-restart / LOBPCG helper details

## Important Contract Decisions

### Opaque public handle layout

Both new public handle structs are intentionally opaque at the public level:

- `sparse_iter_handle_t` currently exposes only `internal_state`
- `sparse_eigs_handle_t` currently exposes only `internal_state`

Why this matters:

- callers get stack-allocation, zero-init, and explicit lifecycle control
- Sprint 49 does not freeze raw internal storage layout as public ABI
- Day 6 can back the new public surface without widening the promise set beyond
  the actual lifecycle contract

### Compatibility stayed first-class

Day 5 intentionally did **not** replace or de-emphasize the old one-shot public
model.

The public contract now reads as:

- one-shot solve/eigensolve calls remain fully supported convenience wrappers
- explicit handles are the opt-in repeated-run path for callers that want
  stable-dimension workspace reuse

Interpretation:

- the migration story remains compatibility-preserving
- existing callers are not forced to adopt the new lifecycle path just because
  it now exists

### Explicit non-goals for Day 5

This batch did **not** yet land:

- public implementation/wrapper routing
- block/minres/BiCGSTAB repeated-run public exposure
- matrix-free repeated-run public exposure
- example or benchmark migration
- README or tutorial migration guidance

That was the right fence:

- Day 5 needed to make the public contract real
- Day 6 can now implement that contract without guessing at the final surface

## Validation

Because `*.h` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Because this was a substantial public-API batch, the stronger reviewed
baseline also ran:

```bash
make quality-review-full
```

That also passed, including:

- reviewed CMake parity still at `53`
- Makefile/CMake parity still `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 597.27 sec`

Targeted touched-family follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/example_iterative`
- `./build/example_eigs`

Representative direct results:

- iterative regression surface stayed fully green (`76 / 76`)
- eigensolver regression surface stayed fully green (`25 / 25`)
- `example_iterative` still converged cleanly both without and with ILU(0)
- `example_eigs` still converged cleanly across all three shipped demos

## Sprint 49 Position After Day 5

The next batch is now much clearer:

1. Day 6 can wire the new public lifecycle declarations to the existing
   internal iterative/eigensolver reuse seams
2. one-shot wrappers can then delegate through the new public lifecycle model
   where appropriate
3. only after the public API is both declared and implemented should Sprint 49
   widen into migration docs and cross-surface compatibility work

## Bottom Line

Day 5 delivered:

- the first real public repeated-run iterative handle surface
- the first real public repeated-run eigensolver handle surface
- compatibility-preserving one-shot continuity
- no leak of raw internal workspace layout
- a fully green required and reviewed validation baseline

That is the right bounded first public API landing for Sprint 49.
