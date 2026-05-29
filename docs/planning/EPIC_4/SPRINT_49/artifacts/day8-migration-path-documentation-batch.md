# Sprint 49 Day 8 Artifact: Migration-Path Documentation Batch

## Purpose

Explain the final Sprint 49 caller model clearly enough that existing users can
remain on the one-shot APIs while repeated-run callers can discover the new
explicit lifecycle handle path without reading only the public headers.

## Main Day 8 Conclusion

Sprint 49 now has a real migration-path explanation instead of only header
contracts and sprint-local notes.

This batch stayed intentionally narrow:

- primary target:
  - `README.md`
- one small supporting handoff surface:
  - `examples/README.md`

The batch did **not** widen into:

- benchmark wording updates
- test additions
- tutorial restructuring
- broad example rewrites

That was the correct boundary for the first migration-doc pass.

## README Migration Guidance

The top-level README now explains the final public repeated-run story in caller
terms.

### One-shot APIs remain first-class

The migration docs now say explicitly that callers can continue to use:

- `sparse_solve_cg(...)`
- `sparse_solve_gmres(...)`
- `sparse_eigs_sym(...)`

That matters because Sprint 49 is a compatibility-preserving public-lifecycle
exposure sprint, not a forced migration sprint.

### Explicit repeated-run handles are opt-in

The README now names the public iterative repeated-run surface:

- `sparse_iter_handle_t`
- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`
- `sparse_iter_handle_free(...)`

It also names the public eigensolver repeated-run surface:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_handle_free(...)`

### The “when to use it” explanation is now explicit

The README now tells callers:

- stay on the one-shot path when solves are one-off or occasional
- use explicit handles when the dimension is stable across repeated runs and
  preserving workspace capacity is worthwhile

This is the highest-value migration explanation Sprint 49 needed before the
cross-surface compatibility sweep.

### The key behavioral truth is now visible

The README now states the most important repeated-run contract clearly:

- reuse preserves allocation capacity, not old numerical iteration state

That prevents a common incorrect inference:

- repeated-run handles are a memory/capacity reuse feature
- they are not a promise to preserve previous Krylov / Ritz /
  search-direction state as a numerical continuation feature

## Example-Surface Handoff

`examples/README.md` now explains the intended scope of the shipped examples:

- they still lean on the one-shot public APIs
- this is deliberate because those APIs remain first-class and simpler for most
  callers
- explicit repeated-run handles exist, but they are an opt-in path for
  stable-dimension repeated work rather than a replacement for the shipped
  one-shot examples

Why this was the right supporting touch:

- it keeps the example surface honest
- it avoids implying that every example should immediately convert
- it lets Day 9/10 decide whether any example should change later as part of
  the compatibility sweep

## Important Boundary Decisions

This batch deliberately did **not** yet land:

- benchmark README wording changes
- benchmark driver code updates
- direct public-handle regression tests
- tutorial rewrites
- maintainer-guide expansion
- broad example conversions to the handle path

That was correct because Day 8’s job was caller guidance, not the full
agreement sweep.

## Targeted Sanity Checks

This was a docs-only batch, so no `make format`, `make lint`, or `make test`
run was required.

Targeted sanity checks were:

```bash
rg -n "Repeated-Run Lifecycle Handles|sparse_iter_handle_|sparse_eigs_handle_|one-shot public APIs|opt-in path" README.md examples/README.md
wc -l README.md examples/README.md
```

The new README migration section was also spot-read in context to confirm:

- the section sits near the existing iterative-solver user guidance
- the lifecycle explanation reads as caller-facing guidance rather than header
  duplication
- the example README handoff remains small and scope-correct

## Sprint 49 Position After Day 8

The remaining queue is now cleaner:

1. the old-vs-new caller path is documented
2. Day 9 can audit the highest-value remaining drift across
   benchmarks/tests/docs
3. Day 10 can land only the smallest coherent agreement batch

## Bottom Line

Day 8 delivered:

- a real top-level migration-path explanation
- explicit confirmation that one-shot APIs remain first-class
- a concrete “when handles are worth it” explanation
- a small supporting examples-surface handoff
- no unnecessary widening into the later compatibility sweep

That is the right migration-doc landing for Sprint 49 Day 8.
