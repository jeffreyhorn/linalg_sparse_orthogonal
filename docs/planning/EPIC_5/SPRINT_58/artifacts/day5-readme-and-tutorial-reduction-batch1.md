# Sprint 58 Day 5 - README and tutorial reduction batch 1

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Land the first bounded top-level docs simplification patch by reducing the
README front-door workflow story, aligning the tutorial to the same one-shot-
first and repeated-run-support boundary, and preserving the validated
example/benchmark truthfulness fence without widening the touched scope into
benchmark, header, or example-doc cleanup yet.

## Landed patch

Touched files:

- `README.md`
- `docs/tutorial.md`

The patch did four bounded things:

1. reduced the top-level README feature ledger into shorter product-level
   workflow summaries
2. added a new top-level `Choose a Workflow` section to `README.md`
3. added a paired `Choose a Workflow First` section to `docs/tutorial.md`
4. added small tutorial follow-through notes near the Cholesky and GMRES
   sections so the repeated-run support boundary is visible where callers are
   already making workflow choices

## Main outcomes

### README front-door simplification

The touched `README.md` sections now make the final public workflow story
visible near the top of the file:

- one-shot direct solves are still the default path
- explicit repeated direct lifecycle is called out only for stable-pattern
  reuse
- repeated iterative handles remain intentionally bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated symmetric eigensolver handle wording remains bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- the strongest workflow-local benchmark proof surfaces are named directly

### Tutorial alignment

The touched tutorial sections now align more clearly with the final public
workflow fence:

- choose the smallest one-shot path first
- move to the explicit repeated direct lifecycle only when the sparsity pattern
  stays fixed across many solves
- move to explicit iterative/eigensolver handles only when dimension-stable
  scratch reuse matters
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

## Preserved boundary

The Day 5 patch intentionally did not touch:

- deep CSC historical performance sections
- deep eigensolver chronology
- benchmark taxonomy organization
- `examples/README.md`
- public headers

This keeps the landed patch inside the Day 4 design boundary.

## Sanity checks

Targeted checks run after the patch:

- `git diff -- README.md docs/tutorial.md`
- `rg -n "Choose a Workflow|example_analysis|bench_refactor|bench_iterative_reuse|bench_eigs_reuse|BiCGSTAB|block iterative|CG|GMRES|MINRES|LOBPCG" README.md docs/tutorial.md`
- `wc -l README.md docs/tutorial.md`

Measured touched-surface result:

- `README.md`: `987 -> 973`
- `docs/tutorial.md`: `415 -> 453`

Interpretation:

- the README reduction is real
- the tutorial grew modestly only because the repeated-run workflow boundary is
  now explicit near the front door

## Conclusion

Day 5 landed the first bounded top-level docs simplification patch:

- `README.md` front-door workflow story is shorter and more product-level
- `docs/tutorial.md` now aligns more clearly to the final repeated-run support
  boundary
- example and benchmark truthfulness anchors remained intact
- benchmark/example/header cleanup remains explicitly deferred

That is enough to move to the Day 6 follow-through pass from a coherent landed
docs baseline.
