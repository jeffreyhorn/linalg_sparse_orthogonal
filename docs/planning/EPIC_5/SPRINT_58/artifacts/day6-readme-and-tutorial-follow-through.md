# Sprint 58 Day 6 - README and tutorial follow-through

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Re-audit the landed README/tutorial state after Day 5, then finish the
strongest remaining top-level drift by normalizing the most visible README
summary sections and product-structure framing without widening the touched
scope into benchmark, examples, or public-header cleanup yet.

## Landed patch

Touched file:

- `README.md`

The Day 6 follow-through patch did three bounded things:

1. normalized the high-signal sparse eigensolver summary so it reads more like
   a stable capability overview and less like sprint-local chronology
2. removed the sprint-stamped wording from the visible repeated-run iterative
   support summary while preserving the real support boundary
3. simplified the `Project Structure` summary so it is less count-heavy and
   less brittle as the repo evolves

## Main outcomes

### Eigensolver summary normalization

The touched eigensolver overview still keeps the real product story visible:

- three concrete backends:
  - grow-m Lanczos
  - thick-restart Lanczos
  - `LOBPCG`
- AUTO backend dispatch
- shift-invert support
- optional refinement
- `bench_eigs` as the broader benchmark surface

The main difference is that the high-signal summary now carries less explicit
sprint-day narration.

### Iterative support-boundary normalization

The touched repeated-run iterative summary still says the real thing:

- repeated-run handles are intentionally bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

The wording is now less tied to sprint history.

### Project-structure framing cleanup

The `Project Structure` summary is now less brittle:

- removed stale counts from high-level directories
- kept examples and benchmarks visible as product surfaces
- updated planning-tree wording to the broader current planning layout

## Preserved boundary

The Day 6 patch intentionally did not touch:

- `docs/tutorial.md`
- `benchmarks/README.md`
- `examples/README.md`
- public headers
- deep CSC historical performance sections

This keeps the patch inside the Sprint 58 top-level docs follow-through fence.

## Sanity checks

Targeted checks run after the patch:

- `git diff -- README.md`
- `rg -n "Sparse Symmetric Eigensolver|bench_eigs|public repeated-run iterative handle support|Project Structure|planning/|CG|GMRES|MINRES|BiCGSTAB|LOBPCG" README.md`
- `wc -l README.md docs/tutorial.md`

Measured touched-surface state after the Day 6 patch:

- `README.md`: `973`
- `docs/tutorial.md`: `453`

## Conclusion

Day 6 landed the bounded top-level docs follow-through patch:

- `README.md` summary layers are more product-level and less sprint-local
- the tutorial intentionally stayed untouched because its post-Day-5 state was
  already close to the target
- the remaining queue is now clearly benchmark, examples, and headers

That is enough to move to the Day 7 public-header audit/design from a cleaner
caller-facing docs baseline.
