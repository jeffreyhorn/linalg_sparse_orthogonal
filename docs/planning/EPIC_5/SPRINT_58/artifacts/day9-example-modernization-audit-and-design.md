# Sprint 58 Day 9 - example modernization audit and design

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Freeze the highest-value example modernization target before touching shipped
example code or docs by auditing the main example surfaces, separating real
caller-story gaps from already-healthy examples, ranking the targets by
truthfulness and caller value, and defining the exact Day 10 landing set plus
non-goals.

## Audited surfaces

Primary shipped example sources:

- `examples/example_analysis.c`
- `examples/example_iterative.c`
- `examples/example_ic_minres.c`
- `examples/example_eigs.c`

Paired example docs:

- `examples/README.md`

## Findings

### `example_analysis.c`

State:

- already the strongest aligned shipped example
- teaches the direct repeated-run lifecycle clearly
- states the same-pattern and reuse boundaries truthfully
- output wording already matches the stable README/tutorial story

Conclusion:

- do not touch in the first example batch unless a contradiction appears

### `example_eigs.c`

State:

- strongest remaining example-side narrative offender
- still carries visible sprint chronology in both comments and runtime banner
- still explains stable backend/preconditioner behavior through sprint-local
  framing rather than product-level wording
- numerical content and proof quality remain strong

Conclusion:

- strongest Day 10 first target
- fix wording, framing, and caller-story clarity without changing behavior

### `examples/README.md`

State:

- mostly aligned structurally after earlier sprint work
- still has a few visible sprint-history and chronology-shaped example entries
- the eigensolver entry is the strongest remaining offender

Conclusion:

- best paired doc target for `example_eigs.c`

### `example_iterative.c`

State:

- already a solid one-shot GMRES + ILU example
- only likely remaining improvement is small wording around the repeated-run
  handle alternative

Conclusion:

- lower-risk secondary target, not the first landed batch

### `example_ic_minres.c`

State:

- dense feature/demo bundle
- more of a scope-density issue than a clear stale-public-wording issue

Conclusion:

- defer from the first modernization batch

## Ranked targets

1. `examples/example_eigs.c`
2. `examples/README.md`
3. `examples/example_iterative.c` only if the landed patch remains very small

## Day 10 landing set

Selected touched set:

- `examples/example_eigs.c`
- `examples/README.md`

Optional follow-through only if still clearly bounded:

- one small wording touch in `examples/example_iterative.c`

## Non-goals

- no broad tutorial rewrite
- no example explosion
- no behavioral redesign
- no support-boundary changes
- no attempt to turn every shipped example into a repeated-run-handle showcase

## Conclusion

The example problem is now reduced to one clear first modernization seam:

- the eigensolver example source plus its README entry

That is enough to move to Day 10 implementation without broadening into a
general example rewrite.
