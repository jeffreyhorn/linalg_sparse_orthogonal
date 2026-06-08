# Sprint 58 Day 10 - example modernization batch

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Land the highest-value bounded example modernization batch by updating the
strongest remaining narrative offender in the shipped example set and aligning
its paired example README entry to the simplified stable product story.

## Touched surfaces

Landed set:

- `examples/example_eigs.c`
- `examples/README.md`

Intentionally deferred:

- `examples/example_analysis.c`
- `examples/example_iterative.c`
- `examples/example_ic_minres.c`

## Landed changes

### `examples/example_eigs.c`

The batch:

- removed stale sprint chronology from the file header and runtime banner
- simplified the backend/preconditioner narrative into product-level wording
- kept the example explicitly one-shot-first while still pointing callers at
  the separate repeated-run handle workflow
- preserved all existing numerical workflows and output structure

Preserved behavior:

- same three workflows
- same residual checks
- same solver/backends exercised

### `examples/README.md`

The batch:

- rewrote the `example_eigs` entry to match the simplified example source
- removed the stale sprint-local eigensolver framing
- removed the remaining `Sprint 54` support-boundary sentence
- preserved the explicit note that repeated-run handles do not broaden to
  `BiCGSTAB` or block iterative workflows

## Measured result

Touched-surface line counts:

- `examples/example_eigs.c`: `285 -> 287`
- `examples/README.md`: stayed `134`

Diff shape:

- `2` files changed
- `22` insertions
- `20` deletions

## Validation

Because `examples/example_eigs.c` changed, the required gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

Focused follow-on:

- `./build/example_eigs` passed

Representative retained output:

- nos4 largest-eigenvalue demo: `5 / 5` converged pairs in `115` Lanczos
  iterations
- KKT nearest-sigma demo: `3 / 3` converged pairs in `6` Lanczos iterations
- explicit LOBPCG on `bcsstk04`: `3 / 3` converged pairs in `62` outer
  iterations, reported residual `8.808e-09`

Drift check:

- `rg -n "Sprint" examples/example_eigs.c examples/README.md`
  returned no matches

## Conclusion

The Day 10 batch stayed inside the Day 9 fence:

- it modernized the strongest remaining shipped example surface
- it aligned the paired example README entry
- it preserved the stable support boundary and one-shot-first posture
- it avoided widening into a broader example rewrite
