# Sprint 55 Day 11 - historical comment reduction sweep

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Remove stale sprint/day narrative from the permanent implementation files
touched earlier in Sprint 55 while preserving durable maintainership comments
about ownership, invariants, and algorithm behavior.

## Touched permanent files

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_thick_restart.c`

No public headers, tests, examples, benchmarks, or build wiring changed.

## Cleanup performed

The sweep normalized comments in four bounded ways:

1. Replaced sprint-history framing with durable file/section ownership notes.
2. Replaced rollout-history comments with actual algorithm or contract
   explanations.
3. Kept the useful numerical commentary:
   - Lanczos recurrence and reorthogonalization semantics
   - grow-m and thick-restart ownership boundaries
   - arrowhead / restart-state invariants
   - LOBPCG role and dispatch boundaries
   - progress/cancel and timer semantics
4. Removed stale chronology from the last remaining iterative progress comments.

## Truthfulness check

The intended cleanup target is now exact:

- `rg -n "Sprint|Day [0-9]+" src/sparse_eigs.c src/sparse_eigs_internal.h src/sparse_eigs_thick_restart.c src/sparse_iterative.c`
  returned no matches after the patch

This means the touched permanent implementation files no longer carry sprint
history as active source commentary.

## Measured result

Post-Day-11 line counts:

- `src/sparse_eigs.c` = `1534`
- `src/sparse_eigs_internal.h` = `631`
- `src/sparse_eigs_thick_restart.c` = `914`
- `src/sparse_iterative.c` = `1985`

Diff-stat summary:

- `src/sparse_eigs.c` = `429` changed lines
- `src/sparse_eigs_internal.h` = `106` changed lines
- `src/sparse_eigs_thick_restart.c` = `110` changed lines
- `src/sparse_iterative.c` = `12` changed lines
- total patch = `217` insertions / `440` deletions

Interpretation:

- this was a real cleanup pass rather than formatting churn
- the eigensolver decomposition files are now more maintainable for future
  readers who were not part of the original sprint history
- the iterative main file kept the Day 10 ownership split intact while losing
  its leftover chronology comments

## Validation

Required code-day validation passed:

- `make format`
- `make lint`
- `make test`

## Conclusion

Sprint 55 Day 11 delivered the planned maintainability sweep:

- stale sprint/day narrative is gone from the Sprint 55 touched permanent
  implementation files
- durable algorithm and ownership commentary still explains the difficult parts
- the patch stayed bounded and comment-only
- the normal code-day validation gate remained green

This closes the Day 11 cleanup task without reopening Sprint 55’s source-split
scope or changing any public solver behavior.
