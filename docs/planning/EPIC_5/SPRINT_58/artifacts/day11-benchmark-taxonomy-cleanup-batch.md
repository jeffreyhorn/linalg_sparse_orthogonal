# Sprint 58 Day 11 - benchmark taxonomy cleanup batch

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Land the bounded benchmark-documentation cleanup by rewriting the highest-signal
benchmark README surface around stable workflow groupings, removing stale
sprint-local taxonomy drift, and keeping the benchmark story aligned with the
live driver set and validated proof roles.

## Touched surfaces

Landed set:

- `benchmarks/README.md`

Intentionally deferred:

- benchmark driver source files
- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- lower-priority public-header follow-through

## Landed changes

### `benchmarks/README.md`

The batch:

- normalized the benchmark summary table so per-driver descriptions read as
  stable workflow/capability summaries instead of sprint-history markers
- added one explicit `Workflow groups` section that splits the shipped
  benchmarks into:
  - one-shot compatibility/comparison
  - direct repeated-run lifecycle
  - iterative public-handle reuse
  - eigensolver public-handle reuse
- removed stale sprint-local wording from:
  - reorder coverage
  - the `bench_main` CLI behavior section
  - the `bench_chol_csc` table entry
  - the `bench_eigs` section header and backend description
  - the `bench_eigs` compare-mode explanation
- preserved the stable benchmark-proof boundaries:
  - `bench_refactor` / `bench_refactor_csc` for direct repeated-run lifecycle
  - `bench_iterative_reuse` for `CG` / `GMRES` / `MINRES`
  - `bench_eigs_reuse` for grow-m / thick-restart / explicit `LOBPCG`

## Measured result

Touched-surface line count:

- `benchmarks/README.md`: `235 -> 248`

Diff shape:

- `1` file changed
- `44` insertions
- `31` deletions

Interpretation:

- the README grew slightly because the taxonomy is now more explicit
- the added lines are workflow-grouping structure, not new benchmark claims

## Validation

This was a docs-only batch, so `make format`, `make lint`, and `make test`
were not required.

Targeted Day 11 sanity checks:

- `./build/bench_refactor` passed
- `./build/bench_iterative_reuse` passed
- `./build/bench_eigs_reuse` passed
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  passed

Important bounded substitution:

- the first no-argument `./build/bench_refactor_csc` run was allowed to proceed
  for several minutes but remained in its larger sweep without completing
  during this docs pass
- the final verification therefore used the narrower `nos4 --repeat 1`
  invocation already used elsewhere in Epic 5

Representative retained outputs:

- `bench_refactor`
  - `tridiag-200 2.90x`
  - `tridiag-500 1.27x`
  - `bcsstk04 1.27x`
  - `nos4 1.35x`
- `bench_refactor_csc nos4 --repeat 1`
  - `speedup_refactor = 1.76x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_iterative_reuse`
  - `cg-tridiag-300 1.18x`
  - `gmres-unsym-220 1.89x`
  - `minres-kkt-42 1.11x`
- `bench_eigs_reuse`
  - `growm-nos4-k5 1.19x`
  - `thick-bcsstk14-k5 1.01x`
  - `lobpcg-diag40-k3 1.01x`
  - `|lambda|max diff = 0.000e+00`

Drift check:

- `rg -n "Sprint" benchmarks/README.md`
  returned no matches

## Conclusion

The Day 11 batch stayed inside the planned fence:

- it simplified the benchmark story around stable workflow groupings
- it removed the highest-value stale sprint-local taxonomy markers
- it preserved the live proof boundaries and benchmark claims
- it avoided widening into benchmark-source or broader docs/header cleanup
