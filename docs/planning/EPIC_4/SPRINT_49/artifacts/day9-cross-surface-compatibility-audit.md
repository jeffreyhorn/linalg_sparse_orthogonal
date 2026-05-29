# Sprint 49 Day 9 Artifact: Cross-Surface Compatibility Audit

## Purpose

Audit the remaining examples, benchmarks, tests, and docs after the Day 8
migration-doc batch so Day 10 can land only the smallest coherent final
agreement sweep.

## Main Day 9 Conclusion

The remaining Sprint 49 compatibility work is now smaller than a generic
“examples/benchmarks/tests/docs cleanup” bucket.

The strongest remaining drift is concentrated in two places:

1. repeated-run benchmark drivers still exercise internal reuse seams rather
   than the final public handle path
2. direct regression coverage for the new public handle entries is still absent

That is the right final queue shape for Day 10.

## Audited Surfaces

### Docs and examples

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/tutorial.md`

### Benchmarks

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`

### Tests

- `tests/test_iterative.c`
- `tests/test_eigs.c`
- broader `tests/` search for direct public-handle references

### Public headers

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

## What Looks Done Enough

### README and examples are no longer the main problem

After Day 8:

- `README.md` now explains the old one-shot path vs the explicit repeated-run
  handle path
- `examples/README.md` now explains why shipped examples still lean on the
  one-shot APIs

Interpretation:

- the top-level caller story is now present
- examples are no longer silently implying that one-shot is the only supported
  model

This means examples and the main README are not the strongest Day 10 targets
any more.

### Public header contract is already clear

The public headers now provide a coherent repeated-run contract:

- iterative handle lifecycle
- eigensolver handle lifecycle
- explicit statement that one-shot entries remain first-class

No serious header naming or ownership drift remains.

## Remaining High-Value Drift

### 1. Reuse benchmarks still prove the internal seam

`bench_iterative_reuse.c` still uses:

- `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres_with_workspace_internal(...)`

`bench_eigs_reuse.c` still uses:

- `sparse_eigs_sym_with_workspace_internal(...)`

Why this now matters:

- these were the right seams before Sprint 49 public exposure
- after Day 5/6, the final caller-facing repeated-run story is the public
  handle path
- leaving the benchmarks purely on the internal seam means the benchmark
  surface does not yet reflect the final public repeated-run model

This is the strongest implementation-side Day 10 target.

### 2. Direct public-handle regression coverage is still absent

The live test tree still has no direct references to:

- `sparse_iter_handle_*`
- `sparse_solve_*_with_handle(...)`
- `sparse_eigs_handle_*`
- `sparse_eigs_sym_with_handle(...)`

Current safety is still coming from:

- one-shot iterative/eigensolver tests
- family-level integration/regression tests

That is good baseline protection, but it does not yet pin the final public
repeated-run contract directly.

This is the strongest regression-side Day 10 target.

## Secondary Drift

### Benchmark docs may need one small alignment touch

`benchmarks/README.md` still focuses on benchmark-local command usage and does
not mention the new public repeated-run handle path.

That is not the primary issue. It is only worth touching if:

- Day 10 changes the repeated-run benchmark drivers themselves
- a small local README clarification keeps the benchmark surface consistent

### Tutorial does not need to move

`docs/tutorial.md` still teaches the one-shot iterative path and matrix-free
path without the new repeated-run handle model.

That is acceptable for Sprint 49 closeout because:

- the tutorial is still functionally correct
- the README now owns the migration-path explanation
- broad tutorial expansion would be larger than the highest-value remaining
  compatibility sweep

## Day 10 Target List

The smallest coherent high-signal Day 10 batch now looks like:

### Primary targets

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- focused direct public-handle regression additions in:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`

### Secondary touch only if needed

- `benchmarks/README.md`

## Day 10 Boundary

Day 10 should preserve behavior while reconciling the final repeated-run model.

Desired outcome:

- repeated-run benchmark evidence reflects the public handle path
- direct public-handle tests pin the new contract
- nearby benchmark docs are clarified only if the code changes make it useful

Non-goals:

- broad example conversion
- tutorial rewrite
- benchmark framework redesign
- large unrelated test churn

## Bottom Line

Day 9 reduced the remaining compatibility sweep to:

- one strong implementation bucket:
  - reuse benchmarks
- one strong regression bucket:
  - direct public-handle tests

Everything else now looks like optional noise relative to those targets.

That is the right bounded setup for Sprint 49 Day 10.
