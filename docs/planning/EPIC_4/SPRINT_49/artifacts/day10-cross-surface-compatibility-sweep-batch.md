# Sprint 49 Day 10 Artifact: Cross-Surface Compatibility Sweep Batch

## Purpose

Align the final repeated-run benchmark evidence and direct regression coverage
with the new public lifecycle handle APIs, without widening into unnecessary
example, tutorial, or benchmark-framework churn.

## Main Day 10 Conclusion

Sprint 49 now has the smallest coherent compatibility sweep it still needed
after the Day 5/6 public lifecycle landing and the Day 8 migration-doc pass.

This batch stayed intentionally narrow:

- benchmark agreement targets:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- direct regression targets:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`

The batch did **not** widen into:

- example rewrites
- tutorial expansion
- benchmark framework redesign
- broad docs follow-ons

That was the correct Day 10 fence.

## Benchmark Agreement Outcome

### Iterative repeated-run benchmark now uses the public handle path

`bench_iterative_reuse.c` no longer proves repeated-run behavior through the
internal-only workspace-backed entry points.

The repeated-run path now uses:

- `sparse_iter_handle_t`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`

Why that matters:

- the benchmark evidence now reflects the final caller-facing repeated-run API
- the internal workspace layer returns to its intended role as implementation
  detail
- the repository surface is now more honest about what callers should actually
  use

### Eigensolver repeated-run benchmark now uses the public handle path

`bench_eigs_reuse.c` now routes the repeated-run path through:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`

That completes the same agreement on the eigensolver side:

- the benchmark no longer proves only an internal reuse seam
- it now proves the real public repeated-run lifecycle path

## Direct Public-Handle Regression Coverage

### Iterative direct-handle tests

`tests/test_iterative.c` now adds bounded direct public-handle coverage for:

- explicit prepare-and-reuse for CG
- GMRES public-handle validation
- zero-initialized on-demand handle growth for GMRES

Interpretation:

- the new public iterative lifecycle contract is now pinned directly
- both explicit prepare and on-demand growth paths are covered
- validation behavior is no longer only implied through implementation details

### Eigensolver direct-handle tests

`tests/test_eigs.c` now adds bounded direct public-handle coverage for:

- explicit prepare-and-reuse for symmetric eigensolve
- public-handle validation
- zero-initialized on-demand handle growth

That is the right Sprint 49 eigensolver test scope:

- it proves the new public repeated-run contract directly
- it avoids duplicating the broader family/backend eigensolver regression suite

## Important Boundary Decisions

This batch deliberately did **not** yet land:

- public-handle conversions in the shipped examples
- a tutorial rewrite around repeated-run handles
- benchmark README churn beyond what the current benchmark output already
  states
- a broader benchmark abstraction layer

That was correct because Day 10’s job was agreement, not surface expansion.

## Validation

Because `*.c` changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Focused follow-ons also passed:

- `./build/test_iterative`
- `./build/test_eigs`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative direct results:

- `test_iterative`: all `78` tests passed
- `test_eigs`: all `27` tests passed
- iterative repeated-run benchmark:
  - CG: `43.9660 ms` one-shot vs `46.9730 ms` handle reuse, `0.94x`
  - GMRES: `30.5820 ms` one-shot vs `28.4040 ms` handle reuse, `1.08x`
- eigensolver repeated-run benchmark:
  - grow-m: `2.1650 ms` one-shot vs `2.0980 ms` handle reuse, `1.03x`
  - thick-restart: `74.7310 ms` one-shot vs `78.1250 ms` handle reuse,
    `0.96x`

Behavior-level parity remained intact:

- iterative repeated-run cases matched one-shot iteration counts and residuals
- eigensolver repeated-run cases matched one-shot iterations, convergence,
  `n_converged`, residuals, and eigenvalues

## Sprint 49 Position After Day 10

The Sprint 49 public lifecycle story is now aligned across the highest-value
surfaces:

1. public headers expose the new bounded lifecycle handle APIs
2. implementation and one-shot wrappers route through those public handles
3. top-level docs explain when callers should stay on one-shot APIs vs adopt
   explicit handles
4. reuse benchmarks now prove the public handle path
5. direct tests now pin the public repeated-run contract

That leaves the final sprint queue much cleaner:

- residual review
- full validation
- Epic 4 closeout and handoff

## Bottom Line

Day 10 delivered:

- reuse benchmarks that now exercise the real public repeated-run handle path
- direct regression coverage for the new iterative and eigensolver handle APIs
- preserved behavior-level parity with the one-shot path
- no unnecessary widening into examples, tutorial, or framework work

That is the right bounded compatibility sweep for Sprint 49 Day 10.
