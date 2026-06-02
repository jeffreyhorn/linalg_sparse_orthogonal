# Sprint 53 Day 8: Indefinite Factor-Many Benchmark Proof

## Purpose

Day 8 closes Sprint 53's strongest remaining proof gap: there was still no
LDL^T-specific factor-many benchmark equivalent to the repeated-run Cholesky
surface. The goal is to add that proof without redesigning the benchmark
framework and without widening the sprint back into general direct-solver API
work.

## Main Day 8 Result

Sprint 53 now has a real bounded indefinite factor-many benchmark surface:

- `bench_refactor_csc` still ships the SPD / Cholesky repeated-run proof
- it now also ships an explicit `--indefinite-kkt` LDL^T mode
- both modes report the same public-vs-direct-CSC timing structure
- both modes now close at round-off residuals after a Day 8 LDL^T permutation
  fix in the shared solve path

This stayed inside the Sprint 53 fence:

- no benchmark-framework redesign
- no new public direct-solver handle
- no broad corpus expansion
- no dispatch redesign beyond the correctness fix surfaced by the new proof

## Touched Code

### `benchmarks/bench_refactor_csc.c`

Day 8 turns the old SPD-only repeated-run driver into a two-workflow benchmark:

- `chol_spd`
  - existing SPD / Cholesky repeated-run workflow
- `ldlt_kkt`
  - new synthetic KKT saddle-point repeated-run workflow

The benchmark now reports:

- `matrix`
- `workflow`
- `analyze_ms`
- `refactor_public_ms`
- `refactor_csc_ms`
- `solve_public_ms`
- `solve_csc_ms`
- `speedup_refactor`
- `res_public`
- `res_csc`

The indefinite mode is intentionally bounded:

1. build one synthetic above-threshold `kkt-150` matrix
2. analyze once with `SPARSE_FACTOR_LDLT`
3. prime the public repeated-run path with `sparse_factor_numeric(...)`
4. per iteration:
   - copy the base KKT matrix
   - perturb numeric values while preserving sparsity pattern
   - time `sparse_refactor_numeric(...)`
   - time the direct CSC completion seam via:
     - `ldlt_csc_prepare_resolved_analysis(...)`
     - `ldlt_csc_factor_with_resolved_analysis(...)`
   - solve and report residuals on the final perturbed matrix

This keeps the measured surface tied directly to the Sprint 53 implementation
work rather than to the older one-shot LDL^T dispatch benchmark.

### `benchmarks/README.md`

The benchmark-local README now matches the live benchmark contract:

- `bench_refactor_csc` is no longer described as one generic repeated-run
  linked-list-vs-CSC story
- it now explicitly documents:
  - the default SPD / Cholesky mode
  - the optional indefinite LDL^T KKT mode
  - the public-vs-direct-CSC timing columns

### `src/sparse_analysis.c`

Day 8 also lands a real correctness fix surfaced by the new indefinite
benchmark:

- `sparse_factor_solve(...)`
  - LDL^T factors no longer receive an extra outer `analysis->perm`
    pre/post-application
- below-threshold LDL^T `sparse_factor_numeric(...)`
  - now composes the outer analysis permutation into the stored factor
    permutation so the factor object carries one consistent final permutation
    contract

Without that, reordered indefinite repeated-run solves could double-permute the
right-hand side or solution.

### `tests/test_integration.c`

Day 8 adds:

- `test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt_amd`

That regression proves the exact reordered indefinite repeated-run case the
benchmark exposed:

1. analyze KKT with `SPARSE_REORDER_AMD`
2. factor once
3. perturb same-pattern values
4. refactor
5. solve back to the exact known solution

## Important Mid-Batch Catch

The first Day 8 indefinite benchmark run was useful precisely because it did
not look healthy:

- the direct CSC completion path solved to round-off
- the public repeated-run path did not

That was not a benchmark flaw. It exposed a real LDL^T permutation mismatch in
the shared solve wrapper. Day 8 therefore became both:

- a benchmark-proof batch
- a bounded correctness batch

The final landing keeps that scope disciplined by fixing only the exact
permutation contract needed for truthful reordered indefinite repeated-run
behavior.

## Measured Results

### `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

- `workflow = chol_spd`
- `analyze_ms = 0.313`
- `refactor_public_ms = 0.157`
- `refactor_csc_ms = 0.110`
- `solve_public_ms = 0.010`
- `solve_csc_ms = 0.004`
- `speedup_refactor = 1.43x`
- `res_public = 8.24e-16`
- `res_csc = 7.06e-16`

### `./build/bench_refactor_csc --indefinite-kkt --repeat 1`

- `workflow = ldlt_kkt`
- `analyze_ms = 0.124`
- `refactor_public_ms = 0.172`
- `refactor_csc_ms = 0.137`
- `solve_public_ms = 0.006`
- `solve_csc_ms = 0.002`
- `speedup_refactor = 1.26x`
- `res_public = 2.96e-16`
- `res_csc = 2.96e-16`

Interpretation:

- Day 8 now leaves measured indefinite same-pattern evidence at round-off
  accuracy
- the direct CSC completion path stays modestly ahead on the bounded KKT proof
  workload
- the benchmark claims now line up with live behavior instead of with an older
  linked-list-only mental model

## Focused Regression / Follow-On Proof

After the Day 8 fix, the most relevant direct reruns stayed clean:

- `./build/test_integration`
  - `36 / 36`
- `./build/test_etree`
  - `97 / 97`
- `./build/example_analysis`
  - residual `4.44e-16`

Those matter here because the Day 8 patch touched:

- shared analysis/factor solve semantics
- analysis-aware factor proof
- repeated-run benchmark proof

## Validation

Because `*.c` changed, Day 8 reran the required code-day gate:

- `make format`
- `make test`
- `make lint`

The focused benchmark / regression follow-ons above were also rerun after the
LDL^T permutation fix.

## Day 8 Operational Result

Sprint 53 now has a materially stronger benchmark-proof surface:

1. an explicit indefinite LDL^T factor-many mode now exists
2. benchmark wording matches the live public-vs-direct-CSC surfaces
3. reordered indefinite repeated-run solves now honor one consistent factor
   permutation contract
4. the sprint no longer depends on unmeasured indefinite factor-many claims

That closes the strongest remaining Day 8 proof gap cleanly enough for Day 9
to audit the remaining caller/documentation queue rather than reopening whether
the LDL^T repeated-run CSC story is actually measured or correct.
