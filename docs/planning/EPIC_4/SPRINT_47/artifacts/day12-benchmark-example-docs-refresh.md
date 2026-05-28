## Sprint 47 Day 12: Benchmark / Example Docs Refresh

### Objective

Refresh the touched benchmark/example documentation so it matches the live
Sprint 47 CLI and helper behavior, without broadening into unrelated tutorial
or architecture-document churn.

### Commands Run

1. Re-read the touched benchmark/example docs and the live runtime surfaces:
   - `sed -n '1,260p' benchmarks/README.md`
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '640,760p' benchmarks/bench_main.c`
   - `sed -n '1,120p' examples/example_eigs.c`
2. Land the bounded Day 12 docs batch:
   - `apply_patch` on:
     - `benchmarks/README.md`
     - `examples/README.md`

### Findings

#### 1. The benchmark doc drift was narrow and specific

After Days 5-8, the main benchmark README needed to reflect the actual
`bench_main` contract more directly:

- real `--help` support
- explicit malformed-input handling
- explicit conflicting-mode rejection
- explicit `--reorder` ownership split

Interpretation:

- Sprint 47 did not need a broad benchmark-doc rewrite
- it needed the touched README to state the live `bench_main` behavior clearly

#### 2. `benchmarks/README.md` now reflects the live `bench_main` contract

The Day 12 update now states:

- `bench_main`'s role as the main LU / Cholesky / SpMV / iterative harness
- the fact that `--help` is a real supported path
- that bad numeric or enum-like arguments fail with explicit diagnostics
- that conflicting modes such as `--spmv --iterative` are rejected
- that `--reorder` remains intentionally limited to:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- that COLAMD comparisons belong in:
  - `bench_reorder`
  - `bench_colamd`

Interpretation:

- the benchmark docs now match the Day 6 and Day 8 runtime contract

#### 3. `examples/README.md` now reflects the live small-example helper convention

The Day 12 update now records the current small-example convention:

- dynamic scratch buffers should route through
  `examples/example_alloc_helpers.h`

It also updates the `example_eigs` description so it matches the live example:

- nos4 largest-eigenvalue demo
- KKT shift-invert demo
- explicit LOBPCG + IC(0) preconditioned `bcsstk04` demo

Interpretation:

- the example docs now reflect both the Day 10 helper adoption direction and
  the actual three-part `example_eigs` runtime story

#### 4. The docs batch stayed bounded

No Day 12 changes were needed in:

- broad top-level `README.md`
- tutorial docs
- benchmark implementation comments
- broader example narrative restructuring

Interpretation:

- Sprint 47 stayed inside the touched benchmark/example documentation surface

### Bottom Line

Sprint 47 Day 12 refreshed the touched benchmark/example docs to match the live
runtime and helper contracts:

- `benchmarks/README.md`
  - now matches the modernized `bench_main` CLI and reorder ownership split
- `examples/README.md`
  - now reflects the small-example helper convention and the full
    `example_eigs` demo scope

The batch stayed intentionally narrow and left broader docs work deferred.
