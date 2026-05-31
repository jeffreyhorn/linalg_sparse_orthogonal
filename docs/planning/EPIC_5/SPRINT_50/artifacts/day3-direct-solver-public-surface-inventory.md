# Sprint 50 Day 3 Artifact: Direct-Solver Public Surface Inventory

## Purpose

Reduce the Sprint 50 direct-lifecycle problem from a generic “state model”
concern to a small set of named public seams grounded in the current headers
and caller-facing documentation.

## Highest-Value Day 3 Inputs

### Primary direct-solver public headers

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

### Public contrast / boundary-setting header

- `include/sparse_qr.h`

### Strongest shipped direct repeated-workflow caller surfaces

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`

### Public caller docs framing the current story

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`

## Public Surface Classes

### 1. Matrix-mutating one-shot factor-and-solve

This bucket is the strongest compatibility-facing direct-solver story today:

- LU
  - `sparse_lu_factor(...)`
  - `sparse_lu_factor_opts(...)`
  - `sparse_lu_solve(...)`
- Cholesky
  - `sparse_cholesky_factor(...)`
  - `sparse_cholesky_factor_opts(...)`
  - `sparse_cholesky_solve(...)`

Caller model:

- preserve the original matrix yourself if needed
- factor a copied matrix in place
- solve through the mutated factor-bearing matrix

This is simple, but it relies heavily on caller discipline around copied
matrices and implicit factor state stored in `SparseMatrix`.

### 2. Factor-object one-shot lifecycle

This bucket already uses explicit owned factor state:

- LDL^T
  - `sparse_ldlt_t`
  - `sparse_ldlt_factor(...)`
  - `sparse_ldlt_factor_opts(...)`
  - `sparse_ldlt_solve(...)`
  - `sparse_ldlt_free(...)`

Caller model:

- keep the original matrix unchanged
- receive an explicit factor object
- solve and free through that object

This is materially closer to an explicit lifecycle model than the LU and
Cholesky one-shot paths, but it still does not make repeated symbolic reuse the
dominant public direct-solver story.

### 3. Explicit analysis / factor / refactor bridge

This is the strongest existing public repeated direct-workflow model:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

Caller model:

1. analyze once
2. factor numerically
3. solve
4. update values with the same sparsity pattern
5. refactor numerically
6. solve again
7. free explicit lifecycle objects

This is already a real public lifecycle. The main Sprint 50 gap is that it
still reads as a specialist bridge rather than the central direct repeated-run
contract.

### 4. Backend / reorder / telemetry side paths

These shape implementation choice more than public lifecycle ownership:

- Cholesky backend selection:
  - `sparse_chol_backend_t`
  - `used_csc_path`
- LDL^T backend selection:
  - `sparse_ldlt_backend_t`
  - `used_csc_path`
- direct-solver reorder fields in the one-shot option structs

These matter for later implementation compatibility, but they are not the
first lifecycle-design target. They refine how factoring occurs, not who owns
the direct repeated-run state.

### 5. QR as a lifecycle contrast surface

QR is not a direct solver, but it informs lifecycle expectations:

- `sparse_qr_t`
- `sparse_qr_factor(...)`
- `sparse_qr_factor_opts(...)`
- `sparse_qr_solve(...)`
- `sparse_qr_free(...)`

It already uses an explicit factor object and explicit free path, but it does
not expose a direct analyze/refactor public story. That makes QR useful as a
boundary-setting comparison surface rather than a Sprint 50 first landing
target.

## Workflow Classification

The current direct-solver public caller stories reduce cleanly to:

- one-shot matrix-copy then in-place factor:
  - LU
  - Cholesky
- one-shot factor object:
  - LDL^T
  - QR as comparison
- analyze-once / factor-many:
  - `sparse_analysis_t` + `sparse_factors_t`

The example and benchmark surfaces are separate proof/demo surfaces:

- `example_analysis.c`
- `bench_refactor.c`
- `bench_refactor_csc.c`

That separation matters because Sprint 50’s first design work should target the
real API buckets, not the verification perimeter.

## Highest-Value Day 3 Gaps

### 1. Hidden mutable matrix state remains concentrated in the one-shot LU / Cholesky paths

Those paths still rely on documentation and examples to teach:

- copy first
- mutate the copied matrix
- keep the original around manually if later workflows need it

That is the strongest remaining implicit-lifecycle seam on the public
direct-solver side.

### 2. The analysis/refactor bridge is explicit but under-centered

The repo already has a public repeated direct workflow, but the dominant public
story is still the simpler one-shot path. The Sprint 50 design job is therefore
integration and centering, not invention.

### 3. Factor-object lifecycle ownership is already a normalized public pattern

LDL^T and QR show that caller-owned factor structs and explicit free calls are
already accepted public API patterns. Sprint 50 does not need to introduce that
idea as something novel to the repo.

## First Landing Targets vs Later Verification Surfaces

### True first landing targets

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

These headers define the real public lifecycle seams Sprint 50 must reconcile.

### Later compatibility / verification surfaces

- `README.md`
- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/README.md`
- `include/sparse_qr.h`

These matter for migration guidance, demos, benchmarks, and contrast, but they
are not the correct first design landing target.

## Highest-Value Day 3 Conclusions

### 1. Sprint 50’s public lifecycle problem is already small enough to name exactly

The repo does not lack lifecycle models. It has several:

- matrix-mutating one-shot
- factor-object one-shot
- analysis/factor/refactor bridge

The real problem is that those models are not yet reconciled into one explicit
direct repeated-run story.

### 2. The strongest design anchor remains `sparse_analysis.h`

That header already exposes the clearest repeated direct workflow and should be
the main public design anchor for the next Sprint 50 artifacts.

### 3. The first landing target is the direct public headers, not the surrounding docs/examples/benchmarks

Day 3 fixes the boundary cleanly enough that Day 4 can now focus on lifecycle
precedents and Day 5 on the actual gap analysis instead of re-litigating what
the public surface even is.
