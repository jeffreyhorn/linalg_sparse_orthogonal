# Sprint 50 Day 10 Artifact: Validation and Landing Plan

## Purpose

Define the validation contract, targeted follow-ons, and implementation order
for the later direct-solver lifecycle landing sprints so Sprint 51 starts from
an explicit execution plan rather than implicit sprint memory.

## Main Day 10 Conclusion

The direct repeated-run public contract is already designed and fenced. The
highest-value remaining design work is therefore operational:

- what later code days must validate
- what binaries are the most relevant targeted follow-ons
- what order minimizes public-surface churn and review risk

## Validation Contract For Later `*.c` / `*.h` Work

### Mandatory gate for any direct lifecycle code change

Later implementation days that modify `*.c` or `*.h` files must run:

- `make format`
- `make lint`
- `make test`

Reason:

- Sprint 50 is preparing public direct-solver lifecycle work, not an isolated
  internal refactor
- header wording, wrapper integration, and lifecycle semantics touch public
  contract surfaces and must clear the full baseline gate

### Stronger default for substantial public API batches

When a batch changes public direct lifecycle headers, core direct-solver
integration code, or repeated-run wrapper behavior, it should also run:

- `make quality-review-full`

Reason:

- this remains the strongest local reviewed baseline
- the later lifecycle work is exactly the kind of public-surface batch where
  reviewed parity should stay explicit

### Truthfulness anchors that should remain visible in substantial landing batches

For reviewed/public batches, the main explicit anchors remain:

- `ctest -N --test-dir build/quality-review-cmake`
- Makefile/CMake parity count
- full reviewed `ctest` pass count

These do not need to run on every docs-only day, but they should stay visible
for meaningful lifecycle landings and the later validation sweep.

## Targeted Follow-Ons For Later Implementation Sprints

The later public direct lifecycle work should rerun touched high-signal direct
surfaces rather than relying only on the monolithic baseline.

### Highest-signal repeated-run adoption surfaces

- `./build/example_analysis`
- `./build/bench_refactor`

Reason:

- these already model the explicit analysis/factor/refactor workflow directly
- they are the clearest caller-facing repeated-run compatibility surfaces

### Important backend/perf follow-on

- `./build/bench_refactor_csc`

Reason:

- it is not the first caller-teaching surface
- but it is still a direct repeated-run benchmark tied to the same analysis
  contract and should be rerun when analysis/refactor behavior is touched

### Core direct regression binaries

- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Reason:

- they cover the strongest family-level direct correctness and internal
  structural seams behind the public lifecycle story

### Optional focused reruns when specific files are touched

- if `include/sparse_analysis.h` or `src/sparse_analysis.c` changes:
  - `./build/example_analysis`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- if `include/sparse_cholesky.h` or Cholesky integration code changes:
  - `./build/test_cholesky`
  - `./build/test_chol_csc`
  - `./build/example_analysis`
- if `include/sparse_ldlt.h` or LDL^T integration code changes:
  - `./build/test_ldlt`
  - `./build/test_ldlt_csc`
- if public docs/examples/benchmarks are the only touched surfaces:
  - targeted binary reruns only where the changed docs claim behavior

## Intended Implementation Order

The later direct lifecycle landing should proceed in the following order.

### 1. Public headers / API surface

Primary likely targets:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Reason:

- the public contract should be visible and reviewable before source-level
  integration spreads across multiple files

### 2. Implementation and wrapper integration

Primary likely targets:

- `src/sparse_analysis.c`
- direct family integration points that need to align one-shot paths with the
  explicit repeated-run story

Reason:

- after the headers stabilize, implementation can align behavior with the
  contract without simultaneously discovering public semantics

### 3. High-signal example / benchmark adoption

Primary early adopters:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

Secondary or later adopters:

- `benchmarks/bench_refactor_csc.c`
- broader example/README surfaces

Reason:

- this keeps the first adoption batch aligned with the explicit repeated-run
  story and avoids broad low-signal surface churn

### 4. Compatibility sweep

Likely focus:

- public wording parity across headers and top-level docs
- example/benchmark alignment with the final repeated-run story
- narrow regression additions for the public lifecycle contract

### 5. Final validation

Expected shape:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- targeted follow-ons for touched direct surfaces

## Explicit Out-Of-Scope Notes For Sprint 50 And Immediate Sprint 51 Landing

### Still out of scope

- broad direct-solver API redesign beyond the Day 8 contract
- generic public direct-handle introduction as the main landing
- raw CSC/native storage exposure
- large tutorial rewrite
- broad benchmark framework redesign
- structural-pattern verifier redesign
- sweeping conversion of all direct examples to the repeated-run path

### Immediate Sprint 51 should also avoid

- solving benchmark/doc adoption everywhere at once
- reopening QR lifecycle scope
- mixing unrelated direct-solver feature expansion into the first lifecycle
  landing batch

## Landing Priorities

### Highest priority

- public contract alignment in headers
- implementation behavior that matches the contract
- repeated-run example/benchmark surfaces that already exist

### Medium priority

- selected docs adoption
- selected regression additions tied directly to the lifecycle contract

### Lower priority

- broader docs polish
- broader benchmark narrative cleanup

## Highest-Value Day 10 Conclusions

### 1. Later implementation sprints now have a clear mandatory validation baseline

`make format`, `make lint`, and `make test` are the non-negotiable code-day
gate, with `make quality-review-full` as the stronger default for substantial
public lifecycle batches.

### 2. The high-signal targeted follow-ons are now explicit

`example_analysis`, `bench_refactor`, `bench_refactor_csc`, and the direct
solver regression binaries are the right later rerun surfaces.

### 3. The landing order is now grounded in the live seam map

Headers first, then implementation, then high-signal adoption, then
compatibility sweep, then final validation.

### 4. Sprint 50 remains bounded even in its landing plan

The plan supports Sprint 51 implementation without quietly reopening
out-of-scope direct-solver redesign work.
