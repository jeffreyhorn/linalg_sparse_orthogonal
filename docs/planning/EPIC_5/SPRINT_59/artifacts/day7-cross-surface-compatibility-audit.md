# Sprint 59 Day 7 - cross-surface compatibility audit

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Reduce the final Sprint 59 integration problem to named caller-story drift
classes across the strongest public workflow, example, benchmark, header, and
proof surfaces before the last reconciliation batch lands.

## Re-audited surfaces

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `tests/test_integration.c`

## Main findings

### 1. No blocker-level support-boundary contradiction remains

The main public surfaces still agree on the stable product fence:

- one-shot APIs remain the default/front-door workflows
- repeated direct solves use the explicit analysis/factors lifecycle
- repeated-run iterative handles remain limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handles remain limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

Conclusion:

- the final integration problem is no longer about conflicting support
  boundaries
- it is now about caller-story precision and top-level wording

### 2. Example/docs alignment is mostly coherent

The example-side story is already explicit:

- shipped examples still lean on one-shot APIs
- `example_analysis` is the strongest shipped repeated-run direct example
- `example_eigs` remains intentionally one-shot even though the repeated-run
  eigensolver handle exists
- iterative handles remain opt-in paths rather than default example flows

Conclusion:

- no example/docs contradiction justifies a new example or a broad example
  rewrite

### 3. Benchmark/docs alignment is also coherent

The benchmark docs already use stable workflow categories directly:

- direct repeated-run lifecycle
- iterative public-handle reuse
- eigensolver public-handle reuse

And the intentional exclusions are already explicit in
`benchmarks/README.md`.

Conclusion:

- benchmark/docs mismatch is no longer a high-priority drift class

### 4. The strongest remaining drift is top-level terminology/positioning

The strongest residual seam is concentrated in:

- `README.md`
- `docs/tutorial.md`

Pattern:

- top-level docs still sometimes say:
  - `repeated direct solves`
  - `repeated iterative solves`
  - `repeated symmetric eigensolves`
- while the tighter example/benchmark/header surfaces more consistently say:
  - explicit repeated-run direct lifecycle
  - explicit iterative handles
  - explicit eigensolver handle

Conclusion:

- the highest-value final reconciliation target is a bounded top-level
  terminology/positioning pass

### 5. Test/story mismatch is low-risk

`tests/test_integration.c` already proves the public lifecycle story with:

- repeated solve + zeroed free behavior
- same-pattern refactor parity
- indefinite repeated-run LDL^T proof
- failure-preservation and mismatch rejection cases

Conclusion:

- the tests are the right proof surfaces
- the top-level docs do not need to enumerate all of those regression details
- test/story mismatch should stay out of the final integration batch

## Drift classes

### Highest-value remaining drift

- top-level terminology/positioning mismatch:
  - `README.md`
  - `docs/tutorial.md`

### Lower-value residual drift

- example/docs emphasis seam
- benchmark/docs emphasis seam
- test/story density mismatch

## Proposed Day 8 landing boundary

The cleanest final integration target is:

- `README.md`
- `docs/tutorial.md`

Goals:

- align top-level terminology with the more precise stable vocabulary already
  used by headers, examples, and benchmark docs
- make repeated-run paths read more clearly as explicit opt-in lifecycle
  workflows

Non-goals:

- public-header cleanup batch
- example README rewrite
- benchmark README rewrite
- broad long-form README history cleanup
- test naming/proof-surface churn

## Conclusion

Day 7 closes with a concrete final integration map:

- no blocker-level support-boundary contradiction remains
- example/docs and benchmark/docs alignment are already mostly coherent
- the strongest remaining drift class is top-level terminology and workflow
  positioning in `README.md` and `docs/tutorial.md`
- test/story mismatch is low-risk and remains outside the final batch
