# Sprint 49 Day 7 Artifact: Post-Landing API Audit

## Purpose

Re-audit the public lifecycle/workspace surface after the Day 5/6 landing so
the remaining Sprint 49 queue is reduced to concrete migration and
compatibility work rather than a generic final-integration bucket.

## Main Day 7 Conclusion

The public lifecycle landing itself is complete and still bounded.

The remaining Sprint 49 work is no longer “finish the API.” It is now:

- migration-path documentation
- cross-surface agreement between docs/examples/benchmarks/tests
- final residual-review bookkeeping

That is the right post-landing state for Epic 4 closeout.

## Audited Surfaces

### Public contract and implementation

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

### Caller-facing and compatibility surfaces

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `tests/`

## What Is Complete

### Public lifecycle exposure is real

Day 5 and Day 6 together now provide:

- public iterative repeated-run handles
- public eigensolver repeated-run handles
- working implementation behind those handles
- compatibility-preserving one-shot wrapper routing

Interpretation:

- callers now have a supported explicit repeated-run API
- the public surface is no longer only declared or only internal

### The landing stayed bounded

No meaningful API sprawl showed up in the live repo state:

- no public matrix-free repeated-run exposure
- no public block/MINRES/BiCGSTAB repeated-run exposure
- no raw internal workspace owners or typed internal views exposed
- no broad public solver-family redesign

That means the Day 5/6 fence held.

## What Is Still Missing

The strongest remaining gaps are now outside the core public headers and source
implementation.

### 1. Migration-path documentation is still missing

The public handle names appear in:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- Sprint 49 notes/artifacts

They do **not** yet appear in the main user-facing repo guidance.

Current consequences:

- `README.md` still does not explain the old one-shot path vs the new explicit
  handle path
- there is not yet a concrete “when reuse is worth it” explanation
- the handle lifecycle is documented in headers, but not yet in the main caller
  guidance

### 2. Cross-surface agreement is incomplete

Examples and benchmark docs still mostly present the older one-shot story:

- `examples/README.md` still reads as one-shot public usage guidance
- `benchmarks/README.md` documents the benchmark surface but not the final
  caller-facing repeated-run handle story

This is not a correctness problem, but it is now the main coherence gap.

### 3. Reuse benchmarks still use internal seams

The two repeated-run benchmark drivers still measure internal reuse paths:

- `bench_iterative_reuse.c`
  - `sparse_solve_*_with_workspace_internal(...)`
- `bench_eigs_reuse.c`
  - `sparse_eigs_sym_with_workspace_internal(...)`

Interpretation:

- this is acceptable as historical evidence for the internal seam
- but it is no longer the final caller-facing repeated-run story after Day 5/6
- Day 9/10 should decide whether the public handle path should become the
  benchmarked path, or whether these drivers remain intentionally internal
  evidence with clarified wording

### 4. Direct public-handle regression coverage is absent

The live `tests/` tree currently has no direct references to:

- `sparse_iter_handle_*`
- `sparse_solve_*_with_handle(...)`
- `sparse_eigs_handle_*`
- `sparse_eigs_sym_with_handle(...)`

Interpretation:

- the required/public-API validation from Day 5/6 proved no regressions through
  the one-shot path and family-level behavior surfaces
- but the final public repeated-run contract still lacks direct behavior-level
  regression coverage

That is a strong Day 9/10 candidate.

## Naming and Ownership Drift

The public headers themselves do not show meaningful naming drift.

The real remaining ownership drift is cross-surface:

- public headers now define the final repeated-run caller contract
- benchmarks still describe repeated-run evidence in internal-workspace terms
- README/examples do not yet present the public repeated-run handle path as a
  supported caller-facing option

That is a documentation and compatibility issue, not another core API issue.

## Day 8 Boundary

Day 8 should be a migration-path documentation batch, not a broad sweep.

Strongest likely targets:

- `README.md`
- one small supporting doc surface if needed:
  - `examples/README.md`
  - or a tightly bounded cross-reference update elsewhere

Day 8 should explain explicitly:

- one-shot calls remain fully supported
- explicit handles are the repeated-run opt-in path
- reuse preserves allocation capacity, not old numerical iteration state
- repeated-run handles are most relevant for stable-dimension repeated solves

## Day 9/10 Boundary

The compatibility sweep should stay narrow and high-signal.

Strongest candidate surfaces:

- repeated-run benchmarks
- direct public-handle regression coverage in `tests/`
- any nearby docs/examples wording still contradicting the final caller story

Important non-goal:

- do not turn the sweep into a broad benchmark or example redesign

## Bottom Line

Day 7 shows that Sprint 49 is no longer blocked on public API exposure.

What remains is now concrete:

- document the old-vs-new caller path
- reconcile the highest-value examples/benchmarks/tests/docs
- finish the final residual-review and validation closeout from there

That is the right narrowed queue for the back half of Sprint 49.
