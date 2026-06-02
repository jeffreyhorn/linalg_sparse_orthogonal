# Sprint 53 Day 9: Cholesky / LDL^T Dispatch Reconciliation Audit

## Purpose

Day 9 audits the live public dispatch story after the Day 4-8 CSC batches.
The goal is not to reopen implementation work. The goal is to identify the
smallest high-signal follow-through targets that still matter before Sprint 53
closes its CSC completion and dispatch queue.

## Main Day 9 Result

The remaining Sprint 53 dispatch queue is now smaller than the original plan
implied:

- the strongest remaining public-story drift is in the top-level `README.md`
- the benchmark-local README is already aligned enough after Day 8
- `include/sparse_ldlt.h` is already the strongest public source of truth for
  the LDL^T CSC dispatch contract
- the CSC-specific regression comments are already aligned enough to leave
  alone unless Day 10 discovers a very small wording mismatch while touching a
  primary target

That leaves Day 10 with a bounded documentation target rather than a broad
"dispatch cleanup" bucket.

## Audit Scope

Day 9 checked the live dispatch story across:

- `README.md`
- `benchmarks/README.md`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`

## What Is Already Aligned Enough

### 1. `include/sparse_ldlt.h` is already the clearest public LDL^T dispatch contract

The LDL^T header already says the important things Sprint 53 needs it to say:

- AUTO vs forced backend selection is explicit
- forced CSC means the CSC pipeline, not an unconditional promise of batched
  supernodal completion
- `used_csc_path` reports the selected numeric path, not an idealized internal
  variant
- the `n == 0` empty-matrix exception is documented

Interpretation:

- Day 10 should not spend time restating this contract in a second header
- `include/sparse_ldlt.h` is already the authoritative family-local wording

### 2. `benchmarks/README.md` is aligned enough after Day 8

The benchmark-local README now correctly distinguishes:

- `bench_refactor`
  - Cholesky analyze-once / refactor-many proof
- `bench_refactor_csc`
  - default SPD / Cholesky repeated-run proof
  - optional indefinite LDL^T KKT repeated-run proof
  - public repeated-run path vs direct CSC completion path

Interpretation:

- the benchmark-local docs no longer look like the main remaining drift source
- Day 10 does not need to reopen benchmark-local wording unless a primary
  README change forces one tiny consistency edit

### 3. CSC-specific tests are already describing the LDL^T pipeline accurately enough

The LDL^T CSC tests and dispatch tests already reflect the current layered
story:

- the scalar BK pre-pass is the authoritative indefinite permutation-resolution
  step
- the CSC pipeline may retain batched completion or resolved scalar fallback
- helper contract violations are no longer silently treated as fallback

Interpretation:

- Sprint 53's regression comments are already closer to the implementation than
  the top-level README is
- tests are not the right Day 10 target unless a docs edit exposes one tiny
  naming mismatch

## Real Remaining Drift

### 1. `README.md` still compresses Cholesky and LDL^T dispatch into a story that is now too coarse

The top-level README still talks about the direct repeated-run workflow well,
but its CSC family wording is now lower-resolution than the live code:

- Cholesky dispatch is comparatively simple:
  - AUTO picks linked-list vs CSC by size
  - forced CSC means the CSC backend
- LDL^T dispatch is intentionally layered:
  - AUTO also picks by size
  - forced CSC means the CSC pipeline
  - that pipeline still begins from the scalar BK pre-pass
  - completion may retain the batched path or fall back to the resolved scalar
    factor

Interpretation:

- the main remaining public-story gap is not missing implementation
- it is top-level wording that still under-explains why the two families are
  intentionally similar at the outer dispatch layer but different internally

### 2. The top-level README still under-centers the new indefinite benchmark proof

Day 8 gave Sprint 53 a new measurable LDL^T factor-many proof surface, but the
top-level README still reads mostly from the older SPD / Cholesky benchmark
mental model.

Interpretation:

- Day 10 should add only a small benchmark-story reconciliation
- that change should point readers toward the benchmark-local README rather
  than duplicating the whole benchmark contract

## Acceptable Family-Local Differences

These are real differences and should remain explicit rather than being
"cleaned up" into fake symmetry:

### Cholesky

- SPD-only path
- no Bunch-Kaufman pivot resolution
- forced CSC means the CSC backend directly
- simpler `used_csc_path` meaning

### LDL^T

- symmetric indefinite path
- scalar BK pre-pass remains authoritative
- forced CSC means the CSC pipeline, not guaranteed batched completion
- `used_csc_path` reports CSC-pipeline selection even when completion falls
  back from the batched path to the resolved scalar factor

Interpretation:

- Day 10 should clarify these differences, not erase them

## Ranked Day 10 Targets

### 1. Primary target: `README.md`

Why first:

- it is now the highest-visibility place where the Cholesky / LDL^T dispatch
  story is still lower-resolution than the code and tests
- one bounded edit can improve both dispatch interpretation and benchmark
  interpretation without reopening large docs work

What to do:

- tighten the CSC dispatch wording so Cholesky and LDL^T are not described as
  if they had identical internal dispatch/completion behavior
- mention the new indefinite repeated-run proof surface briefly
- keep the details compact and point readers to:
  - `include/sparse_ldlt.h`
  - `benchmarks/README.md`

### 2. Secondary target only if the README patch exposes a tiny mismatch: one direct header touch

Likely candidates only if needed:

- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`

Constraint:

- only touch a header if the README clarification would otherwise contradict a
  local header sentence
- do not reopen `include/sparse_ldlt.h` unless a real contradiction appears

### 3. Leave tests and benchmark-local docs alone by default

Reason:

- they are already more precise than the remaining top-level README story

## Deferred Non-Goals

Day 9 explicitly defers:

- tutorial-scale direct-solver rewrite
- broad example rewrite beyond the already-aligned repeated-run example
- benchmark-framework redesign
- new public direct-solver API work
- attempts to make Cholesky and LDL^T dispatch read as perfectly symmetric
- reopening the scalar BK pre-pass as the authoritative indefinite
  permutation-resolution step

## Operational Result

Sprint 53's remaining reconciliation queue is now bounded:

1. the README is the real primary target
2. direct headers are mostly aligned already
3. benchmark-local docs are aligned enough
4. tests already reflect the intended layered LDL^T CSC contract

That is small enough for Day 10 to land one compact public-story batch instead
of drifting into broad documentation churn.
