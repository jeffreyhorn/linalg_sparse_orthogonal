# Sprint 63 Day 3: Internal Path Audit

Date: 2026-06-10
Branch: sprint-63

## Purpose

Reduce the broad Sprint 63 “direct-lifecycle uniformity and CSC/LU
follow-through” claim to a ranked live seam map before choosing the first
bounded implementation target.

## Ranked Audit

### 1. LU lifecycle follow-through is the strongest first target

The LU surfaces are materially improved after Sprint 62, but they still carry
the strongest remaining lifecycle crossover:

- the public wrapper story now clearly distinguishes:
  - one-shot direct use on a fresh matrix or `sparse_copy()`
  - repeated-run reuse through the explicit `analysis` / `factors` lifecycle
- reordered one-shot LU already preserves the caller matrix on cancel/failure
- the implementation still contains the strongest wrapper-to-lifecycle bridge
  through the default-compatible shared-lifecycle path and publish-back logic

Strongest live files:

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`
- `tests/test_sparse_lu.c`

Why this ranks first:

- the remaining LU problem is not broad public confusion anymore
- it is lifecycle/result/factor-state coherence where the one-shot wrapper and
  shared repeated-run machinery still meet

### 2. Cholesky owns the strongest CSC repeated-run uniformity seam

The Cholesky public story is also materially cleaner now:

- reordered one-shot preservation is already hardened
- the header already states the shipped reordered-path preservation rule
- the explicit repeated-run lifecycle remains the stable public reuse path

The remaining strongest asymmetry is behind the public interface:

- linked-list versus CSC path differences
- CSC conversion and write-back behavior
- analysis-aware repeated-run coherence on the CSC side
- backend/working-format dispatch that is less uniform internally than the
  public lifecycle story suggests

Strongest live files:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Why this ranks second:

- this is now more of an internal repeated-run uniformity seam than a basic
  caller-story defect
- it is the strongest current CSC follow-through target after LU

### 3. LDL^T is lower-risk and should stay later unless contradicted

The LDL^T surfaces are less urgent than LU and Cholesky for this sprint:

- the family-local ownership model is already explicit
- the one-shot path is less entangled with the shared lifecycle than LU
- CSC complexity exists, but it is not the strongest first contradiction in
  the current direct repeated-run story

Strongest live files:

- `include/sparse_ldlt.h`
- `src/sparse_ldlt.c`
- `src/sparse_ldlt_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`

Why this stays later:

- Sprint 63 should not widen into LDL^T merely to make the direct families
  look symmetrical
- it should move only if a later design pass exposes a concrete contradiction
  that LU and Cholesky do not already cover

### 4. QR remains a comparison/deferred surface

QR still matters as a contrast surface for caller expectations, but it is not
the right first target for Sprint 63. The current live direct-lifecycle
pressure is more concrete in LU and the CSC-backed Cholesky path.

## Proof Surface Ranking

The current proof burden already has a natural home:

1. `tests/test_integration.c`
2. `tests/test_chol_csc.c`
3. `tests/test_sparse_lu.c`
4. later workflow-proof follow-through:
   - `examples/example_analysis.c`
   - `benchmarks/bench_refactor.c`

Implication:

- Sprint 63 does not need a new lifecycle test harness
- the next step should be a bounded design pass over the existing integration
  and family-local proof homes

## Day 4 Target

The exact Day 4 design target is now fixed:

1. design the first LU lifecycle follow-through landing
2. fix the second-cut Cholesky/CSC repeated-run fence
3. keep LDL^T in the later lane unless needed
4. keep QR deferred

## Exit State

Sprint 63 now has a ranked live seam map instead of a generic lifecycle
uniformity backlog:

- LU is the strongest first implementation target
- Cholesky CSC repeated-run uniformity is the strongest second target
- LDL^T is cleaner and lower-risk than the sprint headline implied
- the proof burden already has a clear home in the current integration and
  family-local surfaces
