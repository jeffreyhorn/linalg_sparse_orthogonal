# Sprint 62 Day 8: Post-LU Lifecycle/Wrapper Audit

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Re-audit the remaining direct wrapper/lifecycle queue after the Day 6-7 LU
landing so Sprint 62 can move from a broad direct-usability backlog to one
exact next convergence slice.

## Audit Scope

### Re-read sources

- `docs/planning/EPIC_6/SPRINT_62/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
- `docs/planning/EPIC_6/SPRINT_62/artifacts/day7-one-shot-hardening-batch2.md`

### Live direct-family surfaces reviewed

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `tests/test_integration.c`
- `tests/test_cholesky.c`
- `tests/test_ldlt.c`
- top-level direct-usage wording in:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`

## Main Result

### 1. LU is no longer the strongest remaining direct-usability seam

Days 6-7 already landed the highest-value LU usability fixes:

- reused one-shot state now rejects explicitly
- reordered LU one-shot failure/cancel no longer publishes partial caller
  mutation
- the public LU header now states the strengthened one-shot rule truthfully

That means Sprint 62 should not keep treating LU as the default next target.

### 2. Cholesky now owns the strongest remaining wrapper/lifecycle mismatch

The Cholesky one-shot path still carries the strongest live risk combination:

- the API remains in-place and mutation-first
- reordered factorization still publishes the permuted working state onto the
  caller matrix before numeric success is known
- linked-list cancel-at-step-0 is documented as leaving a non-bit-identical
  matrix
- CSC path progress semantics still lag behind the linked-list path

This is now the clearest next Sprint 62 convergence target.

### 3. LDL^T is mostly a compatibility-follow-through surface for this sprint

LDL^T is materially cleaner than LU and Cholesky on the core Sprint 62 axis:

- family-local factor state lives in `sparse_ldlt_t`
- factorization does not mutate the input matrix
- cancellation already preserves the input matrix bit-identically
- repeated-run convergence already maps naturally to the shared
  `analysis` / `factors` lifecycle

That makes LDL^T a low-priority code target for the rest of Sprint 62.

## Ranked Remaining Queue

### Move now in Sprint 62

- bounded Cholesky one-shot mutation/publication hardening
- bounded Cholesky cancel/lifecycle wording follow-through

### Compatibility-only for now

- LU wording follow-through only where touched later
- LDL^T wording normalization if needed for the final sprint story
- preservation of the existing one-shot family boundaries

### Defer beyond Sprint 62

- backend-wide CSC progress callback parity
- QR convergence work
- hidden-copy semantics that erase one-shot mutation
- broad direct-family API redesign

## Exact Day 9 Target

The next bounded design batch should target:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

Optional only if the proof burden forces it:

- `tests/test_cholesky.c`
- small touched wording in `README.md` / `docs/tutorial.md`

The Day 9 design question is:

- how much of the LU Day 6-7 preservation model should move onto Cholesky
- which existing Cholesky cancellation/publication semantics remain
  compatibility-only
- how to reduce caller surprise without blurring the line between one-shot
  direct usage and the explicit repeated-run lifecycle

## Non-Goals Reconfirmed

- no reopening the repeated-run workflow fence
- no broad configuration-surface rewrite
- no packaging/platform spillover
- no broad QR or backend-policy batch
- no fake convergence via hidden automatic copies

## Day 8 Exit State

Sprint 62’s remaining direct-usability queue is now smaller and concrete:

- LU first-package work is complete enough to leave alone
- Cholesky is the strongest next code target
- LDL^T is mostly a compatibility-follow-through lane
- Day 9 can proceed from one exact bounded Cholesky convergence design
