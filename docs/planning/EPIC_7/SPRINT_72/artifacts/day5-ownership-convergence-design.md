# Sprint 72 Day 5: Ownership Convergence Design

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Define the bounded implementation contract for Sprint 72's first convergence
batch so code work can improve direct-workflow ownership clarity without
widening into a broad matrix-model rewrite.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day4-first-product-model-boundary.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day11-epic7-target-synthesis.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day12-epic7-architecture-contract.md`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`

## Day 5 Design Conclusions

### 1. `SparseMatrix` keeps a bounded compatibility-shell role

The first Sprint 72 landing fixes `SparseMatrix` as the public owner of:

- mutable sparse construction and edit flow
- Matrix Market and generic interop shell behavior
- one-shot direct-workflow compatibility
- permutation-bearing matrix-shell publication for callers that still choose
  the one-shot lane
- factored-state compatibility markers needed by the one-shot lane

It should not keep reading like the owner of:

- reusable symbolic analysis
- long-lived factor/workspace state
- the best repeated-run direct workflow
- the long-term product identity of the strongest compressed direct paths

### 2. The repeated-run lifecycle is the clearer owner of reusable symbolic and factor/workspace state

The first convergence design fixes the repeated-run side as the clearer owner
of:

- reusable symbolic and permutation preparation
- same-pattern refactorable numeric flow
- explicit factor/workspace lifetime separate from the matrix shell
- the strongest cross-family long-run direct workflow

So the public relationship should read as:

- one-shot family lanes remain supported
- repeated-run analysis/factor surfaces are the clearer reuse lane
- the matrix shell is not the right long-term place to accumulate more solver
  ownership

### 3. The first code batch should target ownership language and factor-state mechanics

The best first bounded implementation batch is now explicit:

- clarify the ownership split in:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- tighten the matrix-shell ownership mechanics in:
  - `src/sparse_matrix.c`

The strongest likely Day 6-7 landing themes are:

- clearer invalidation/reset behavior around matrix mutation versus factored
  compatibility state
- clearer handoff wording between one-shot family APIs and repeated-run
  analysis/factor APIs
- clearer statement that compressed paths publish back through the matrix shell
  for compatibility, not because the matrix shell is the real long-lived
  factor owner

Explicitly not in the first batch:

- CSC or CSR conversion redesign
- compressed-path publication/writeback redesign
- new family-local factor types
- removal of existing one-shot public entry points

### 4. The first-batch non-touch set is now fixed

Sprint 72 Day 5 fixes the first-batch non-touch set:

- unrelated solver families outside the first ownership lane
- capability or type surfaces
- packaging/platform/install/workflow files
- broad public-doc cleanup spill
- giant proof-surface redesign
- deep compressed-path internal files unless the first batch truly forces a
  bounded follow-through

## Preserved Compatibility Checklist

- one-shot family entry points remain supported
- repeated-run analysis/factor surfaces remain the shared reuse lane
- `SparseMatrix` keeps factored/permutation compatibility behavior needed by
  existing callers
- no new capability claim is introduced
- no packaging/platform/install truth surface moves

## Exit State

Sprint 72 Day 5 closes with:

1. one explicit ownership convergence design
2. one fixed non-touch set for the first code batch
3. one preserved compatibility checklist for Day 6 implementation work
