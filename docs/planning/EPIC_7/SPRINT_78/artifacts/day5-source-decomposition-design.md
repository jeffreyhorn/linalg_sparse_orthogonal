# Sprint 78 Day 5 - Source Decomposition Design

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Define the bounded implementation contract for the first Sprint 78 source decomposition batch before any code moves begin.

## Main Result
Sprint 78 now has one explicit first implementation contract:
- stay inside the LDL^T CSC implementation and internal-contract lane
- clarify ownership rather than widen the subsystem
- prefer one bounded helper extraction or regrouping seam instead of a broad breakup

## Day 6 Ownership Split
The strongest Day 6 ownership split is now fixed:
- orchestration owner:
  - `src/sparse_ldlt_csc.c`
- helper/internal contract owner:
  - `src/sparse_ldlt_csc_internal.h`
  - one extracted helper cluster only if the cleanup truly needs it
- already-extracted supernodal dense-panel owner:
  - `src/sparse_ldlt_csc_supernodal.c`
- proof-sensitive boundary owner:
  - `tests/test_ldlt_csc.c` only if local helper extraction truly forces family-local regression touchpoints

## Best First Seam
The best Day 6 seam is not the supernodal helper file.

It is the mixed scalar/native plus compatibility/writeback cluster still living in `src/sparse_ldlt_csc.c`, especially around:
- conversion paths
- writeback/public transplant path
- validation and wrapper/orchestration glue
- native-kernel helper cluster and row-adjacency support

The design intent is bounded maintainability payoff:
- reduce mixed-role review pressure
- clarify which helpers are local implementation detail versus stable internal contract
- avoid turning the batch into a subsystem rewrite or proof-tax multiplier

## Preserved Compatibility Checklist
Day 6 must preserve:
- current public LDL^T behavior and error surface
- current family-local scalar/native versus batched-supernodal execution behavior
- current proof ownership:
  - no new proof owner implied by implementation cleanup alone
- local helper visibility and current call sequencing where behavior depends on it
- the Day 2 reviewed validation contract and rerun set

## Exact First-Batch Non-Touch Set
- no unrelated giant-test cleanup
- no public-header or API-surface edits
- no broader chronology scrub outside the touched implementation seam
- no iterative, Cholesky CSC, LU CSR, eigensolver, benchmark, packaging, or platform work
- no wider taxonomy cleanup across other direct-solver families

## Exit State
- The first Sprint 78 source batch is explicitly designed before edits begin.
- The landing is bounded to LDL^T CSC implementation ownership cleanup.
- Compatibility and non-goal fences are fixed in writing before Day 6 code changes.
