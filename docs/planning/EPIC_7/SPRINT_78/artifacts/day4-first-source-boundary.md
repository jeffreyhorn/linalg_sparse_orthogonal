# Sprint 78 Day 4 - First Source Boundary

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Freeze the first Sprint 78 implementation fence so the sprint starts from one bounded LDL^T CSC maintainability lane rather than from a mixed large-source, giant-test, and chronology backlog.

## Main Result
Sprint 78 now has one explicit first implementation fence instead of a generic large-source backlog.

Required first landing:
- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_internal.h`

Support only if the first landing forces it:
- `src/sparse_ldlt_csc_supernodal.c`
- `tests/test_ldlt_csc.c`
- `tests/test_ldlt.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Explicitly deferred:
- `src/sparse_iterative.c`
- `src/sparse_chol_csc.c`
- `src/sparse_lu_csr.c`
- giant-test architecture work
- public-header or API-surface cleanup

## Boundary Interpretation
The strongest first Sprint 78 fence is now fixed as an implementation decomposition and helper-ownership lane:
- not chronology cleanup first
- not giant-test cleanup first
- not a broader direct-solver family rewrite

The useful Day 4 clarification is explicit:
- `src/sparse_ldlt_csc.c` is still the strongest first target because it mixes scalar/native kernel ownership, compatibility/writeback mechanics, lifecycle helpers, and top-level batched-supernodal orchestration.
- `src/sparse_ldlt_csc_internal.h` belongs in the first batch because the most valuable cleanup is clarifying the internal ownership boundary itself, not just moving lines mechanically.
- `src/sparse_ldlt_csc_supernodal.c` stays support-only because it already reads as an extracted, comparatively well-bounded helper cluster.

## Support-Surface Reading
- `tests/test_ldlt_csc.c` is the strongest likely proof owner if the first batch forces local regression updates, but it is not a first-batch center by default.
- `tests/test_ldlt.c` and `tests/test_integration.c` stay lower-weight support surfaces because the first batch should preserve the current public LDL^T contract.
- `docs/maintainer_guide.md` remains support-only because no new proof-ownership contradiction has been forced yet.

## Strongest Non-Goals
- no broad LDL^T algorithm rewrite
- no subsystem or public API redesign
- no helper explosion without clear ownership gain
- no giant-test cleanup pulled into the first implementation batch
- no unrelated backend, capability, benchmark, packaging, or platform work

## Exit State
- The first Sprint 78 implementation fence is explicit before design begins.
- The LDL^T CSC implementation and internal-contract lane is fixed as the required first batch.
- Lower-value or higher-risk source and proof work is clearly separated from the first landing.
