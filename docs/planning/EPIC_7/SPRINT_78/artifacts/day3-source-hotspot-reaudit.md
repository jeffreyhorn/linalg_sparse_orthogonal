# Sprint 78 Day 3 - Source Hotspot Re-audit

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Re-rank the largest remaining implementation files by review pain, ownership ambiguity, chronology burden, and bounded extraction value so Sprint 78 starts from a real current hotspot map rather than from raw size alone.

## Main Result
Sprint 78's broad large-source problem is now reduced to one ranked contradiction map instead of one generic “largest files first” bucket.

The strongest current implementation-hotspot ranking is:
- first:
  - `src/sparse_ldlt_csc.c`
- second:
  - `src/sparse_iterative.c`
- third:
  - `src/sparse_chol_csc.c`
- fourth:
  - `src/sparse_lu_csr.c`
- later / lower-value or higher-risk targets:
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_eigs.c`
  - `src/sparse_svd.c`
  - `src/sparse_matrix.c`

## Why the First Target Is `src/sparse_ldlt_csc.c`
`src/sparse_ldlt_csc.c` is the strongest remaining contradiction center because it still combines too many durable roles in one permanent review surface:
- CSC lifecycle and validation helpers
- linked-list compatibility conversion and writeback
- row-adjacency support
- scalar elimination and solve
- top-level orchestration for the batched supernodal LDL^T path

That makes it the best first Sprint 78 source target:
- highest mixed-ownership density
- strongest bounded extraction potential
- real chronology/comment burden from multiple sprint-era sections
- strong review payoff without forcing a public API redesign

## Ranked Contradiction Map
### `src/sparse_iterative.c`
Large and dense, but already more segmented than the LDL^T CSC lane:
- handle/workspace lifecycle
- shared stagnation and residual helpers
- CG / GMRES / MINRES / BiCGSTAB
- block and matrix-free variants

It is still a major hotspot, but it reads more like a very large coherent family surface than the strongest mixed compatibility/orchestration seam.

### `src/sparse_chol_csc.c`
Still a real hotspot because it owns:
- conversion
- scalar elimination
- solve
- factor/writeback publication
- backend dispatch and compatibility shims

It ranks behind `src/sparse_ldlt_csc.c` because earlier Epic 7 work already reduced some of its strongest ambiguity and because more backend-local detail has already moved into the paired supernodal file.

### `src/sparse_lu_csr.c`
Still large, but comparatively more mechanical and family-local:
- conversion
- structural validation
- scalar and block elimination
- solves
- dense-block utilities

That lowers first-batch payoff relative to the CSC LDL^T and Cholesky seams.

## Lower-Ranked but Important Later Targets
- `src/sparse_qr.c` is large, but it reads more like one algorithm-family surface than the strongest mixed-role contradiction center.
- `src/sparse_ldlt.c` is history-heavy, but the stronger residual maintainability pressure has moved to the CSC-backed LDL^T lane.
- `src/sparse_eigs.c` is large, but the explicit backend/workspace split and extracted thick-restart file reduce first-batch payoff.
- `src/sparse_svd.c` still carries real chronology debt, but it is narrower and lower-value than the direct-factor and iterative hotspots.
- `src/sparse_matrix.c` is permanent and large, but its strongest ownership contradiction was already reduced earlier in Epic 7 and reopening it now would carry higher compatibility risk.

## Current Contradiction Classes
The strongest large-source contradiction classes are now explicit:
- mixed orchestration plus numeric/detail ownership in one file
- helper density without one obvious extracted seam
- chronology/comment spill from multiple sprint-era landings
- compatibility/writeback mechanics living beside core numeric kernels

## Exit State
- The broad Sprint 78 source problem is reduced to a concrete seam ranking.
- `src/sparse_ldlt_csc.c` is fixed as the strongest first implementation hotspot.
- Day 4 can now proceed from a real current-state source ranking.
