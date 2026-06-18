# Sprint 78 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 78 large-source and giant-test maintainability baseline grounded in the live tree, the maintained reviewed validation contract, the Sprint 70 hotspot inventory, and the current permanent proof-owner surfaces.

### Actions
- Re-read the Sprint 78 plan in `docs/planning/EPIC_7/SPRINT_78/PLAN.md` and the Sprint 78 section in `docs/planning/EPIC_7/PROJECT_PLAN.md`.
- Re-read the strongest Sprint 70 hotspot and architecture-fence inputs:
  - `docs/planning/EPIC_7/SPRINT_70/artifacts/day8-support-surface-fence.md`
  - `docs/planning/EPIC_7/SPRINT_70/artifacts/day12-epic7-architecture-contract.md`
- Re-read the Sprint 77 closeout handoff in `docs/planning/EPIC_7/SPRINT_77/artifacts/day14-closeout-and-handoff.md`.
- Rechecked the maintained reviewed wrapper surface with `make -n quality-review-full`.
- Re-materialized the reviewed CMake parity tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Re-read the strongest current proof-ownership and documentation-policy support surfaces:
  - `docs/maintainer_guide.md`
  - `README.md`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 78 touch surfaces across implementation and permanent proof-owner tests.
- Rechecked the strongest likely giant-test queue terminology and chronology burden references with targeted `rg`.

### Findings
- Sprint 78 starts from the same strongest local reviewed baseline as Sprint 77:
  - `make quality-review-full`
- Reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The highest-value Sprint 78 pressure is now clearly narrowed to:
  - source hotspot rerank
  - bounded source decomposition
  - giant-test rerank
  - bounded giant-test architecture batch
  - chronology/comment cleanup on touched permanent files
  - docs and proof-ownership alignment
  - validation and closeout
- The strongest likely Sprint 78 implementation hotspots are now explicit from the live tree:
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_lu_csr.c` = `1665`
  - `src/sparse_chol_csc.c` = `1564`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_matrix.c` = `1125`
- The strongest likely Sprint 78 giant-test and permanent proof-owner hotspots are now explicit from the live tree:
  - `tests/test_chol_csc.c` = `4690`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_etree.c` = `2962`
  - `tests/test_graph.c` = `2925`
  - `tests/test_iterative.c` = `2841`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_svd.c` = `2766`
  - `tests/test_integration.c` = `2543`
  - `tests/test_reorder_nd.c` = `2287`
- The strongest likely support and policy surfaces are already clear before deeper audit:
  - `docs/maintainer_guide.md` remains the main proof-ownership and documentation-policy authority
  - `README.md` remains a lower-weight support surface that names stable maintainer-policy ownership without owning the full maintainability contract
- Sprint 70's hotspot work still matters directly to Sprint 78:
  - `tests/test_reorder_nd.c` remains the densest live chronology-heavy giant-test seam
  - `tests/test_chol_csc.c` remains the strongest family-local giant-test seam
  - Sprint 78 should spend large-source and giant-test budget only on the highest-value residual hotspots
- The maintained Sprint 78 non-goal fence is already clear and must be preserved:
  - no broad subsystem redesign disguised as maintainability cleanup
  - no giant-test breakup campaign across every proof owner
  - no chronology purge that removes durable algorithm explanation
  - no reopening of unrelated product, capability, backend, benchmark, packaging, or platform work

### Validation
- Rechecked `make -n quality-review-full`.
- Rebuilt the reviewed CMake tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live large-source and giant-test hotspot maps from direct `wc -l` measurement plus targeted terminology scans.

### Day 1 Exit State
- Sprint 78 no longer starts from a generic “cleanup big files” prompt.
- The implementation and giant-test workstreams, strongest likely touch surfaces, and preserved non-goal fence are fixed in writing.
- The branch is clean after the Day 1 baseline commit.

## Day 2 - Validation Baseline

### Goal
Reconfirm the Sprint 78 implementation-day validation contract and the live proof-surface split across reviewed CMake proof owners, giant-test surfaces, representative examples, and maintainer-policy ownership before any source or giant-test restructuring lands.

### Actions
- Re-read the Sprint 78 Day 2 plan expectations in `docs/planning/EPIC_7/SPRINT_78/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner tests and representative examples most likely to matter in Sprint 78:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_etree`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Re-read the strongest current maintainer-policy ownership section in `docs/maintainer_guide.md` for the already-fixed direct and proof-owner split.
- Reconfirmed the Sprint 70-derived giant-test queue and non-goal reading against the live Sprint 78 scope.

### Findings
- Sprint 78 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 78 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial maintainability or proof-architecture batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the key Sprint 78 proof surfaces most likely to be stressed by large-source or giant-test work:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_etree`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
- The representative example surfaces most likely to matter remain explicit:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- The strongest current proof-owner split is already fixed in policy:
  - `tests/test_reorder_nd.c` owns the shared ND compatibility/default-policy proof surface
  - `tests/test_chol_csc.c` owns the family-local large-`n` analysis-backed Cholesky CSC handoff and publish-back ownership surfaces
  - `tests/test_integration.c` owns the public one-shot vs explicit repeated-run Cholesky parity and matrix-shell reset boundaries
  - `tests/test_fuzz.c` remains bounded seeded generative follow-through for the large-`n` CSC-backed Cholesky lifecycle parity lane
- The highest-signal Sprint 78 rerun set is now fixed around the likely touched maintainability seams:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_etree`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most likely to matter in Sprint 78.
- Re-read the strongest current direct-family proof-ownership section in `docs/maintainer_guide.md`.
- Reconfirmed the Sprint 70-derived giant-test queue and non-goal reading remain aligned with Sprint 78.

### Day 2 Exit State
- Sprint 78 now has one explicit implementation-day validation contract.
- The live proof split across giant-test owners, public-oracle owners, and representative examples is fixed in writing.
- The highest-signal rerun set is explicit before source-hotspot and giant-test audit work begins.

## Day 3 - Source Hotspot Re-audit

### Goal
Re-rank the largest remaining implementation sources by real review pain, mixed ownership density, chronology burden, and bounded extraction value so Sprint 78 starts from one explicit current contradiction map instead of a generic “largest files first” list.

### Actions
- Re-read the Sprint 78 Day 3 plan expectations in `docs/planning/EPIC_7/SPRINT_78/PLAN.md`.
- Rechecked the live raw `wc -l` implementation hotspot map across `src/*.c`.
- Re-read the file-level ownership and section structure of the strongest likely Sprint 78 implementation hotspots:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_lu_csr.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_eigs.c`
  - `src/sparse_svd.c`
  - `src/sparse_matrix.c`
- Rechecked section-structure and helper density with targeted `rg` over those files.
- Re-read the preserved Sprint 70 architecture fence for large-source cleanup:
  - spend cleanup budget only on the highest-value residual hotspots
  - do not widen into broad subsystem redesign

### Findings
- Sprint 78's broad large-source problem is now reduced to one ranked contradiction map instead of one generic “big files are bad” bucket:
  - strongest first target:
    - `src/sparse_ldlt_csc.c`
  - strongest second target:
    - `src/sparse_iterative.c`
  - strongest third target:
    - `src/sparse_chol_csc.c`
  - strongest fourth target:
    - `src/sparse_lu_csr.c`
  - strong but lower-value or higher-risk later targets:
    - `src/sparse_qr.c`
    - `src/sparse_ldlt.c`
    - `src/sparse_eigs.c`
    - `src/sparse_svd.c`
    - `src/sparse_matrix.c`
- `src/sparse_ldlt_csc.c` is the strongest current contradiction center because it still combines too many durable roles in one permanent review surface:
  - CSC lifecycle and validation helpers
  - linked-list compatibility conversion and writeback
  - row-adjacency support
  - scalar elimination and solve
  - top-level orchestration for the batched supernodal LDL^T path
- That makes `src/sparse_ldlt_csc.c` the best first target by Sprint 78 rules:
  - highest mixed-ownership density
  - strongest extraction potential without reopening public API design
  - strong chronology burden from multiple sprint-era sections still living in one file
  - bounded proof risk because the family-local proofs already exist rather than needing a new proof family
- `src/sparse_iterative.c` is the strongest second target because it is very large and spans many solver families, but it is already more segmented than the LDL^T CSC lane:
  - handle/workspace lifecycle
  - shared stagnation and residual helpers
  - CG / GMRES / MINRES / BiCGSTAB
  - block and matrix-free variants
- The useful Day 3 distinction is that `src/sparse_iterative.c` is large and dense, but it already reads more like a big coherent family surface than a mixed compatibility/backend/orchestration bottleneck.
- `src/sparse_chol_csc.c` remains a real hotspot because it still owns:
  - conversion
  - scalar elimination
  - solve
  - factor/writeback publication
  - backend dispatch and compatibility shims
- But it ranks behind `src/sparse_ldlt_csc.c` because Sprint 72 and Sprint 75 already reduced some of its strongest ownership ambiguity, and its paired supernodal file has already absorbed more backend-local detail.
- `src/sparse_lu_csr.c` remains large and multi-section, but it reads more family-local and mechanically segmented:
  - conversion
  - structural validation
  - scalar and block elimination
  - solves
  - dense-block utilities
- The lower-ranked large sources are not equally urgent:
  - `src/sparse_qr.c` is large, but it reads more like one algorithm-family surface than the strongest mixed-ownership seam
  - `src/sparse_ldlt.c` is history-heavy, but the stronger residual maintainability pain has moved to the CSC-backed LDL^T lane
  - `src/sparse_eigs.c` is large, but its explicit backend/workspace split and extracted thick-restart file reduce first-batch payoff
  - `src/sparse_svd.c` still carries real chronology debt, but it is narrower and lower-value than the direct-factor and iterative hotspots
  - `src/sparse_matrix.c` is permanent and large, but Sprint 72 and Sprint 74 already reduced its strongest ownership contradiction, and reopening it now would carry higher compatibility risk than the stronger family-local hotspots
- The strongest current large-source contradiction classes are now explicit:
  - mixed orchestration plus numeric/detail ownership in one file
  - helper density without one obvious extracted seam
  - chronology/comment spill from multiple sprint-era landings
  - compatibility/writeback mechanics living beside core numeric kernels

### Validation
- Rechecked the live `wc -l` source hotspot map.
- Re-read the file headers and first section structures of the strongest likely implementation hotspots.
- Rechecked section boundaries and helper density with targeted `rg`.
- Reconfirmed the Sprint 70 architecture fence still requires bounded, payoff-driven cleanup rather than a broad breakup pass.

### Day 3 Exit State
- Sprint 78 now has one explicit implementation-hotspot ranking instead of a generic large-source backlog.
- `src/sparse_ldlt_csc.c` is fixed as the strongest first implementation hotspot.
- Day 4 can now freeze a real first source boundary from the current maintainability map.

## Day 4 - Source Boundary Freeze

### Goal
Refine the Day 3 source ranking into one explicit first Sprint 78 implementation fence so the next design pass starts from a bounded LDL^T CSC ownership lane rather than a mixed large-source and giant-test backlog.

### Actions
- Re-read the Sprint 78 Day 4 plan expectations in `docs/planning/EPIC_7/SPRINT_78/PLAN.md`.
- Re-read the Day 3 source-hotspot audit and reranked it against:
  - review payoff
  - extraction clarity
  - compatibility risk
  - bounded Sprint 78 value
- Re-read `src/sparse_ldlt_csc_internal.h` to confirm the current internal contract split between:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_supernodal.c`
- Re-read the current file header in `src/sparse_ldlt_csc_supernodal.c` to check whether that extracted cluster already has the right ownership shape.
- Rechecked the strongest likely proof-owner and public-oracle support surfaces:
  - `tests/test_ldlt_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`

### Findings
- Sprint 78 now has one explicit first implementation fence instead of a generic large-source backlog:
  - required first landing:
    - `src/sparse_ldlt_csc.c`
    - `src/sparse_ldlt_csc_internal.h`
  - support only if the first landing forces it:
    - `src/sparse_ldlt_csc_supernodal.c`
    - `tests/test_ldlt_csc.c`
    - `tests/test_ldlt.c`
    - `tests/test_integration.c`
    - `docs/maintainer_guide.md`
  - explicitly deferred:
    - `src/sparse_iterative.c`
    - `src/sparse_chol_csc.c`
    - `src/sparse_lu_csr.c`
    - giant-test architecture work
    - public-header or API-surface cleanup
- The strongest first Sprint 78 fence is now fixed as an implementation decomposition and helper-ownership lane, not chronology cleanup first and not a broader direct-solver family rewrite.
- `src/sparse_ldlt_csc_internal.h` belongs in the first landing because the strongest likely cleanup value is not merely moving lines out of `src/sparse_ldlt_csc.c`.
- It is clarifying which helpers and contracts are genuinely local to:
  - scalar/native LDL^T CSC ownership
  - shared CSC LDL^T internal contract
  - already-extracted supernodal helper ownership
- `src/sparse_ldlt_csc_supernodal.c` stays support-only because its header and file-level ownership already read comparatively well:
  - extracted dense-panel path
  - extract/writeback
  - diagonal-block factor/update
  - panel solve
  - driver-local row-map helper
- That makes it the wrong place to start unless the Day 6 landing truly needs internal contract follow-through there.
- The strongest likely proof owner is `tests/test_ldlt_csc.c`, but it stays support-only for the first boundary because the first batch target is implementation maintainability, not proof taxonomy yet.
- `tests/test_ldlt.c` and `tests/test_integration.c` remain lower-weight support surfaces because the first batch should not change the public LDL^T contract.
- `docs/maintainer_guide.md` remains support-only because no proof-ownership or documentation-placement contradiction has been forced yet.
- The strongest non-goal fence is now explicit:
  - no broad LDL^T algorithm rewrite
  - no public API redesign
  - no helper explosion without ownership gain
  - no giant-test cleanup pulled into the first implementation batch
  - no unrelated backend, capability, benchmark, packaging, or platform work

### Validation
- Re-read the Day 3 source-hotspot ranking.
- Re-read the LDL^T CSC internal and supernodal companion ownership surfaces.
- Rechecked the strongest likely proof-owner and policy support files for whether they truly belong in the first batch.
- Reconfirmed the Sprint 70 and Sprint 78 non-goal fence still requires bounded cleanup with clear payoff.

### Day 4 Exit State
- The first Sprint 78 implementation fence is explicit before design begins.
- The required first landing is fixed to the LDL^T CSC implementation and internal contract lane.
- Lower-value or higher-risk source and proof work is now clearly separated from the first batch.

## Day 5 - Source Decomposition Design

### Goal
Define one explicit bounded implementation contract for the first Sprint 78 source decomposition batch so Day 6 lands a real LDL^T CSC ownership cleanup rather than a mechanical file split.

### Actions
- Re-read the Sprint 78 Day 5 plan expectations in `docs/planning/EPIC_7/SPRINT_78/PLAN.md`.
- Re-read the Day 4 source boundary against the first-batch files:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
- Re-read the current ownership split across:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `src/sparse_ldlt_csc_supernodal.c`
- Re-read the strongest current writeback, validation, wrapper, and supernodal-helper clusters inside `src/sparse_ldlt_csc.c`.
- Rechecked the strongest likely proof-sensitive support surfaces:
  - `tests/test_ldlt_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`

### Findings
- Sprint 78 now has one explicit first implementation contract:
  - the first batch should stay inside the LDL^T CSC implementation and internal-contract lane
  - it should clarify ownership, not widen the subsystem
  - it should prefer bounded helper extraction or regrouping around one obvious seam rather than a broad multi-file breakup
- The strongest Day 6 ownership split is now fixed:
  - orchestration owner:
    - `src/sparse_ldlt_csc.c`
  - helper/internal contract owner:
    - `src/sparse_ldlt_csc_internal.h`
    - one extracted helper cluster if the cleanup truly needs it
  - already-extracted supernodal dense-panel owner:
    - `src/sparse_ldlt_csc_supernodal.c`
  - proof-sensitive boundary owner:
    - `tests/test_ldlt_csc.c` only if a local helper extraction changes family-local test touchpoints
- The best Day 6 seam is not the supernodal helper file.
- It is the mixed scalar/native plus compatibility/writeback cluster still living in `src/sparse_ldlt_csc.c`, especially around:
  - conversion paths
  - writeback/public transplant path
  - validation and wrapper/orchestration glue
  - native-kernel helper cluster and row-adjacency support
- The preserved Day 6 compatibility checklist is now explicit:
  - preserve current public LDL^T behavior and error surface
  - preserve current family-local scalar/native versus batched-supernodal execution behavior
  - preserve current proof ownership:
    - no new proof owner implied by implementation cleanup alone
  - preserve local helper visibility and current call sequencing where behavior depends on it
  - preserve the current reviewed validation path and rerun set fixed on Day 2
- The exact first-batch non-touch set is also fixed:
  - no unrelated giant-test cleanup
  - no public-header or API-surface edits
  - no broader chronology scrub outside the touched implementation seam
  - no iterative, Cholesky CSC, LU CSR, eigensolver, benchmark, packaging, or platform work
  - no wider taxonomy cleanup across other direct-solver families
- The strongest practical design conclusion is that Day 6 should aim for a bounded implementation ownership cleanup of the LDL^T CSC local seam, not a maximal split:
  - enough extraction or regrouping to reduce mixed-role review pressure
  - not so much movement that the batch becomes a subsystem rewrite or a proof-tax multiplier

### Validation
- Re-read the Day 4 first-source boundary.
- Re-read the strongest first-batch implementation and internal-contract surfaces.
- Rechecked the likely proof-sensitive support files to keep them support-only by default.
- Reconfirmed the Day 2 validation contract and Sprint 78 non-goal fence remain unchanged.

### Day 5 Exit State
- The first Sprint 78 source batch is explicitly designed before edits begin.
- The landing is bounded to LDL^T CSC implementation ownership cleanup.
- Compatibility, validation, and non-goal fences are fixed in writing before Day 6 code changes.
