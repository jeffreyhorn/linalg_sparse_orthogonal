# Sprint 53 Day 3: Analysis-Aware Indefinite Path Audit

## Purpose

Day 3 audits the live analysis-aware LDL^T indefinite CSC path so Sprint 53
can start implementation from named fallback, dispatch, and proof seams rather
than from a generic “complete the path” instruction.

## Main Day 3 Conclusion

The shared analysis-aware indefinite CSC path is already real, but its
remaining cost and clarity issues are concentrated in one place:

- `sparse_factor_numeric(...)` already routes large LDL^T problems through a
  dedicated shared helper
- the helper already reuses caller analysis directly when the scalar BK
  pre-pass permutation matches the caller analysis
- the helper still owns the expensive fallback sequence when that permutation
  diverges:
  - scalar pre-pass
  - pre-permute
  - reanalyze
  - rebuild CSC factor
  - attempt supernodal factor
  - fall back to scalar factor if needed

That means Sprint 53 does not need to invent the path. It needs to reduce or
better bound the hidden orchestration still clustered inside that helper and
strengthen proof around the intended indefinite repeated-run workloads.

## What Already Exists

### Shared repeated-run direct surface

`include/sparse_analysis.h` already documents the real current contract:

- Cholesky CSC path directly reuses caller analysis on larger repeated-run
  problems
- LDL^T CSC path directly reuses caller analysis when the scalar BK pre-pass
  does not introduce extra swaps
- otherwise it rebuilds symbolic analysis only on the pre-permuted matrix

That is not a placeholder contract. It matches the live implementation in
`src/sparse_analysis.c`.

### Shared implementation path

`src/sparse_analysis.c` already provides:

- `factor_ldlt_with_analysis_csc(...)`
- `perm_matches_analysis_reorder(...)`
- `sparse_validate_analysis_input_matrix(...)`

So the shared public repeated-run path already has:

- analysis-family validation
- dimension validation
- original-state validation
- cheap `source_nnz` gross-structure validation
- direct LDL^T CSC routing above `SPARSE_CSC_THRESHOLD`

### One-shot public dispatch path

`src/sparse_ldlt.c` already provides a coherent one-shot public LDL^T CSC
story:

- AUTO / LINKED_LIST / CSC dispatch
- `used_csc_path` telemetry
- scalar pre-pass + pre-permute + analyze +
  `ldlt_csc_from_sparse_with_analysis(...)`
- supernodal attempt with scalar fallback
- writeback to public `sparse_ldlt_t`

This means the user-facing one-shot surface is already operational even though
its CSC orchestration is still somewhat duplicated relative to the shared path.

## Main Residual Seams

### 1. Shared-path fallback still rebuilds too much state

`factor_ldlt_with_analysis_csc(...)` still performs a full scalar BK pre-pass
every time, and when the resulting permutation diverges from the caller's
analysis it:

1. builds a pre-permuted matrix
2. reruns `sparse_analyze(..., REORDER_NONE)`
3. rebuilds the CSC factor from that derived analysis

This is the strongest remaining implementation seam because it directly
governs both first factorization and the repeated-run refresh path.

### 2. `sparse_refactor_numeric(...)` still re-enters the same full helper

The refactor path is still a safe wrapper around a fresh call to
`sparse_factor_numeric(...)`, so the LDL^T CSC repeated-run story still pays
the same orchestration costs on every refactor:

- scalar BK pre-pass
- possible pre-permute + reanalysis
- CSC factor rebuild
- possible supernodal fallback

That is now the clearest factor-many seam for Sprint 53.

### 3. Shared and one-shot CSC orchestration still live in parallel

The one-shot CSC path in `ldlt_factor_csc_path(...)` and the shared repeated-
run CSC path in `factor_ldlt_with_analysis_csc(...)` both own similar
indefinite orchestration logic:

- scalar pre-pass
- permutation resolution
- analysis-aware CSC build
- supernodal attempt
- fallback

This is not a public API problem; it is an internal CSC ownership and
reasoning problem.

### 4. Proof is stronger on dispatch routing than on indefinite factor-many behavior

Already well covered:

- AUTO dispatch routing
- forced CSC backend behavior
- CSC kernel invariants and cross-backend solve agreement

Still weaker than it should be:

- no LDL^T-specific repeated-run benchmark equivalent to the Cholesky
  factor-many proof
- no equally direct indefinite same-pattern benchmark centered on the shared
  analysis/factors path

## Important Boundary: Why This Is Harder Than Cholesky

`ldlt_csc_from_sparse_with_analysis(...)` already documents the real internal
boundary:

- SPD inputs can call it directly
- indefinite batched use still requires the caller to resolve the BK pivot
  permutation first and analyze the pre-permuted matrix

That means the Sprint 53 goal should not be to pretend indefinite LDL^T is as
simple as Cholesky. The goal is to make the extra steps better owned, better
bounded, and better proved.

## Ranked Sprint 53 Target List

1. Reduce or better bound hidden scalar-prepass / reanalysis work inside the
   shared LDL^T CSC repeated-run path.
2. Tighten shared-vs-one-shot LDL^T CSC orchestration so the dispatch story is
   easier to reason about.
3. Add real indefinite factor-many benchmark proof on intended workloads.
4. Refresh public regression coverage around the shared indefinite repeated-run
   path.
5. Reconcile README / header wording only after the CSC ownership seams are
   clearer.

## Explicit Non-Goals

Day 3 explicitly rejects these as Sprint 53 targets:

- public direct-solver redesign
- raw CSC/native storage exposure
- full structural-pattern verifier redesign
- broad tutorial or example rewrite
- generic direct-handle abstraction

## Operational Result

Sprint 53 now has a concrete CSC implementation starting point:

- the path exists
- the fallback/rebuild seam is named
- the duplicated orchestration seam is named
- the proof asymmetry is named
- the ranked Phase 3 target list is fixed

That is enough to move into Day 4 from a real audit rather than from a generic
completion backlog.
