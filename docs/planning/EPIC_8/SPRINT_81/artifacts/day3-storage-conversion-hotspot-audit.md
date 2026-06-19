# Sprint 81 Day 3 - Storage / Conversion Hotspot Audit

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Re-rank the strongest linked-list-first storage, conversion, and repeated-run
workflow costs against the live direct workflows so Sprint 81 starts from one
ranked contradiction map instead of one generic storage-modernization bucket.

## Main Result

Sprint 81's broad product/storage problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - public mutable matrix-shell and mutation/publication center
- strongest second target:
  - repeated-run direct-workflow factor path that still rebuilds linked-list
    permuted copies on the small-problem lane
- strongest third target:
  - family-local one-shot direct wrappers that still keep the linked-list shell
    as the visible compatibility center
- strongest support-only but real target:
  - proof and benchmark surfaces that currently normalize the linked-list shell
    rather than a compressed-first reading

## Strongest Current Contradiction

The strongest current contradiction center is still the public matrix shell:

- `include/sparse_matrix.h` still presents the public matrix API as the
  orthogonal linked-list shell
- `src/sparse_matrix.c` still concentrates mutable construction, insertion,
  transpose, copy, and shell lifecycle around pointer-heavy row/column walks
  and slab-node allocation
- this means the highest-value Sprint 81 first move is the matrix-shell
  construction/import/publication seam itself, not a later wrapper-only cleanup

The strongest useful Day 3 conclusion is therefore explicit:

- Sprint 81 should start with the public matrix-shell contradiction
- it should not start with a broad repeated-run wrapper rewrite
- it should not widen into backend, capability, or package lanes

## Secondary Structural Seam

The strongest second contradiction is now fixed to the repeated-run direct
workflow path:

- `src/sparse_analysis.c` already owns the repeated-run direct lane
- but its smaller-problem factorization routes still go through
  `build_permuted_copy(...)` and rebuild linked-list shells before factoring
- that makes repeated-run workflow convergence the strongest likely second
  batch, not the first implementation center

## Lower-Order But Still Real Seam

The one-shot direct wrappers remain real, but lower-order, support context:

- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`

These still keep the linked-list shell as the compatibility owner, but they no
longer read as the best first batch center once the public matrix shell and the
repeated-run factor path are reranked directly.

## Proof / Benchmark Context

The strongest proof-tier context for Sprint 81 is now fixed too:

- `tests/test_sparse_matrix.c` is the family-local shell/lifecycle proof owner
- `tests/test_integration.c` is the public repeated-run and cross-workflow
  proof owner
- `benchmarks/bench_refactor_csc.c` is the strongest benchmark-side
  measurability surface for repeated-run direct workflows

These are support-tier surfaces for Sprint 81's first batch unless the
implementation truly forces them to move.

## Exit State

- Sprint 81 now has one ranked live storage/workflow contradiction map.
- The first implementation center is fixed to the public matrix-shell
  construction/import/publication seam.
- The repeated-run direct path is fixed as the strongest likely second seam,
  not the first landing center.
