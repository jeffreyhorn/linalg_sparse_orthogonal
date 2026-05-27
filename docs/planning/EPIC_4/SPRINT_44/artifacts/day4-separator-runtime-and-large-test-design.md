# Sprint 44 Day 4 Artifact: Separator, Runtime, and Large-Test Design

## Purpose

Define the separator-lifting extraction boundary, the runtime/parser cleanup
ownership model, and the bounded first large-test maintainability target set
before Sprint 44 begins the main implementation batches.

## 1. Separator-Lifting Module Boundary

### Proposed target file

- `src/sparse_graph_separator.c`

### Separator-owned extraction set

- separator strategy enums:
  - `sep_lift_strategy_t`
  - `sep_lift_weight_t`
- separator policy parsers:
  - `parse_sep_lift_strategy(...)`
  - `parse_sep_lift_weight(...)`
- separator policy helpers:
  - `is_per_vertex_strategy(...)`
  - `per_vertex_score_cmp_desc(...)`
- separator conversion entry point:
  - `graph_edge_separator_to_vertex_separator(...)`

### What should remain outside the separator module

- `graph_uncoarsen(...)`
- `graph_partition_seed_coarsest(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- `graph_partition_should_retry_with_forced_hem(...)`

Reason:

- separator lifting is the final projection/policy step
- the top-level partition composition seam still sequences hierarchy build,
  coarsest split, FM, uncoarsening, separator lifting, and retry policy

## 2. Runtime / Config Parsing Ownership Model

Sprint 44 should not begin with a generic parser file.

The live parsing ownership is:

### FM-owned parsing

- annealing schedule parsing
- thick-restart perturbation parsing
- gain-noise schedule parsing

### Separator-owned parsing

- separator lift strategy parsing
- separator lift weight parsing

### Residual orchestration-owned parsing

- finest-pass count parsing
- finest strategy selection parsing
- ensemble selector-list parsing
- intermediate-pass parsing
- sep=0 retry / forced-HEM composition

### Day 4 decision

- Day 5 should move FM parser logic with FM
- Day 6 should move separator parser logic with separator lifting
- Day 8 should clean up only the remaining orchestration-scoped parser/config
  logic

Interpretation:

- runtime cleanup is a post-extraction simplification step, not an independent
  first-wave split

## 3. Shared-Header Strategy for Separator Extraction

Current behavior-level shared seam:

- `graph_edge_separator_to_vertex_separator(...)`

Day 4 decision:

- keep that as the main separator behavior seam in
  `src/sparse_graph_internal.h`
- do not expose separator parsers or scoring helpers through broader graph
  headers
- prefer translation-unit-local policy helpers in the separator file

Result:

- Day 6 should need minimal shared-header expansion, if any

## 4. Large-Test Maintainability Target Set

The highest-volume test binaries remain:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_svd.c` = `3746`
- `tests/test_ldlt_csc.c` = `3637`
- `tests/test_qr.c` = `3291`

But the first helper-consolidation batch should follow the strongest actual
helper seams, not file size alone.

### `tests/test_qr.c`

Strongest current seam:

- `compare_dense_sparse_qr(...)`

Likely next helper opportunities:

- repeated dense-vs-sparse factor/solve/compare harnesses
- repeated reconstruction/residual setup
- repeated sparse-mode fixture wrappers

Assessment:

- very strong Day 12 candidate

### `tests/test_chol_csc.c`

Strongest current seam:

- repeated supernodal cross-check / roundtrip / dispatch fixture harnesses

Likely next helper opportunities:

- supernodal scalar-vs-batched comparison harnesses
- repeated SPD fixture builders
- dispatch residual helpers

Assessment:

- strong audit candidate, but likely best handled through one bounded helper
  batch rather than broad consolidation

### `tests/test_ldlt_csc.c`

Strongest current seam:

- repeated indefinite fixture builders and two-pass factor harnesses

Likely next helper opportunities:

- KKT fixture builders
- repeated solve-residual helpers
- repeated factor-state comparison harnesses

Assessment:

- strong audit candidate; likely Day 12 candidate if QR remains too narrow

### `tests/test_svd.c`

Strongest current seam:

- repeated dense fixture setup for full/economy/output comparisons

Likely next helper opportunities:

- 16×8 full-mode fixture builder
- repeated low-rank corpus-safety loops
- repeated orthogonality/reconstruction comparison harnesses

Assessment:

- strong audit candidate, but slightly more likely to benefit from a later
  focused batch than the first Day 12 landing

## 5. Bounded Day 11 / Day 12 Plan

### Day 11 audit target set

- `tests/test_qr.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`
- `tests/test_svd.c`

### Day 12 likely implementation shape

Best first target:

- `tests/test_qr.c`

Best second target:

- whichever of `test_chol_csc.c`, `test_ldlt_csc.c`, or `test_svd.c` shows the
  clearest repeated helper/fixture pattern after the Day 11 audit

### Explicit non-goals

- no file splitting
- no test-framework redesign
- no broad helper extraction across all four files at once
- no behavior changes bundled into maintainability cleanup

## 6. Fixed Sprint 44 Mid-Sprint Order

The Day 4 design locks the graph implementation order:

1. Day 5:
   - FM extraction
2. Day 6:
   - separator extraction
3. Day 7:
   - residual runtime/orchestration audit
4. Day 8:
   - runtime/orchestration cleanup

This keeps parser ownership honest and keeps the residual graph cleanup tied to
the post-extraction reality rather than to the pre-extraction file shape.

## Bottom Line

Day 4 makes three things explicit:

- separator lifting should extract into `src/sparse_graph_separator.c`
- runtime/parser cleanup should follow FM and separator ownership instead of
  starting from a generic parser module
- the first large-test maintainability batch should target real helper seams,
  with `tests/test_qr.c` as the strongest likely first landing and one of
  `test_chol_csc.c`, `test_ldlt_csc.c`, or `test_svd.c` as the second bounded
  candidate
