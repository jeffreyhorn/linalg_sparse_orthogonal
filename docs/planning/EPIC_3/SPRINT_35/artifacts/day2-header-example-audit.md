# Sprint 35 Day 2 Artifact: Header Example Audit

## Audited Header Set

Day 2 audited the first-pass installed-header surface identified on Day 1:

- `include/sparse_iterative.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`

## Main Result

The installed-header public example surface is already much closer to the Sprint 35 target state than the sprint plan initially implied.

### What is already aligned

All seven audited headers currently show public options-struct examples using designated initializers rather than brittle positional initialization.

This means Sprint 35 does **not** need to spend its header budget on a broad mechanical conversion pass analogous to earlier code-cleanup sprints.

## File-By-File Classification

### `include/sparse_iterative.h`

Status: `keep`

- Public usage examples already use:
  - `sparse_iter_opts_t opts = { .max_iter = 1000, .tol = 1e-10 };`
  - `sparse_gmres_opts_t opts = { .max_iter = 500, .restart = 30, .tol = 1e-10 };`
- No positional-initializer drift found in the public examples.
- Larger iterative truthfulness debt remains in `docs/tutorial.md`, not in this header.

### `include/sparse_reorder.h`

Status: `keep`

- Public examples are already explicit and designated-style where applicable.
- No direct initializer cleanup required.
- Possible later wording reconciliation may still be useful when Sprint 35 unifies reorder-mode explanations across docs.

### `include/sparse_svd.h`

Status: `update`

This file contains the strongest real header debt found on Day 2:

- `sparse_svd_opts_t::economy` says full SVD mode (`economy = 0`) was enabled in Sprint 29 Day 3.
- But the `sparse_svd_compute()` return documentation still says:
  - `compute_uv is set without economy (full SVD not implemented)`
- The `sparse_svd_partial()` prose also contradicts itself:
  - one part says singular vectors are recovered when `opts->compute_uv` is set
  - the `@param opts` line still says singular vectors are not computed

Interpretation:

- this is real public behavior-doc drift
- it is higher-value to fix than any initializer-style work in this file

### `include/sparse_lu.h`

Status: `keep`

- Public example and options snippet already use designated initializers.
- No header-example syntax debt found.

### `include/sparse_cholesky.h`

Status: `keep`

- Public options snippet is already designated-initializer based.
- No direct header example rewrite required on Day 2 evidence.

### `include/sparse_ldlt.h`

Status: `keep`

- Public options example is already designated-initializer based.
- No direct initializer cleanup required.

### `include/sparse_analysis.h`

Status: `keep`

- Public workflow/examples already use designated initializers.
- No broad syntax cleanup needed.

## Queue Narrowing

Day 2 narrows the Sprint 35 header queue to:

Priority A:

- `include/sparse_svd.h` public behavior wording reconciliation

Priority B:

- cross-header wording consistency checks, only where later README/tutorial cleanup reveals mismatches

Not supported by Day 2 evidence:

- a large installed-header positional-initializer backlog
- a need to treat all audited headers as rewrite targets

## Implication For Day 3+

The maintainer/public example standard should still be written, but Day 4 and Day 5 should be scoped as:

- targeted header cleanup
- wording reconciliation
- public-truthfulness fixes

not:

- a repo-wide header example conversion campaign

## Bottom Line

Day 2’s most important result is negative evidence: most installed-header examples are already using the right syntax.

The strongest remaining header debt is not snippet style. It is contradictory public wording in `include/sparse_svd.h`, while the larger remaining Sprint 35 queue still appears to live in tutorial/README/example consistency work rather than in installed-header mechanical cleanup.
