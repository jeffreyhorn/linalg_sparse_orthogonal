# Sprint 35 Day 1 Artifact: Public Docs Baseline

## Sprint 34 Baseline Reconfirmed

Sprint 35 inherits the Sprint 34 enforced state, not a broken quality flow:

- `make format`: passed at Sprint 34 close
- `make lint`: passed at Sprint 34 close
- `make test`: passed at Sprint 34 close
- `make quality-review-compile`: passed at Sprint 34 close
- `make quality-review`: passed at Sprint 34 close
- `make quality-review-cmake-compile`: passed at Sprint 34 close
- `make quality-review-cmake`: passed at Sprint 34 close
- `ctest -N --test-dir build/quality-review-cmake`: `53` registered tests
- `ctest --test-dir build/quality-review-cmake --output-on-failure`: `53 / 53` passed
- `make deadcode-report`: passed at Sprint 34 close
- `make deadcode-check`: passed at Sprint 34 close

Implication:

- Sprint 35 should treat the reviewed quality wrappers and active-suite count as fixed invariants while public docs/examples are rewritten.

## Public-Surface Inventory Snapshot

- installed public headers: `18`
- shipped top-level example programs: `12`
- high-value public docs in immediate sprint scope: `6`
  - `README.md`
  - `docs/tutorial.md`
  - `docs/algorithm.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `INSTALL.md`

## Highest-Signal Day 1 Findings

### 1. The biggest immediate public drift is tutorial/API-usage truthfulness

`docs/tutorial.md` still refers to stale option-type names:

- `sparse_cg_opts_t`
- `sparse_ilu_opts_t`

Current public headers instead expose:

- `sparse_iter_opts_t`
- `sparse_gmres_opts_t`

This is the strongest Day 1 signal because it is a real public-surface mismatch between the docs and the current installed API.

### 2. Public designated-initializer usage is already widespread

Installed headers and public examples already show designated-initializer usage in many places, including:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- `include/sparse_reorder.h`
- `include/sparse_iterative.h`
- `include/sparse_svd.h`
- `README.md`
- `docs/algorithm.md`
- multiple `examples/*.c` programs

Interpretation:

- Sprint 35 should audit header examples carefully, but Day 1 does not support assuming a large leftover positional-initializer backlog in installed headers.

### 3. The likely dominant Sprint 35 queue is cross-surface consistency

The first-day evidence points to a consistency problem more than a pure snippet-style problem:

- tutorial vs header type-name truthfulness
- README/tutorial/example wording alignment
- reorder/precondition language consistency
- maintainer standard cleanup for public examples that already mostly use the preferred style

## Likely First-Fix Queue

Priority A:

- `docs/tutorial.md` iterative/ILU sections
- `README.md` public example snippets and usage wording

Priority B:

- installed-header example/style audit across:
  - `include/sparse_iterative.h`
  - `include/sparse_reorder.h`
  - `include/sparse_svd.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_analysis.h`

Priority C:

- shipped public-example consistency pass across:
  - `examples/example_iterative.c`
  - `examples/example_matrix_free.c`
  - `examples/example_ic_minres.c`
  - `examples/example_ldlt.c`
  - `examples/example_colamd.c`
  - `examples/example_analysis.c`

## Day 1 Bottom Line

Sprint 35 starts from a clean enforcement baseline and a bounded public-surface queue.

The strongest current public mismatch is not “headers still teach positional initialization everywhere.” It is that public documentation and examples are not yet fully synchronized on current type names, wording, and usage guidance, even though the designated-initializer style has already propagated through much of the installed API surface.
