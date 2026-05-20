# Sprint 35 Day 5: Header Cleanup Batch II

## Scope

Close the remaining installed-header consistency pass after Day 4's targeted
`include/sparse_svd.h` truthfulness fix.

Day 5 is not a second large conversion batch. The Day 2 audit was still
correct: the residual header queue is mainly wording and style consistency in
the most visible public examples.

## Main Result

Day 5 closed the remaining installed-header cleanup by aligning the other
high-traffic public header examples with the Sprint 35 Day 3 contract:

- designated initializers for non-default examples
- `NULL` for pure-default option paths
- current behavior wording instead of ambiguous or overly broad guidance

The touched headers were:

- `include/sparse_iterative.h`
- `include/sparse_reorder.h`
- `include/sparse_analysis.h`

## What Changed

### 1. `include/sparse_iterative.h` now teaches one consistent options style

The CG and GMRES top-level usage snippets now use the same multi-line
designated-initializer style already used in the stronger README / algorithm
examples.

The preconditioned-GMRES snippet also now defines its own `opts` object
locally instead of relying on the preceding example block implicitly. That
makes the snippet stand on its own as public guidance.

This is a style/clarity cleanup, not a behavior change.

### 2. `include/sparse_reorder.h` now matches the same example presentation

The COLAMD / QR example still taught the right API, but it used a compact
single-line struct literal while nearby headers increasingly used the Day 3
designated-initializer presentation.

Day 5 makes that example match the standard:

```c
sparse_qr_opts_t opts = {
    .reorder = SPARSE_REORDER_COLAMD,
};
```

This keeps the column-only reorder guidance intact while making the public
example surface more uniform.

### 3. `include/sparse_analysis.h` now states the symmetric-analysis rule more directly

`sparse_analysis_opts_t.reorder` already accepted `SPARSE_REORDER_COLAMD`, but
the public wording could read as if COLAMD were a first-class peer to the
normal symmetric analysis reorder set.

Day 5 tightens that contract:

- the normal symmetric analysis path is `NONE`, `RCM`, `AMD`, or `ND`
- `COLAMD` remains accepted
- but the header now says more directly that `sparse_analyze()` applies it
  symmetrically, and that the QR column-only path lives elsewhere

That makes the header less likely to mis-teach COLAMD as the default analysis
choice for symmetric factor workflows.

## End-State Interpretation

Sprint 35's installed-header cleanup is now effectively complete:

- no broad positional-initializer backlog remained
- the one true behavior contradiction in `include/sparse_svd.h` is closed
- the remaining high-traffic examples now present one public style contract
- the remaining larger truthfulness queue is outside the headers

That residual queue is now clearly:

- `docs/tutorial.md`
- `README.md`
- selected example-facing docs and shipped examples

## Bottom Line

Day 5 finishes the header layer. The public installed headers now teach a
consistent options-style contract and more precise reorder / analysis wording,
so Sprint 35 can move to README/tutorial cleanup without carrying hidden
header drift forward.
