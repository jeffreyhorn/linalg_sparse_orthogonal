# Sprint 34 Day 11: initializer regression audit

## Goal

Audit the reviewed Sprint 34 target set for newly surfaced positional options-struct initializers and convert any real regression-prevention cleanup sites without reopening the already-closed Sprint 32 warning backlog.

## Audit finding

The reviewed tree did not reopen the historical `-Wmissing-field-initializers` queue. That backlog remains closed.

The real remaining reviewed-target drift was narrower:

- non-zero positional `sparse_analysis_opts_t` initializers

These sites were still mechanically valid because `sparse_analysis_opts_t` currently has only two fields:

- `factor_type`
- `reorder`

But they are brittle against future struct growth and inconsistent with the designated-initializer standard already established for reviewed options structs.

## Cleanup performed

Converted the full reviewed non-zero `sparse_analysis_opts_t` batch to designated form across:

- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `examples/example_analysis.c`
- `benchmarks/bench_amd_qg.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_reorder.c`
- `tests/test_sprint19_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`
- `tests/test_etree.c`
- `tests/test_colamd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_chol_csc.c`

## What was not treated as regression debt

The remaining reviewed-tree `*_opts_t` literal found by the narrowed query is:

- `tests/test_eigs.c`
  - `sparse_svd_opts_t svd_opts = {0};`

That is a zero-init sentinel, not the Day 11 non-zero positional regression class.

## Validation

After the cleanup:

- `make format`: passed
- `make lint`: passed
- `make test`: passed

## Result

Sprint 34 Day 11 closed the real reviewed-target initializer drift without reopening the earlier warning backlog:

- no new warning debt
- no partial cleanup queue left for this regression class
- reviewed-target quality gates stayed green after the conversions
