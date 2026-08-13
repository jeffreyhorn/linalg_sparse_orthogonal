# Sprint 155 Day 9 Header Cleanup Summary

## Purpose

Day 9 finished the selected Sprint 155 public-header cleanup batch by editing
only Doxygen/comment text in `include/sparse_eigs.h` and
`include/sparse_analysis.h`.

## Eigensolver Header Cleanup

- shortened top-level convergence and ownership prose while preserving the
  Wu/Simon residual contract and caller-owned result-buffer contract;
- clarified v2.2.0 options/result source-rebuild warnings without adding
  package, shared-library, or dynamic ABI claims;
- shortened `max_iterations`, `reorthogonalize`, `precond`,
  `lobpcg_soft_lock`, and `progress_cb` comments;
- preserved backend, shift-invert, LOBPCG, soft-locking, vector reliability,
  and progress/cancel boundaries.

## Analysis Header Cleanup

- shortened the repeated-run direct-solver introduction;
- clarified analyze/factor/refactor/free lifecycle and one-shot API handoff;
- clarified symbolic-analysis and numeric-factor ownership boundaries;
- kept same-pattern refactor language explicit and avoided implying full
  structural-pattern verification;
- preserved error contracts and ownership/freeing requirements.

## Completed Batch

The selected Sprint 155 header cleanup batch is now complete:

- `include/sparse_ldlt.h` and `include/sparse_ic.h` were cleaned on Day 8.
- `include/sparse_eigs.h` and `include/sparse_analysis.h` were cleaned on
  Day 9.

## Declaration Preservation Evidence

- `day9-header-declarations-before.txt`
- `day9-header-declarations-after.txt`
- `day9-header-declarations-normalized-diff.txt`

The normalized diff is empty after stripping file/line prefixes and sorting
declaration-like text.

## Deferred Header Register Update

No new deferred-header issues were found. The Day 6 deferred-header register
remains unchanged.

## Claim Scan

The unsupported-claim scan returned no matches across the selected batch. No
shared-library, dynamic ABI, package-manager, runtime-loader, broad Windows
parity, portable-performance, external-parity, or state-of-the-art claims were
introduced.

## Validation

Commands run:

```sh
git diff --check
make format && make lint && make test
```

Results:

- `git diff --check` passed before and after the full gate.
- `make format && make lint && make test` passed.
- The final test output ended with `All tests passed.`

## Day 10 Handoff

Day 10 should use the cleaned headers as input for API reference baseline and
publication planning. It should inventory existing Doxygen/API surfaces, decide
whether to add direct API index guidance or a generated-reference plan, and
preserve static-first/package/ABI claim boundaries.
