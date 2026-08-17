# Sprint 164 Day 8: Header Organization And Cross-Link Cleanup

## Purpose

Align selected public headers and public workflow docs so users can move from
first-use guidance to exact declarations without pulling maintainer-only policy
into the headers.

Day 8 stayed inside the selected Sprint 164 public-header batch:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `docs/solver_selection.md`

## Header Organization Cleanup

`include/sparse_iterative.h` now has an explicit shared callback/options/result
section before the solver-specific declarations. This makes the top of the
header easier to scan before callers reach repeated-run handles and one-shot
solver entries.

The selected headers were reviewed for duplicate comment blocks, stale
references, and inconsistent terminology. No public declaration changes were
made.

## Cross-Link Updates

`include/sparse_matrix.h` now points first-use matrix-shell readers to:

- `docs/cookbook.md`
- `docs/tutorial.md`
- `docs/solver_selection.md`
- `examples/README.md`

`include/sparse_iterative.h` now points iterative-solver readers to the same
public workflow path before using the header as the exact declaration and
option/result reference.

`include/sparse_eigs.h` now points eigensolver readers to the same public
workflow path, keeps `docs/algorithm.md` as the algorithm-note owner, and adds
explicit sibling-header references for:

- `sparse_ldlt.h` for shift-invert factorization;
- `sparse_svd.h` for the related rectangular decomposition;
- `sparse_iterative.h` for the preconditioner callback type shared with LOBPCG.

`docs/solver_selection.md` now links the eigensolver section to the fuller
tutorial and cookbook eigensolver walkthroughs in addition to the exact public
header.

## Maintainer-Only Detail Boundary

The public headers keep concise navigation and exact API contracts only. They
do not pull in generated-reference policy, hosted-CI policy, package/ABI
interpretation, or historical benchmark methodology. Those remain owned by
`docs/api_reference.md`, `docs/maintainer_guide.md`, `INSTALL.md`, and
benchmark/report documentation.

## Deferred Cross-Link Gaps

- Broader public-header navigation for non-selected headers remains deferred
  outside the Sprint 164 selected batch.
- Generated API HTML freshness remains a Day 9 policy check rather than a Day
  8 source-controlled HTML update.
- Any README/table-wide public-header index reshaping remains deferred because
  existing README and API reference entries already route to the selected
  headers.

## Declaration Preservation

The selected public-header normalized declaration checksum stayed unchanged
after formatting and after the full gate:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

This matches the Day 4 baseline checksum.

## Validation

Commands run:

```sh
make format
make lint && make test
git diff --check
```

Additional Day 8 checks:

- normalized selected-header declaration checksum compared against the Day 4
  baseline before and after the full gate;
- scoped claim scan over selected public headers plus README/API/tutorial/
  cookbook/solver-selection/maintainer documentation;
- generated-output status check for `build`, `docs/api/html`,
  `scripts/__pycache__`, and `tests/__pycache__`;
- anchor presence check for the new tutorial and cookbook eigensolver links.

## Completion Criteria

- Selected headers and public docs use consistent user-facing navigation.
- Users can navigate from API comments to maintained docs and runnable
  examples.
- Maintainer-only policy remains out of public headers.
- Public declarations remain unchanged.
- Required C/header quality gate passes.
