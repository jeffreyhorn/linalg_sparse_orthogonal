# Sprint 164 Day 4 Declaration Baseline Capture

## Purpose

Day 4 captures the before-state declaration evidence for the selected Sprint
164 public-header cleanup batch and records the current API-reference and
documentation reference state before any header edits begin.

Selected headers:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`

## Baseline Capture

The Day 3 normalized declaration capture command was run. It strips comments,
preserves declaration-like source order, writes local ignored artifacts under
`build/sprint164/declarations/`, and records a SHA-256 checksum for the
combined selected-header bundle.

Captured checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41  build/sprint164/declarations/selected-public-headers.before.normalized.txt
```

Generated local declaration files:

| Local Generated File | Lines |
| --- | ---: |
| `build/sprint164/declarations/sparse_iterative.h.normalized.txt` | 151 |
| `build/sprint164/declarations/sparse_eigs.h.normalized.txt` | 102 |
| `build/sprint164/declarations/sparse_matrix.h.normalized.txt` | 88 |
| `build/sprint164/declarations/selected-public-headers.before.normalized.txt` | 346 |

These generated files remain local ignored evidence and are not committed.

## Current Generated API Reference State

Current local API page coverage check:

```sh
python3 scripts/check_api_docs_coverage.py
```

Result:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

Selected-header local generated pages are present:

| Header | Reference Page | Source Page | State |
| --- | --- | --- | --- |
| `include/sparse_iterative.h` | `docs/api/html/sparse__iterative_8h.html` | `docs/api/html/sparse__iterative_8h_source.html` | present |
| `include/sparse_eigs.h` | `docs/api/html/sparse__eigs_8h.html` | `docs/api/html/sparse__eigs_8h_source.html` | present |
| `include/sparse_matrix.h` | `docs/api/html/sparse__matrix_8h.html` | `docs/api/html/sparse__matrix_8h_source.html` | present |

Sprint 158 policy still applies: generated HTML is local-only, ignored output.
After header-comment edits, `make docs-check` must be run before closeout.

## Documentation Reference Map

| Selected Header | Current References |
| --- | --- |
| `include/sparse_iterative.h` | README API overview and repeated-run handle section; `docs/api_reference.md`; tutorial iterative examples; solver-selection iterative section; maintainer public-header policy. |
| `include/sparse_eigs.h` | README capability overview, backend description, API overview, and repeated-run handle section; `docs/api_reference.md`; tutorial eigensolver section; cookbook eigensolver notes; solver-selection eigensolver section; maintainer public-header and evidence policy. |
| `include/sparse_matrix.h` | README first-use and API overview sections; `docs/api_reference.md`; tutorial matrix construction and Matrix Market sections; cookbook CSR/CSC/Matrix Market routes; solver-selection data-entry and diagnostics sections; maintainer public-header policy. |

Key exact-reference findings from the Day 4 scan:

- `docs/api_reference.md` lists all three selected headers in the source of
  truth table.
- `docs/tutorial.md` includes all three selected headers in examples or route
  tables.
- `docs/cookbook.md` references `sparse_eigs.h` directly and references
  matrix construction APIs tied to `sparse_matrix.h`.
- `docs/solver_selection.md` links directly to all three selected headers.
- README links all three selected headers in the API overview and names their
  first-use APIs.

## Pre-Existing Inconsistency Notes

No declaration drift exists yet because selected public headers have not been
edited on Sprint 164.

Pre-existing documentation notes before cleanup:

- local generated API HTML exists in the working checkout and currently passes
  page coverage, but remains ignored local output;
- `docs/api_reference.md` correctly says generated HTML can be stale unless
  `make docs-check` has just passed;
- README contains the word `deprecated` in repeated-run guidance, but this is
  existing workflow wording and not an introduced Day 4 inconsistency;
- `include/sparse_matrix.h` contains a stale-handle warning in API-local
  lifecycle wording; this is a valid cleanup target, not declaration drift.

## Day 5 Handoff

Day 5 may begin ownership and lifetime comment cleanup on:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`

Controls to preserve:

- use the Day 4 checksum as the before-state declaration baseline;
- keep edits comment-only unless an explicit reviewed exception is recorded;
- do not edit generated `docs/api/html/`;
- preserve package, ABI, runtime-loader, backend-superiority, performance,
  platform, and state-of-the-art non-claims.

## Validation Notes

Day 4 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 4.

## Completion Check

- Selected declarations are captured before edits.
- Current generated API-reference state is recorded.
- Current documentation references are mapped.
- Pre-existing notes are separated from introduced drift.
