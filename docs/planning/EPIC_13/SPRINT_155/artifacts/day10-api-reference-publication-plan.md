# Sprint 155 Day 10 API Reference Publication Plan

## Purpose

Day 10 inventories the current API reference surface and defines how Sprint 155
should publish or link API reference material without widening package, ABI,
platform, ecosystem, or performance claims.

## Current Reference Surfaces

| Surface | Current role | Day 10 finding |
| --- | --- | --- |
| `Doxyfile` | Doxygen configuration for checked-in public headers under `include/`. | Uses `INPUT = include/`, `FILE_PATTERNS = *.h`, and emits HTML to `docs/api/html/`. |
| `Makefile` target `docs` | Local API reference regeneration command. | Runs `doxygen Doxyfile` and reports `docs/api/html/` as output. |
| `docs/api/html/` | Generated Doxygen HTML reference. | Present, but not complete for the current checked-in public header set. |
| `README.md` | High-level documentation map and command list. | Mentions `make docs` as the Doxygen API-reference command. |
| `docs/tutorial.md` | First-use learning path. | Points users to public headers and generated API reference for exact declarations and ownership contracts. |
| `docs/maintainer_guide.md` | Maintainer policy surface. | Owns generated-output, report, packaging, and header-cleanup boundaries. |

## Header Coverage Inventory

Current checked-in public headers under `include/`:

| Header | Generated HTML page in `docs/api/html/` |
| --- | --- |
| `include/sparse_analysis.h` | Missing |
| `include/sparse_bidiag.h` | Present |
| `include/sparse_cholesky.h` | Present |
| `include/sparse_csr.h` | Present |
| `include/sparse_dense.h` | Present |
| `include/sparse_eigs.h` | Missing |
| `include/sparse_ic.h` | Missing |
| `include/sparse_ilu.h` | Present |
| `include/sparse_iterative.h` | Present |
| `include/sparse_ldlt.h` | Missing |
| `include/sparse_lu.h` | Present |
| `include/sparse_lu_csr.h` | Missing |
| `include/sparse_matrix.h` | Present |
| `include/sparse_qr.h` | Present |
| `include/sparse_reorder.h` | Present |
| `include/sparse_svd.h` | Present |
| `include/sparse_types.h` | Present |
| `include/sparse_vector.h` | Present |

The installed header surface also includes generated `sparse_version.h`, built
from `include/sparse_version.h.in` into `build/include/sparse_version.h`.
Because `Doxyfile` currently reads only `include/*.h`, the generated installed
version header is outside the current API-reference input set.

## Gaps

1. The generated HTML reference is incomplete for current checked-in public
   headers. The missing pages include the Day 8 and Day 9 cleaned headers:
   `sparse_ldlt.h`, `sparse_ic.h`, `sparse_eigs.h`, and
   `sparse_analysis.h`.
2. `sparse_lu_csr.h` is also missing from the generated HTML surface.
3. The generated installed version header is not represented because the
   Doxygen input is `include/`, not the configured build include directory.
4. There is no concise maintainer-facing freshness rule that says when
   `docs/api/html/` must be regenerated and what validation should accompany
   that refresh.
5. There is no small user-facing API reference index page that separates:
   exact declarations in headers, generated HTML, tutorial/cookbook workflows,
   and maintainer-only policy.

## Decision

Sprint 155 should do both:

1. Add direct API-reference guidance for users.
2. Add a generated-reference publication plan for maintainers.

Day 11 should not attempt a broad reference rewrite. It should add a compact
source-controlled Markdown entry point and maintainer guidance, then optionally
refresh generated HTML only if the diff remains reviewable.

## Proposed Day 11 Implementation

1. Add `docs/api_reference.md` as the user-facing API reference index.
2. Link `docs/api_reference.md` from `README.md` and `docs/tutorial.md`.
3. Add maintainer instructions to `docs/maintainer_guide.md` for regenerating
   and reviewing `docs/api/html/`.
4. Keep generated HTML ownership explicit:
   - source of truth: public headers under `include/` plus generated
     `sparse_version.h` when included by an explicit build-aware Doxygen
     configuration;
   - generated output: `docs/api/html/`;
   - regeneration command: `make docs`;
   - review rule: generated HTML changes should be bundled only with the
     header/source comment changes that justify them or with a dedicated
     reference refresh.
5. Decide whether to update `Doxyfile` to include the generated version header
   through a stable generated-header path. If this cannot be done without
   making the local build directory a documentation prerequisite, document that
   version macros remain covered by `README.md`, `VERSION`, and install tests
   rather than generated Doxygen.
6. If generated HTML is refreshed, record:
   - Doxygen version;
   - command used;
   - header coverage before/after;
   - warnings, if any;
   - generated-output diff summary.

## Freshness Semantics

Generated API reference is fresh only when all of the following are true:

1. The current branch has run `make docs`.
2. The generated page inventory covers the intended Doxygen input set.
3. Any Doxygen warnings are recorded and triaged.
4. The generated output is committed with the corresponding source/header
   documentation change or in a dedicated reference-refresh commit.
5. The review description states whether generated output changed.

Generated API reference should be treated as stale or partial when:

- public header comments changed after the last `make docs` refresh;
- new public headers do not have matching generated pages;
- generated installed headers are outside the configured Doxygen input set;
- generated output is not committed with the source/header change that
  explains it.

## Claim Boundaries

API reference text may say:

- public headers own exact declarations, types, options, result structs, and
  ownership/freeing contracts;
- `make docs` generates local Doxygen HTML from the configured public-header
  input set;
- installed packages remain static-first unless a later project explicitly
  changes that contract.

API reference text must not imply:

- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity from fixture-local oracle rows;
- portable runtime guarantees from local benchmarks;
- broad completeness beyond the configured public-header input set.

## Validation Plan

Day 10 changes are documentation/planning only, so C quality gates are not
required. The required local check is:

```sh
git diff --check
```

Day 11 should run `git diff --check` after documentation edits. If Day 11
refreshes generated HTML with `make docs`, it should also capture Doxygen
warnings and the generated-page coverage inventory.

## Day 11 Handoff

Implement the lightweight API reference entry point first. Then add maintainer
publication guidance. Refresh generated HTML only if the resulting generated
diff is understandable and can be reviewed separately from policy wording.
