# Sprint 155 Day 12 Preservation And Reconciliation

## Purpose

Day 12 reconciled the Sprint 155 tutorial, public-header, API reference, and
maintainer guidance work after Days 8 through 11. The focus was preservation
evidence, cross-link correctness, installed-header expectations, generated API
reference freshness, and claim boundaries.

## Declaration Preservation

Edited public headers:

- `include/sparse_ldlt.h`
- `include/sparse_ic.h`
- `include/sparse_eigs.h`
- `include/sparse_analysis.h`

Evidence files:

- `day8-header-declarations-before.txt`
- `day8-header-declarations-after.txt`
- `day8-header-declarations-normalized-diff.txt`
- `day9-header-declarations-before.txt`
- `day9-header-declarations-after.txt`
- `day9-header-declarations-normalized-diff.txt`
- `day12-header-declarations-current.txt`
- `day12-header-declarations-normalized-diff.txt`

The Day 8, Day 9, and Day 12 normalized declaration diffs are all empty after
stripping file/line prefixes and sorting declaration-like text.

The Day 12 aggregate check uses `day8-header-declarations-before.txt` as the
single full pre-cleanup baseline because the Day 8 and Day 9 declaration scans
both cover the same four-header batch.

## Installed Header Expectations

The install configuration remains aligned with the edited checked-in headers:

- CMake installs checked-in `include/*.h` headers under
  `${CMAKE_INSTALL_INCLUDEDIR}/sparse`;
- CMake excludes `*.h.in` and installs generated `sparse_version.h`
  separately from the build include directory;
- the edited checked-in headers remain in `include/` with unchanged names;
- no installed-header rename, include-guard rename, generated-version-header
  change, or install-surface change was made.

## Cross-Document Reconciliation

The user-facing API reference path is now:

- `README.md` -> `docs/api_reference.md`;
- `docs/tutorial.md` -> `docs/api_reference.md`;
- `docs/cookbook.md` -> `docs/api_reference.md`;
- `docs/api_reference.md` -> public headers under `include/` and generated
  Doxygen HTML under `docs/api/html/`;
- `docs/maintainer_guide.md` -> generated-reference freshness and publication
  rules.

The stale tutorial phrase "API reference surface" was replaced with a direct
link to `api_reference.md`.

## Generated API Reference Freshness

Day 12 keeps the Day 10 and Day 11 decision: do not refresh generated HTML as
part of this reconciliation pass.

The current generated HTML inventory is partial for the checked-in public
header set:

- checked-in public headers: `18`;
- generated `sparse__*_8h.html` pages: `13`;
- missing generated pages:
  - `sparse__analysis_8h.html`;
  - `sparse__eigs_8h.html`;
  - `sparse__ic_8h.html`;
  - `sparse__ldlt_8h.html`;
  - `sparse__lu__csr_8h.html`.

This is now documented as a freshness boundary rather than hidden behind the
new API reference page.

## Maintainer Guidance Update

`docs/maintainer_guide.md` now says public-header/API-comment cleanup should
also check whether:

- `docs/api_reference.md` needs a header table or ownership update;
- generated API HTML should be refreshed;
- generated API HTML should instead be explicitly treated as stale/partial.

## Claim Scan

The unsupported-claim scan across touched docs and edited headers found only
explicit non-claim wording. No dynamic ABI, shared-library, package-manager,
runtime-loader, broad Windows parity, external-library parity, portable
performance, or state-of-the-art claim was introduced.

## Validation

Commands run:

```sh
git diff --check
test -f docs/api_reference.md && test -f docs/api/html/index.html && test -d include
```

Results:

- `git diff --check` passed.
- Link-target checks passed.
- The stale phrase scan for `API reference surface` and `generated API
  reference` returned no matches.
- Day 8, Day 9, and Day 12 normalized declaration diffs are `0` bytes.

Day 12 changed documentation only. No C source or public header declarations
were edited on Day 12, so the full C quality gate is deferred to Day 13 unless
Day 13 changes that scope.

## Day 13 Checklist

- Run `git diff --check`.
- Re-run the API reference link-target checks.
- Re-run the unsupported-claim scan across README, tutorial, cookbook,
  maintainer guide, API reference, and edited headers.
- Confirm Day 8, Day 9, and Day 12 declaration normalized diffs remain empty.
- Decide whether Day 13 should run a broader docs or generated-reference gate.
