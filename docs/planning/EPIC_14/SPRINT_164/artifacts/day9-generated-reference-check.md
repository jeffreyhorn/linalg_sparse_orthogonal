# Sprint 164 Day 9: Generated Reference Policy Check

## Purpose

Day 9 checked whether the selected public-header cleanup batch remains
compatible with the maintained generated API-reference policy from Sprint 158.
The selected headers were:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

The check focused on generated Doxygen coverage, stale or unsupported wording,
and whether any gaps belonged to generated-reference policy/tooling rather than
to the Sprint 164 header-comment cleanup.

## Tooling Inspected

- `Makefile`
  - `docs` runs `doxygen Doxyfile`.
  - `api-docs-coverage` runs `scripts/check_api_docs_coverage.py`.
  - `docs-check` composes both steps.
- `Doxyfile`
  - `INPUT = include/`
  - `FILE_PATTERNS = *.h`
  - `RECURSIVE = NO`
  - `OUTPUT_DIRECTORY = docs/api`
  - `GENERATE_HTML = YES`
- `docs/api_reference.md`
  - Documents `docs/api/html/` as local-only generated output.
  - Routes exact declarations back to checked-in public headers.
- `docs/maintainer_guide.md`
  - Defines the Sprint 158 policy: generated Doxygen HTML is local-only,
    ignored, and current only for the checkout where `make docs-check` passed.
  - Explicitly forbids using generated API HTML to imply hosted docs,
    source-controlled generated output, shared-library ABI support, package
    manager distribution, external-library parity, or portable runtime
    guarantees.

## Generated Coverage Result

`make docs-check` passed:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

Selected-header generated pages were present after the gate:

- `docs/api/html/sparse__matrix_8h.html`
- `docs/api/html/sparse__matrix_8h_source.html`
- `docs/api/html/sparse__iterative_8h.html`
- `docs/api/html/sparse__iterative_8h_source.html`
- `docs/api/html/sparse__eigs_8h.html`
- `docs/api/html/sparse__eigs_8h_source.html`

## Stale Link And Claim Scan

A scoped scan of the selected generated reference/source pages for workflow
links and unsupported generated-reference claims found only the expected
repo-relative workflow navigation rendered from the selected header comments:

- `docs/solver_selection.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `examples/README.md`
- `docs/algorithm.md` in the eigensolver header page

No selected generated page introduced unsupported wording for:

- hosted generated documentation publication
- source-controlled generated HTML
- dynamic ABI compatibility
- shared-library support
- package-manager distribution
- broad Windows Makefile or Windows `pkg-config` parity
- external-library parity
- portable runtime guarantees
- state-of-the-art performance claims

`git status --short -- docs/api/html` produced no source-controlled generated
HTML churn.

## Policy Alignment

The selected header batch is compatible with the Sprint 158 generated-reference
policy:

- checked-in public headers under `include/` remain the exact declaration and
  call-site contract source of truth;
- generated Doxygen HTML remains local-only validation output;
- generated `sparse_version.h` remains outside the Doxygen input set and is
  governed by `VERSION`, `include/sparse_version.h.in`, install artifacts, and
  install-validation tests;
- generated-reference wording does not expand package, ABI, hosted-docs,
  platform-parity, or performance claims.

## Gaps Separated From Header Cleanup

No additional Sprint 164 header-comment edits were required on Day 9.

The remaining generated-reference limitations are policy/tooling boundaries,
not selected-header cleanup defects:

- `docs/api/html/` is ignored local output and is not a publication artifact.
- Doxygen renders repo-relative markdown paths as text navigation, not as a
  hosted documentation contract.
- Installed generated headers are intentionally outside the current Doxygen
  checked-in header coverage model.
- Broader generated API publication, if desired, belongs to generated-reference
  publication work rather than this header-cleanup sprint.

## Validation

- `make docs-check`
- selected generated reference/source page presence check
- scoped selected generated-page stale-link and unsupported-claim scan
- `git status --short -- docs/api/html`
