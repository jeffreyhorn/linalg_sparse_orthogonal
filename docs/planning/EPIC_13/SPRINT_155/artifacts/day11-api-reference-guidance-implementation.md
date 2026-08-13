# Sprint 155 Day 11 API Reference Guidance Implementation

## Purpose

Day 11 implemented the Day 10 API reference publication plan without refreshing
generated HTML. The implementation adds a stable user-facing Markdown entry
point and maintainer publication rules while keeping generated Doxygen output
as an explicitly freshness-governed convenience view.

## Changes

- Added `docs/api_reference.md`.
- Linked the API reference from:
  - `README.md`;
  - `docs/tutorial.md`;
  - `docs/cookbook.md`.
- Added generated API-reference publication and freshness guidance to
  `docs/maintainer_guide.md`.
- Updated `docs/planning/EPIC_13/SPRINT_155/WORKING_NOTES.md`.

## API Reference Index Scope

`docs/api_reference.md` now:

- lists the checked-in public headers under `include/`;
- states that public headers are the source of truth for declarations and
  call-site contracts;
- explains that installed packages include generated `sparse_version.h`;
- points to generated Doxygen HTML under `docs/api/html/`;
- warns that generated HTML is a convenience view of the configured Doxygen
  input set, not a broader support claim;
- routes users back to README, tutorial, cookbook, solver-selection, install,
  and maintainer docs for workflow and policy topics.

## Generated HTML Decision

Generated HTML was not refreshed on Day 11. Day 10 found that the checked-in
`docs/api/html/` tree is partial for the current header set, and refreshing it
would create a large generated-output diff. The Day 11 implementation instead
records the publication rule and leaves the generated refresh as an explicit
reviewable action.

## Freshness Rule Added

The maintainer guide now says generated API HTML is fresh only when maintainers
have run `make docs`, triaged Doxygen warnings, checked generated page
coverage, committed generated output with the corresponding source/header
change or a dedicated refresh, and described generated-output changes in the
review.

## Claim Boundaries Preserved

The new API reference wording does not imply:

- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable performance;
- broad completeness beyond the configured Doxygen input set;
- state-of-the-art coverage.

## Validation

Commands run:

```sh
git diff --check
test -f docs/api_reference.md && test -f docs/api/html/index.html && test -d include
```

Results:

- `git diff --check` passed.
- Link-target checks for the new API reference, generated HTML index, and
  public-header directory passed.
- The unsupported-claim scan found only explicit non-claim wording.

## Day 12 Handoff

Day 12 should reconcile tutorial, public-header, and API-reference references
after the new index. It should run link/claim scans, preserve declaration
evidence from Days 8-9, and decide whether any generated-reference refresh
belongs in this sprint or should remain a deferred explicit refresh.
