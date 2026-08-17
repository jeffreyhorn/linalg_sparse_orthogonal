# Day 14 Closeout Handoff

## Sprint 158 Outcome

Sprint 158 closed the generated API HTML publication residual with an explicit
no-commit product decision and a recurring local freshness guard.

Final decision:

- keep generated Doxygen API HTML local-only under `docs/api/html/`;
- keep `docs/api/` ignored;
- do not add hosted Doxygen API publication in Sprint 158;
- do not commit generated HTML;
- use `make docs-check` as the maintained local generation and page-coverage
  guard;
- keep `docs/api_reference.md` and checked-in public headers as the
  source-controlled API reference authority.

## Delivered Changes

| Area | Delivered |
| --- | --- |
| Doxygen baseline | Captured Doxygen version, warning inventory, generated file/page inventory, local path scan, and generated version-header behavior. |
| Publication decision | Compared committed HTML, hosted HTML, and local-only guard paths; selected local-only generated output. |
| Coverage guard | Added `scripts/check_api_docs_coverage.py`, `make api-docs-coverage`, and `make docs-check`. |
| Warning closure | Fixed selected Doxygen warnings in public-header comments without declaration or behavior changes. |
| Docs alignment | Updated API reference, maintainer guide, and README command inventory to describe the selected policy. |
| Validation | Ran docs guard, whitespace checks, claim scans, and the full public-header quality gate. |
| Handoff | Recorded residuals and Sprint 159 hosted-report prerequisites. |

## Artifact Inventory

| Day | Artifact |
| --- | --- |
| 1 | `day1-api-docs-intake.md` |
| 2 | `day2-doxygen-baseline.md` |
| 3 | `day3-public-header-coverage-map.md` |
| 4 | `day4-warning-triage-policy.md` |
| 5 | `day5-publication-options.md` |
| 6 | `day6-publication-decision.md` |
| 7 | `day7-page-coverage-check-design.md` |
| 8 | `day8-coverage-implementation.md` |
| 9 | `day9-warning-fix-batch.md` |
| 10 | `day10-policy-alignment.md` |
| 11 | `day11-publication-finalization.md` |
| 12 | `day12-validation-evidence.md` |
| 13 | `day13-claim-reconciliation.md` |
| 14 | `day14-closeout-handoff.md` |

## Final Validation Evidence

Latest recorded validation:

```text
make docs-check
make format && make lint && make test
git diff --check
```

Results:

- `make docs-check` passed with no Doxygen warnings and complete generated page
  coverage for 18 checked-in public headers.
- `make format && make lint && make test` passed after the public-header
  comment fixes.
- `git diff --check` passed.
- trailing-whitespace scans over touched docs/planning/source surfaces passed.
- focused claim scans found only explicit non-claims, historical option
  analysis, or unrelated bounded-evidence language.
- final Day 14 `make docs-check`, `git diff --check`, and trailing-whitespace
  scan passed after closeout edits.

## Final Tracking State

Generated API HTML remains ignored:

```text
!! docs/api/
```

No generated HTML is tracked or staged.

Intended tracked changes for Sprint 158 are:

- `Makefile`;
- `README.md`;
- `docs/api_reference.md`;
- `docs/maintainer_guide.md`;
- `include/sparse_iterative.h`;
- `include/sparse_lu_csr.h`;
- `include/sparse_types.h`;
- `scripts/check_api_docs_coverage.py`;
- `docs/planning/EPIC_14/SPRINT_158/**`.

## Residuals

| Residual | Disposition |
| --- | --- |
| Hosted generated API HTML publication | Not selected for Sprint 158. Requires separate hosted-docs artifact retention and support-tier policy. |
| Committed `docs/api/html/` | Not selected for Sprint 158. Avoids generated-output churn while preserving source-header-first API authority. |
| Generated `sparse_version.h` Doxygen page | Not selected under the current checked-in-header Doxygen input set. Version macro behavior remains install-artifact evidence. |
| Broad generated API reference completeness | Explicit non-claim. Coverage applies only to configured checked-in public-header input pages. |
| Hosted generated report promotion | Deferred to Sprint 159 and kept separate from generated API HTML publication. |

## Sprint 159 Handoff

Sprint 159 should begin from these settled Sprint 158 boundaries:

- generated API HTML is local-only and ignored;
- `make docs-check` is the API-doc freshness and page-coverage guard;
- `docs/api_reference.md` and checked-in public headers own exact API
  declarations and call-site contracts;
- ignored generated HTML is not hosted, release, package, ABI, platform,
  performance, parity, or state-of-the-art evidence.

Recommended Sprint 159 hosted-report prerequisites:

1. Select claim-bearing generated oracle/comparison report families for hosted
   promotion.
2. Keep non-selected report families explicitly local-only or advisory.
3. Define artifact retention, branch freshness, and failure semantics before
   adding hosted jobs.
4. Keep hosted report evidence separate from Doxygen API HTML publication.
5. Add public and maintainer wording naming exactly which hosted rows are
   reviewed evidence.
6. Preserve package, ABI, platform, performance, external-library parity, and
   state-of-the-art non-claims unless independently validated.

## Retrospective Inputs

What closed:

- generated API docs now have a deterministic local guard;
- selected Doxygen warnings are fixed;
- generated output policy is unambiguous;
- live docs and maintainer policy match the selected local-only path;
- validation evidence is complete for the touched public-header and docs
  surfaces.

What remains intentionally out of scope:

- hosted Doxygen API publication;
- committed generated HTML;
- hosted generated report evidence, which belongs to Sprint 159;
- broader product claims not supported by Sprint 158 evidence.
