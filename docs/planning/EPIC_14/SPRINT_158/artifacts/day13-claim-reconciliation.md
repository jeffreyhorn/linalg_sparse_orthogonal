# Day 13 Claim Reconciliation

## Scope

Day 13 reconciles Sprint 158 generated API documentation claims, artifacts, and
Sprint 159 handoff boundaries before closeout.

The selected Sprint 158 publication path remains local-only generated API HTML
with a recurring freshness guard:

```text
make docs-check
```

Generated HTML under `docs/api/html/` remains ignored local output, not hosted
or source-controlled evidence.

## Claim Audit

| Surface | Audit result |
| --- | --- |
| `README.md` | Lists `make docs-check` as the local generated API page-coverage command and routes exact declaration readers to `docs/api_reference.md` and public headers. No hosted/generated API HTML claim was introduced. |
| `docs/api_reference.md` | Describes generated HTML as local-only, ignored, and current only for the checkout where `make docs-check` has just passed. Keeps public headers as exact declaration authority. |
| `docs/maintainer_guide.md` | Defines `make docs-check` interpretation, ignored generated-output semantics, generated `sparse_version.h` policy, and explicit non-claims. |
| `docs/tutorial.md` | Already routes exact declaration readers to `docs/api_reference.md` and public headers. No generated HTML publication claim was introduced. |
| `Makefile` | Owns `docs`, `api-docs-coverage`, and `docs-check`; no hosted publication or generated-output tracking change. |
| `.gitignore` | Continues to ignore `docs/api/`. |

## Unsupported Claim Scan

The Day 13 scan checked live docs and Sprint 158 artifacts for:

- hosted generated API freshness;
- source-controlled generated HTML;
- release evidence from ignored local generated output;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad platform parity;
- external-library parity;
- portable performance;
- state-of-the-art coverage.

Findings:

- live API-reference and maintainer-guide matches are explicit non-claims or
  local-only boundaries;
- README and tutorial matches are existing non-claim language outside the
  generated API publication path;
- Sprint 158 artifact matches are either selected-policy evidence or historical
  option analysis.

No unsupported generated API documentation claim was introduced by Sprint 158.

## Artifact-To-Item Reconciliation

| Sprint 158 item | Status | Evidence |
| --- | --- | --- |
| 1. Doxygen Baseline | Closed | Day 1 intake, Day 2 Doxygen baseline, Day 3 public-header coverage map. |
| 2. Publication Decision | Closed | Day 5 publication options, Day 6 publication decision, Day 11 publication finalization. |
| 3. Coverage Check | Closed | Day 7 coverage-check design, Day 8 `scripts/check_api_docs_coverage.py`, `api-docs-coverage`, and `docs-check`. |
| 4. Warning Triage | Closed | Day 4 warning triage and Day 9 warning fix batch. |
| 5. Docs Alignment | Closed | Day 10 policy alignment, Day 11 README command inventory update. |
| 6. Validation | Closed | Day 12 validation evidence: `make docs-check`, `make format && make lint && make test`, `git diff --check`, trailing-whitespace scan. |
| 7. Closeout | In progress | Day 13 reconciliation complete; Day 14 should write final closeout and Sprint 159 handoff. |

## Residuals

| Residual | Status | Rationale |
| --- | --- | --- |
| Hosted generated API HTML publication | Not selected | Sprint 158 deliberately chose local-only generated API HTML. Hosted Doxygen publication would require a separate artifact retention/support-tier policy. |
| Committed `docs/api/html/` | Not selected | Sprint 158 avoided generated-output review churn and kept checked-in headers as API authority. |
| Generated `sparse_version.h` Doxygen page | Not selected | Current Doxygen input is checked-in `include/*.h`; generated install headers remain owned by install artifacts, `VERSION`, and install-validation tests. |
| Broad API/reference completeness claim | Blocked | `docs-check` proves coverage for the configured checked-in public-header input set only. |
| Hosted generated report promotion | Deferred to Sprint 159 | Sprint 159 owns hosted freshness promotion for selected oracle/comparison report rows, not Doxygen API HTML publication. |

## Sprint 159 Handoff Draft

Sprint 159 should start from the Sprint 158 policy boundary:

- generated API HTML is local-only and ignored;
- `make docs-check` is the local API-doc freshness guard;
- source-controlled API truth remains `docs/api_reference.md` plus checked-in
  public headers;
- ignored generated output is not hosted, release, package, ABI, platform,
  performance, parity, or state-of-the-art evidence.

Concrete Sprint 159 prerequisites for hosted generated report work:

1. Select which generated oracle/comparison report families are claim-bearing
   enough to promote into hosted evidence.
2. Keep non-selected report families explicitly local-only or advisory.
3. Define artifact retention, branch freshness, and failure semantics before
   adding hosted jobs.
4. Separate hosted report evidence from generated API HTML; do not imply
   Doxygen HTML hosting unless a future sprint explicitly funds that lane.
5. Add public and maintainer wording that names exactly which hosted rows are
   reviewed evidence and which remain local-only.
6. Preserve non-claims for package, ABI, platform, performance,
   external-library parity, and state-of-the-art coverage unless separately
   validated.

## Completion Check

- Generated API documentation claims are evidence-bound.
- Sprint 158 deliverables are closed or explicitly residualized.
- Sprint 159 starts with concrete hosted-report prerequisites and boundaries.

## Validation

Day 13 changed planning documentation only. Validation focused on claim and
documentation hygiene:

```text
git diff --check
```

Result: passed.

A trailing-whitespace scan over README, API reference, maintainer guide,
tutorial, and Sprint 158 planning artifacts passed.

A claim-sensitive scan over README, API reference, maintainer guide, tutorial,
and the Day 13 artifact found only explicit non-claims, existing unrelated
bounded-evidence language, or the Day 13 audit terms themselves.
