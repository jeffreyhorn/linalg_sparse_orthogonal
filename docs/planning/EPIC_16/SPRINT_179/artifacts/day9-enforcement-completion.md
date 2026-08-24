# Sprint 179 Day 9: Enforcement Completion

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Complete the first implementation path for the strengthened local-only
generated API HTML decision. Day 9 tightens path handling, documents command
assumptions, confirms generated output cannot be accidentally committed or
published through current guards, and records remaining risks before Day 10
freshness/staging hardening.

## Implementation Completed

| File | Day 9 completion |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | Added checked-file existence failures before wording checks. |
| `scripts/check_api_docs_local_only.sh` | Added workflow-publication checks for generated API output path references. |
| `docs/maintainer_guide.md` | Retains Day 8 Sprint 179 product-decision wording. |

## Path Handling Notes

| Path or command | Day 9 behavior |
| --- | --- |
| script root detection | `SCRIPT_DIR` and `ROOT_DIR` continue to derive the repository root from the script location. |
| wording files | Guard now checks file existence before searching `README.md`, `docs/api_reference.md`, and `docs/maintainer_guide.md`. |
| generated output paths | Guard checks `docs/api`, `docs/api/html`, and `docs/api/html/index.html` through `git -C "$ROOT_DIR" check-ignore`. |
| tracked output scan | Guard uses `git -C "$ROOT_DIR" ls-files docs/api`. |
| staged output scan | Guard uses `git -C "$ROOT_DIR" diff --cached --name-only -- docs/api`. |
| visible untracked output scan | Guard uses `git -C "$ROOT_DIR" ls-files --others --exclude-standard docs/api`. |
| workflow scan | Guard scans `$ROOT_DIR/.github/workflows` when present and passes explicitly when no workflow directory exists. |

## Workflow Publication Guard

The selected Sprint 179 product status does not allow generated API HTML
publication, retained artifacts, or committed output. Day 9 adds workflow scans
that reject:

- `docs/api/html`
- `docs/api/`

inside `.github/workflows`.

This is intentionally scoped to generated API output paths. It does not block a
future local-only CI check from running `make api-docs-freshness`, but it does
block checked-in workflows from referencing the generated output tree as an
upload, deployment, or publication path while the Sprint 179 decision is in
force.

## Generated Output Commit-Policy Evidence

Day 9 validation confirms:

| State | Guard result |
| --- | --- |
| `docs/api/` ignored | Pass |
| `docs/api/html/` ignored | Pass |
| `docs/api/html/index.html` ignored | Pass |
| tracked generated API files | Pass: none found |
| staged generated API files | Pass: none found |
| non-ignored untracked generated API files | Pass: none found |
| generated API output path in workflows | Pass: none found |

## Validation Evidence

Direct guard output included:

```text
api-docs-local-only: no workflow generated API HTML output path references ok
api-docs-local-only: no workflow generated API output tree references ok
api-docs-local-only: passed
```

Full freshness output included:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
api-docs-local-only: no workflow generated API HTML output path references ok
api-docs-local-only: no workflow generated API output tree references ok
api-docs-local-only: passed
```

## Remaining Risks And Deferrals

| Risk or deferral | Day 9 status |
| --- | --- |
| Doxygen warnings are still not fatal by configuration. | Deferred to Day 10 freshness/staging review; Day 3 observed no warning output. |
| Workflow scan is string-based. | Accepted for the local-only decision; if generated API workflows are added later, a structured workflow guard should replace or narrow it. |
| No hosted or artifact publication metadata exists. | Intentional after Day 6 rejection of hosted/artifact status. |
| No generated API CI lane exists. | Intentional for the selected local-only status unless a future sprint chooses local-only CI proof. |
| Examples and Markdown adoption docs are not Doxygen inputs. | Intentional; generated API HTML remains declaration-level local reference. |

## Day 9 Deliverables

- completed implementation path
- path-handling notes
- generated output commit-policy evidence
- remaining-risk list
- Day 9 enforcement artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation satisfies the Day 6 acceptance requirements. | Complete | Local-only status, wording, staging, and workflow-publication checks now run in `api-docs-local-only`. |
| Local and CI command assumptions are documented. | Complete | Path handling and workflow publication guard sections above. |
| Accidental generated-output publication paths are guarded. | Complete | Guard rejects generated API output path references in `.github/workflows`. |
