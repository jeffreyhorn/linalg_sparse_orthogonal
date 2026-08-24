# Sprint 179 Day 12: Focused Verification

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Run the focused generated API and local-only guard checks required by the
Sprint 179 strengthened local-only product decision. Day 12 validates docs
generation, configured public-header page coverage, local-only staging, wording
guards, workflow publication guards, and whitespace hygiene.

## Commands Run

```bash
make docs
make docs-check
bash -n scripts/check_api_docs_local_only.sh
bash scripts/check_api_docs_local_only.sh
make api-docs-freshness
python3 scripts/check_api_docs_coverage.py
git ls-files docs/api
git diff --cached --name-only -- docs/api
git ls-files --others --exclude-standard docs/api
git diff --check
```

## Docs Generation Record

`make docs` passed:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
```

Interpretation:

- Doxygen generation runs from checked-in `Doxyfile`.
- Output remains under `docs/api/html/`.
- Generated output remains ignored local output and was not staged.

## Docs Check Record

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

Coverage status:

| Check | Result |
| --- | --- |
| checked-in public headers | 18 |
| generated reference pages | 18 |
| generated source pages | 18 |
| generated `sparse_version.h` Doxygen page | Not expected; separate installed-header policy row. |

## Strengthened Local-Only Guard Record

`bash -n scripts/check_api_docs_local_only.sh` passed with no syntax output.

`bash scripts/check_api_docs_local_only.sh` passed and reported:

```text
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: Doxyfile INPUT local-only contract ok
api-docs-local-only: Doxyfile FILE_PATTERNS local-only contract ok
api-docs-local-only: Doxyfile RECURSIVE local-only contract ok
api-docs-local-only: Doxyfile OUTPUT_DIRECTORY local-only contract ok
api-docs-local-only: Doxyfile GENERATE_HTML local-only contract ok
api-docs-local-only: Doxyfile HTML_OUTPUT local-only contract ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: README.md local-only freshness wording ok
api-docs-local-only: docs/api_reference.md local-only generated output wording ok
api-docs-local-only: docs/api_reference.md not hosted or source-controlled wording ok
api-docs-local-only: docs/maintainer_guide.md Sprint 179 product decision wording ok
api-docs-local-only: docs/maintainer_guide.md not hosted, artifact-published, or release evidence wording ok
api-docs-local-only: no workflow generated API HTML output path references ok
api-docs-local-only: no workflow generated API output tree references ok
api-docs-local-only: passed
```

## Aggregate Freshness Record

`make api-docs-freshness` passed. This proves the selected local generated API
freshness path still composes correctly:

1. regenerate Doxygen HTML;
2. check configured public-header generated page coverage;
3. check Doxyfile input/output contract;
4. enforce ignored/tracked/staged/non-ignored generated output policy;
5. enforce source-controlled local-only wording;
6. reject workflow references to generated API output publication paths.

## Staging And Whitespace Record

| Command | Result |
| --- | --- |
| `git ls-files docs/api` | Empty output. |
| `git diff --cached --name-only -- docs/api` | Empty output. |
| `git ls-files --others --exclude-standard docs/api` | Empty output. |
| `git diff --check` | Passed with no output. |

## Focused Negative-Coverage Notes

The strengthened guard now has scoped failure branches for:

- missing checked-in docs required for product-status wording;
- Doxyfile input/output contract drift;
- tracked generated API files;
- staged generated API files;
- visible non-ignored generated API files;
- missing local-only wording in README, API reference, or maintainer guide;
- workflow references to generated API output paths.

Day 12 did not mutate repository files to force each failure branch. The direct
positive guard run and syntax check are the maintained focused verification for
this documentation/script batch; integrated reconciliation on Day 13 should
inspect whether additional negative-test harnessing is worth adding or should
remain deferred.

## Day 12 Deliverables

- docs-generation validation record
- docs-check validation record
- generated API check record
- freshness and staging guard record
- Day 12 focused verification artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected generated API path passes its focused checks. | Complete | `make docs`, `make docs-check`, direct guard, and `make api-docs-freshness` all passed. |
| Whitespace and staged-file checks are clean. | Complete | `git diff --check` passed; generated API tracked/staged/non-ignored scans were empty. |
| Validation commands are reproducible from the artifact. | Complete | Commands and representative output are recorded above. |
