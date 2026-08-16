# Day 1 API Docs Intake

## Scope

Day 1 establishes Sprint 158 scope, artifact layout, generated API
documentation inputs, local tooling state, stop conditions, and the Day 2
Doxygen baseline handoff.

Sprint 158 owns T157-01/C157-01 from Sprint 157: generated API reference policy
must become explicit and reviewable, either through published generated HTML
with freshness evidence or through a guarded local-only policy.

## Branch And Tool Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-158` |
| Starting commit | `ab98ce4d32173617de3dc009e0a07e446d157042` |
| Starting commit summary | `ab98ce4d Merge pull request #175 from jeffreyhorn/sprint-157` |
| Doxygen binary | `/usr/local/bin/doxygen` |
| Doxygen version | `1.16.1` |
| Current generated-doc tracking state | `!! docs/api/` from `git status --ignored=matching --short docs/api` |
| Existing local generated HTML | `docs/api/html/` exists with 100 top-level generated files and remains ignored local context |

## Sprint 157 Inputs Reviewed

| Input | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day9-evidence-contract-templates.md` | Defines required API docs evidence fields: command, warnings, page coverage, version-header policy, and publication decision. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day10-quality-surface-map.md` | Defines required validation for generated API docs and escalation to full C/header gates if public headers change. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day12-risk-register-and-sprint158-handoff.md` | Defines generated API risks, mitigations, required Sprint 158 artifacts, and stop conditions. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day14-sprint-closeout-and-sprint158-handoff.md` | Confirms Sprint 158 should start from target T157-01 and claim C157-01. |

## Doxygen Configuration Baseline

| Configuration | Day 1 value |
| --- | --- |
| `PROJECT_NAME` | `linalg_sparse_orthogonal` |
| `INPUT` | `include/` |
| `FILE_PATTERNS` | `*.h` |
| `RECURSIVE` | `NO` |
| `OUTPUT_DIRECTORY` | `docs/api` |
| `GENERATE_HTML` | `YES` |
| `HTML_OUTPUT` | `html` |
| `EXTRACT_ALL` | `NO` |
| `EXTRACT_STATIC` | `NO` |
| `EXTRACT_PRIVATE` | `NO` |
| `WARN_IF_UNDOCUMENTED` | `YES` |
| `WARN_IF_DOC_ERROR` | `YES` |
| `WARN_NO_PARAMDOC` | `YES` |
| `WARN_AS_ERROR` | `NO` |
| `QUIET` | `YES` |

## API Documentation Input Inventory

| Input | Owner role | Day 1 note |
| --- | --- | --- |
| `Doxyfile` | Generator configuration | Defines input, output, extraction, and warning behavior. |
| `Makefile` `docs` target | Generation command | Runs `doxygen Doxyfile` and reports `docs/api/html/`. |
| `include/*.h` | Source-header-first public API authority | Current checked-in public header source set contains 18 files. |
| `include/sparse_version.h.in` | Generated installed version-header template | Must be explicitly classified for generated API docs coverage. |
| `docs/api_reference.md` | User-facing API reference entry point | Currently points users at source headers and generated HTML boundary. |
| `docs/maintainer_guide.md` | Maintainer policy | Owns generated Doxygen freshness and source-header-first interpretation. |
| `README.md` | User discovery | Links users to API reference and maintainer guide. |
| `docs/tutorial.md` | First-use guide | Routes users to API reference after first workflow. |
| `.gitignore` | Generated-output policy | Ignores `docs/api/`, so generated HTML is not source-controlled today. |

## Checked-In Public Header Source Set

| # | Header |
| ---: | --- |
| 1 | `include/sparse_analysis.h` |
| 2 | `include/sparse_bidiag.h` |
| 3 | `include/sparse_cholesky.h` |
| 4 | `include/sparse_csr.h` |
| 5 | `include/sparse_dense.h` |
| 6 | `include/sparse_eigs.h` |
| 7 | `include/sparse_ic.h` |
| 8 | `include/sparse_ilu.h` |
| 9 | `include/sparse_iterative.h` |
| 10 | `include/sparse_ldlt.h` |
| 11 | `include/sparse_lu.h` |
| 12 | `include/sparse_lu_csr.h` |
| 13 | `include/sparse_matrix.h` |
| 14 | `include/sparse_qr.h` |
| 15 | `include/sparse_reorder.h` |
| 16 | `include/sparse_svd.h` |
| 17 | `include/sparse_types.h` |
| 18 | `include/sparse_vector.h` |

The generated installed `sparse_version.h` is not checked in as a public
header. It is produced from `include/sparse_version.h.in`; Sprint 158 must
decide whether generated API docs should include the template, generated
install header behavior, or an explicit exclusion.

## Initial Risks And Stop Conditions

| Risk | Day 1 control |
| --- | --- |
| Generated HTML is mistaken for source-controlled evidence. | Treat existing `docs/api/html/` as ignored local context until Day 5/6 publication decision. |
| Doxygen warnings are hidden by quiet output. | Day 2 must capture command output and warning log behavior, not just exit status. |
| Page coverage misses public headers. | Day 3 must build coverage from checked-in `include/*.h` and explicitly handle generated version header behavior. |
| Source-header-first policy is weakened. | Keep public headers as declaration authority until publication decision says otherwise. |
| Header edits trigger unplanned validation scope. | Any `.c` or public `.h` edit requires `make format && make lint && make test`. |
| Generated docs imply unsupported product claims. | Scan changed public docs for generated, hosted, package, ABI, platform, parity, performance, and state-of-the-art wording. |

## Day 2 Handoff

Day 2 should run the Doxygen baseline without changing publication policy:

1. Run `make docs` from the repository root.
2. Capture command, Doxygen version, exit status, stdout/stderr, and warning
   text.
3. Inventory generated output under `docs/api/html/`.
4. Record whether output includes stale files, local paths, timestamps, or
   review-noise risks.
5. Keep `docs/api/` ignored unless the later publication decision changes it.

## Completion Check

- Sprint 158 scope is tied to the Epic 14 project plan and Sprint 157 handoff.
- Doxygen availability and generated-output tracking state are recorded.
- API documentation inputs and the checked-in public-header source set are
  identified.
- Stop conditions block unsupported generated-evidence and broad product
  claims before generation work begins.
