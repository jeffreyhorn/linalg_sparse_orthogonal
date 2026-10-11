# Sprint 213 Day 2: Current Local-Only Baseline

## Summary

Day 2 captures the current generated API local-only behavior before any
publication policy decision. The baseline confirms that local Doxygen
generation, generated-page coverage, local-only staging checks, workflow
publication checks, and source-controlled routing checks all pass on the
current branch.

The generated Doxygen tree remains ignored local output. It is current for the
checked-in public headers in this checkout only, and it is not hosted,
retained, committed, or release evidence.

## Commands Run

| Command | Result |
| --- | --- |
| `make api-docs-freshness` | Passed. Doxygen generated local HTML; coverage checked 18 public headers, 18 generated reference pages, and 18 generated source pages; local-only and routing guards passed. |
| `python3 tests/test_api_docs_coverage.py && python3 tests/test_api_docs_local_only_guard.py && python3 tests/test_api_docs_routing.py` | Passed. Standalone coverage, local-only, and routing regression suites completed. |
| `git check-ignore -v docs/api docs/api/html docs/api/html/index.html include/sparse_version.h` | Passed. `docs/api/` and descendants are ignored by `.gitignore:44`; generated `include/sparse_version.h` is ignored by `.gitignore:48`. |
| `git status --short --ignored docs/api include/sparse_version.h` | Passed. Generated docs appear as ignored `!! docs/api/` only. |

The generated `docs/api/` tree is ignored local output and was not added to
the branch.

## Generated Output Snapshot

| Field | Observed value |
| --- | --- |
| Doxygen input | `include/`, `*.h`, non-recursive |
| Output root | `docs/api/` |
| HTML output | `docs/api/html/` |
| Index page | `docs/api/html/index.html` present |
| Checked-in public headers | 18 |
| Generated reference pages | 18 |
| Generated source pages | 18 |
| Generated installed header policy | `sparse_version.h` is a separate installed-header policy row, not an expected generated Doxygen page |
| Git tracking status | ignored local output only |

## Current Local-Only Guard Behavior

| Guard | Current proof |
| --- | --- |
| Coverage | Generated reference/source pages exist and are fresh for each checked-in public header selected by `Doxyfile`. |
| Ignore/staging | `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` are ignored; no generated API files are staged, tracked, or non-ignored. |
| Doxyfile settings | `INPUT=include/`, `FILE_PATTERNS=*.h`, `RECURSIVE=NO`, `OUTPUT_DIRECTORY=docs/api`, `GENERATE_HTML=YES`, and `HTML_OUTPUT=html` satisfy the local-only contract. |
| Documentation wording | README, API reference, and maintainer guide carry required local-only wording. |
| Workflow publication | Current workflows do not reference generated API output paths or generated API publication semantics. Existing `upload-artifact` steps are for other evidence lanes. |
| Routing | Seven routing documents pass; generated API publication links are absent; source-controlled API entry point is `docs/api_reference.md`. |

## Documentation Baseline

| Surface | Baseline wording |
| --- | --- |
| `README.md` | Users run `make docs-check` or `make api-docs-freshness` and use `docs/api_reference.md`; generated HTML is local-only ignored output, not hosted, retained, committed, or release evidence. |
| `INSTALL.md` | The support/readiness matrix classifies generated API HTML as local-only with no hosted API publication, retained generated-doc artifact, committed generated HTML, or completeness beyond Doxyfile-selected checked-in headers. |
| `docs/api_reference.md` | Checked-in headers own exact declarations; generated HTML is local-only output current only after `make api-docs-freshness`; routing rejects generated-output and unsupported hosted publication links. |
| `docs/maintainer_guide.md` | Sprint 204 is the current policy owner; patches that add hosted URLs, Pages deployment, artifact uploads, or committed `docs/api/` content must reopen the product decision. |

## Policy Baseline For Day 3

| Candidate policy | Baseline implication |
| --- | --- |
| Stronger local-only | Starts from a passing validation chain and would mainly harden guard fixtures, wording, and workflow scanning. |
| Hosted generated API HTML | Requires new deployment workflow, hosted URL policy, freshness ordering, stale-output rollback, routing exceptions, and user/maintainer docs. |
| Retained generated-doc artifact | Requires narrow upload path, artifact retention/naming, artifact freshness proof, non-release wording, and guard exceptions. |
| Committed generated HTML | Requires changing ignore/staging policy, controlling generated review noise, preserving freshness checks, and documenting source/generated ownership. |

## Day 2 Outcome

Item 213.1 now has evidence-backed current-state data. The baseline
distinguishes source-controlled API routes from generated Doxygen output and
records the exact local-only contract that any Sprint 213 policy change must
replace or preserve.

Day 3 should compare local-only, hosted Pages, retained artifacts, and
committed generated HTML against this baseline before the Day 5 decision.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

