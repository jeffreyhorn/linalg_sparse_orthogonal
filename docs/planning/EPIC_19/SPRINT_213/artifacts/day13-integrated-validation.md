# Sprint 213 Day 13: Integrated Validation

## Purpose

Day 13 validates the Sprint 213 stronger local-only generated API policy as an
integrated documentation and guard chain. The selected policy remains unchanged:
generated API HTML is local ignored output, while hosted generated API HTML,
retained generated-doc artifacts, and committed generated HTML remain unclaimed
future options.

## Validation Matrix

| Command | Result | Evidence |
| --- | --- | --- |
| `make docs-check` | Pass | Doxygen regenerated `docs/api/html/`; coverage passed with 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` retained as a separate generated-header policy row. |
| `python3 tests/test_api_docs_coverage.py` | Pass | Standalone API docs coverage regression suite completed without failure. |
| `python3 tests/test_api_docs_local_only_guard.py` | Pass | Local-only workflow, staging, publication, folded-command, quoted-comment, and `.yaml` workflow regressions completed without failure. |
| `python3 tests/test_api_docs_routing.py` | Pass | Routing, required-route, publication-link, encoded-link, documentation-marker, and residual-option regressions completed without failure. |
| `make api-docs-freshness` | Pass | Serialized docs generation, coverage, local-only, and routing validation completed; routing checked seven documents and confirmed generated API publication links absent. |
| `git diff --check` | Pass | No whitespace errors in the final Day 13 diff after writing the notes and artifact. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No C source or header files are modified. |
| `git status --short --branch` | Informational | Branch remains `sprint-213` with Sprint 213 documentation, generated API guard, and regression-test changes pending. |

## Integrated Surfaces

| Surface | Day 13 evidence |
| --- | --- |
| Generated page coverage | `make docs-check`, the standalone coverage suite, and `make api-docs-freshness` all keep the checked-in public-header page contract intact. |
| Local-only workflow policy | The standalone local-only suite and aggregate freshness target keep generated API HTML out of workflow publication, staging, artifact, and tracked-file surfaces. |
| Source-controlled route policy | The standalone routing suite and aggregate freshness target keep user routes on source-controlled API documentation and reject generated/hosted publication links. |
| User and maintainer documentation | Required markers in README, INSTALL, API reference, and maintainer guide remain present under the routing/local-only guards. |
| Planning residuals | Sprint 213 planning records stronger local-only closure and leaves hosted generated API HTML, retained generated-doc artifacts, and committed generated HTML unclaimed. |

## Full C Gate Decision

`git diff --name-only -- '*.c' '*.h'` produced no files. Day 13 does not run
`make format && make lint && make test` because the sprint instruction requires
that full C quality gate only when C source or header files changed.

## Outcome

Item 213.6 is complete for integrated validation. The selected generated API
policy now has command-backed evidence across docs generation, coverage,
local-only workflow detection, routing/link validation, aggregate freshness, and
workspace hygiene.

Day 14 should perform final closeout review, reconcile the Sprint 213 notes and
artifacts, and prepare retrospective inputs without changing the selected
generated API publication policy.
