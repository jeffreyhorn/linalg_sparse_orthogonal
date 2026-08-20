# Day 1: API Docs Intake

## Purpose

Establish the Sprint 173 generated API HTML baseline before choosing or
implementing any publication path.

## Active Planning Source

The active Sprint 173 source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 173: Generated API HTML Publication Closure".

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but that path is
not the active Sprint 173 planning source. Sprint 173 continues from Epic 15.

## Sprint 173 Scope

Sprint 173 must decide and implement one supported generated API HTML path:

| Candidate path | Current status at Day 1 | Day 1 interpretation |
| --- | --- | --- |
| Hosted generated API HTML | Not selected | No hosted API-doc publication claim exists. |
| Committed generated API HTML | Not selected | `docs/api/` remains ignored; do not stage generated HTML. |
| CI artifact-only generated API HTML | Not selected | No artifact-only API-doc support claim exists yet. |
| Local-only generated API HTML | Current inherited policy | Local generated HTML is current only after `make docs-check` passes in the active checkout. |

Day 1 does not select the final path. It preserves the decision space for Day
3 and Day 4.

## Inherited Sprint 172 Handoff

Sprint 172 improved one public-header input family:

- `include/sparse_lu.h` was cleaned for local contract comments and workflow
  organization.
- `docs/tutorial.md` was aligned with the public six-argument
  `sparse_lu_refine(...)` signature.
- `scripts/check_lu_header_docs_guard.sh` was added as a focused LU
  header/docs drift guard.
- Sprint 172 explicitly did not regenerate, stage, publish, or claim fresh
  generated API HTML.

Sprint 173 can treat Sprint 172 as better generated-doc input, not as generated
HTML freshness or publication evidence.

## Current Generated API HTML Surface

### Configuration

`Doxyfile` currently defines:

- `INPUT = include/`;
- `FILE_PATTERNS = *.h`;
- `RECURSIVE = NO`;
- `OUTPUT_DIRECTORY = docs/api`;
- `GENERATE_HTML = YES`;
- `HTML_OUTPUT = html`;
- `WARN_IF_UNDOCUMENTED = YES`;
- `WARN_IF_DOC_ERROR = YES`;
- `WARN_NO_PARAMDOC = YES`;
- `WARN_AS_ERROR = NO`.

### Commands

The Makefile currently defines:

| Target | Behavior |
| --- | --- |
| `make docs` | Runs `doxygen Doxyfile` and writes HTML under `docs/api/html/`. |
| `make api-docs-coverage` | Runs `python3 scripts/check_api_docs_coverage.py`. |
| `make docs-check` | Runs `make docs` and then page coverage. |

### Coverage Check

`scripts/check_api_docs_coverage.py` currently:

- requires `docs/api/html/index.html`;
- enumerates checked-in `include/*.h`;
- expects a Doxygen reference page and source page for each checked-in public
  header;
- excludes generated installed `sparse_version.h` from Doxygen page
  expectations;
- reports checked-in header, reference-page, and source-page counts.

### Documentation Entry Points

`docs/api_reference.md` currently:

- treats checked-in headers under `include/` as the source of truth for exact
  declarations and call-site contracts;
- describes `make docs-check` as the local generated HTML validation command;
- states generated HTML under `docs/api/html/` is local-only generated output,
  ignored by the repository, and current only after `make docs-check` passes in
  the active checkout.

`docs/maintainer_guide.md` currently:

- keeps `docs/api_reference.md` as the user-facing API reference entry point;
- records the Sprint 158 policy that `docs/api/html/` is local-only and
  ignored rather than committed or hosted;
- says local generated output under `docs/api/html/` is not
  source-controlled, hosted, or release evidence;
- warns not to imply dynamic ABI compatibility, shared-library support,
  package-manager distribution, broad Windows Makefile or Windows `pkg-config`
  parity, portable runtime guarantees, hosted documentation publication,
  source-controlled generated HTML, or completeness beyond the configured
  Doxygen input set.

`README.md` exposes:

- `make docs`;
- `make docs-check`;
- `docs/api_reference.md` as the API reference entry point.

### Ignore/Staging State

`.gitignore` currently ignores:

- `docs/api/`;
- generated install header `include/sparse_version.h`;
- coverage and build outputs.

The working tree currently has ignored generated HTML files under
`docs/api/html/`. These files are local generated output and should not be
staged unless a later publication decision explicitly selects committed HTML.

## Claim Boundaries

Sprint 173 starts with no claim for:

- hosted generated API docs;
- source-controlled generated API docs;
- artifact-only generated API docs;
- generated API freshness without a just-passed local check;
- Doxygen coverage of generated installed headers;
- package-manager provider support;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra status.

## Day 1 Completion Check

Day 1 completion criteria are met:

- Sprint 173 scope is tied to the active Epic 15 project plan.
- Generated API HTML publication choices are explicit before implementation.
- Unsupported package, ABI, platform, performance, external-parity, and
  state-of-the-art claims remain protected.

No `.c` or `.h` files changed on Day 1, so the full C quality gate is not
required for this day.
