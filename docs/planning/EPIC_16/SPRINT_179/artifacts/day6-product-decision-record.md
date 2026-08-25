# Sprint 179 Day 6: Product Decision Record

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Decision

Sprint 179 selects **strengthened local-only generated API HTML** as the
supported generated API HTML product status.

Generated Doxygen HTML remains:

- generated from checked-in public headers under `include/`;
- written locally under `docs/api/html/`;
- ignored by source control through `docs/api/`;
- validated locally with `make api-docs-freshness`;
- not hosted, artifact-published, committed, or release evidence.

The supported user-facing API path remains:

1. `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, and
   `docs/solver_selection.md` for workflow selection and adoption guidance.
2. `docs/api_reference.md` for the compact API reference index.
3. Checked-in public headers under `include/` for exact declarations and
   call-site contracts.
4. `make api-docs-freshness` for branch-local generated Doxygen HTML
   inspection.

## Rationale

Day 5 recommended strengthened local-only status because it best fits the
current evidence and repository maintenance model.

| Evidence | Rationale |
| --- | --- |
| `make api-docs-freshness` already passes. | The project has a maintained local generation, coverage, and staging proof. |
| `docs/api/` is ignored and untracked. | Current repository policy deliberately keeps generated HTML out of source-controlled diffs. |
| No generated API workflow lane exists. | Hosted or retained artifact status would require new CI ownership before any public claim could be true. |
| Public headers and Markdown docs are source-of-truth surfaces. | Local-only generated HTML preserves current API authority and avoids treating generated output as hand-edited source. |
| Day 3 found no configured-input page or warning blocker. | Local generated HTML is usable as a refreshed local view without requiring publication. |

## Alternatives Considered

| Alternative | Decision | Reason |
| --- | --- | --- |
| Hosted generated API HTML publication | Rejected for Sprint 179 | No Pages/deployment path, publication metadata, hosted freshness guard, or workflow drift guard currently exists. |
| Retained CI artifact | Rejected for Sprint 179 | Would require a new generated API CI lane, artifact name/retention policy, fail-closed upload behavior, and metadata contract. |
| Committed generated HTML output | Rejected for Sprint 179 | Conflicts with `.gitignore` and the current local-only guard; would create large generated diffs and weaken review focus. |
| Strengthened local-only status | Accepted | Matches current evidence, keeps generated output derived and ignored, and closes ambiguous product status with lower implementation risk. |

## Implementation Acceptance Requirements

Days 7-12 must implement or verify the selected local-only status against these
requirements:

| Area | Acceptance requirement |
| --- | --- |
| Freshness | `make api-docs-freshness` remains the selected generated API freshness proof and regenerates output before checking required pages. |
| Coverage | Generated reference and source pages remain required for every checked-in public header under the configured `Doxyfile` input set. |
| Staging | `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` remain ignored; tracked, staged, or visible non-ignored generated API files fail the guard. |
| Navigation | README, API reference, maintainer guide, and support-tier wording point users to `docs/api_reference.md`, public headers, and local generation commands without implying hosted publication. |
| Guard discoverability | The local-only guard contract is easy for maintainers and reviewers to find. |
| Claim boundary | Public docs explicitly distinguish local generated HTML from hosted, source-controlled, artifact-published, or release evidence. |

## Supported Claims After This Decision

Sprint 179 may support these claims once the remaining implementation and
verification days complete:

- The project has a maintained local generated API HTML path.
- `make api-docs-freshness` regenerates Doxygen HTML, checks configured public
  header page coverage, and enforces local-only generated-output staging.
- `docs/api_reference.md` and checked-in public headers remain the supported
  source-controlled API reference path.
- Generated API HTML is current only for the branch and checkout where
  `make api-docs-freshness` has just passed.
- Generated API HTML covers the configured checked-in public headers under
  `include/`, not every documentation or adoption surface.

## Unsupported Claims After This Decision

Documentation, workflow comments, sprint artifacts, and support-tier wording
must not imply:

- hosted generated API documentation;
- retained CI artifact publication for generated API HTML;
- source-controlled generated API HTML;
- generated API HTML as release evidence;
- generated API completeness beyond configured `Doxyfile` inputs;
- generated API pages for examples, tutorial, cookbook, solver-selection,
  install, maintainer, or planning docs;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad platform parity;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable performance or runtime guarantees;
- state-of-the-art claim coverage from generated HTML.

## Guard And Documentation Follow-Through

| Day | Follow-through |
| --- | --- |
| Day 7 | Design the strengthened local-only implementation path and decide whether an additional drift guard is needed. |
| Day 8 | Implement the first batch of guard or documentation-contract changes. |
| Day 9 | Complete remaining enforcement details and reconcile path assumptions. |
| Day 10 | Tighten freshness/staging guard behavior as needed. |
| Day 11 | Update navigation and claim wording across public and maintainer docs. |
| Day 12 | Run focused generated API validation and whitespace checks. |

## Pass/Fail Contract

The selected product status passes when:

- one product status is documented as authoritative: strengthened local-only;
- generated API HTML remains ignored derived output;
- `make api-docs-freshness` remains the named freshness/staging command;
- docs navigation does not imply a hosted, artifact-published, or
  source-controlled generated API surface;
- future generated API status changes would require an explicit new product
  decision.

The selected product status fails if:

- generated HTML status remains ambiguous;
- generated HTML can be cited as hosted, release, or source-controlled evidence;
- generated files can be staged accidentally without a guard failure;
- README, API reference, maintainer guide, or support-tier wording implies
  publication evidence that does not exist;
- workflow changes add generated API publication without metadata and
  fail-closed checks.

## Day 6 Deliverables

- generated API HTML product decision
- implementation acceptance requirements
- supported and unsupported claim list
- rejected alternatives record
- Day 6 product decision record

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| One supported path is selected. | Complete | Strengthened local-only generated API HTML status is selected above. |
| Implementation work has a clear pass/fail contract. | Complete | Acceptance requirements and pass/fail contract are listed above. |
| Documentation updates cannot overstate the selected path. | Complete | Supported and unsupported claim lists define allowed wording boundaries. |
