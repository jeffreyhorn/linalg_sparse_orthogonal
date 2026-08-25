# Sprint 179 Day 7: Implementation Design

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Design the implementation path for the Day 6 product decision before changing
behavior. Sprint 179 selected strengthened local-only generated API HTML, so
Day 7 scopes implementation to guard hardening and documentation clarity rather
than hosted publication, retained CI artifacts, or committed generated output.

## Selected Path

The selected path is:

- keep `docs/api/html/` as local generated Doxygen output;
- keep `docs/api/` ignored and untracked;
- keep `make api-docs-freshness` as the supported validation command;
- keep `docs/api_reference.md` and public headers under `include/` as the
  source-controlled API reference path;
- reject hosted, artifact-published, and committed generated API HTML for
  Sprint 179.

## Implementation Owner Files

| File or surface | Planned role | Change expectation |
| --- | --- | --- |
| `scripts/check_api_docs_local_only.sh` | Primary local-only generated-output guard. | Candidate for Day 8/Day 10 tightening so the script also validates product-status wording or points maintainers to the decision. |
| `scripts/check_api_docs_coverage.py` | Required generated page coverage guard. | Preserve behavior; only change if Day 10 finds a concrete coverage-policy gap. |
| `Makefile` | Command namespace for docs generation and validation. | Preserve `make docs`, `make docs-check`, and `make api-docs-freshness`; consider adding a clearer status/contract target only if it reduces ambiguity. |
| `.gitignore` | Enforces ignored generated API output path. | Preserve `docs/api/` ignore rule. |
| `README.md` | User-facing command list and claim boundary. | Tighten wording if needed so local-only status is visible without implying hosted docs. |
| `docs/api_reference.md` | Compact API reference entry point and generated HTML interpretation. | Tighten local-only product-status wording and source-of-truth guidance if needed. |
| `docs/maintainer_guide.md` | Maintainer reproduction and support-tier interpretation. | Update Sprint 179 decision language and guard expectations. |
| `.github/workflows/*.yml` | Hosted CI workflows. | No planned edits under the selected local-only status unless a drift guard is added outside publication lanes later. |
| `docs/api/html/` | Ignored generated output. | Do not edit or commit generated HTML. |

## Command And Artifact Naming Plan

| Name | Decision |
| --- | --- |
| `make docs` | Keep as local Doxygen generation command. |
| `make docs-check` | Keep as generation plus configured-header page coverage. |
| `make api-docs-local-only` | Keep as local-only generated-output staging guard. |
| `make api-docs-validate` | Keep as combined docs-check plus local-only guard. |
| `make api-docs-freshness` | Keep as the documented supported command for local generated API freshness. |
| Hosted artifact name | None; hosted/artifact publication is rejected for Sprint 179. |
| Publication metadata path | None; no hosted or artifact publication metadata is created under the local-only decision. |

## Failure Message Design

Future guard edits should keep failures concrete and local-only specific:

| Failure | Message intent |
| --- | --- |
| `docs/api/` is not ignored | Explain that generated API HTML must remain ignored local output unless a future product decision selects committed output. |
| generated API files are tracked | Explain that local-only generated HTML must not be source-controlled. |
| generated API files are staged | Explain that staged generated API files must be unstaged unless a future product decision selects committed output. |
| generated API files are visible as non-ignored untracked output | Explain that generated HTML belongs under ignored `docs/api/`. |
| required local-only wording disappears | Explain which source-controlled doc must continue to identify generated API HTML as local-only and not hosted/artifact-published/release evidence. |

## Freshness Verification Design

The strengthened local-only design keeps freshness local and reproducible:

1. `make api-docs-freshness` runs `make docs`.
2. `make docs` regenerates Doxygen HTML from `Doxyfile`.
3. `scripts/check_api_docs_coverage.py` requires `docs/api/html/index.html`.
4. The coverage script requires reference and source pages for every checked-in
   public header under configured input path `include/`.
5. `scripts/check_api_docs_local_only.sh` enforces ignored, untracked,
   unstaged, and non-visible untracked generated output under `docs/api/`.

This design intentionally does not claim CI freshness, hosted freshness,
release freshness, or publication freshness.

## Staging Policy Design

| Generated output state | Selected behavior |
| --- | --- |
| Ignored local output under `docs/api/html/` | Allowed after local generation. |
| Tracked files under `docs/api/` | Rejected by local-only guard. |
| Staged files under `docs/api/` | Rejected by local-only guard. |
| Non-ignored untracked files under `docs/api/` | Rejected by local-only guard. |
| Committed generated HTML | Rejected for Sprint 179 unless a future product decision replaces this one. |

## Navigation Update Design

Navigation should point users through supported source-controlled docs first:

1. README remains the front door and should name `docs/api_reference.md` as the
   API reference entry point.
2. `docs/api_reference.md` should state that generated HTML is a local
   convenience view, not hosted/source-controlled/artifact-published output.
3. `docs/maintainer_guide.md` should identify Sprint 179 as the current
   product decision replacing the older Sprint 158-only wording.
4. Tutorial, cookbook, and solver-selection docs do not need generated API
   navigation changes unless Day 11 finds wording that implies hosted generated
   docs.

## Workflow Design

No workflow publication or upload path should be added under the selected
Sprint 179 decision.

| Workflow topic | Day 7 design |
| --- | --- |
| Generated API artifact upload | Do not add. |
| GitHub Pages deployment | Do not add. |
| Hosted publication metadata | Do not add. |
| CI docs freshness lane | Optional only if later implementation explicitly wants CI to prove local-only status; not required for the Day 6 decision. |
| Workflow guard | Add only if a future workflow starts touching generated API docs or publication wording. |

## Day 8/Day 9 Implementation Candidates

The strongest implementation candidates are:

1. Add local-only product-status wording checks to
   `scripts/check_api_docs_local_only.sh`.
2. Update maintainer guide wording from the historical Sprint 158 policy to the
   current Sprint 179 strengthened local-only decision.
3. Tighten README and API reference wording only where needed to make the
   supported path obvious.
4. Preserve all generated HTML as ignored output and avoid editing
   `docs/api/html/`.

## Out Of Scope

- Hosted generated API documentation.
- Retained CI generated API artifacts.
- Committed generated API HTML.
- Generated API release evidence.
- Doxygen examples/tutorial/cookbook publication.
- Dynamic ABI, shared-library, package-manager, platform-parity, external
  parity, performance, runtime, or state-of-the-art claims.

## Day 7 Deliverables

- implementation file list
- command and artifact naming plan
- freshness and staging design
- navigation update design
- Day 7 implementation design artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Planned edits map directly to the Day 6 decision. | Complete | All planned changes preserve strengthened local-only generated API HTML status. |
| Freshness and staging behavior has a testable design. | Complete | `make api-docs-freshness` and local-only generated-output states are defined above. |
| No implementation area is left without an owner. | Complete | Owner file table assigns scripts, Make targets, docs, ignore policy, workflows, and generated output handling. |
