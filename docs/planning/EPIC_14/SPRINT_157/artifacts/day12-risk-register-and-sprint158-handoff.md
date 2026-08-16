# Day 12 Risk Register And Sprint 158 Handoff

## Scope

Day 12 consolidates risks from Sprint 157 baseline, residual selection,
evidence contracts, quality mapping, and claim-register work. It also drafts
the Sprint 158 handoff for generated API HTML publication closure.

The primary Day 12 target is C157-01: generated API reference policy becomes
explicit and reviewable, either through published generated HTML with freshness
evidence or through a guarded local-only policy.

## Epic 14 Risk Register

| Risk ID | Risk | Claim surface | Likelihood | Impact | Owner | Mitigation | Stop condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R157-01 | Generated API HTML is committed or cited without warning triage. | C157-01 | Medium | High | Documentation/API owner | Require `make docs`, captured warning log, warning disposition, and publication decision. | Any untriaged warning remains while generated HTML is treated as fresh evidence. |
| R157-02 | Generated API HTML page coverage misses public headers. | C157-01 | Medium | High | Documentation/API owner | Add or document a page-coverage check for intended `include/*.h` inputs. | Missing page coverage is found and no explicit exclusion is recorded. |
| R157-03 | Local ignored generated files are mistaken for source-controlled pass evidence. | C157-01, C157-02 | High | High | Report and documentation owners | Keep `docs/api/`, `build/`, and `coverage/` support-tier wording explicit until selected publication/hosted gates land. | Any public claim cites ignored local output without a freshness or publication gate. |
| R157-04 | Source-header-first policy is weakened by generated-doc publication. | C157-01, C157-07 | Medium | Medium | Documentation/API and header owners | Keep exact declarations owned by checked-in public headers; generated HTML is a view over configured inputs. | Docs imply generated HTML is broader or more authoritative than public headers. |
| R157-05 | Hosted report promotion accidentally promotes advisory families. | C157-02 | Medium | High | CI/report owners | Sprint 159 must select claim-bearing families and leave coverage, dead-code, benchmark, large-matrix, and optional-data rows advisory unless explicitly selected. | Hosted lane or docs imply all generated families are reviewed claim evidence. |
| R157-06 | QR or partial-SVD comparison wording becomes broad parity. | C157-03, C157-04 | Medium | High | QR/SVD and comparison owners | Require fixture-local metrics, dependency provenance, tolerance policy, skip/defer semantics, and non-claims. | Docs imply broad QR/SVD, LAPACK, NumPy, SciPy, SuiteSparse, or ecosystem parity. |
| R157-07 | Windows package decision conflates CMake support with Makefile or `pkg-config` parity. | C157-05 | High | High | Platform/package owner | Sprint 162 must separate Makefile and `pkg-config` parity decisions and preserve CMake-first wording. | Windows CMake install/downstream proof is described as Windows Makefile or `pkg-config` execution proof. |
| R157-08 | Performance publication is overread as portable superiority. | C157-06 | Medium | High | Benchmark/report owner | Require methodology fields, selected row classification, threshold semantics, and explicit non-superiority wording. | Timing rows are published without hardware/compiler/build/thread/repeat/caveat context. |
| R157-09 | Header cleanup changes declarations accidentally. | C157-07 | Medium | High | Header owners | Require before/after normalized declaration capture and full C/header gates for header edits. | Normalized declarations drift without explicit API review. |
| R157-10 | Static-first hardening accidentally introduces shared or ABI metadata. | C157-08 | Low | High | Package/ABI owner | Run static deferral guard and install/export checks for package metadata changes. | Shared metadata, selectors, export/import macros, or ABI promises appear without product decision. |
| R157-11 | CI expected counts or lane names drift from actual support tiers. | C157-02, C157-05 | Medium | Medium | CI owner | Use Day 10 CI reconciliation checklist for workflow changes. | Expected counts, hosted artifact semantics, or lane names are updated without evidence. |
| R157-12 | Final claim audit misses unfunded work. | C157-09 | Medium | High | Epic/product owner | Sprint 166 must reconcile evidence inventory, public claims, project-plan status, and residual queue. | Any unsupported state-of-the-art, package, ABI, platform, performance, or parity claim lacks a residual or rejection. |

## Prioritized Risks

| Priority | Risk IDs | Rationale |
| --- | --- | --- |
| 1 | R157-01, R157-02, R157-03, R157-04 | Sprint 158 starts next and can fail quickly if generated API docs scope is unclear. |
| 2 | R157-05, R157-06, R157-07 | These risks can turn bounded evidence into broad unsupported claims. |
| 3 | R157-08, R157-09, R157-10 | These touch later implementation surfaces with higher validation cost. |
| 4 | R157-11, R157-12 | These are closeout and governance risks that require recurring audit discipline. |

## Sprint 158 Handoff Draft

### Objective

Close the generated API reference residual with either:

1. committed or otherwise published generated HTML plus warning/page-coverage
   evidence; or
2. an explicit no-commit/local-only product decision with a recurring guard and
   source-header-first wording.

### Starting Sources

| Source | Sprint 158 use |
| --- | --- |
| `Doxyfile` | Defines Doxygen input set and generated output behavior. |
| `Makefile` `docs` target | Runs `doxygen Doxyfile` and writes `docs/api/html/`. |
| `docs/api_reference.md` | User-facing API reference entry point and current generated HTML boundary. |
| `docs/maintainer_guide.md` generated Doxygen section | Current freshness rules for `docs/api/html/`. |
| `include/*.h` | Source of truth for public declarations and intended page coverage. |
| `include/sparse_version.h.in` | Installed generated version-header template whose Doxygen treatment must be explicit. |
| `.gitignore` | Currently ignores `docs/api/`; publication decision must address tracking. |
| Day 5 generated-artifact baseline | Confirms no tracked `docs/api` files and generated output is local-only today. |
| Day 9 API docs template | Defines required evidence fields for Sprint 158. |
| Day 10 quality map | Defines docs, generated docs, and header validation requirements. |
| Day 11 claim register | Defines C157-01 and docs that must move together. |

### Day 1 Prerequisites For Sprint 158

1. Confirm branch starts from merged Sprint 157 baseline.
2. Confirm `doxygen` is installed or record a tooling blocker.
3. Confirm `git status --ignored=matching --short docs/api` before generation.
4. Run or plan `make docs` and capture stdout/stderr.
5. Inventory generated pages under `docs/api/html/`.
6. Build the intended public-header coverage list from checked-in
   `include/*.h`.
7. Decide how `include/sparse_version.h.in` and generated
   `sparse_version.h` should be represented in generated API docs.
8. Keep source-header-first wording unchanged until the publication decision is
   made.

### Required Sprint 158 Artifacts

| Artifact | Minimum content |
| --- | --- |
| Doxygen baseline | Command, tool availability, warning count, warning log location or excerpt, generated output path. |
| Page coverage inventory | Intended public header list, generated page list, missing pages, explicit exclusions. |
| Publication decision | Commit generated HTML, publish via another route, or retain local-only policy; include rationale and support tier. |
| Warning triage | Fixed warnings, accepted warnings, blockers, and owners. |
| Docs alignment | Updates or no-change rationale for `docs/api_reference.md`, `docs/maintainer_guide.md`, README, tutorial, and header-doc policy wording. |
| Validation summary | `make docs`, page/warning checks, `git diff --check`, direct whitespace scan for untracked docs, and full C/header gate if headers change. |
| Sprint 159 handoff | Generated-artifact wording needed for hosted oracle/comparison freshness promotion. |

## Sprint 158 Stop Conditions

- `make docs` cannot run and the tooling blocker cannot be resolved locally.
- Doxygen warnings are present without triage.
- Generated page coverage misses intended public headers without recorded
  exclusions.
- Generated output is committed while `.gitignore`, review guidance, or docs
  still describe it as ignored local-only output.
- Generated output remains local-only but docs imply it is checked-in,
  published, complete, or fresh.
- Public header comments change without declaration-preservation proof and
  `make format && make lint && make test`.
- API reference wording implies dynamic ABI, shared-library support,
  package-manager distribution, broad platform parity, external parity,
  portable performance, or state-of-the-art coverage.

## Mitigation And Deferral Rules

| Scenario | Required response |
| --- | --- |
| Doxygen is unavailable | Record blocker, do not fabricate generated-doc evidence, and keep API docs local-only/source-header-first. |
| Generated HTML has warnings | Triage and fix selected warnings or record explicit exclusions with owner/blocker. |
| Generated page coverage is incomplete | Fix input configuration or document exclusions; do not publish completeness wording. |
| Publication decision is no-commit/local-only | Add or preserve guard/docs making ignored generated output an explicit product decision. |
| Publication decision is commit generated HTML | Update `.gitignore`/tracking policy, commit generated output with corresponding source policy, and record review guidance. |
| Header comments need cleanup | Use Sprint 164 declaration-preservation rules if cleanup exceeds Sprint 158 docs-policy scope. |
| Public docs need broader claim updates | Use Day 11 docs ownership checklist and preserve rejected claims. |

## Day 13 Inputs

Day 13 should reconcile this risk/handoff artifact against all Sprint 157
artifacts:

- C157-01 and T157-01 must still map cleanly to Sprint 158;
- Day 5 generated-output baseline and this handoff must agree on current
  `docs/api/` ignored status;
- Day 9 and Day 10 evidence/quality requirements must remain consistent;
- Day 11 rejected claims must be preserved in Sprint 158 stop conditions.

## Completion Check

- Sprint 158 can start without rediscovering generated API docs scope.
- Risks have owners, mitigations, and stop conditions.
- API docs work is bounded by source-header-first policy.
- Doxygen warning, generated-output tracking, public-header coverage, and claim
  wording risks are explicit.
