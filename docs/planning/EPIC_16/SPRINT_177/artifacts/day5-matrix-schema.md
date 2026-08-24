# Sprint 177 Day 5: Evidence Status Matrix Schema

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_177/`
**Status:** Complete

## Purpose

Define the evidence/status matrix schema that will become the working
authority for Epic 16 claim governance. Day 5 designs the schema and initial
row set; Day 6 will populate current evidence values from repository state.

## Schema Columns

| Column | Required | Meaning |
| --- | --- | --- |
| Row ID | Yes | Stable matrix identifier for references from later Sprint 177 artifacts. |
| Surface | Yes | Product, workflow, package, documentation, report, or code surface being governed. |
| Residual IDs | Yes | Day 2-3 residual IDs that motivate the row. |
| Support tier | Yes | Current tier wording such as reviewed, hosted selected, local-only, advisory, deferred, or unsupported. |
| Evidence status | Yes | Current status value using the row semantics below. |
| Evidence locality | Yes | Local, hosted, hybrid, source-controlled, decision-only, or none. |
| Owner files | Yes | Files that own behavior, wording, metadata, or tests for the surface. |
| Validation commands | Yes | Exact command set that proves or inspects the row. |
| Artifact path | Yes | Sprint artifact, generated artifact, workflow artifact, or `none` if not applicable. |
| Claim boundary | Yes | Positive statement that is supported by the evidence. |
| Non-claims | Yes | Explicit unsupported interpretations that must not be implied. |
| Next action | Yes | Populate, select, defer, guard, promote, or close in a later sprint. |

## Evidence Status Semantics

| Status | Meaning | Use when |
| --- | --- | --- |
| `pass` | A row has a maintained validation command and the current evidence is expected to pass. | A local or hosted command directly proves the scoped claim. |
| `hosted` | A row has reviewed CI evidence on one or more named hosted platforms. | The claim relies on CI execution or uploaded workflow artifacts. |
| `local-only` | A row has a maintained local command but no reviewed hosted proof. | The claim is intentionally local or publication is not yet promoted. |
| `advisory` | A row records navigation, measurement, or support context without fail-closed proof. | Missing or stale data should guide maintainers without widening claims. |
| `defer` | A row is a deliberate product or governance deferral with owner wording. | Closure is not selected now, but the non-claim must stay explicit. |
| `unsupported` | A row identifies an unsupported capability or platform interpretation. | The correct outcome is to reject or avoid the claim. |
| `unknown` | A temporary Sprint 177 state before Day 6 population. | The owner files are known but current evidence has not yet been assigned. |

## Evidence Locality Semantics

| Locality | Meaning |
| --- | --- |
| `local` | Maintained command runs locally and produces the relevant proof. |
| `hosted` | Reviewed CI lane is the primary proof source. |
| `hybrid` | Local command and hosted CI both matter to the claim. |
| `source-controlled` | Checked-in metadata or documentation is the proof owner. |
| `decision-only` | Product decision or explicit deferral is the closure mechanism. |
| `none` | No proof exists; use with `unsupported` or unresolved rows only. |

## Initial Matrix Rows

These rows are the Day 5 starting set. Current support tiers and statuses are
left as `unknown` where Day 6 still needs to inspect exact evidence.

| Row ID | Surface | Residual IDs | Support tier | Evidence status | Evidence locality | Owner files | Validation commands | Artifact path | Claim boundary | Non-claims | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESM-001 | Evidence/status matrix authority | S177-R14 | sprint-local | unknown | source-controlled | Sprint 177 artifacts | `git diff --check` | `day6-populated-matrix.md` | Matrix rows govern selected Epic 16 claim interpretation. | Not a runtime or package proof. | Populate on Day 6. |
| ESM-002 | Static-first package install contract | S177-R03, S177-R04 | reviewed/static-first | unknown | hybrid | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install tests | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; deferral scripts | existing install and package artifacts | Static archive install and metadata are maintained for selected platforms. | No package-manager, shared-library, dynamic ABI, or runtime-loader support. | Populate and preserve boundaries. |
| ESM-003 | Package-manager provider support | S177-R03 | deferred/unsupported | unknown | decision-only | package deferral script, `README.md`, `INSTALL.md`, maintainer guide | `bash scripts/package_manager_deferral_check.sh` | future decision artifact | Provider support is either explicitly selected or explicitly deferred. | No registry, tap, recipe, binary package, upgrade, or provider parity claim. | Decide candidate shape before Sprint 180. |
| ESM-004 | Shared-library and dynamic ABI support | S177-R04 | deferred/unsupported | unknown | decision-only | `CMakeLists.txt`, static deferral script, package templates, public docs | `bash scripts/static_package_deferral_check.sh` | future decision artifact | Static-first-only package posture remains intentional. | No shared-library packaging, symbol ABI, loader, or import/export guarantee. | Keep deferred unless fully selected. |
| ESM-005 | Generated API HTML status | S177-R02 | local-only/decision | unknown | local | `Doxyfile`, public headers, API docs scripts, maintainer guide | `make api-docs-freshness` | future publication or local-only artifact | Generated API output is either published with freshness proof or explicitly local-only. | No hosted API publication claim until promoted. | Populate current state and select closure direction. |
| ESM-006 | Selected oracle freshness | S177-R06, S177-R11 | selected/local-plus-hosted | unknown | hybrid | `scripts/run_corpus_oracle.py`, corpus data, workflows, report tests | `make report-index-oracle-freshness` | generated oracle artifacts | Selected QR/partial-SVD oracle rows are fresh for named fixtures. | No broad oracle, macOS oracle, Windows report, or broad report-index claim. | Populate hosted/local split. |
| ESM-007 | Selected comparison freshness | S177-R07, S177-R10, S177-R11 | selected/local-plus-hosted | unknown | hybrid | comparison script, selected reference helpers, workflows, workflow guard test | `make report-index-comparison-freshness`; workflow guard test | generated comparison artifacts | Selected QR, partial-SVD, and LU comparison rows are fresh for named fixtures. | No unselected comparison, external-library parity, package, ABI, or state-of-the-art claim. | Populate and identify manifest target. |
| ESM-008 | Windows report freshness | S177-R05 | deferred/decision | unknown | hosted/decision-only | Windows workflow, report scripts, public support wording | Windows CI lane or explicit deferral guard | future Sprint 182 artifact | Windows report freshness is either selected for one safe family or explicitly deferred. | No broad Windows report, Makefile parity, or pkg-config parity claim. | Select by Day 7. |
| ESM-009 | Selected performance publication | S177-R08 | advisory/hosted-selected | unknown | hybrid | benchmark scripts, benchmark docs, Linux workflow | `make bench-canonical-report-freshness`; hosted freshness checker | canonical benchmark artifacts | One selected benchmark report has methodology and freshness evidence. | No portable performance, timing superiority, backend superiority, or state-of-the-art claim. | Populate current hosted row. |
| ESM-010 | Allocation-failure evidence | S177-R01 | local focused | unknown | local | iterative solver implementation, allocation internals, `tests/test_iterative.c` | `make iterative-allocation-failure-gate` | Sprint 176 and future Sprint 178 artifacts | Selected solver-family cleanup paths can be proven under injected allocation failure. | No broad allocation-failure guarantee across all solvers or constructors. | Select next subsystem. |
| ESM-011 | Public header and API coherence | S177-R09 | local reviewed | unknown | local | `include/*.h`, `docs/api_reference.md`, tutorial/cookbook docs | `make docs-check`; `make api-docs-freshness`; `make test` if code changes | future Sprint 184 artifact | Selected header family declarations and docs remain coherent. | No whole-library API redesign or ABI compatibility claim. | Populate and pick family. |
| ESM-012 | Platform support tiers | S177-R05, S177-R06, S177-R11 | reviewed tiered | unknown | hybrid | `README.md`, `INSTALL.md`, maintainer guide, workflows | platform CI plus docs grep/deferral checks | support-tier artifacts | Linux, macOS, and Windows support claims stay tiered and explicit. | No broad platform parity, Windows Makefile parity, or all-report parity. | Populate wording owners. |
| ESM-013 | Registration and workflow drift | S177-R10, S177-R13 | local guard candidate | unknown | local | `Makefile`, `CMakeLists.txt`, workflows, report tests, build metadata | source-list check, workflow guard tests, report tests | future Sprint 181 artifact | Selected target lists can be centralized or guarded against drift. | No broad workflow correctness proof unless every lane is guarded. | Select manifest strategy. |
| ESM-014 | Large review-surface maintainability | S177-R13 | advisory/local | unknown | local | large solver tests, report tooling, maintainer guide | existing tests for selected owner after refactor | future Sprint 185 artifact | One selected review surface can be reduced without behavior change. | No broad maintainability completion across all large files. | Pick exact extraction candidate. |

## Alignment With Existing Vocabulary

The schema intentionally mirrors current repository terms:

- `claim boundary` and `non-claims` match normalized report-index vocabulary.
- `reviewed`, `hosted`, `local-only`, `advisory`, `deferred`, and
  `unsupported` match maintainer-guide and install wording.
- `source-controlled` rows match package/report-family metadata that is
  reviewed in Git rather than regenerated as runtime evidence.
- `hosted` rows must name exact CI lanes and cannot imply broader platform
  support than those lanes run.
- `advisory` rows can help navigation and methodology, but cannot carry
  product claims without an accompanying pass/hosted/local-only proof.

## Day 6 Population Rules

Day 6 should populate this schema using these rules:

1. Every public support claim gets at least one row.
2. Every unsupported or deferred capability gets explicit non-claims.
3. Hosted rows must name their workflow file and job scope.
4. Local-only rows must name the command that maintainers can run.
5. Decision-only rows must point to a sprint decision artifact or create one
   later in Epic 16.
6. If a row has no current validation command, keep status `unknown` or
   `unsupported`; do not promote it to `advisory` just to avoid a gap.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Matrix distinguishes evidence from non-claims | Complete | Separate evidence status, evidence locality, claim boundary, and non-claims columns. |
| Row semantics match existing vocabulary | Complete | Status/locality definitions align with report-index and maintainer-guide terms. |
| Future deliverables can update matrix without ambiguity | Complete | Stable row IDs, owner files, validation commands, and next actions are defined. |
