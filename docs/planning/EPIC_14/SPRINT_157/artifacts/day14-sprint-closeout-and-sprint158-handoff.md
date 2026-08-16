# Day 14 Sprint Closeout And Sprint 158 Handoff

## Scope

Day 14 closes Sprint 157 by finalizing the artifact index, confirming each
Sprint 157 project-plan item has a completed artifact, preserving the Sprint
158 generated API documentation handoff, and recording final residuals and open
questions.

Sprint 157 changed planning documentation only. It did not change C sources,
headers, scripts, build metadata, package metadata, or CI workflows.

## Final Artifact Index

| Day | Artifact | Status | Closeout note |
| --- | --- | --- | --- |
| 1 | `day1-sprint-intake.md` | Complete | Established branch baseline, scope correction, stop conditions, category map, and Sprint 158 seed. |
| 2 | `day2-code-public-surface-inventory.md` | Complete | Captured source/public API inventory, largest-file hotspots, and source-list baseline. |
| 3 | `day3-test-ci-baseline.md` | Complete | Captured test counts, CMake enumeration, CI support tiers, Windows CTest baseline, and validation surfaces. |
| 4 | `day4-documentation-claim-baseline.md` | Complete | Captured public docs, positive claims, non-claims, support-tier owners, and claim-scan result. |
| 5 | `day5-generated-artifact-baseline.md` | Complete | Captured generated API, corpus, oracle, comparison, benchmark, sentinel, coverage, dead-code, and report-index baselines. |
| 6 | `day6-package-abi-platform-baseline.md` | Complete | Captured static-first package support, Windows package deltas, shared-library blockers, and metadata ownership. |
| 7 | `day7-residual-consolidation.md` | Complete | Consolidated Epic 13 residuals and Epic 14 review gaps into claim-oriented residuals. |
| 8 | `day8-target-selection.md` | Complete | Selected Epic 14 targets T157-01 through T157-09 and explicit non-goals. |
| 9 | `day9-evidence-contract-templates.md` | Complete | Defined reusable evidence templates for each selected target family. |
| 10 | `day10-quality-surface-map.md` | Complete | Defined validation commands by change type, package/build surface, CI workflow, and generated evidence family. |
| 11 | `day11-claim-target-register.md` | Complete | Published accepted target claims, rejected broad claims, evidence owners, docs ownership, and claim-change checklist. |
| 12 | `day12-risk-register-and-sprint158-handoff.md` | Complete | Published Epic 14 risks, mitigations, stop conditions, and Sprint 158 handoff draft. |
| 13 | `day13-baseline-reconciliation.md` | Complete | Reconciled all prior artifacts against selected targets, claims, quality gates, project-plan scopes, and deferrals. |
| 14 | `day14-sprint-closeout-and-sprint158-handoff.md` | Complete | Finalizes Sprint 157 closeout, validation notes, residuals, open questions, and Sprint 158 handoff. |

## Project-Plan Item Status

| Sprint 157 item | Planned work | Completed artifacts | Status |
| --- | --- | --- | --- |
| Item 1: Baseline Inventory | Capture source, header, test, script, benchmark, example, docs, corpus, generated report, and package inventory. | Days 1-6 artifacts. | Complete. |
| Item 2: Residual Selection | Convert Epic 13 residuals into selected Epic 14 targets, long-horizon deferrals, and explicit non-goals. | Days 7-8 artifacts. | Complete. |
| Item 3: Evidence Contract | Define recurring evidence requirements for API docs, hosted reports, comparison rows, Windows package decisions, and performance reports. | Day 9 artifact. | Complete. |
| Item 4: Claim Target Register | Publish claim targets and non-claims for state-of-the-art, external parity, performance, package, Windows, ABI, and docs surfaces. | Day 11 artifact. | Complete. |
| Item 5: Quality Surface Map | Map required validation commands for docs, scripts, C/header, build-system, package, CI, and generated artifacts. | Day 10 artifact. | Complete. |
| Item 6: Risk And Handoff | Publish sprint working notes, risk register, and Sprint 158 API-doc handoff. | Day 12 artifact and `WORKING_NOTES.md`. | Complete. |
| Item 7: Closeout | Reconcile completed artifacts with the plan and update residuals. | Days 13-14 artifacts. | Complete. |

## Final Sprint 158 Handoff

Sprint 158 should start from target T157-01 and claim C157-01.

### Objective

Close the generated API reference residual with either:

1. committed or otherwise published generated HTML plus warning/page-coverage
   evidence; or
2. an explicit no-commit/local-only product decision with a recurring guard and
   source-header-first wording.

### Required Starting Context

| Source | Why it matters |
| --- | --- |
| `Doxyfile` | Owns Doxygen input/output configuration. |
| `Makefile` `docs` target | Runs `doxygen Doxyfile` and writes `docs/api/html/`. |
| `docs/api_reference.md` | User-facing API reference entry and current generated HTML boundary. |
| `docs/maintainer_guide.md` generated Doxygen section | Current freshness and non-claim policy for generated HTML. |
| `include/*.h` | Source of truth for public declarations and page coverage. |
| `include/sparse_version.h.in` | Installed generated version-header template whose API-doc policy must be explicit. |
| `.gitignore` | Currently ignores `docs/api/`; publication decision must address this. |
| Day 5 artifact | Baseline for generated output and ignored tracking state. |
| Day 9 artifact | API docs evidence template. |
| Day 10 artifact | Generated API docs validation map. |
| Day 11 artifact | C157-01 claim owner and docs ownership. |
| Day 12 artifact | Risk register and detailed Sprint 158 handoff. |

### Day 1 Actions For Sprint 158

1. Confirm branch starts from merged Sprint 157 baseline.
2. Confirm `doxygen` availability or record tooling blocker.
3. Capture `git status --ignored=matching --short docs/api` before generation.
4. Run or schedule `make docs` with stdout/stderr capture.
5. Inventory generated pages under `docs/api/html/`.
6. Build expected page coverage from checked-in `include/*.h`.
7. Decide how `include/sparse_version.h.in` and generated `sparse_version.h`
   are represented in generated docs.
8. Preserve source-header-first wording until the publication decision is made.

### Sprint 158 Stop Conditions

- `make docs` cannot run and the tooling blocker cannot be resolved locally.
- Doxygen warnings are untriaged.
- Generated page coverage misses intended public headers without exclusions.
- Generated output is committed while tracking/docs still describe it as
  ignored local-only output.
- Generated output stays local-only while docs imply it is checked in,
  published, complete, or fresh.
- Header comments change without declaration-preservation proof and
  `make format && make lint && make test`.
- API docs imply dynamic ABI, shared-library support, package-manager
  distribution, broad platform parity, external parity, portable performance,
  or state-of-the-art coverage.

## Final Residuals And Open Questions

Sprint 157 itself has no unresolved planning item. It intentionally leaves
implementation residuals to later Epic 14 sprints:

| Residual/open question | Owner sprint | Closeout state |
| --- | --- | --- |
| Generated API HTML publication or guarded local-only decision | 158 | Handoff complete; implementation not started in Sprint 157. |
| Hosted selected oracle/comparison freshness promotion | 159 | Selected target and evidence template complete. |
| One bounded QR comparison family | 160 | Selected target and evidence template complete. |
| One bounded partial-SVD comparison family | 161 | Selected target and evidence template complete. |
| Windows package parity decision | 162 | Selected as product-decision closure, not guaranteed parity promotion. |
| Methodology-bound performance publication | 163 | Selected target with non-superiority boundary. |
| Public header/API coherence batch | 164 | Selected target with declaration-preservation gate. |
| Static-first package boundary hardening | 165 | Selected target; shared-library/dynamic ABI remain non-claims. |
| Final Epic 14 claim recalibration and residual queue | 166 | Selected closeout target. |

Long-horizon non-goals remain package-manager distribution, full
shared-library product support, dynamic ABI compatibility, broad ecosystem
parity, portable performance superiority, broad Windows parity, runtime/backend
API promotion, and unqualified state-of-the-art status.

## Validation Notes

Sprint 157 Day 14 validation requirement is documentation-only:

- run `git diff --check`;
- run a direct trailing-whitespace scan on `docs/planning/EPIC_14/SPRINT_157`
  because the sprint directory is still untracked;
- do not run `make format && make lint && make test` unless `.c` or `.h` files
  changed.

## Completion Check

- All Sprint 157 project-plan items have a completed artifact.
- Sprint 158 can begin from a concrete generated API docs handoff.
- Final residuals and open questions are recorded.
- Documentation validation remains the only required local validation for this
  docs-only sprint work.
