# Sprint 177 Day 6: Populated Evidence Status Matrix

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_177/`
**Status:** Complete

## Purpose

Populate the Day 5 evidence/status matrix with the current repository state.
This artifact turns the Day 4 owner-file inventory and Day 5 schema into a
current support-tier map for Day 7 target selection.

## Population Rules Applied

- Rows use the Day 5 schema and status vocabulary.
- Hosted rows name the workflow surface that owns the reviewed evidence.
- Local-only rows name the maintained command that must pass before citing the
  evidence.
- Deferred and unsupported rows include the non-claim that must stay visible.
- Source-controlled rows are treated as ownership evidence, not as proof that
  a generator or validation command just ran.

## Populated Evidence Matrix

| Row ID | Surface | Support tier | Evidence status | Locality | Current evidence | Validation commands | Claim boundary | Non-claims | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESM-001 | Evidence/status matrix authority | sprint-local | pass | source-controlled | Sprint 177 Day 5 schema and Day 6 populated matrix artifacts. | `git diff --check` | Matrix rows are the Sprint 177 working authority for selected Epic 16 claim interpretation. | Not runtime, package, solver, hosted CI, or release evidence. | Use on Day 7 for target selection. |
| ESM-002 | Static-first package install contract | reviewed static-first | hosted | hybrid | Linux has reviewed Make install/pkg-config, CMake install/export, and static deferral proof; macOS has reviewed Make install/pkg-config and CMake install/export; Windows has reviewed CMake install/downstream validation and metadata-only `sparse.pc` inspection. | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh`; hosted package lanes. | Maintained package contract is static archive install/export with exact version metadata and no shared artifacts. | No package-manager support, shared-library packaging, dynamic ABI, runtime-loader behavior, Windows Makefile parity, or Windows pkg-config execution parity. | Preserve; do not widen without a new product decision. |
| ESM-003 | Package-manager provider support | deferred unsupported | defer | decision-only | Sprint 171 provider support remains a formal deferral; guard checks provider recipe absence and public non-claims. | `bash scripts/package_manager_deferral_check.sh` | Package-manager support is intentionally not provided unless a future provider-specific decision selects it. | No registry, tap, recipe, binary package, upgrade, provider availability, or package-manager parity claim. | Consider only a bounded provider decision in later Epic 16 scope. |
| ESM-004 | Shared-library and dynamic ABI support | deferred unsupported | defer | decision-only | Sprint 170 selected static-first-only posture; CMake rejects `BUILD_SHARED_LIBS=ON`; deferral guard blocks shared ABI metadata drift. | `bash scripts/static_package_deferral_check.sh` | Static-first-only package posture is the maintained product contract. | No shared-library packaging, symbol visibility policy, import/export ABI, dynamic ABI compatibility, loader metadata, installed shared consumer, or runtime-loader validation. | Keep deferred unless a full shared-library project is funded. |
| ESM-005 | Generated API HTML status | local-only | local-only | local | Sprint 158/173 policy keeps `docs/api/html/` generated, ignored, untracked, unstaged, and local-only. | `make api-docs-freshness` | Local Doxygen HTML can be regenerated and validated for configured public-header inputs. | No hosted documentation publication, source-controlled generated HTML, artifact-published generated HTML, release evidence, or completeness beyond configured Doxygen inputs. | Candidate for Sprint 179 publication/local-only closure. |
| ESM-006 | Selected oracle freshness | selected local plus Linux hosted | hosted | hybrid | Local `make report-index-oracle-freshness` regenerates selected QR/partial-SVD oracle output; reviewed Linux hosted report-freshness lane runs selected oracle gate and uploads split artifacts. | `make report-index-oracle-freshness`; Linux hosted report-freshness lane. | Selected QR and partial-SVD oracle rows are fresh for named fixtures on local and reviewed Linux hosted evidence surfaces. | No broad oracle proof, broad report-index freshness, macOS selected oracle freshness, Windows report freshness, release proof, package/ABI support, or state-of-the-art claim. | Use as baseline; decide whether macOS oracle promotion competes with other targets. |
| ESM-007 | Selected comparison freshness | selected local plus Linux/macOS hosted | hosted | hybrid | Local comparison freshness covers selected QR, partial-SVD, and LU targets; reviewed Linux and macOS hosted lanes run selected comparison freshness and upload selected artifacts. | `make report-index-comparison-freshness`; `python3 tests/test_selected_comparison_workflow.py`; Linux/macOS hosted lanes. | Selected QR, partial-SVD, and linked-list LU comparison rows are fresh for named fixture-local comparisons. | No unselected comparison family, broad external-library parity, Windows report freshness, package/ABI support, performance claim, release proof, or state-of-the-art claim. | Strong candidate for manifest centralization before further expansion. |
| ESM-008 | Windows report freshness | explicit non-claim | defer | decision-only | Windows remains CMake-first with reviewed CTest and CMake install/downstream validation, but no Windows report freshness lane. | Windows CI CMake tests/install proof; future deferral or lane guard. | Windows supports the reviewed CMake-first subset and static-first CMake install/downstream proof. | No Windows report freshness, Makefile parity, pkg-config execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. | Select promotion or formal deferral closure for Sprint 182. |
| ESM-009 | Selected performance publication | selected hosted/advisory | hosted | hybrid | Local canonical report freshness and reviewed Linux hosted selected-performance lane cover the selected `bench_refactor_csc` row for `nos4.mtx --repeat 1`. | `make bench-canonical-report-freshness`; Linux hosted selected-performance freshness checker. | One selected benchmark report has methodology and freshness metadata. | No raw timing gate, portable performance guarantee, performance superiority, backend superiority, external-library parity, package/ABI support, release proof, or state-of-the-art performance. | Preserve current row; only expand with methodology-bound scope. |
| ESM-010 | Allocation-failure evidence | focused local | pass | local | Sprint 176 added focused CG/GMRES/MINRES repeated-run handle allocation-failure proof. | `make iterative-allocation-failure-gate` | Iterative repeated-run handle prepare/growth cleanup is covered under selected injected allocation-failure scenarios. | No broad allocation-failure guarantee across direct solvers, eigensolvers, matrix construction, package/install flows, generated-report tooling, or unrelated allocation paths. | Select exactly one additional subsystem if Sprint 178 closes S177-R01. |
| ESM-011 | Public header and API coherence | reviewed local | pass | local | Public headers and API reference are maintained through docs/API checks; recent header coherence work focused on selected families. | `make docs-check`; `make api-docs-freshness`; `make test` if C/header behavior changes. | Selected public-header documentation and declarations can be kept coherent with local checks. | No whole-library API redesign, API freeze, generated HTML hosting, package ABI, or dynamic ABI guarantee. | Select one header family for Sprint 184. |
| ESM-012 | Platform support tiers | reviewed tiered | hosted | hybrid | README, INSTALL, maintainer guide, and workflows state Linux strongest reviewed source; macOS reviewed static-first install/export plus selected comparison; Windows reviewed CMake-first subset plus CMake install/downstream proof. | Hosted Linux/macOS/Windows CI; package deferral checks; docs review. | Platform support tiers are explicit and bounded to named reviewed lanes. | No broad platform parity, Windows Makefile parity, Windows pkg-config execution parity, broad report parity, package-manager support, shared-library support, or dynamic ABI support. | Keep as claim-calibration anchor for Day 7 and Sprint 186. |
| ESM-013 | Registration and workflow drift | local guard candidate | local-only | local | Source-list parity is guarded; selected report/workflow target lists remain repeated across Makefile, scripts, tests, workflows, and docs. | `make source-list-check`; `python3 tests/test_selected_comparison_workflow.py`; report-index tests. | Some registration drift is guarded, and selected workflow guard coverage exists. | No single canonical manifest yet for all selected oracle/comparison/performance workflow targets; no broad workflow correctness proof. | Strong candidate for Sprint 181 selected-target manifest closure. |
| ESM-014 | Large review-surface maintainability | advisory local | advisory | local | Day 4 identified large solver tests and report tooling; no behavior-preserving split has been selected for Epic 16 yet. | Existing family tests after selected refactor; `git diff --check` for docs-only inventory. | One selected review surface could be reduced with targeted tests. | No broad maintainability completion across large test, solver, tooling, and documentation surfaces. | Select exact extraction target for Sprint 185. |

## Support-Tier Breakdown

| Tier | Rows | Interpretation |
| --- | --- | --- |
| Reviewed/hosted | ESM-002, ESM-006, ESM-007, ESM-009, ESM-012 | CI or hosted artifact evidence exists for a named lane and a bounded claim. |
| Local-only/pass | ESM-001, ESM-005, ESM-010, ESM-011, ESM-013 | Maintained local commands or source-controlled artifacts define current evidence. |
| Advisory | ESM-014 | Useful current context exists, but it is not fail-closed product evidence. |
| Deferred/unsupported | ESM-003, ESM-004, ESM-008 | The correct current state is explicit deferral or non-claim preservation. |

## Hosted Versus Local Split

| Surface | Hosted evidence | Local evidence | Current boundary |
| --- | --- | --- | --- |
| Package install | Linux, macOS, and Windows named package/install lanes | install scripts and static/package deferral scripts | Static-first package only. |
| Selected oracle | Linux selected oracle freshness lane | `make report-index-oracle-freshness` | No macOS/Windows oracle parity. |
| Selected comparison | Linux and macOS selected comparison freshness lanes | `make report-index-comparison-freshness` | No Windows report freshness or unselected comparison proof. |
| Selected performance | Linux selected-performance freshness lane | `make bench-canonical-report-freshness` | Freshness/methodology, not timing superiority. |
| Generated API HTML | none | `make api-docs-freshness` | Local-only generated HTML. |
| Allocation failure | none | `make iterative-allocation-failure-gate` | Focused iterative repeated-run handle proof only. |

## Rows Requiring Day 7 Selection Attention

| Row | Reason |
| --- | --- |
| ESM-003 | Package-manager support is deferred; Day 7 must decide whether Epic 16 selects a provider proof, a stronger deferral, or leaves it as non-goal. |
| ESM-005 | Generated API HTML is local-only; Day 7 must decide whether publication or stronger local-only closure is selected. |
| ESM-007 / ESM-013 | Selected comparison evidence is hosted but target metadata is duplicated; Day 7 should decide whether manifest centralization is selected before expansion. |
| ESM-008 | Windows report freshness is a non-claim; Day 7 must choose promotion or formal deferral closure. |
| ESM-010 | Allocation-failure evidence is narrow; Day 7 should select one additional subsystem or explicitly keep the current boundary. |
| ESM-014 | Maintainability risk is visible but not closed; Day 7 should select one large review surface if it remains in Epic 16 scope. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every major public claim surface has a matrix row | Complete | Package, API docs, reports, comparisons, performance, platform, ABI, allocation-failure, public-header, workflow-drift, and maintainability rows are populated. |
| Unsupported surfaces are recorded as non-claims | Complete | Package-manager, shared-library/dynamic ABI, Windows report freshness, broad platform parity, and broad performance claims are explicit non-claims. |
| Selected hosted evidence is distinguishable from local-only evidence | Complete | Hosted/local split table separates Linux/macOS/Windows CI proof from local-only API and allocation-failure proof. |
