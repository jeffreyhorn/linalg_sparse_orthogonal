# Sprint 186 Day 4: Public Claim Inventory

## Purpose

Inventory public and maintainer-facing claims affected by Epic 16 before any
claim-calibration edits. This artifact separates earned claims from protected
non-claims and gives Days 5 through 7 a document-family checklist.

## Source Inputs

| Source | Day 4 use |
| --- | --- |
| `artifacts/day3-reconciled-evidence-matrix.md` | Final status and residual classifications for Sprints 177-185. |
| `README.md` | Primary public claim surface for features, validation, API docs, reports, packaging, Windows, and non-claims. |
| `INSTALL.md` | Public install, platform, package, and support-tier claim surface. |
| `docs/maintainer_guide.md` | Maintainer-facing owner, gate, non-claim, report, package, and generated API guidance. |
| `docs/api_reference.md` | Supported source-controlled API documentation entry point. |
| `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md` | User documentation affected by QR/header, solver-selection, report, and comparison wording. |
| `packaging/homebrew/README.md` | Selected local Homebrew formula proof boundary. |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Homebrew local proof template and static/shared boundary checks. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected report target authority, claim scopes, support tiers, non-claims, expected rows, and workflow metadata. |
| `scripts/package_manager_deferral_check.sh` | Guarded package-manager provider non-claim and Homebrew proof-path wording. |
| `scripts/static_package_deferral_check.sh` | Static-first package and unsupported shared-library/ABI claim guard. |
| `scripts/check_api_docs_local_only.sh` | Generated API HTML local-only product decision guard. |
| `scripts/check_qr_header_docs_guard.sh` | QR header/docs unsupported-claim and declaration-boundary guard. |

## Earned Claims Inventory

| Claim family | Earned claim | Evidence rows | Current claim surfaces | Guard or validation owner |
| --- | --- | --- | --- | --- |
| Allocation-failure proof | `sparse_matmul()` workspace allocation has deterministic cleanup, stale-output suppression, and retry evidence. | 178.1-178.6 | `README.md`; `docs/maintainer_guide.md` | `make matmul-allocation-failure-gate`; `tests/test_matmul_allocation_failure_gate_registration.py` |
| Generated API status | Generated API HTML is a local-only regenerated convenience view; the supported source-controlled API path is `docs/api_reference.md` plus headers under `include/`. | 179.1-179.6 | `README.md`; `docs/api_reference.md`; `docs/maintainer_guide.md`; `scripts/check_api_docs_local_only.sh` | `make api-docs-freshness`; `make api-docs-validate`; `scripts/check_api_docs_local_only.sh` |
| Package-provider decision | Homebrew is the selected local formula/tap proof path with source-controlled template, notes, script, and guard behavior. | 180.1-180.6 | `README.md`; `INSTALL.md`; `packaging/homebrew/README.md`; `docs/maintainer_guide.md` | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; `scripts/homebrew_local_formula_proof.sh` |
| Selected report targets | Selected oracle, comparison, and benchmark report metadata has one manifest authority. | 181.1-181.6 | `README.md`; `docs/maintainer_guide.md`; report-index docs; `tests/corpus/manifests/selected_report_targets.tsv` | `python3 scripts/validate_corpus_schema.py`; report normalizer and selected workflow tests |
| Windows report status | Windows report freshness is formally deferred; Windows support remains CMake/MSVC configure, build, `ctest`, and static-first install/downstream validation. | 182.1-182.6 | `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; selected manifest | selected workflow/report guards; Windows deferral artifact |
| Selected comparison family | The selected `cholesky_spd_tridiag_5` comparison report is fresh for named rows against the source-controlled dense Cholesky reference helper. | 183.1-183.6 | `README.md`; `docs/maintainer_guide.md`; report-index docs; selected manifest | `python3 tests/test_run_external_comparison.py`; `make report-index-comparison-freshness` |
| QR header coherence | QR public header comments are clearer and declarations are organized without public declaration-set drift. | 184.1-184.6 | `include/sparse_qr.h`; `README.md`; `docs/api_reference.md`; `docs/solver_selection.md`; examples/docs | `make qr-header-docs-guard`; `make api-docs-validate`; declaration diff evidence |
| LDLT CSC review surface | The selected LDLT CSC test review surface was reduced through family-local helper headers while preserving behavior and registration. | 185.1-185.6 | `docs/maintainer_guide.md`; Sprint 185 artifacts | `make ldlt-csc-helper-guard`; `make source-list-check`; full C gate |

## Protected Non-Claims

| Non-claim | Evidence source | Surfaces that must preserve it |
| --- | --- | --- |
| Broad allocation-failure cleanup coverage is not claimed beyond selected iterative and `sparse_matmul()` proof lanes. | Sprint 178 retrospective; Day 3 matrix | `README.md`; `docs/maintainer_guide.md` |
| Hosted generated API HTML, retained CI generated API artifacts, and committed generated output are not claimed. | Sprint 179 retrospective; local-only guard | `README.md`; `docs/api_reference.md`; `docs/maintainer_guide.md`; `.github/workflows/*`; `docs/api/` staging state |
| Homebrew support, Homebrew/core readiness, bottles, Linuxbrew support, public tap support, vcpkg/Conan/pkgsrc support, and broad package-manager support are not claimed. | Sprint 180 retrospective; package-manager guard | `README.md`; `INSTALL.md`; `packaging/homebrew/README.md`; `docs/maintainer_guide.md`; package metadata templates |
| Full Homebrew proof success is not claimed while standalone license metadata is absent. | Day 3 reconciliation; `scripts/homebrew_local_formula_proof.sh` | `README.md`; `INSTALL.md`; `packaging/homebrew/README.md`; `docs/maintainer_guide.md` |
| Windows selected report freshness and broad Windows generated-report parity are not claimed. | Sprint 182 retrospective; Windows deferral decision | `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; selected manifest; workflow docs |
| Broad Cholesky correctness, broad SPD coverage, SuiteSparse/Eigen/LAPACK/NumPy/SciPy parity, and external-library ecosystem parity are not claimed by the selected Cholesky comparison. | Sprint 183 retrospective; selected manifest | `README.md`; `docs/maintainer_guide.md`; report docs; selected manifest |
| QR cleanup does not add public API declarations, ABI guarantees, package/platform support, performance claims, or broad external parity. | Sprint 184 retrospective; QR guard | `include/sparse_qr.h`; `README.md`; `docs/api_reference.md`; `docs/solver_selection.md`; examples/docs |
| LDLT CSC helper extraction does not claim solver behavior, correctness expansion, performance, new test binaries, production-source changes, or public API changes. | Sprint 185 retrospective; PR #205 review fix | `docs/maintainer_guide.md`; Sprint 185/186 closeout docs |
| Shared-library support, dynamic ABI compatibility, runtime-loader behavior, portable performance, release readiness, and state-of-the-art status remain non-claims unless future evidence exists. | Epic 16 project plan; Sprint 177 claim-boundary freeze; Day 3 matrix | `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; package docs; report docs |

## Unsupported, Overbroad, Stale, Or Missing Claim Candidates

Day 4 did not make calibration edits. These candidates are the checklist for
Days 5 through 7.

| ID | Surface | Candidate issue | Calibration action |
| --- | --- | --- | --- |
| D4-CAL-001 | `README.md` package/install sections | README already states package-manager support is not provided, but final closeout should confirm Homebrew is described only as a local proof path blocked by missing standalone license metadata. | Day 5 verify and adjust wording if it implies support rather than proof-path evidence. |
| D4-CAL-002 | `INSTALL.md` support split | INSTALL must preserve the package-manager deferral entry and current Homebrew proof blocker. | Day 5 verify support split and package-manager non-claim text. |
| D4-CAL-003 | `packaging/homebrew/README.md` | Homebrew proof docs must not read like a user installation method. | Day 5 preserve local-only proof wording and unsupported Homebrew/core, bottle, Linuxbrew, tap, and broad provider claims. |
| D4-CAL-004 | `README.md` and `docs/maintainer_guide.md` Windows sections | Windows support wording must keep report freshness deferred and avoid broad Windows parity. | Day 6 verify CMake/MSVC-only support boundary and selected report non-claim. |
| D4-CAL-005 | report-index and selected report docs | Selected manifest authority must reflect seven selected rows and no Windows selected-report platform promotion. | Day 6 verify manifest wording, selected rows, support tiers, and non-claims. |
| D4-CAL-006 | `docs/api_reference.md` and generated API docs guidance | Generated API HTML must remain local-only and source-controlled API docs must remain the supported entry point. | Day 7 verify local-only wording and generated output staging guard expectations. |
| D4-CAL-007 | QR-facing docs and public header references | QR header coherence must remain a declaration-preserving docs cleanup claim. | Day 7 verify no broad QR behavior, API addition, ABI, or external parity wording. |
| D4-CAL-008 | comparison report docs and manifest | Cholesky comparison must remain selected-fixture-only. | Day 6 or Day 7 verify selected Cholesky target wording and broad external-parity non-claims. |
| D4-CAL-009 | Sprint 185/Sprint 186 closeout docs | Sprint 185 retrospective predates the PR #205 review fix. | Day 8 or Day 12 include `a64c1bc0` in final Epic evidence rather than rewriting historical Sprint 185 retrospective. |
| D4-CAL-010 | state-of-the-art/support-tier language across public docs | Final closeout must not imply state-of-the-art, portable performance, release readiness, broad platform parity, or package/ABI support. | Days 5-7 scan edited public docs and keep non-claims explicit. |

## Document-Family Calibration Checklist

| Day | Document family | Checklist |
| --- | --- | --- |
| Day 5 | `README.md`, `INSTALL.md`, package docs | Verify allocation-failure, package-manager, Homebrew proof, shared-library/ABI, and Windows package metadata wording. |
| Day 6 | `docs/maintainer_guide.md`, report docs, selected manifest docs | Verify maintainer validation owners, selected report target authority, Windows report deferral, Cholesky comparison boundaries, and residual references. |
| Day 7 | `docs/api_reference.md`, generated API docs guidance, public header/example docs | Verify generated API local-only status, QR declaration-preserving cleanup, generated docs input/output boundaries, and example claim scope. |
| Day 8 | `docs/planning/EPIC_16/PROJECT_PLAN.md` | Mark final statuses using the Day 3 matrix and link evidence without changing scope history. |
| Day 12 | `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md` | Summarize earned claims, non-claims, residuals, and validation evidence in one final closeout artifact. |

## Validation

Day 4 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required. Required validation:

```sh
git diff --check
```
