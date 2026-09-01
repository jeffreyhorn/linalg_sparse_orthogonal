# Epic 16 Residual Queue

## Purpose

This file is the next-epic handoff for Epic 16 residuals. It turns the Sprint
186 reconciled evidence matrix and retrospective residuals into prioritized,
deduplicated work items with exact closure targets, owner surfaces, expected
evidence, validation commands, and deferral horizons.

## Queue Summary

| Priority | Residual ID | Theme | Deferral horizon |
| ---: | --- | --- | --- |
| 1 | R186-PKG-LICENSE | Homebrew local proof blocker | Near-term product/legal metadata decision |
| 2 | R186-WIN-PWSH | PowerShell validation environment | Near-term validation infrastructure |
| 3 | R186-WIN-REPORT-FRESHNESS | Windows selected report freshness | Hosted evidence and manifest promotion decision |
| 4 | R186-HOSTED-API | Generated API publication | Product documentation decision |
| 5 | R186-BROAD-COMPARISON | Future bounded comparison breadth | Incremental implementation |
| 6 | R186-REVIEW-SURFACE-NEXT | Future review-surface reduction | Incremental maintainability |

## Priority 1: R186-PKG-LICENSE

| Field | Value |
| --- | --- |
| Source | Sprint 180, item 180.6. |
| Current status | Residualized. |
| Owner surfaces | Repository root license metadata; `packaging/homebrew/sparse-lu-ortho.rb.in`; `packaging/homebrew/README.md`; `scripts/homebrew_local_formula_proof.sh`; package/install docs. |
| Why it remains | No root `LICENSE`, `COPYING`, or `NOTICE` file exists, so the Homebrew proof script correctly stops before formula rendering and install proof success. |
| Closure target | Add approved standalone license metadata or explicitly select an alternate formula license strategy. |
| Expected evidence | Homebrew formula renders, local source archive/checksum are created, install succeeds, installed static files are checked, `brew test` passes, uninstall succeeds, and temporary proof outputs are cleaned. |
| Validation commands | `bash scripts/homebrew_local_formula_proof.sh`; `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; install/downstream checks selected by the implementation sprint. |
| Claim boundary | Until closed, Homebrew remains a local proof path, not Homebrew support, Homebrew/core readiness, bottle support, Linuxbrew support, public tap support, or broad package-manager support. |
| Deferral horizon | Near-term product/legal metadata decision. |

## Priority 2: R186-WIN-PWSH

| Field | Value |
| --- | --- |
| Source | Sprint 182, item 182.6. |
| Current status | Residualized. |
| Owner surfaces | PowerShell-capable local or hosted environment; Windows workflow/report scripts; selected workflow guard tests; Windows deferral artifact. |
| Why it remains | Local `pwsh` is unavailable, so PowerShell parse/workflow validation could not run locally. |
| Closure target | Run the PowerShell parse/workflow checks in an environment with `pwsh`, or document a hosted-only validation owner. |
| Expected evidence | PowerShell syntax/parse validation passes for selected Windows report scripts and workflow snippets, with exact command output recorded. |
| Validation commands | PowerShell parse check selected by the future implementation sprint; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 scripts/validate_corpus_schema.py`. |
| Claim boundary | Until closed, local validation cannot claim PowerShell report-script proof. |
| Deferral horizon | Near-term validation infrastructure. |

## Priority 3: R186-WIN-REPORT-FRESHNESS

| Field | Value |
| --- | --- |
| Source | Sprint 182, items 182.2 and 182.3. |
| Current status | Renewed and narrowed by Sprint 190. |
| Owner surfaces | `.github/workflows/windows-ci.yml`; selected report target manifest; report generator/normalizer scripts; Windows report freshness deferral artifact; maintainer/report docs. |
| Why it remains | Sprint 190 wires one bounded hosted Windows selected Cholesky comparison freshness job and local CMake-probe generator path, but hosted `windows-2022` evidence has not been observed in this local sprint pass and the source selected-target manifest still does not list `windows`. |
| Closure target | Review hosted `selected-comparison-freshness` evidence for `cholesky-spd-tridiag-5`, then either promote exactly that manifest row to `windows` metadata or retain the staged boundary with refreshed blockers. |
| Expected evidence | `.github/workflows/windows-ci.yml` job `selected-comparison-freshness`; artifact `sprint190-windows-selected-comparison-cholesky`; exact target `cholesky-spd-tridiag-5`; six expected row IDs; required generated files; target-specific freshness command; hosted Windows pass evidence; and aligned selected manifest metadata if promotion lands. |
| Validation commands | `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_run_external_comparison.py`; `python3 tests/test_validate_windows_powershell.py`; `python3 scripts/validate_corpus_schema.py`; `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake`; `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`; hosted `selected-comparison-freshness` on `windows-2022`. |
| Claim boundary | Until hosted evidence and manifest promotion are reviewed together, Windows has one bounded selected Cholesky workflow path only; broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, broad selected comparison freshness, and broad Windows generated-report parity remain non-claims. |
| Deferral horizon | Hosted evidence and manifest promotion decision. |

## Priority 4: R186-HOSTED-API

| Field | Value |
| --- | --- |
| Source | Sprint 179, item 179.2. |
| Current status | Narrowed. |
| Owner surfaces | `Doxyfile`; `docs/api_reference.md`; `docs/maintainer_guide.md`; `scripts/check_api_docs_local_only.sh`; future documentation publication workflow if selected. |
| Why it remains | Sprint 179 intentionally selected strengthened local-only generated API HTML instead of hosted, retained-artifact, or committed generated output. |
| Closure target | Revisit hosted publication, retained CI artifacts, or committed generated output only if product value justifies the additional publication and freshness guard work. |
| Expected evidence | Publication decision record, configured publication path, freshness guard, staging guard, docs navigation update, and proof that generated output policy matches the selected support tier. |
| Validation commands | `make api-docs-validate`; `make api-docs-freshness`; new publication/retention guard if hosted or retained output is selected; `git diff --check`. |
| Claim boundary | Until closed, generated API HTML remains local-only and ignored; `docs/api_reference.md` plus checked-in public headers remain the source-controlled API reference path. |
| Deferral horizon | Product documentation decision. |

## Priority 5: R186-BROAD-COMPARISON

| Field | Value |
| --- | --- |
| Source | Sprint 183 and selected comparison closeout. |
| Current status | Residualized as future breadth, not a failed Epic 16 item. |
| Owner surfaces | `scripts/run_external_comparison.py`; `tests/test_run_external_comparison.py`; selected report target manifest; report-family manifests; solver docs and maintainer guide. |
| Why it remains | Epic 16 added one bounded Cholesky comparison family; broad external comparison parity remains intentionally unclaimed. |
| Closure target | Add future comparison evidence one bounded family at a time with exact fixture, metric, tolerance, report, manifest, support-tier, and non-claim evidence. |
| Expected evidence | Source-controlled fixtures or generators, selected comparison target row, generated project/baseline observations, dependency status, study output, normalized report rows, docs updates, and selected freshness proof. |
| Validation commands | `python3 tests/test_run_external_comparison.py`; `make report-index-comparison-freshness`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; relevant focused C tests. |
| Claim boundary | Until each bounded family is selected and proven, do not claim broad SuiteSparse, Eigen, LAPACK, NumPy, SciPy, PETSc, Trilinos, package ecosystem, or solver-family parity. |
| Deferral horizon | Incremental implementation, one selected family at a time. |

## Priority 6: R186-REVIEW-SURFACE-NEXT

| Field | Value |
| --- | --- |
| Source | Sprint 185 review-surface reduction handoff. |
| Current status | Residualized as future maintainability work, not a failed Epic 16 item. |
| Owner surfaces | Candidate large test/source files; maintainer guide helper-ownership sections; Make/CMake/source-list guards; future cluster-specific guard. |
| Why it remains | Sprint 185 selected exactly one cluster, `tests/test_ldlt_csc.c`; other large review surfaces remain outside that bounded scope. |
| Closure target | Select exactly one future large review surface and repeat the behavior-preserving extraction pattern. |
| Expected evidence | Candidate inventory, selection rationale, no-behavior-change design, focused extraction, owner-surface docs, registration guard, focused tests, source-list check, and full C gate if `.c` or `.h` files change. |
| Validation commands | Future focused cluster test; future cluster guard; `make source-list-check`; `make format && make lint && make test` when `.c` or `.h` files change; `git diff --check`. |
| Claim boundary | Do not claim solver behavior, correctness expansion, performance, production API changes, or broad review-surface cleanup from a single selected extraction. |
| Deferral horizon | Incremental maintainability work after a new single-cluster selection. |

## Long-Horizon Deferrals

These remain intentionally outside the prioritized near-term queue unless a
future epic explicitly selects them:

- unqualified state-of-the-art sparse linear algebra positioning;
- broad external ecosystem parity;
- portable performance or backend superiority proof;
- shared-library support and dynamic ABI compatibility;
- broad Windows package/platform parity;
- broad generated report hosting or publication;
- release packaging evidence.

## Source Evidence

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [EPIC_16_RETROSPECTIVE.md](./EPIC_16_RETROSPECTIVE.md)
- [SPRINT_186/artifacts/day3-reconciled-evidence-matrix.md](./SPRINT_186/artifacts/day3-reconciled-evidence-matrix.md)
- [SPRINT_186/artifacts/day8-project-plan-status-update.md](./SPRINT_186/artifacts/day8-project-plan-status-update.md)
- [SPRINT_186/artifacts/day10-focused-integrated-validation.md](./SPRINT_186/artifacts/day10-focused-integrated-validation.md)
- [SPRINT_186/artifacts/day11-full-repository-quality-gate.md](./SPRINT_186/artifacts/day11-full-repository-quality-gate.md)
