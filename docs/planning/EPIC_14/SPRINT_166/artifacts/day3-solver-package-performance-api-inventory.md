# Sprint 166 Day 3: Solver, Package, Performance, And API Evidence Inventory

## Purpose

Day 3 completes the final evidence inventory started on Day 2 by covering the solver comparison, Windows package, performance publication, public-header/API, and static-first package-boundary evidence from Sprints 161-165. The output is an evidence map for Sprint 166 validation design, hosted CI reconciliation, claim audit, project-plan reconciliation, and Epic 14 closeout.

## Source Inputs

| Source | Day 3 use |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_161/RETROSPECTIVE.md` | Partial-SVD comparison publication closure. |
| `docs/planning/EPIC_14/SPRINT_162/RETROSPECTIVE.md` | Windows package parity decision closure. |
| `docs/planning/EPIC_14/SPRINT_163/RETROSPECTIVE.md` | Methodology-bound performance publication closure. |
| `docs/planning/EPIC_14/SPRINT_164/RETROSPECTIVE.md` | Public-header/API coherence closure. |
| `docs/planning/EPIC_14/SPRINT_165/RETROSPECTIVE.md` | Static-first package boundary hardening closure. |
| `docs/planning/EPIC_14/SPRINT_161/artifacts/day14-closeout.md` | Final partial-SVD validation and closeout record. |
| `docs/planning/EPIC_14/SPRINT_162/artifacts/day14-closeout.md` | Final Windows package validation and non-claim record. |
| `docs/planning/EPIC_14/SPRINT_163/artifacts/day14-closeout.md` | Final performance publication validation and non-superiority record. |
| `docs/planning/EPIC_14/SPRINT_164/artifacts/day14-closeout.md` | Final public-header/API validation and declaration-preservation record. |
| `docs/planning/EPIC_14/SPRINT_165/artifacts/day14-closeout-and-handoff.md` | Final static-first package validation, residuals, and Sprint 166 handoff. |

## Partial-SVD Comparison Evidence Map

| Surface | Evidence owner | Sprint 161 close state | Claim boundary |
| --- | --- | --- | --- |
| Selected partial-SVD comparison target | `scripts/run_external_comparison.py --target partial-svd-diag6-k2` | Added descriptor-backed generation for `partial_svd_diag6_k2`. | One diagonal top-k fixture only. |
| Source-controlled report metadata | `tests/corpus/manifests/report_families.tsv` | Added `comparison/partial_svd_diag6_k2`. | Metadata is contract/context, not pass evidence by itself. |
| Selected comparison freshness | `make report-index-comparison-freshness` | Regenerates two QR targets plus `partial-svd-diag6-k2`; strict freshness covers 22 generated rows plus three contract rows. | Local selected comparison freshness only unless hosted artifacts prove the same selected surface. |
| Runner regression tests | `tests/test_run_external_comparison.py` | Covers target dispatch, generated artifacts, row IDs, metadata, support tier, and optional dependency context. | Harness behavior and selected rows, not broad solver correctness. |
| Normalizer regression tests | `tests/test_normalize_report_index.py` | Covers complete, missing, unexpected, duplicate, stale, fail, skip, and defer selected-row states. | Row-state semantics only. |
| Documentation | README, maintainer guide, solver-selection docs, corpus docs, report-index schema docs | Describes selected QR plus partial-SVD comparison freshness. | No broad SVD, partial-SVD, external-library, hosted, release, platform, package, ABI, performance, or state-of-the-art claim. |

Sprint 166 claim audit should preserve the Sprint 161 positive claim exactly: one selected fixture-local partial-SVD comparison family for `partial_svd_diag6_k2`, with singular-value, residual, orthogonality, and diagonal projector diagnostics against the selected source-controlled dense SVD reference helper.

## Windows Package Decision Evidence Map

| Surface | Evidence owner | Sprint 162 close state | Claim boundary |
| --- | --- | --- | --- |
| Windows package tier | `.github/workflows/windows-ci.yml` | Windows package validation remains CMake-first and static-first. | Not broad Windows platform parity. |
| Installed Windows CMake package proof | Windows hosted CMake install/downstream lane | Installs static `.lib`, headers, CMake package metadata, and checks maintained/exact-version/mismatch consumers. | Hosted Windows evidence, not locally reproduced on macOS. |
| Windows `sparse.pc` | Windows workflow and docs | Metadata-only inspection. | No Windows `pkg-config` command execution parity. |
| Windows Makefile install/uninstall | `scripts/static_package_deferral_check.sh`, Windows workflow wording | Retained as explicit non-claim and guarded against unselected execution. | No Windows Makefile parity claim. |
| Package guard | `scripts/static_package_deferral_check.sh` | Fails if docs/workflow drift into unselected Windows package execution or wording. | Guard does not add support by itself. |
| Public docs | README, INSTALL, maintainer guide | CMake-first Windows support wording aligned. | No package-manager, shared-library, dynamic ABI, runtime-loader, broad Windows, performance, or state-of-the-art claim. |

Sprint 166 should carry Windows Makefile parity and Windows `pkg-config` execution parity as retained non-claims unless a new provider-backed proof is added.

## Performance Publication Evidence Map

| Surface | Evidence owner | Sprint 163 close state | Claim boundary |
| --- | --- | --- | --- |
| Canonical benchmark report | `make bench-canonical-report` | Selected local performance-publication command with four canonical rows. | Local-only threshold-free measurements; no portable performance or superiority claim. |
| Performance sentinels | `make performance-sentinels` | Selected sentinel command with 19 rows. | S5 is the only hard local wall-check gate; S2/S3 are threshold-free backend-context rows. |
| Methodology fields | benchmark/sentinel report scripts and normalizer output | Rows carry status, support tier, claim boundary, fixture/workload, repeat, warmup, variance, baseline, threshold, backend context, and methodology notes. | Methodology metadata does not convert local rows into hosted/release proof. |
| Report-index normalization | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | Preserves benchmark/sentinel methodology text. | Navigation metadata, not release proof. |
| Documentation | README, `docs/benchmarking.md`, maintainer guide, report-index schema docs | States local-only, threshold-free, and non-superiority boundaries. | No package, ABI, platform, backend superiority, external parity, or state-of-the-art claim. |

Sprint 166 validation may run selected benchmark/report checks, but public claims must continue to cite Sprint 163 rows as methodology-bound local evidence only.

## Public Header And API Evidence Map

| Surface | Evidence owner | Sprint 164 close state | Claim boundary |
| --- | --- | --- | --- |
| Selected header batch | `include/sparse_matrix.h`, `include/sparse_iterative.h`, `include/sparse_eigs.h` | Cleaned ownership, lifetime, output-buffer, result-state, callback, repeated-run handle, and backend-routing wording. | No public declaration changes. |
| Declaration preservation | Sprint 164 normalized declaration capture | Baseline and final checksum both `513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41`. | Local recorded evidence; no maintained helper target yet. |
| Generated API docs | `make docs-check` | Passed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. | Generated API HTML remains local-only and ignored. |
| Public docs | README, tutorial, solver-selection docs | Eigensolver result type and AUTO routing wording aligned with selected headers. | No backend superiority, portable performance, package, ABI, runtime-loader, hosted-docs, external parity, or state-of-the-art claim. |
| Residual header cleanup | Sprint 164 residuals | Non-selected headers remain future cleanup candidates. | Residual comment cleanup does not invalidate selected declaration-preserving claim. |

Sprint 166 should not reopen public declarations unless an explicit API change is selected. Documentation cleanup can cite Sprint 164 as evidence only for the selected header batch.

## Static-First Package Boundary Evidence Map

| Surface | Evidence owner | Sprint 165 close state | Claim boundary |
| --- | --- | --- | --- |
| Static package contract | `CMakeLists.txt`, Make install, CMake install/export, `sparse.pc` | Maintained static archive package surface. | No shared-library or dynamic ABI support. |
| Shared-library deferral | `BUILD_SHARED_LIBS=ON` CMake rejection and `scripts/static_package_deferral_check.sh` | Fail-closed shared-library request path remains guarded. | Rejection is a deferral guard, not shared support. |
| Public-header ABI wording | `include/sparse_cholesky.h` | Replaced stale "`ABI break`" wording with source-rebuild options-layout wording. | No dynamic ABI compatibility policy. |
| Package guard | `scripts/static_package_deferral_check.sh` | Rejects premature export/import and static/shared ABI macro scaffolding, shared metadata, static/shared selectors, unsupported docs wording, and Windows unselected package execution. | Guard preserves non-claims. |
| Make install/`pkg-config` proof | `tests/test_install.sh` | Passed with 23 checks, including filesystem-identity installed path checks and downstream compile/link/run consumers. | Unix-like static archive proof only. |
| CMake install/export proof | `tests/test_cmake_install.sh` | Passed with 27 checks, including static imported target metadata, exact-version consumer success, and mismatched-version rejection. | Static archive package metadata; exact version is not dynamic ABI. |
| Package report-index checks | `python3 scripts/normalize_report_index.py --family package --check` and `--check-freshness` | Six source-controlled advisory package rows passed. | Source-controlled proof-owner rows are advisory, not package-manager distribution. |
| Documentation | README, INSTALL, maintainer guide, CMake comments | Static-first support and retained non-claims aligned. | No runtime-loader, package-manager, broad platform, performance, or state-of-the-art claim. |

Sprint 166 should use Sprint 165 as the package-boundary baseline and carry its residual register forward.

## Cross-Surface Non-Claim Matrix

| Evidence surface | Non-claims to preserve in Sprint 166 |
| --- | --- |
| Partial-SVD comparison | no broad SVD or partial-SVD correctness; no raw singular-vector identity; no vector sign/orientation identity; no repeated-spectrum ordering; no external-library parity; no hosted/release/platform/package/ABI/performance/state-of-the-art proof |
| Windows package | no Windows Makefile parity; no Windows `pkg-config` execution parity; no package-manager support; no shared-library support; no dynamic ABI compatibility; no runtime-loader behavior; no broad Windows parity |
| Performance publication | no portable performance; no backend superiority; no release benchmark claim; no hosted performance proof; no package/ABI/platform proof; no state-of-the-art performance claim |
| Public header/API | no public declaration changes; no dynamic ABI compatibility; no hosted generated API HTML publication; no package/runtime/backend/performance/external-parity/state-of-the-art claim |
| Static package boundary | no shared-library support; no dynamic ABI compatibility; no runtime-loader behavior; no package-manager distribution; no static/shared selector support; no broad platform package parity |

## Final Validation Planning Inputs

Day 4 should consider these command owners when designing the strongest feasible final validation baseline:

- full C gate if any `.c` or `.h` files change: `make format && make lint && make test`;
- selected comparison freshness: `make report-index-comparison-freshness`;
- selected oracle freshness: `make report-index-oracle-freshness`;
- report-index checks: `python3 scripts/normalize_report_index.py --check` and selected family-specific freshness commands;
- generated API docs: `make docs-check`;
- performance publication: `make bench-canonical-report` and `make performance-sentinels`;
- package boundary: `bash scripts/static_package_deferral_check.sh`, `bash tests/test_install.sh`, and `bash tests/test_cmake_install.sh`;
- focused Python checks: `python3 tests/test_normalize_report_index.py` and `python3 tests/test_run_external_comparison.py`;
- corpus metadata: `python3 scripts/validate_corpus_schema.py`;
- documentation hygiene: `git diff --check` and targeted claim scans.

## Reconciliation Items Carried Forward

| Item | Source | Later sprint day |
| --- | --- | --- |
| Hosted comparison artifact name/content may lag current three-family selected comparison freshness surface. | Day 2 inventory; `.github/workflows/ci.yml`; Sprint 159/160/161 closeouts. | Day 7 hosted CI reconciliation. |
| Sprint 164 declaration preservation command is recorded but not a maintained helper target. | Sprint 164 retrospective residual. | Day 11 project-plan reconciliation and residual queue. |
| Sprint 163 performance rows remain local-only and have limited repeat/warmup/variance methodology. | Sprint 163 retrospective residual. | Day 8 claim audit and Day 13 residual queue. |
| Windows Makefile and Windows `pkg-config` execution parity remain retained non-claims. | Sprint 162 and Sprint 165 retrospectives. | Day 9 claim audit and Day 13 residual queue. |
| Shared-library support, dynamic ABI policy, runtime-loader behavior, and package-manager distribution remain future product decisions. | Sprint 165 residual register. | Day 9 claim audit and Day 13 residual queue. |

## Day 4 Handoff

Day 4 should turn the Day 2 and Day 3 evidence inventory into a validation design. It should choose a strongest feasible local baseline, mark hosted-only evidence that cannot be locally reproduced, and separate hard gates from advisory/source-controlled/local-only checks before executing expensive validation on Days 5 and 6.

## Validation Notes

Day 3 changed only Sprint 166 planning artifacts. No `.c`, `.h`, source, script, workflow, or public documentation files were modified, so the full C quality gate and focused generated/package/performance commands were not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Solver, package, performance, and API evidence surfaces are mapped. | Complete | Partial-SVD, Windows package, performance, public-header/API, and static-first package maps are recorded. |
| Each surface has validation and non-claim boundaries attached. | Complete | Evidence maps and cross-surface non-claim matrix attach command owners and retained boundaries. |
| Sprint 165 package handoff is ready for final validation planning. | Complete | Static-first package boundary map and final validation planning inputs include Sprint 165 guard, install, CMake, report-index, and residual evidence. |
