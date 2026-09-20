# Epic 18 Residual Queue

## Purpose

This file is the next-epic handoff for the current Epic 18 closeout state. It
turns Sprint 197 through Sprint 206 evidence into prioritized residual work
with exact closure targets, owner surfaces, expected evidence, validation
commands, and claim boundaries.

The queue is intentionally conservative. Sprint 197 remains historical
final-validation evidence with a numbering caveat. Sprints 198 through 205 are
closed for their selected scopes or explicit re-deferrals. Sprint 206 is
closed through Day 14 as the explicit Epic 18 closeout branch. Residuals below
should not be read as support promotion beyond the exact selected evidence
recorded for each closed sprint.

## Queue Summary

| Priority | Residual ID | Theme | Horizon |
| ---: | --- | --- | --- |
| 1 | E18-RQ-001 | Package-manager distribution beyond local proof | Near-term product/legal/provider decision |
| 2 | E18-RQ-002 | Selected Windows Cholesky freshness promotion | Hosted evidence and manifest promotion decision |
| 3 | E18-RQ-003 | Additional allocation-failure owner proof | Near-term selected reliability proof |
| 4 | E18-RQ-004 | Additional review-surface reduction | Incremental maintainability |
| 5 | E18-RQ-005 | Additional hosted selected benchmark freshness | Hosted platform evidence |
| 6 | E18-RQ-006 | Windows QR incompatible comparison promotion | Hosted Windows comparison evidence |
| 7 | E18-RQ-007 | Generated API publication policy | Product/docs infrastructure decision |
| 8 | E18-RQ-008 | Adoption and diagnostics follow-up | Documentation/product UX follow-up |
| 9 | E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Long-horizon product/platform policy |
| 10 | E18-RQ-010 | State-of-the-art evidence program | Long-horizon methodology and research proof |

## Priority 1: E18-RQ-001

| Field | Value |
| --- | --- |
| Theme | Package-manager distribution beyond local proof. |
| Source | Epic 17 residual queue; Sprint 198; Sprint 206 Days 3, 5, 9, and 11. |
| Current status | Selected Sprint 198 local proof is closed. Broader package-manager distribution remains residual. |
| Owner surfaces | Root license metadata; `packaging/homebrew/`; `scripts/homebrew_local_formula_proof.sh`; README; INSTALL; maintainer guide; package/static guards; install tests; any future provider recipe or tap. |
| Why it remains | Sprint 198 proved a developer-mode local Homebrew static source formula path. It did not create Homebrew/core readiness, bottles, Linuxbrew support, public tap maintenance, vcpkg/Conan/pkgsrc/distro packages, binary packages, or a user-facing package-manager install path. |
| Closure target | Decide exact provider scope, maintain approved license/package metadata, run provider-specific install/test/uninstall proof, update guards, and promote docs only to the support tier earned by reviewed evidence. |
| Expected evidence | Formula or provider recipe renders; source archive/checksum are reproducible; install succeeds on the claimed platform tier; installed static files are checked; downstream compile/link/test succeeds; uninstall/cleanup succeeds; non-claims outside the proven provider path remain guarded. |
| Validation commands | `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` for local proof; provider-specific proof command for broader support; `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `make support-docs-guard`; `make docs-check`; full C gate if `.c` or `.h` files change. |
| Claim boundary | Do not claim Homebrew/core readiness, bottles, Linuxbrew, public tap maintenance, vcpkg, Conan, pkgsrc, distro/system packages, binary packages, broad package-manager distribution, or package-manager user support until the exact provider path is proved and documented. |

## Priority 2: E18-RQ-002

| Field | Value |
| --- | --- |
| Theme | Selected Windows Cholesky freshness promotion. |
| Source | Epic 17 residual queue; Sprint 199; Sprint 206 Days 2-4 and 11. |
| Current status | Sprint 199 closed as re-deferral. Guarded workflow evidence exists; selected Windows freshness promotion remains unearned. |
| Owner surfaces | `.github/workflows/windows-ci.yml`; `tests/corpus/manifests/selected_report_targets.tsv`; `scripts/run_external_comparison.py`; `scripts/normalize_report_index.py`; corpus README; report-index schema docs; maintainer guide; README; INSTALL. |
| Why it remains | Existing evidence is guarded workflow evidence, not selected manifest promotion. Hosted Windows artifacts, generated support tier, selected metadata, and non-claim wording must be promoted together. |
| Closure target | Inspect hosted Windows selected Cholesky artifacts, verify exact target rows and artifact paths, promote or re-defer manifest metadata, and recalibrate docs and guards together. |
| Expected evidence | Hosted Windows selected comparison job passes; uploaded artifact contains only the expected selected Cholesky bundle; target `cholesky-spd-tridiag-5` emits expected rows; path normalization handles Windows separators; support tier, workflow platforms, claim scope, and non-claims match the evidence. |
| Validation commands | `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_run_external_comparison.py`; `make windows-powershell-guard`; target-specific selected comparison freshness command; hosted Windows workflow evidence review. |
| Claim boundary | Until hosted evidence and metadata promotion are reviewed together, claim only guarded workflow evidence. Do not claim broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, broad selected comparison freshness, or broad Windows generated-report parity. |

## Priority 3: E18-RQ-003

| Field | Value |
| --- | --- |
| Theme | Additional allocation-failure owner proof. |
| Source | Epic 17 residual queue; Sprint 200; Sprint 206 Days 2 and 11. |
| Current status | Sprint 200 closed for selected `sparse_symbolic_lu()` owner proof. Broader allocation-failure coverage remains residual. |
| Owner surfaces | Selected symbolic, analysis, etree, direct-solver, matrix-construction, or output-publication owner; deterministic allocation harness; focused tests; Make/CTest labels; README; INSTALL; maintainer guide. |
| Why it remains | Sprint 200 proves one selected symbolic LU owner only. Other allocation-heavy owners remain separate candidates. |
| Closure target | Select exactly one additional owner, record cleanup/publication/retry/caller-input invariants, extend deterministic failure injection, add regressions, add a focused gate, and update claim docs. |
| Expected evidence | Failed allocation returns the expected status; partial state is cleaned; stale outputs are suppressed; caller-owned inputs are preserved; retry after reset succeeds; focused gate and registration guard prevent drift. |
| Validation commands | New focused owner gate; new registration guard if applicable; relevant CTest label; focused owner binary; `make source-list-check`; `make format && make lint && make test` if `.c` or `.h` files change; `make docs-check`. |
| Claim boundary | Claim only the selected owner and selected allocation path. Do not claim broad allocation-failure coverage, OS OOM behavior, concurrent allocation-hook behavior, generated-tooling reliability, package/install reliability, or state-of-the-art reliability. |

## Priority 4: E18-RQ-004

| Field | Value |
| --- | --- |
| Theme | Additional review-surface reduction. |
| Source | Epic 17 residual queue; Sprint 201; Sprint 206 Days 2 and 11. |
| Current status | Sprint 201 closed for the selected `tests/test_svd.c` rank, pseudoinverse, and dense low-rank helper cluster. Broader large-surface cleanup remains residual. |
| Owner surfaces | Large QR, LDLT, SVD, etree, integration, graph, direct-solver, or helper surfaces; guard scripts/tests; maintainer guide; source-list and CMake registration. |
| Why it remains | Sprint 201 reduced one selected SVD helper cluster only. Other large QR, LDLT, etree, integration, graph, direct-solver, helper, source, and test surfaces remain separate candidates. |
| Closure target | Select one high-risk cluster, record no-behavior-change boundaries, extract or refactor only where reviewability improves, add ownership guards, and prove behavior with focused and required full validation. |
| Expected evidence | Candidate ranking; selected-cluster rationale; behavior-preservation notes; extraction diff; focused tests; guard coverage; source-list/CMake parity when registration changes. |
| Validation commands | Cluster-specific focused tests; relevant helper guard or new guard; source-list/CMake parity when registration changes; `make format && make lint && make test` if `.c` or `.h` files change. For the Sprint 201 SVD cluster, preserve `make svd-helper-guard` and `python3 tests/test_svd_helper_guard.py`. |
| Claim boundary | Do not claim new solver behavior, public API change, numerical tolerance change, performance improvement, or broad review-surface cleanup from one selected reduction. |

## Priority 5: E18-RQ-005

| Field | Value |
| --- | --- |
| Theme | Additional hosted selected benchmark freshness. |
| Source | Epic 17 residual queue; Sprint 202; Sprint 206 Days 2 and 11. |
| Current status | Sprint 202 closed one macOS hosted selected freshness lane for `SRT-BENCH-REFACTOR-CSC-NOS4`. Broader benchmark/platform/performance claims remain residual. |
| Owner surfaces | Benchmark workflow YAML; selected target manifest; `scripts/check_bench_canonical_freshness.py`; report normalizer; benchmark docs; maintainer guide; README; INSTALL. |
| Why it remains | Sprint 202 adds one bounded macOS hosted selected lane. It does not prove portable performance, timing thresholds, Windows selected benchmark freshness, broad benchmark-family publication, or state-of-the-art performance. |
| Closure target | Add another hosted selected benchmark freshness lane only after selecting an exact row/platform, preserving methodology-bound, threshold-free, non-portable interpretation unless a future sprint explicitly designs thresholds. |
| Expected evidence | Hosted platform pass; exact selected benchmark bundle; selected CSV matches manifest contract; methodology metadata records platform/compiler/build flags/repeat policy; docs keep non-portable wording. |
| Validation commands | `make bench-canonical-report-freshness`; `python3 tests/test_bench_canonical_freshness.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_performance_docs.py`; `python3 tests/test_normalize_report_index.py`; hosted workflow evidence review. |
| Claim boundary | Do not claim portable performance, timing thresholds, Linux/macOS performance parity, Windows selected benchmark freshness, broad benchmark-family publication, package-manager distribution, package/ABI support, backend superiority, release benchmark readiness, or state-of-the-art performance. |

## Priority 6: E18-RQ-006

| Field | Value |
| --- | --- |
| Theme | Windows QR incompatible comparison promotion. |
| Source | Epic 17 residual queue; Sprint 203; Sprint 206 Days 2 and 11. |
| Current status | Sprint 203 closed as re-deferral. Local QR incompatible generator/freshness evidence and guard coverage exist; hosted Windows/MSVC proof and hosted artifact inspection remain absent. |
| Owner surfaces | Windows workflow; QR incompatible comparison target; comparison runner; selected manifest; normalizer; corpus docs; maintainer guide; README; INSTALL. |
| Why it remains | Sprint 203 generated local selected QR incompatible evidence and hardened manifest, workflow, normalizer, docs, and Windows claim-boundary guards, but did not add hosted Windows/MSVC execution evidence or promote selected target Windows metadata. |
| Closure target | Add MSVC/CMake proof for `qr-incompatible-ls`, fix Windows-safe generation/path handling as needed, inspect artifacts, promote exact selected metadata if evidence supports it, and retain broad QR parity non-claims. |
| Expected evidence | Windows CMake probe builds and runs; generated rows match expected QR incompatible target output; artifact paths normalize correctly; manifest metadata matches the promoted platform scope. |
| Validation commands | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls`; `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`; `python3 tests/test_run_external_comparison.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_validate_windows_powershell.py`; focused QR solve tests; hosted Windows comparison workflow evidence review before any future promotion. |
| Claim boundary | Do not claim broad QR least-squares parity, broad external-library parity, Windows selected oracle freshness, Windows benchmark freshness, or broad Windows report freshness. |

## Priority 7: E18-RQ-007

| Field | Value |
| --- | --- |
| Theme | Generated API publication policy. |
| Source | Sprint 204; Sprint 206 Days 6, 10, and 11. |
| Current status | Sprint 204 closed stronger local-only generated API policy. Hosted publication, retained generated-doc artifacts, and committed generated HTML remain residual options only. |
| Owner surfaces | `docs/api_reference.md`; `docs/maintainer_guide.md`; `Doxyfile`; generated API ignore rules; docs/API freshness scripts; README; INSTALL; workflows if publication is added. |
| Why it remains | Sprint 204 selected the stronger local-only path and implemented generated-page freshness, local-only staging, workflow non-publication, API routing, and Makefile wiring guards. Any future publication path is a product decision that must replace the local-only policy with matching proof. |
| Closure target | Reopen only if a future sprint deliberately selects hosted publication, retained artifact publication, or committed generated output and implements matching freshness, link, staging, workflow, retention, and claim-boundary evidence. |
| Expected evidence | Current closure evidence is `SPRINT_204/RETROSPECTIVE.md`, `SPRINT_204/artifacts/day14-closeout-review.md`, and passing `make api-docs-freshness`; future publication evidence needs new workflow/link/retention proof before claims change. |
| Validation commands | Current closure: `make docs-check`; `make api-docs-freshness`; `python3 tests/test_api_docs_coverage.py`; `python3 tests/test_api_docs_local_only_guard.py`; `python3 tests/test_api_docs_routing.py`. Future hosted or artifact publication also needs workflow publication checks and hosted evidence review; full C gate if headers change. |
| Claim boundary | Do not claim hosted API docs, artifact-published generated HTML, committed generated HTML, ABI completeness, package support, or release evidence unless the selected policy explicitly proves it. |

## Priority 8: E18-RQ-008

| Field | Value |
| --- | --- |
| Theme | Adoption and diagnostics follow-up. |
| Source | Sprint 205; Sprint 206 Days 3, 6, and 11. |
| Current status | Sprint 205 closed support truth consolidation, compact problem-shape quick reference, diagnostics vocabulary normalization, and claim-guard alignment. Future work is optional product UX follow-up. |
| Owner surfaces | README; INSTALL; tutorial; cookbook; solver selection; examples; benchmark docs; API reference; maintainer guide; docs claim guards. |
| Why it remains | Sprint 205 closed the selected documentation consolidation scope. The residual remains as a future product-writing bucket for additional adoption experience work outside the compact quick-reference/support-truth/diagnostics/guard scope. |
| Closure target | Reopen only for additional adoption UX changes with their own owner surfaces, wording design, guard coverage, and validation evidence. |
| Expected evidence | Current closure evidence is `SPRINT_205/WORKING_NOTES.md`, `SPRINT_205/artifacts/day13-integrated-validation.md`, and `SPRINT_205/artifacts/day14-closeout-review.md`; future adoption changes need updated design artifacts, edited docs, guard updates, and focused validation. |
| Validation commands | Current closure: `make support-docs-guard`; `make api-docs-freshness`; `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; `python3 tests/test_validate_windows_powershell.py`; `python3 tests/test_selected_performance_docs.py`; `git diff --check`; full C gate if headers change. |
| Claim boundary | Simplified wording must not imply package-manager support, broad platform parity, portable performance, release readiness, dynamic ABI support, or state-of-the-art status. |

## Long-Horizon Deferrals

| Residual ID | Theme | Closure target |
| --- | --- | --- |
| E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Define release criteria, semantic versioning policy, shared-library exports, symbol visibility, SONAME/install-name/RPATH, DLL/import-library behavior, ABI compatibility policy, package selectors, and runtime-loader validation before making public claims. |
| E18-RQ-010 | State-of-the-art evidence program | Define external baselines, versions, fixtures, matrix suites, workloads, tolerances, platforms, compilers, package provenance, reliability semantics, benchmark methodology, acceptance thresholds, and hosted evidence review before any broad state-of-the-art claim. |

## Final Claim Decision

No stronger broad Epic 18 public or maintainer support claim is earned by the
current branch.

The earned evidence is selected and bounded: Sprint 198 local developer-mode
Homebrew proof; Sprint 199 and Sprint 203 re-deferrals; Sprint 200 selected
symbolic LU allocation-failure owner proof; Sprint 201 selected SVD helper
review-surface reduction; Sprint 202 selected macOS benchmark freshness;
Sprint 204 local-only generated API policy; Sprint 205 support/adoption
consolidation; and Sprint 206 reconciliation, claim calibration, project-plan
status, validation, retrospective, residual-handoff, consistency hardening, and
final closeout evidence through Day 14.

Package-manager distribution, selected Windows freshness promotion, broader
allocation-failure coverage, broader review-surface cleanup, broader benchmark
platform freshness, Windows QR promotion, hosted generated API publication,
release readiness, shared-library/dynamic ABI support, portable performance,
broad ecosystem parity, and state-of-the-art status remain unpromoted.

## Source Evidence

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [EPIC_18_RETROSPECTIVE.md](./EPIC_18_RETROSPECTIVE.md)
- [SPRINT_197/WORKING_NOTES.md](./SPRINT_197/WORKING_NOTES.md)
- [SPRINT_197/artifacts/day8-project-plan-status.md](./SPRINT_197/artifacts/day8-project-plan-status.md)
- [SPRINT_198/RETROSPECTIVE.md](./SPRINT_198/RETROSPECTIVE.md)
- [SPRINT_199/RETROSPECTIVE.md](./SPRINT_199/RETROSPECTIVE.md)
- [SPRINT_200/RETROSPECTIVE.md](./SPRINT_200/RETROSPECTIVE.md)
- [SPRINT_201/RETROSPECTIVE.md](./SPRINT_201/RETROSPECTIVE.md)
- [SPRINT_202/RETROSPECTIVE.md](./SPRINT_202/RETROSPECTIVE.md)
- [SPRINT_203/RETROSPECTIVE.md](./SPRINT_203/RETROSPECTIVE.md)
- [SPRINT_204/RETROSPECTIVE.md](./SPRINT_204/RETROSPECTIVE.md)
- [SPRINT_205/RETROSPECTIVE.md](./SPRINT_205/RETROSPECTIVE.md)
- [SPRINT_206/WORKING_NOTES.md](./SPRINT_206/WORKING_NOTES.md)
- [SPRINT_206/artifacts/day2-outcome-reconciliation.md](./SPRINT_206/artifacts/day2-outcome-reconciliation.md)
- [SPRINT_206/artifacts/day10-broad-quality-gates.md](./SPRINT_206/artifacts/day10-broad-quality-gates.md)
- [SPRINT_206/artifacts/day11-epic-retrospective-draft.md](./SPRINT_206/artifacts/day11-epic-retrospective-draft.md)
- [SPRINT_206/artifacts/day12-residual-queue-draft.md](./SPRINT_206/artifacts/day12-residual-queue-draft.md)
- [SPRINT_206/artifacts/day13-consistency-hardening.md](./SPRINT_206/artifacts/day13-consistency-hardening.md)
- [SPRINT_206/artifacts/day14-closeout-review.md](./SPRINT_206/artifacts/day14-closeout-review.md)
