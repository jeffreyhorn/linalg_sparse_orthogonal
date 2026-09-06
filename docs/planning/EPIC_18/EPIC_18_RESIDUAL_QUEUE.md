# Epic 18 Residual Queue

## Purpose

This file is the next-epic handoff for the current Epic 18 closeout state. It
turns Sprint 197 Day 13 residual triage into prioritized work items with exact
closure targets, owner surfaces, expected evidence, validation commands, and
claim boundaries.

The queue is intentionally conservative. The requested `SPRINT_197` branch
executes the final-validation scope that `PROJECT_PLAN.md` labels as Sprint
206, while Sprints 198 through 205 have no branch-local implementation
artifacts yet. These residuals should not be read as completed Epic 18 work or
as support promotion.

## Queue Summary

| Priority | Residual ID | Theme | Deferral horizon |
| ---: | --- | --- | --- |
| 1 | E18-RQ-001 | Homebrew/package-manager support blocker | Near-term product/legal metadata decision |
| 2 | E18-RQ-002 | Selected Windows Cholesky freshness promotion | Hosted evidence and manifest promotion decision |
| 3 | E18-RQ-003 | Additional allocation-failure owner proof | Near-term selected reliability proof |
| 4 | E18-RQ-004 | Additional review-surface reduction | Incremental maintainability |
| 5 | E18-RQ-005 | Additional hosted selected benchmark freshness | Hosted platform evidence |
| 6 | E18-RQ-006 | Windows QR incompatible comparison promotion | Hosted Windows comparison evidence |
| 7 | E18-RQ-007 | Generated API publication policy | Product/docs infrastructure decision |
| 8 | E18-RQ-008 | Adoption and diagnostics simplification | Documentation coherence |
| 9 | E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Long-horizon product/platform policy |
| 10 | E18-RQ-010 | State-of-the-art evidence program | Long-horizon methodology and research proof |

## Priority 1: E18-RQ-001

| Field | Value |
| --- | --- |
| Theme | Homebrew/package-manager support blocker. |
| Source | Epic 17 residual queue; Epic 18 Sprint 198 plan; Sprint 197 Day 2-3 evidence ledger. |
| Current status | Pending future execution; support remains unclaimed. |
| Owner surfaces | Root license metadata; `packaging/homebrew/sparse-lu-ortho.rb.in`; `packaging/homebrew/README.md`; `scripts/homebrew_local_formula_proof.sh`; README; INSTALL; maintainer guide; package/static guards; install tests. |
| Why it remains | No approved standalone root license metadata or exact Homebrew formula license identifier exists on this branch. Legal/package metadata cannot be invented during closeout. |
| Closure target | Add approved root license metadata, set the exact Homebrew license identifier, run the selected Homebrew proof, update guards, and promote documentation only to the exact support tier earned by proof output. |
| Expected evidence | Local formula proof exits `0`; formula renders; source archive and checksum are created; install succeeds; installed static files are checked; `brew test` passes; uninstall succeeds; proof outputs are cleaned; docs retain non-claims outside the proven path. |
| Validation commands | `bash scripts/homebrew_local_formula_proof.sh`; `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `make docs-check`; full C gate only if `.c` or `.h` files change. |
| Claim boundary | Until closed, do not claim Homebrew support, Homebrew/core readiness, bottles, Linuxbrew support, public tap support, vcpkg, Conan, pkgsrc, distro/system package support, package-manager support, or package-manager distribution. |

## Priority 2: E18-RQ-002

| Field | Value |
| --- | --- |
| Theme | Selected Windows Cholesky freshness promotion. |
| Source | Epic 17 residual queue; Epic 18 Sprint 199 plan; Sprint 197 Day 3 and Day 7 claim-boundary reviews. |
| Current status | Pending future execution; guarded workflow path remains unpromoted. |
| Owner surfaces | `.github/workflows/windows-ci.yml`; `tests/corpus/manifests/selected_report_targets.tsv`; `scripts/run_external_comparison.py`; `scripts/normalize_report_index.py`; corpus README; report-index schema docs; maintainer guide; README; INSTALL. |
| Why it remains | Existing docs describe one guarded Windows Cholesky workflow path, but this branch has no hosted Windows artifact review or selected manifest promotion evidence. |
| Closure target | Inspect hosted Windows selected Cholesky artifacts, verify exact target rows and artifact paths, promote or re-defer manifest metadata, and recalibrate docs and guards together. |
| Expected evidence | Hosted Windows selected comparison job passes; uploaded artifact contains only the expected selected Cholesky bundle; target `cholesky-spd-tridiag-5` emits expected rows; path normalization handles Windows separators; support tier, workflow platforms, claim scope, and non-claims match the evidence. |
| Validation commands | `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_normalize_report_index.py`; `python3 tests/test_run_external_comparison.py`; `make windows-powershell-guard`; target-specific selected comparison freshness command; hosted Windows workflow evidence review. |
| Claim boundary | Until hosted evidence and metadata promotion are reviewed together, claim only a guarded workflow path. Do not claim broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, broad selected comparison freshness, or broad Windows generated-report parity. |

## Priority 3: E18-RQ-003

| Field | Value |
| --- | --- |
| Theme | Additional allocation-failure owner proof. |
| Source | Epic 17 residual queue; Epic 18 Sprint 200 plan; Sprint 197 Day 2-3 evidence ledger. |
| Current status | Pending future execution. |
| Owner surfaces | Selected symbolic, analysis, etree, direct-solver, matrix-construction, or output-publication owner; deterministic allocation harness; focused tests; Make/CTest labels; README; INSTALL; maintainer guide. |
| Why it remains | Prior evidence covers selected owners only. This branch adds no new owner selection, harness reachability, regression, or focused gate. |
| Closure target | Select exactly one additional owner, record cleanup/publication/retry/caller-input invariants, extend deterministic failure injection, add regressions, add a focused gate, and update claim docs. |
| Expected evidence | Failed allocation returns the expected status; partial state is cleaned; stale outputs are suppressed; caller-owned inputs are preserved; retry after reset succeeds; focused gate and registration guard prevent drift. |
| Validation commands | New focused owner gate; new registration guard if applicable; relevant CTest label; focused owner binary; `make source-list-check`; `make format && make lint && make test` if `.c` or `.h` files change; `make docs-check`. |
| Claim boundary | Claim only the selected owner and selected allocation path. Do not claim broad allocation-failure coverage, OS OOM behavior, concurrent allocation-hook behavior, generated-tooling reliability, package/install reliability, or state-of-the-art reliability. |

## Priority 4: E18-RQ-004

| Field | Value |
| --- | --- |
| Theme | Additional review-surface reduction. |
| Source | Epic 17 residual queue; Epic 18 Sprint 201 plan; Sprint 197 Day 3 evidence conflict review. |
| Current status | Pending future execution. |
| Owner surfaces | Large QR, LDLT, SVD, etree, integration, graph, direct-solver, or helper surfaces; guard scripts/tests; maintainer guide; source-list and CMake registration. |
| Why it remains | No new candidate ranking, selected cluster, behavior-preservation invariant, helper extraction, guard, or focused regression exists on this branch. |
| Closure target | Select one high-risk cluster, record no-behavior-change boundaries, extract or refactor only where reviewability improves, add ownership guards, and prove behavior with focused and required full validation. |
| Expected evidence | Candidate ranking; selected-cluster rationale; extraction diff; behavior-preservation notes; focused tests; guard coverage; source-list/CMake parity when registration changes. |
| Validation commands | Cluster-specific focused tests; relevant helper guard or new guard; `make source-list-check`; CMake parity if registration changes; `make format && make lint && make test` if `.c` or `.h` files change. |
| Claim boundary | Do not claim new solver behavior, public API change, numerical tolerance change, performance improvement, or broad review-surface cleanup from one selected reduction. |

## Priority 5: E18-RQ-005

| Field | Value |
| --- | --- |
| Theme | Additional hosted selected benchmark freshness. |
| Source | Epic 17 residual queue; Epic 18 Sprint 202 plan; Sprint 197 Day 4-7 claim audits. |
| Current status | Pending future execution. |
| Owner surfaces | Benchmark workflow YAML; selected target manifest; `scripts/check_bench_canonical_freshness.py`; report normalizer; benchmark docs; maintainer guide; README; INSTALL. |
| Why it remains | This branch adds no selected platform/row decision, hosted workflow, artifact review, benchmark freshness tests, or methodology metadata. |
| Closure target | Add one hosted selected benchmark freshness lane for one exact platform/row pair and preserve methodology-bound, threshold-free, non-portable interpretation. |
| Expected evidence | Hosted platform pass; exact selected benchmark bundle; selected CSV matches manifest contract; methodology metadata records platform/compiler/build flags/repeat policy; docs keep non-portable wording. |
| Validation commands | `make bench-canonical-report-freshness`; `python3 tests/test_bench_canonical_freshness.py`; selected manifest tests; report normalizer tests; hosted platform workflow evidence review; `make docs-check`. |
| Claim boundary | Do not claim portable performance, timing thresholds, backend superiority, platform parity, release benchmark readiness, or state-of-the-art performance. |

## Priority 6: E18-RQ-006

| Field | Value |
| --- | --- |
| Theme | Windows QR incompatible comparison promotion. |
| Source | Epic 17 residual queue; Epic 18 Sprint 203 plan; Sprint 197 Day 3 and Day 7 claim reviews. |
| Current status | Pending future execution. |
| Owner surfaces | Windows workflow; QR incompatible comparison target; comparison runner; selected manifest; normalizer; corpus docs; maintainer guide; README; INSTALL. |
| Why it remains | The QR incompatible comparison remains local/selected evidence; this branch adds no MSVC/CMake generation proof or hosted Windows artifact review. |
| Closure target | Add MSVC/CMake proof for `qr-incompatible-ls`, fix Windows-safe generation/path handling as needed, inspect artifacts, promote exact selected metadata if evidence supports it, and retain broad QR parity non-claims. |
| Expected evidence | Windows CMake probe builds and runs; generated rows match expected QR incompatible target output; artifact paths normalize correctly; manifest metadata matches the promoted platform scope. |
| Validation commands | `python3 tests/test_run_external_comparison.py`; `python3 tests/test_normalize_report_index.py`; selected manifest tests; selected comparison workflow tests; focused QR solve tests; `make windows-powershell-guard`; hosted Windows comparison workflow evidence review. |
| Claim boundary | Do not claim broad QR least-squares parity, broad external-library parity, Windows selected oracle freshness, Windows benchmark freshness, or broad Windows report freshness. |

## Priority 7: E18-RQ-007

| Field | Value |
| --- | --- |
| Theme | Generated API publication policy. |
| Source | Epic 18 Sprint 204 plan; Sprint 197 Day 5 and Day 7 generated API audits. |
| Current status | Pending future execution; generated API remains local-only. |
| Owner surfaces | `docs/api_reference.md`; `docs/maintainer_guide.md`; `Doxyfile`; generated API ignore rules; docs/API freshness scripts; README; INSTALL; workflows if publication is added. |
| Why it remains | This branch does not decide hosted publication, retained artifact publication, committed generated output, or stronger local-only policy beyond existing guards. |
| Closure target | Make a product decision, implement matching publication or local-only guards, update routing docs, and validate freshness/link/staging behavior. |
| Expected evidence | Decision record; implemented guard/workflow/link behavior; `make api-docs-freshness` passes; generated output policy is reflected consistently in user and maintainer docs. |
| Validation commands | `make docs-check`; `make api-docs-freshness`; link/publication checks if added; workflow checks if hosted or artifact publication is added; full C gate if headers change. |
| Claim boundary | Do not claim hosted API docs, artifact-published generated HTML, committed generated HTML, ABI completeness, package support, or release evidence unless the selected policy explicitly proves it. |

## Priority 8: E18-RQ-008

| Field | Value |
| --- | --- |
| Theme | Adoption and diagnostics simplification. |
| Source | Epic 18 Sprint 205 plan; Sprint 197 Day 4 and Day 6 public-doc audit. |
| Current status | Pending future execution. |
| Owner surfaces | README; INSTALL; tutorial; cookbook; solver selection; examples; benchmark docs; API reference; maintainer guide; docs claim guards. |
| Why it remains | This branch audited public docs but did not design or implement a compact problem-shape quick reference, support truth consolidation, diagnostics vocabulary normalization, or claim guards. |
| Closure target | Add a compact problem-shape quick reference, centralize support truth, normalize diagnostics wording, and add/update guards so simplified wording does not broaden claims. |
| Expected evidence | Public doc audit; quick-reference design; edited docs; claim guard updates; docs checks; generated API checks if API routing changes. |
| Validation commands | `make docs-check`; `make api-docs-freshness` if API docs change; package/static deferral guards if install/support wording changes; `make windows-powershell-guard` if Windows wording changes; full C gate if headers change. |
| Claim boundary | Simplified wording must not imply package-manager support, broad platform parity, portable performance, release readiness, dynamic ABI support, or state-of-the-art status. |

## Long-Horizon Deferrals

| Residual ID | Theme | Closure target |
| --- | --- | --- |
| E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Define release criteria, semantic versioning policy, shared-library exports, symbol visibility, SONAME/install-name/RPATH, DLL/import-library behavior, ABI compatibility policy, package selectors, and runtime-loader validation before making public claims. |
| E18-RQ-010 | State-of-the-art evidence program | Define external baselines, versions, fixtures, matrix suites, workloads, tolerances, platforms, compilers, package provenance, reliability semantics, benchmark methodology, acceptance thresholds, and hosted evidence review before any broad state-of-the-art claim. |

## Final Claim Decision

No stronger Epic 18 public or maintainer support claim is earned by the current
branch.

The only earned claim is that the requested `SPRINT_197` final-validation path
has planning, reconciliation, claim-audit, project-plan status, validation, and
residual-handoff evidence. Package-manager support, Windows freshness,
additional reliability proof, additional review-surface reduction, benchmark
platform freshness, Windows QR comparison, generated API publication, adoption
simplification, release readiness, shared-library/dynamic ABI support, portable
performance, broad ecosystem parity, and state-of-the-art status remain
unpromoted.

## Source Evidence

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [EPIC_18_RETROSPECTIVE.md](./EPIC_18_RETROSPECTIVE.md)
- [SPRINT_197/WORKING_NOTES.md](./SPRINT_197/WORKING_NOTES.md)
- [SPRINT_197/artifacts/day2-outcome-ledger.md](./SPRINT_197/artifacts/day2-outcome-ledger.md)
- [SPRINT_197/artifacts/day3-evidence-conflicts.md](./SPRINT_197/artifacts/day3-evidence-conflicts.md)
- [SPRINT_197/artifacts/day6-public-recalibration.md](./SPRINT_197/artifacts/day6-public-recalibration.md)
- [SPRINT_197/artifacts/day7-maintainer-api-recalibration.md](./SPRINT_197/artifacts/day7-maintainer-api-recalibration.md)
- [SPRINT_197/artifacts/day8-project-plan-status.md](./SPRINT_197/artifacts/day8-project-plan-status.md)
- [SPRINT_197/artifacts/day10-focused-validation-log.md](./SPRINT_197/artifacts/day10-focused-validation-log.md)
- [SPRINT_197/artifacts/day11-full-quality-gate-log.md](./SPRINT_197/artifacts/day11-full-quality-gate-log.md)
