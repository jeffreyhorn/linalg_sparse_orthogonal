# Codex Gap-Closure Todo - 2026-09-04

## Purpose

This todo translates the 2026-09-04 Codex review into a step-by-step plan for
closing the highest-value remaining gaps. It favors complete closures over
partial progress across many surfaces.

This file lives alongside the corresponding Epic 18 review under
`docs/planning/EPIC_18/reviews/`.

## Closure Strategy

Epic 18 should close eight concrete gap families:

1. package-manager/Homebrew proof completion;
2. selected Windows Cholesky freshness promotion or formal re-deferral;
3. one additional selected allocation-failure proof owner;
4. one additional large QR/direct-solver review-surface reduction;
5. one additional hosted selected benchmark freshness platform;
6. Windows QR incompatible comparison promotion if MSVC evidence supports it;
7. generated API HTML publication decision;
8. final support, release, and state-of-the-art claim calibration.

Every closure should end with:

- exact owner files;
- source-controlled validation commands;
- local or hosted evidence, explicitly labeled;
- public documentation that states only the earned support level;
- maintainer documentation that records proof ownership and residual
  boundaries;
- project-plan status and retrospective updates.

## Step-By-Step Plan

### Phase 1: Baseline And Selection

1. Create Epic 18 planning artifacts under `docs/planning/EPIC_18/`.
2. Convert the 2026-09-04 review findings and Epic 17 residual queue into an
   Epic 18 gap ledger.
3. Deduplicate residuals that point to the same owner or claim boundary.
4. Select the gaps that can be fully closed within ten 14-day sprints.
5. Record explicit non-goals for broad state-of-the-art status, broad
   external-library parity, portable performance, shared-library ABI support,
   release readiness, and broad Windows parity.
6. Define validation gates for docs-only, package, workflow, report,
   benchmark, API publication, and C/header changes.

### Phase 2: Package-Manager/Homebrew Closure

1. Resolve approved standalone root license metadata.
2. Select the exact Homebrew formula license identifier.
3. Update `packaging/homebrew/sparse-lu-ortho.rb.in` and Homebrew docs.
4. Run `scripts/homebrew_local_formula_proof.sh` through archive/checksum,
   render, install, `brew test`, uninstall, and cleanup.
5. Update `scripts/package_manager_deferral_check.sh` and
   `scripts/static_package_deferral_check.sh` so support cannot be claimed
   without the proof.
6. Recalibrate README, INSTALL, package docs, and maintainer guidance to the
   exact earned support level.
7. Preserve non-claims for Homebrew/core, bottles, Linuxbrew, public taps,
   binary packages, and other package managers unless explicitly proven.

### Phase 3: Selected Windows Cholesky Freshness Promotion

1. Inspect hosted Windows workflow evidence for the selected Cholesky lane.
2. Verify artifact contents, selected row IDs, target key, support tier,
   workflow platforms, and freshness metadata.
3. Promote selected target manifest metadata only if hosted evidence supports
   the exact claim.
4. Add or update normalizer tests for Windows artifact paths and selected
   target filtering.
5. Update README, INSTALL, corpus docs, and maintainer guide to distinguish
   promoted selected freshness from broad Windows report freshness.
6. Rerun selected manifest, workflow, PowerShell, normalizer, and freshness
   gates.

### Phase 4: Additional Allocation-Failure Owner

1. Rank candidate owners by allocation complexity, user impact, and existing
   harness reachability.
2. Select exactly one owner, such as `sparse_symbolic_lu()`,
   `sparse_analyze()`, or a direct-solver publication path.
3. Record cleanup, stale-output, caller-owned input, publication, and retry
   invariants before code changes.
4. Extend deterministic failure injection only as needed for the selected
   owner.
5. Add regression tests for failed allocation, cleanup, stale-output
   suppression, and retry-after-reset behavior.
6. Add focused Make/CTest labels and registration guards.
7. Update README, INSTALL, maintainer docs, and claim boundaries.
8. Run focused validation plus `make format && make lint && make test` if
   `.c` or `.h` files change.

### Phase 5: Additional Review-Surface Reduction

1. Re-rank large source and test files after Epic 17.
2. Select one cluster whose extraction can be behavior-preserving.
3. Capture current invariants and focused tests before moving code.
4. Extract helpers or source modules only where reviewability improves.
5. Add ownership/registration guards so the extracted surface does not drift.
6. Run source-list, focused tests, CMake parity if registration changes, and
   the full C gate if C/header files change.
7. Document the reduced surface and retained non-claims.

### Phase 6: Hosted Selected Benchmark Freshness

1. Select exactly one additional hosted platform or benchmark row.
2. Define artifact scope, row identity, methodology metadata, and expected
   freshness behavior.
3. Add workflow, manifest, and checker changes for the selected lane.
4. Add tests for missing, stale, duplicate, deferred, and path-normalization
   cases.
5. Update benchmark docs and support/readiness wording without claiming
   portable performance or timing thresholds.
6. Inspect hosted evidence before promotion.

### Phase 7: Windows QR Incompatible Comparison

1. Run the QR incompatible comparison generator through MSVC/CMake.
2. Fix only Windows path, generator, or CMake issues needed for the selected
   target.
3. Add tests for Windows artifact path normalization and selected row
   filtering.
4. Promote manifest metadata only if hosted evidence proves the exact target.
5. Update docs with selected Windows QR incompatible freshness boundaries.
6. Preserve non-claims for broad QR least-squares parity and broad Windows
   report freshness.

### Phase 8: Generated API Publication Decision

1. Decide whether generated API HTML remains local-only or gains hosted
   publication.
2. If local-only, strengthen freshness and staging guards and simplify public
   routing.
3. If hosted, add workflow generation, artifact or Pages publication,
   freshness checks, link validation, and publication semantics.
4. Update `docs/api_reference.md`, INSTALL, maintainer guide, and
   support/readiness matrix.
5. Preserve non-claims for ABI compatibility, package-manager distribution,
   and completeness beyond configured headers.

### Phase 9: Final Calibration And Closeout

1. Reconcile all Epic 18 project-plan items.
2. Update public and maintainer claim surfaces from the final evidence.
3. Run focused gates and full quality gates according to changed surfaces.
4. Publish an Epic 18 retrospective.
5. Publish or update a residual queue for any remaining broad package,
   platform, ABI, performance, external-parity, release, reliability, or
   state-of-the-art gaps.
6. Confirm final docs do not overclaim.

## Definition Of Done For Any Gap

A gap is closed only when all are true:

- selected scope is explicit;
- owner files and evidence commands are source-controlled;
- local and hosted evidence are labeled separately;
- tests or guards fail on stale, missing, or overbroad evidence;
- docs state only the earned support level;
- retained non-claims remain visible;
- residuals have exact closure conditions or are intentionally removed.

## Candidate Validation Commands

Use these commands where relevant:

```sh
git diff --check
make format-check
make source-list-check
make lint
make test
make quality-review
make quality-review-cmake
make quality-review-full
make docs-check
make api-docs-freshness
make qr-header-docs-guard
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/homebrew_local_formula_proof.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_validate_windows_powershell.py
make windows-powershell-guard
make report-index-oracle-freshness
make report-index-comparison-freshness
make bench-canonical-report-freshness
make symbolic-allocation-failure-gate
make qr-external-ref-helper-guard
make performance-sentinels
```

Run `make format && make lint && make test` whenever `.c` or `.h` files
change.

## Final Non-Goals

Do not claim these unless a sprint explicitly selects and proves them:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity;
- portable performance superiority;
- shared-library and dynamic ABI support;
- release readiness;
- broad Windows parity;
- broad package-manager distribution;
- hosted generated API publication beyond an explicit decision;
- broad allocation-failure coverage.
