# Codex Gap-Closure Todo - 2026-08-28

## Purpose

This todo translates the 2026-08-28 Codex review into a step-by-step plan for
closing the highest-value gaps. It favors complete closures over partial
progress across too many surfaces.

## Closure Strategy

Epic 17 should close six concrete gap families:

1. package-manager proof completion;
2. Windows validation and report freshness;
3. one bounded external comparison plus one hosted performance/comparison
   publication lane;
4. one large source/test maintainability cluster;
5. user-facing adoption/API simplification;
6. reliability evidence for one selected allocation/failure-path owner.

Every closure should end with:

- exact owner files;
- tests or scripts that enforce the behavior;
- documentation that states the earned claim and retained non-claims;
- validation output recorded in sprint artifacts;
- project-plan status updated at closeout.

## Step-By-Step Plan

### Phase 1: Baseline And Selection

1. Create Epic 17 baseline artifacts under `docs/planning/EPIC_17/`.
2. Convert review findings into a source-controlled gap ledger.
3. Reconcile Epic 16 residuals with new review findings.
4. Select only the gaps that can be fully closed inside ten 14-day sprints.
5. Record explicit non-goals for broad state-of-the-art, broad external
   parity, shared-library ABI support, and portable performance superiority.
6. Define validation gates for documentation-only, script, workflow,
   package, report, and C/header changes.

### Phase 2: Package-Manager Proof Completion

1. Decide the standalone license metadata strategy.
2. Add the approved root license metadata file or record a formal alternate
   formula license strategy.
3. Update Homebrew formula metadata to use the selected license strategy.
4. Run the local Homebrew formula proof through render, archive/checksum,
   install, installed-file inspection, `brew test`, uninstall, and cleanup.
5. Update package-manager guards so support cannot be claimed unless the proof
   passes.
6. Update README, INSTALL, Homebrew docs, and maintainer guidance with the
   precise earned support level.
7. Preserve non-claims for Homebrew/core, bottles, Linuxbrew, public tap, and
   broad package-manager support unless they are explicitly proven.

### Phase 3: Windows Validation And Report Freshness

1. Inventory PowerShell scripts, workflow snippets, report generators, and
   selected report target metadata.
2. Add a PowerShell parse/workflow validation command that can run locally
   when `pwsh` exists and in hosted Windows CI.
3. Decide whether Windows report freshness will be promoted or explicitly
   re-deferred.
4. If promoted, select exactly one Windows-safe report freshness lane.
5. Add selected manifest metadata for the Windows lane.
6. Wire the hosted Windows workflow to regenerate only the selected artifacts.
7. Add artifact upload scope and freshness checks.
8. Update maintainer/report docs and support-tier wording.
9. Preserve non-claims for broad Windows parity, Windows Makefile parity,
   Windows `pkg-config` execution parity, package-manager support, and broad
   generated-report freshness.

### Phase 4: External Evidence Lane

1. Choose one bounded comparison or performance claim that matters to users.
2. Pick exact fixtures, dimensions, solver family, backend policy, metrics,
   tolerances, and dependency expectations.
3. Add or extend source-controlled fixtures/generators.
4. Extend comparison/report scripts and manifest rows.
5. Add tests for parser behavior, dependency deferral, status rows, tolerance
   decisions, and stale artifact handling.
6. Add a hosted freshness lane with bounded runtime and uploaded artifacts.
7. Update benchmark/report documentation so the new evidence remains scoped.
8. Preserve non-claims for broad SuiteSparse/PETSc/Trilinos/Eigen/SciPy parity
   and portable performance superiority.

### Phase 5: Maintainability Cluster

1. Rank large source and test files by review risk, ownership complexity, and
   user-facing importance.
2. Select exactly one cluster to improve.
3. Record no-behavior-change invariants and current focused tests.
4. Extract family-local helper headers or source modules only where ownership
   becomes clearer.
5. Add an ownership/registration guard similar to existing helper guards.
6. Run focused tests, source-list checks, and the full C quality gate if C or
   header files changed.
7. Update maintainer notes with the new ownership boundary.

### Phase 6: Adoption And API Simplification

1. Audit README, tutorial, cookbook, examples, solver-selection guide, and API
   reference for duplicated caveats and adoption friction.
2. Create a compact production-readiness/support matrix that separates user
   truth from sprint history.
3. Add or improve a minimal external installed-consumer tutorial.
4. Normalize diagnostics language across direct, iterative, QR/SVD, and
   eigensolver workflows.
5. Keep public headers as exact declarations and move narrative workflow
   guidance into docs where possible.
6. Validate local markdown links, Doxygen coverage, examples, install proofs,
   and claim guards.

### Phase 7: Reliability Evidence

1. Select one allocation/failure-path owner not covered by prior deterministic
   proof.
2. Map ownership, publication points, cleanup invariants, and retry semantics.
3. Extend or add deterministic failure injection.
4. Add regression tests for failed allocation, cleanup, stale-output
   suppression, and successful retry.
5. Add a focused Make/CTest target.
6. Update docs with the exact earned claim and retained non-claims.
7. Run focused validation and full C quality gates.

### Phase 8: Final Calibration And Closeout

1. Reconcile all Epic 17 project-plan items.
2. Update the public claim inventory and remove stale or unsupported wording.
3. Run focused and full validation according to changed surfaces.
4. Publish an Epic 17 retrospective.
5. Publish a residual queue for remaining state-of-the-art, ABI, platform,
   package, performance, and comparison gaps.
6. Confirm the final public statement does not overclaim.

## Definition Of Done For Any Gap

A gap is closed only when all are true:

- the selected scope is explicit;
- code or docs implement exactly that scope;
- the validation command is source-controlled and documented;
- required local or hosted checks pass;
- public docs state the earned claim narrowly;
- retained non-claims remain visible;
- residual work is either absent or explicitly moved to a future queue.

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
make api-docs-validate
make api-docs-freshness
make qr-header-docs-guard
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/homebrew_local_formula_proof.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
make report-index-oracle-freshness
make report-index-comparison-freshness
make bench-canonical-report-freshness
make performance-sentinels
make coverage
```

Run `make format && make lint && make test` whenever `.c` or `.h` files change.

## Final Non-Goals

Do not try to close these unless a sprint explicitly selects and proves them:

- unqualified state-of-the-art sparse linear algebra status;
- broad external ecosystem parity;
- portable performance superiority;
- shared-library and dynamic ABI support;
- broad Windows parity;
- broad package-manager distribution;
- hosted generated API publication beyond an explicit product decision.

