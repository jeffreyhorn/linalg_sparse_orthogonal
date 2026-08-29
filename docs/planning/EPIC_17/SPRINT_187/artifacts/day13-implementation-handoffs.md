# Sprint 187 Day 13: Implementation Handoffs

## Purpose

Package the selected Epic 17 closures into implementation-ready handoffs for
Sprints 188 through 195. Each handoff links the selected gap, acceptance gates,
owner surfaces, validation commands, retained non-goals, and remaining
pre-closeout questions.

## Handoff Index

| Future sprint | Closure target | Primary gate artifact | Supporting artifacts |
| --- | --- | --- | --- |
| Sprint 188 | Homebrew proof completion | `day7-package-acceptance-gates.md` | `day3-residual-reconciliation.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 189 | PowerShell validation ownership | `day8-windows-acceptance-gates.md` | `day4-owner-surface-inventory.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 190 | Windows selected report freshness decision | `day8-windows-acceptance-gates.md` | `day3-residual-reconciliation.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 191 | Bounded external comparison family | `day9-comparison-performance-gates.md` | `day2-review-intake-matrix.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 192 | Methodology-bound performance evidence lane | `day9-comparison-performance-gates.md` | `day2-review-intake-matrix.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 193 | Selected large review-surface reduction | `day10-maintainability-reliability-gates.md` | `day4-owner-surface-inventory.md`, `day5-gap-ranking-and-feasibility.md`, `day12-quality-surface-map.md` |
| Sprint 194 | Adoption and API coherence simplification | `day11-adoption-documentation-gates.md` | `day4-owner-surface-inventory.md`, `day6-closure-target-selection.md`, `day12-quality-surface-map.md` |
| Sprint 195 | Selected reliability and failure-path proof | `day10-maintainability-reliability-gates.md` | `day4-owner-surface-inventory.md`, `day5-gap-ranking-and-feasibility.md`, `day12-quality-surface-map.md` |

## Sprint 188 Handoff: Homebrew Proof Completion

### Goal

Close `E17-GAP-001 / R186-PKG-LICENSE` by resolving standalone license
metadata and proving the local Homebrew formula workflow without implying
Homebrew/core, bottle, Linuxbrew, public tap, shared-library, or broad
package-manager support.

### Start With

- `day7-package-acceptance-gates.md` for formula, proof-script, package-guard,
  and documentation acceptance gates.
- `day3-residual-reconciliation.md` for the inherited Epic 16 residual.
- `day6-closure-target-selection.md` for selected scope and retained
  non-goals.
- `day12-quality-surface-map.md` for package, Homebrew, docs, and C/header
  quality requirements.

### Owner Files

- Root license metadata: `LICENSE`, `COPYING`, or `NOTICE` if adopted.
- Formula template: `packaging/homebrew/sparse-lu-ortho.rb.in`.
- Proof script: `scripts/homebrew_local_formula_proof.sh`.
- Package guards: `scripts/package_manager_deferral_check.sh`,
  `scripts/static_package_deferral_check.sh`.
- Docs: `README.md`, `INSTALL.md`, `packaging/homebrew/README.md`,
  `docs/maintainer_guide.md`.
- Selected report index rows if package support wording is normalized through
  report metadata.

### First Implementation Steps

1. Decide the exact approved license metadata strategy before editing formula
   claims.
2. Make `SPARSE_HOMEBREW_LICENSE` deterministic and consistent with the root
   metadata and archive contents.
3. Harden the proof script so render, archive, checksum, install, downstream
   CMake `test do`, uninstall, and cleanup are all checked.
4. Update package guards so Homebrew wording remains tied to the proven local
   proof state.
5. Update docs with the earned local support level and retained non-claims.

### Required Validation

- `SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh`
- `scripts/package_manager_deferral_check.sh`
- `scripts/static_package_deferral_check.sh`
- `python3 scripts/normalize_report_index.py --family package --check`
- `python3 scripts/normalize_report_index.py --family package --check-freshness`
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

The repository has an auditable local Homebrew proof, package guards reject
unsupported wording, and docs describe exactly the proven static local formula
support level.

## Sprint 189 Handoff: PowerShell Validation Ownership

### Goal

Close `E17-GAP-002 / R186-WIN-PWSH` by adding an owned validation command for
PowerShell workflow material while keeping Windows report freshness as a
separate Sprint 190 decision.

### Start With

- `day8-windows-acceptance-gates.md` for PowerShell validation acceptance
  gates and retained Windows non-claims.
- `day4-owner-surface-inventory.md` for local environment notes, including
  absent local `pwsh`.
- `day12-quality-surface-map.md` for workflow, script, docs, and hosted
  evidence requirements.

### Owner Files

- `.github/workflows/windows-ci.yml`
- Selected report target manifest and schema tests when workflow names or
  artifact assumptions are referenced.
- PowerShell validation script or Make target added by the sprint.
- `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md`.

### First Implementation Steps

1. Inventory every PowerShell snippet, artifact name, selected report target,
   and Windows workflow assumption.
2. Add a validation command with local skip semantics when `pwsh` is missing
   and real execution in hosted Windows CI.
3. Wire the command into hosted Windows validation without publishing or
   promoting a selected report freshness artifact.
4. Add guard tests that fail on stale artifact names, unsupported workflow
   drift, or accidental report freshness promotion.
5. Document the command, local skip, hosted ownership, and retained non-claims.

### Required Validation

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- The new PowerShell validation command or Make target.
- Hosted Windows workflow evidence before claiming hosted validation.
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

PowerShell workflow material has an explicit owner command, hosted Windows CI
proves it, and docs continue to separate validation ownership from Windows
report freshness or broad Windows parity.

## Sprint 190 Handoff: Windows Selected Report Freshness Decision

### Goal

Close or renew `E17-GAP-003 / R186-WIN-REPORT-FRESHNESS` by either promoting
one Windows-safe selected report freshness lane or publishing a stronger
formal deferral with guard evidence.

### Start With

- `day8-windows-acceptance-gates.md` for both promotion and renewed-deferral
  acceptance criteria.
- Sprint 189's PowerShell validation result.
- `day12-quality-surface-map.md` for workflow, manifest, generated report, and
  hosted evidence requirements.

### Owner Files

- `.github/workflows/windows-ci.yml`
- `tests/corpus/manifests/selected_report_targets.tsv`
- `scripts/normalize_report_index.py`
- Report generator or freshness script selected by the sprint.
- Manifest/schema tests and selected workflow tests.
- `README.md`, `docs/maintainer_guide.md`, and report documentation.

### First Implementation Steps

1. Choose exactly one Windows-safe lane for promotion, or explicitly choose a
   renewed deferral.
2. If promoting, define generator command, artifact pattern, required files,
   expected rows, `.exe` handling, platform metadata, and upload policy.
3. If deferring, write the deferral artifact with blockers, revisit criteria,
   guard evidence, and docs updates.
4. Update manifest/schema/freshness guards so unsupported Windows report
   claims fail review.
5. Calibrate docs to the chosen outcome.

### Required Validation

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- Selected workflow/freshness tests added by Sprint 190.
- `python3 scripts/normalize_report_index.py --check-freshness` for the
  selected scope when promotion is chosen.
- Hosted Windows workflow evidence before claiming Windows report freshness.
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

There is no ambiguous middle state: one Windows-safe freshness lane is proven
and documented, or the formal deferral is renewed with stronger guards and
explicit revisit criteria.

## Sprint 191 Handoff: Bounded External Comparison Family

### Goal

Close `E17-GAP-004 / R186-BROAD-COMPARISON` by adding exactly one bounded
external comparison family with source-controlled fixtures, manifest metadata,
freshness checks, and claim-safe documentation.

### Start With

- `day9-comparison-performance-gates.md` for comparison-family acceptance
  fields and validation commands.
- `day2-review-intake-matrix.md` for comparison credibility and parity risks.
- `day6-closure-target-selection.md` for selected closure boundaries.
- `day12-quality-surface-map.md` for report, script, workflow, and C/header
  quality requirements.

### Owner Files

- `scripts/run_external_comparison.py`
- `tests/corpus/manifests/selected_report_targets.tsv`
- `scripts/normalize_report_index.py`
- `tests/test_selected_report_targets_manifest.py`
- `tests/test_selected_comparison_workflow.py`
- Selected fixture/generator files.
- `README.md`, `docs/maintainer_guide.md`, solver docs, and report docs.

### First Implementation Steps

1. Select one solver family, fixture, reference implementation, dependency
   policy, metric set, tolerance set, and support tier.
2. Add fixture or generator material in source control with deterministic
   row identities.
3. Extend runner output and manifest metadata for dependency status, study
   rows, expected rows, artifact patterns, owner, and introduced version.
4. Add tests for parser behavior, dependency skips, tolerance failures,
   stale-output rejection, and row normalization.
5. Update docs with exact comparison meaning and retained broad-parity
   non-claims.

### Required Validation

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `make report-index-comparison-freshness`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

One new comparison family is fresh, reproducible, bounded, tested, and
documented without implying broad external solver parity.

## Sprint 192 Handoff: Methodology-Bound Performance Evidence Lane

### Goal

Close the selected performance-evidence gap by promoting one benchmark lane
to methodology-bound hosted evidence with explicit metadata and no portable
performance superiority claim.

### Start With

- `day9-comparison-performance-gates.md` for performance-lane acceptance
  fields.
- `day2-review-intake-matrix.md` for performance claim risk.
- `day12-quality-surface-map.md` for benchmark, report, workflow, and hosted
  evidence requirements.

### Owner Files

- `benchmarks/**`
- `scripts/bench_canonical_report.sh`
- `scripts/check_bench_canonical_freshness.py`
- `tests/corpus/manifests/selected_report_targets.tsv`
- `.github/workflows/ci.yml`
- Benchmark docs and report index documentation.

### First Implementation Steps

1. Confirm `bench_refactor_csc` on `nos4.mtx --repeat 1` or select a stronger
   single lane with written rationale.
2. Add required metadata for platform, compiler, runner context, build flags,
   CPU, build mode, threads, artifact path, command, repeat semantics, and
   methodology notes.
3. Add hosted freshness with artifact upload and bounded runtime.
4. Decide whether the lane remains threshold-free or receives one conservative
   regression sentinel.
5. Update report normalization and docs with `not_portable_performance_claim`.

### Required Validation

- `python3 scripts/validate_corpus_schema.py`
- `make bench-canonical-report-freshness`
- `python3 scripts/check_bench_canonical_freshness.py --mode local`
- `python3 scripts/check_bench_canonical_freshness.py --mode hosted`
- `python3 scripts/normalize_report_index.py --family benchmark --check-freshness`
- `python3 tests/test_selected_comparison_workflow.py`
- Hosted workflow artifact evidence before claiming hosted performance
  evidence.
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

One hosted benchmark lane has complete methodology metadata, fresh artifacts,
and calibrated claim language that avoids portable performance or
state-of-the-art assertions.

## Sprint 193 Handoff: Selected Large Review-Surface Reduction

### Goal

Close `E17-GAP-006 / R186-REVIEW-SURFACE-NEXT` by reducing exactly one
high-risk source or test review surface while preserving behavior and full
validation.

### Start With

- `day10-maintainability-reliability-gates.md` for candidate sizes,
  no-behavior-change invariants, and C quality gates.
- `day5-gap-ranking-and-feasibility.md` for feasibility ranking.
- `day12-quality-surface-map.md` for C implementation, test, source-list, and
  maintainer-doc validation.

### Owner Files

- Selected from the ranked large surfaces, including candidates such as
  `tests/test_qr.c`, `tests/test_ldlt_csc.c`, `tests/test_integration.c`,
  `tests/test_svd.c`, `tests/test_ldlt.c`, `tests/test_etree.c`,
  `tests/test_iterative.c`, `tests/test_graph.c`,
  `src/sparse_ldlt_csc.c`, or `src/sparse_lu_csr.c`.
- Related helper headers/sources, test registration, `Makefile`, and
  `CMakeLists.txt` if source membership changes.
- `docs/maintainer_guide.md` for boundary documentation.

### First Implementation Steps

1. Re-rank the candidate cluster against current branch sizes and risk.
2. Select one cluster and record exact behavior-preservation invariants.
3. Design helper/source boundaries, cleanup ownership, status precedence,
   diagnostics preservation, and global-state restoration behavior.
4. Extract only the selected cluster and keep test registration/source-list
   parity intact.
5. Add or update a focused ownership guard for the new boundary.

### Required Validation

- Focused test binary or CTest target for the selected owner.
- `make source-list-check`
- `make quality-review-cmake-compile` if source lists or CMake membership
  change.
- `make ldlt-csc-helper-guard` or `make large-matrix-guardrails` when the
  selected cluster touches those owners.
- Header documentation guards if public headers change.
- `make format && make lint && make test` for any `.c` or `.h` change.

### Done State

One large review surface is smaller or clearer, the new boundary is guarded,
and behavior is preserved by focused and full C validation.

## Sprint 194 Handoff: Adoption And API Coherence Simplification

### Goal

Close the selected adoption/API coherence gap by consolidating user-facing
support truth, reducing workflow duplication, and cleaning selected public
header narrative without changing solver behavior.

### Start With

- `day11-adoption-documentation-gates.md` for adoption, support-matrix,
  installed-consumer, diagnostics, and header documentation gates.
- Completed outcomes from Sprints 188, 190, 191, and 192.
- `day12-quality-surface-map.md` for docs, header, examples, install, and
  generated API checks.

### Owner Files

- `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`,
  `docs/solver_selection.md`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`.
- `examples/README.md` and selected example sources.
- Selected public headers under `include/`.
- Local generated API docs only as freshness output, not as hosted
  publication.

### First Implementation Steps

1. Audit user-facing docs for duplicated support truth and stale workflow
   guidance.
2. Add a compact support/readiness matrix covering source build, installed
   static package, package manager, Windows, macOS/Linux, generated reports,
   external comparison, performance, API docs, and reliability.
3. Add or improve the installed consumer tutorial for Make/pkg-config and
   CMake `find_package` paths.
4. Normalize diagnostics wording for status codes, residuals, convergence,
   retry, cleanup, and unsupported breadth.
5. Move workflow narrative out of selected headers while preserving public
   declarations and Doxygen coverage.

### Required Validation

- `git diff --check`
- Markdown link check for changed docs.
- `make docs-check`
- `make api-docs-freshness`
- Header documentation guards when headers change.
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make examples`
- `make format && make lint && make test` if any `.c` or `.h` files change.

### Done State

Adoption docs present one coherent support truth, installed-consumer workflows
are easier to follow, diagnostics wording is consistent, and header narrative
is reduced without API behavior changes.

## Sprint 195 Handoff: Selected Reliability And Failure-Path Proof

### Goal

Close the selected reliability proof gap by adding deterministic failure-path
evidence for one allocation-heavy or failure-prone owner beyond existing
iterative and matrix-multiply proof lanes.

### Start With

- `day10-maintainability-reliability-gates.md` for reliability-owner criteria,
  failure-path invariants, and global-state restoration rules.
- `day5-gap-ranking-and-feasibility.md` for reliability feasibility.
- Sprint 193's selected review-surface result to avoid overlapping owner
  churn.
- `day12-quality-surface-map.md` for allocation-failure, C/test, docs, and
  focused-gate checks.

### Owner Files

- Selected owner source/test files under `src/` and `tests/`.
- Existing proof models: iterative allocation failure gate, matrix-multiply
  allocation failure gate, and public lifecycle retry tests.
- `Makefile`, `CMakeLists.txt`, and test registration if adding a new focused
  binary or source.
- `README.md` and `docs/maintainer_guide.md` for exact reliability claim
  wording.

### First Implementation Steps

1. Select one owner by allocation density, cleanup complexity, user impact,
   and deterministic testability.
2. Record cleanup invariants, publication points, retry-after-reset semantics,
   stale-output suppression expectations, and unsupported breadth.
3. Add or extend deterministic failure injection for that owner only.
4. Add tests for failed allocation, cleanup, stale-output suppression, reset,
   and successful retry.
5. Add a focused gate and documentation that states the bounded reliability
   proof without generalizing to all failure paths.

### Required Validation

- New focused reliability gate or selected existing focused gate.
- Test registration guard if a new test target is added.
- `make source-list-check` and CMake parity checks if source lists change.
- `make format && make lint && make test` for `.c` or `.h` changes.
- Docs checks for updated reliability wording.

### Done State

One new owner has deterministic failure-path proof, reset and cleanup behavior
is guarded, global state is restored before assertion early returns, and docs
state the exact reliability evidence earned.

## Cross-Sprint Ordering Notes

| Dependency | Ordering rule |
| --- | --- |
| Sprint 188 before Sprint 194 | Adoption docs should consume the final package support wording instead of predicting it. |
| Sprint 189 before Sprint 190 | Report freshness promotion or deferral should rely on the owned PowerShell validation result. |
| Sprint 190 before Sprint 194 | Support/readiness matrix should reflect the actual Windows report freshness decision. |
| Sprint 191 before Sprint 192 | Comparison work must not expand performance claims; performance lane wording should stay separate. |
| Sprint 193 before Sprint 195 | Reliability owner selection should avoid the same files if Sprint 193 performs large helper extraction. |

## Pre-Closeout Open Questions

| Sprint | Question | Closeout disposition needed |
| --- | --- | --- |
| 188 | What exact license identifier and root metadata file are approved for Homebrew formula rendering? | Must be answered before formula support can be claimed. |
| 189 | What command name owns PowerShell validation, and where should local `pwsh` skip semantics be documented? | Must be explicit before hosted validation wiring is reviewable. |
| 190 | Which exact Windows-safe report lane should be promoted, or should the sprint renew deferral instead? | Must be decided at Sprint 190 start after Sprint 189 evidence. |
| 191 | Which bounded comparison family, fixture, dependency, and tolerance set are selected? | Must be narrowed before runner or manifest edits begin. |
| 192 | Should the lane remain `bench_refactor_csc` on `nos4.mtx --repeat 1`, and should it stay threshold-free? | Must be confirmed before hosted evidence is promoted. |
| 193 | Which large source/test cluster is selected, and what exact invariants preserve behavior? | Must be recorded before extraction. |
| 194 | Where will the compact support/readiness matrix live, and which headers are eligible for narrative cleanup? | Must be set after Sprints 188, 190, 191, and 192 land. |
| 195 | Which reliability owner is selected, and does it overlap with Sprint 193 changes? | Must be resolved before failure-injection work starts. |
| 196 | Which earned claims survive final validation, and which remain explicit non-claims? | Must be based on actual Sprint 188-195 outcomes. |

## Day 13 Validation Scope

Day 13 changes are planning documentation only. No C or public header files
should be modified by this handoff package. Required validation is limited to
documentation hygiene and link/path sanity checks unless later edits expand
the changed-file surface.
