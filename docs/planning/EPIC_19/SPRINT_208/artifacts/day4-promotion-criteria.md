# Sprint 208 Day 4: Promotion Criteria

## Purpose

Define objective evidence thresholds for promoting selected Windows Cholesky
freshness or deliberately re-deferring it. Day 4 does not make the final
promotion decision; Day 5 applies these criteria to the Day 2 hosted evidence
and Day 3 traceability.

## Promotion Criteria

Sprint 208 may promote selected Windows Cholesky freshness only when every
criterion below is true in the same source-controlled branch.

| # | Criterion | Required evidence |
| ---: | --- | --- |
| 1 | Current hosted Windows run inspected | Latest relevant `Windows CI` run on `master` is inspected with run ID, URL, head SHA, timestamps, and conclusion recorded. |
| 2 | Exact selected job succeeded | `Windows selected Cholesky comparison freshness (MSVC)` completed with `conclusion=success`. |
| 3 | Artifact is available | `sprint190-windows-selected-comparison-cholesky` is present, unexpired, downloadable, and has artifact ID, size, expiry, and digest recorded. |
| 4 | Artifact membership is exact | Artifact contains exactly `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| 5 | Row identity is exact | `study.tsv` contains exactly the six manifest-owned `comparison_cholesky_spd_tridiag_5_*` row IDs. |
| 6 | Row status is clean | Every selected row has `status=pass`, expected status reason, current source commit, `source_branch=master`, and `worktree_state=clean`. |
| 7 | Platform and compiler are bounded | Rows record `platform=windows-amd64` and `compiler=cmake-probe:Visual Studio 17 2022:Release`. |
| 8 | Path handling is guarded | Normalizer tests prove backslash, mixed separator, absolute Windows suffix, near-match rejection, missing rows, stale rows, wrong target rows, duplicate rows, extra rows, and artifact mismatch behavior for the selected lane. |
| 9 | Manifest metadata is positive and aligned | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` lists `.github/workflows/windows-ci.yml`, `selected-comparison-freshness`, `sprint190-windows-selected-comparison-cholesky`, and `windows` in platform-aligned positions. |
| 10 | Support and claim fields agree | Manifest support tier, claim scope, and non-claims do not contradict hosted Windows selected Cholesky freshness. |
| 11 | Generated evidence wording agrees | Generated `study.tsv` rows and `summary.md` do not retain `no hosted CI proof` or `no Windows report freshness` for the promoted Windows selected Cholesky lane. |
| 12 | Public and maintainer docs agree | README, INSTALL, corpus docs, schema docs, maintainer guide, and planning docs use the same narrow promoted wording. |
| 13 | Non-claims stay explicit | Broad Windows report freshness, Windows oracle freshness, Windows benchmark freshness, QR incompatible Windows freshness, package/ABI support, performance, release, external-library parity, and state-of-the-art status remain explicit non-claims. |

## Re-Deferral Criteria

Sprint 208 must re-defer selected Windows Cholesky freshness if any promotion
criterion is unmet.

| Re-deferral trigger | Required closeout behavior |
| --- | --- |
| No current hosted run is found or inspected | Record the missing evidence as a blocker and keep manifest Windows metadata absent. |
| Selected job fails, is cancelled, skipped, or ambiguous | Record job status and keep claim as guarded workflow or re-deferred evidence only. |
| Artifact is missing, expired, not downloadable, or has wrong name | Keep manifest Windows metadata absent and record artifact blocker. |
| Artifact membership is missing, extra, or broad | Keep or add workflow/guard checks for exact six-file upload scope. |
| Row set is stale, missing, duplicated, extra, wrong-target, or failed | Keep selected freshness unpromoted and add or verify diagnostics. |
| Path matching can accept near-match or wrong artifact paths | Keep selected freshness unpromoted until normalizer coverage is strengthened. |
| Manifest metadata cannot be aligned by platform | Keep `workflow_platforms` without `windows` and strengthen future-promotion tests. |
| Generated rows still say `local_only`, `no hosted CI proof`, or `no Windows report freshness` | Re-defer unless generated support tier and non-claim wording are updated consistently. |
| Public or maintainer docs cannot be reconciled | Re-defer and record residual doc/claim boundary work. |

## Decision-To-Change Map

| Surface | Promotion path | Re-deferral path |
| --- | --- | --- |
| `selected_report_targets.tsv` | Add Windows workflow file, job, artifact, and platform to the exact Cholesky row; adjust support tier, claim scope, and non-claims only if generated evidence and docs also agree. | Keep Windows absent; ensure non-claims and introduced evidence record remain accurate. |
| Manifest contract tests | Assert exact Windows Cholesky metadata, support tier, claim scope, non-claims, required files, expected rows, and artifact alignment. | Assert Windows remains absent and future-promotion allowlist remains exact. |
| Windows workflow | Preserve exact selected job and six-file upload; change only if promotion needs metadata alignment. | Preserve bounded guarded workflow; strengthen absence or drift guards if needed. |
| PowerShell guards | Validate promoted metadata and claim boundaries if Windows becomes manifest-selected. | Continue validating bounded workflow and no selected manifest Windows platform. |
| Normalizer tests | Add any missing duplicate, extra, artifact mismatch, or support-tier/non-claim diagnostics before promotion. | Keep existing path and row diagnostics; add gaps found during review. |
| Generator/summary wording | Remove or qualify conflicting `no hosted CI proof` and `no Windows report freshness` for the exact promoted lane. | Leave generated local-only wording intact and document it as the reason for re-deferral. |
| README and INSTALL | Explain the exact promoted selected Windows Cholesky lane and retained non-claims. | Explain latest hosted evidence was reviewed but selected freshness remains re-deferred. |
| Corpus/schema docs | Update selected manifest authority and support-tier interpretation. | Retain no-Windows selected manifest wording and residual condition. |
| Maintainer guide | Add promotion runbook and validation commands. | Preserve re-deferral runbook and residual evidence requirements. |
| Planning docs | Mark Sprint 208 items closed as promotion or re-deferral with evidence links. | Mark Sprint 208 as closed re-deferral and list blockers. |

## Validation Requirements

### Promotion

Promotion requires all of:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
make windows-powershell-guard
python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
make docs-check
git diff --check
```

If `.c` or `.h` files are modified, promotion also requires:

```sh
make format && make lint && make test
```

### Re-Deferral

Re-deferral requires:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make windows-powershell-guard
git diff --check
```

If normalizer, generator, docs, or C/header files are modified, the
corresponding focused tests and gates from the promotion set become required
for re-deferral too.

## Day 4 Outcome

Day 4 completes the objective decision rules for item 208.2. Day 5 must apply
these rules. Based on Day 2-Day 3 evidence, artifact availability and path
normalization are currently favorable, while manifest metadata and generated
claim semantics remain the primary blockers to promotion.

