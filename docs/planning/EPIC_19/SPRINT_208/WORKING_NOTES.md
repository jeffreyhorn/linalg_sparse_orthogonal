# Sprint 208 Working Notes

## Sprint Goal

Fully promote or deliberately re-defer the selected Windows Cholesky freshness
lane using hosted evidence, manifest metadata, documentation, and guards.

## Current Branch

- Branch: `sprint-208`
- Plan: `docs/planning/EPIC_19/SPRINT_208/PLAN.md`
- Epic source: `docs/planning/EPIC_19/PROJECT_PLAN.md`, Sprint 208

## Item Checklist

| Item | Name | Final disposition | Primary surfaces | Evidence |
| --- | --- | --- | --- | --- |
| 208.1 | Hosted Artifact Intake | Complete for Day 2: latest hosted Windows run `35731703320` and artifact `10696020870` inspected. | GitHub Actions Windows run, artifact `sprint190-windows-selected-comparison-cholesky`, generated comparison files, row IDs, workflow logs | Hosted artifact ledger with run IDs, artifact membership, paths, row IDs, commit SHAs, and blocker classification |
| 208.2 | Manifest Promotion Decision | Complete: continued re-deferral selected on Day 5 because generated support tier and non-claim wording still contradict promotion. | `tests/corpus/manifests/selected_report_targets.tsv`, `tests/corpus/schemas/report_index_fields.md`, Sprint 199 decision artifacts | Re-deferral decision with exact manifest/support-tier/non-claim rationale |
| 208.3 | Metadata And Guard Implementation | Complete for the re-deferral path: manifest and PowerShell guards enforce current Linux/macOS-only selected metadata and exact future-promotion prerequisites. | selected target manifest, `.github/workflows/windows-ci.yml`, `scripts/validate_windows_powershell.py`, selected workflow tests | Strengthened absence guard and future-promotion contract |
| 208.4 | Normalizer Regression Coverage | Complete: Cholesky-specific Windows-path duplicate and unexpected-row regressions were added. | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, selected report target tests | Windows path, selected filtering, duplicate-row, unexpected-row, stale-row, and artifact mismatch regression evidence |
| 208.5 | Documentation Calibration | Complete: public, maintainer, corpus, schema, and planning docs align with reviewed bounded workflow evidence and retained re-deferral. | README, INSTALL, `tests/corpus/README.md`, `docs/maintainer_guide.md`, schema docs, planning docs | Claim-safe re-deferred wording |
| 208.6 | Validation And Closeout | Complete: integrated validation passed and closeout records final residuals. | manifest tests, workflow tests, PowerShell guard, normalizer tests, freshness command, docs checks, C gate if needed | Integrated validation artifact and final closeout ledger |

## Day 1 Evidence Map

| Evidence source | Day 1 interpretation |
| --- | --- |
| `docs/planning/EPIC_18/SPRINT_199/RETROSPECTIVE.md` | Sprint 199 reviewed hosted Windows evidence for exact `cholesky-spd-tridiag-5` but re-deferred selected Windows freshness promotion. |
| `docs/planning/EPIC_18/SPRINT_199/artifacts/day2-hosted-artifact-inventory.md` | Prior hosted run `34269219871` on `master` produced artifact `sprint190-windows-selected-comparison-cholesky` with six expected files and six passing Cholesky rows. This is inherited evidence, not current Sprint 208 proof. |
| `docs/planning/EPIC_18/SPRINT_199/artifacts/day4-manifest-decision.md` | Manifest promotion was deliberately re-deferred because Windows path normalization, generated `support_tier=local_only`, generated non-claims, and diagnostics had to move together before promotion. |
| `docs/planning/EPIC_18/SPRINT_199/artifacts/day14-closeout-review.md` | Final Sprint 199 disposition retained guarded workflow evidence and non-claims for promoted Windows selected freshness, broad Windows report freshness, package/ABI support, performance, release, and state-of-the-art status. |
| `docs/planning/EPIC_18/SPRINT_206/artifacts/day2-outcome-reconciliation.md` | Epic 18 closeout kept Sprint 199 as a closed re-deferral with guarded workflow evidence only. |
| `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md` | Epic 19 closure track requires fetching latest hosted Windows artifacts, verifying target identity, artifact membership, paths, support tier, manifest contract, and either promotion or explicit re-deferral with stronger absence proof. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Current `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` row lists Linux and macOS workflow metadata only, remains `support_tier=local_only`, and retains `no Windows report freshness`. |
| `.github/workflows/windows-ci.yml` | Current Windows workflow contains a bounded `selected-comparison-freshness` job for `cholesky-spd-tridiag-5` and uploads only the six selected Cholesky files. |
| `scripts/validate_windows_powershell.py` | Current PowerShell validation owns the bounded selected Cholesky workflow tokens, artifact name, upload files, and broad Windows non-claim markers. |
| `tests/test_selected_report_targets_manifest.py` | Current tests enforce no Windows selected manifest platform while re-deferred and include a future allowlist for exact Cholesky metadata if promotion is earned. |
| `tests/test_selected_comparison_workflow.py` | Current tests guard the Windows workflow as bounded Cholesky-only evidence and reject broad selected report freshness lanes. |
| `tests/test_normalize_report_index.py` | Current normalizer regressions include Windows artifact path handling, stale selected rows, and selected target mismatch diagnostics for Cholesky and related Windows paths. |
| README, INSTALL, corpus README, maintainer guide | Current docs distinguish guarded Windows Cholesky workflow evidence from promoted selected Windows freshness. |

## Current Windows Cholesky Claim Boundary

Earned or inherited evidence currently covers only:

- one bounded Windows workflow path for `cholesky-spd-tridiag-5`;
- target-specific generation via `scripts/run_external_comparison.py
  --target cholesky-spd-tridiag-5`;
- target-specific freshness checking via `scripts/normalize_report_index.py
  --family comparison --require-generated comparison --check-freshness
  --selected-target cholesky-spd-tridiag-5`;
- upload of the six selected Cholesky files under
  `build/comparison/cholesky_spd_tridiag_5/`;
- prior Sprint 199 hosted evidence review for run `34269219871` and artifact
  `sprint190-windows-selected-comparison-cholesky`;
- source-controlled guards that keep the Windows lane bounded while selected
  manifest metadata remains Linux/macOS only.

The following remain explicit non-claims until Sprint 208 records stronger
evidence and updates all required surfaces together:

- promoted selected Windows Cholesky freshness;
- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- QR incompatible Windows comparison freshness;
- unselected Windows comparison families;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support or package-manager platform parity;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows parity;
- portable performance claims;
- release readiness;
- external-library ecosystem parity;
- state-of-the-art status.

## Initial Surface Inventory

| Surface | Day 1 role |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Positive selected-target authority. Current Cholesky row omits `windows`; Sprint 208 may promote or preserve this absence. |
| `tests/corpus/schemas/report_index_fields.md` | Schema and support-tier contract explaining why the bounded Windows path is not yet manifest promotion. |
| `.github/workflows/windows-ci.yml` | Hosted Windows selected Cholesky job owner and artifact upload scope. |
| `scripts/validate_windows_powershell.py` | Workflow and claim-boundary validator for Windows PowerShell material. |
| `scripts/normalize_report_index.py` | Freshness and selected target filtering implementation. |
| `scripts/run_external_comparison.py` | Selected Cholesky comparison generator and row source. |
| `tests/test_selected_report_targets_manifest.py` | Manifest absence and future-promotion contract coverage. |
| `tests/test_selected_comparison_workflow.py` | Workflow upload scope, artifact, target, and Windows non-claim guard coverage. |
| `tests/test_normalize_report_index.py` | Windows path and selected freshness diagnostic regression coverage. |
| `tests/test_run_external_comparison.py` | External comparison generator behavior and selected target row emission coverage. |
| README | User-facing selected report freshness wording. |
| INSTALL | Support/readiness matrix and Windows selected Cholesky status wording. |
| `tests/corpus/README.md` | Corpus manifest and selected comparison interpretation. |
| `docs/maintainer_guide.md` | Maintainer validation, guard ownership, and claim-boundary runbook. |
| Epic 18 Sprint 199 artifacts | Historical evidence and re-deferral rationale. |
| Epic 19 project plan and reviews | Current sprint objective and closeout track. |

## Initial Validation Matrix

| Validation | Purpose | Day 1 status |
| --- | --- | --- |
| `gh run list` / `gh run view` / `gh run download` for Windows selected Cholesky evidence | Fetch and inspect current hosted run and artifact bundle. | Candidate Day 2 command; not run on Day 1 intake. |
| `python3 tests/test_selected_report_targets_manifest.py` | Enforce selected manifest contract, current Windows absence, and future exact Cholesky allowlist. | Candidate Day 7, Day 13, and Day 14 command. |
| `python3 tests/test_selected_comparison_workflow.py` | Enforce bounded Windows workflow and selected Cholesky artifact scope. | Candidate Day 8, Day 13, and Day 14 command. |
| `python3 tests/test_normalize_report_index.py` | Validate selected freshness filtering, Windows paths, stale/missing/wrong rows, and diagnostics. | Candidate Day 10, Day 13, and Day 14 command. |
| `python3 tests/test_run_external_comparison.py` | Validate selected comparison generator behavior if target or row semantics change. | Candidate Day 10 or Day 13 command. |
| `make windows-powershell-guard` | Run source-controlled Windows workflow and PowerShell guard tests. | Candidate Day 8, Day 13, and Day 14 command. |
| `make windows-powershell-validate` | Locally validate PowerShell if `pwsh` is available; missing `pwsh` remains environment residual. | Candidate command; classify exit `2` carefully if local `pwsh` is unavailable. |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Prove current generated selected Cholesky rows are fresh to local `HEAD`. | Candidate Day 13 command after generation or fixture updates. |
| `make docs-check` | Validate docs and generated API coverage when public or maintainer docs change. | Candidate Day 13 command. |
| `make format && make lint && make test` | Full C quality gate. | Required only if Sprint 208 modifies `.c` or `.h` files. |
| `git diff --check` | Whitespace validation for all changed files. | Required before closeout. |

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Prior Sprint 199 hosted evidence is mistaken for current Sprint 208 promotion proof. | Manifest or docs could overstate freshness based on stale external evidence. | Day 2 must fetch latest hosted evidence or explicitly record blocker status. |
| Manifest adds `windows` without matching support tier, generated non-claims, workflow metadata, and docs. | Source-controlled claim surfaces become contradictory. | Day 4 criteria require all promotion surfaces to move together. |
| Hosted artifact names or paths are correct but row metadata still says `local_only` or `no Windows report freshness`. | A promoted support claim would conflict with generated evidence. | Day 3-Day 5 must classify support-tier and generated wording before implementation. |
| Windows path normalization accepts near-match or wrong-target artifacts. | Wrong evidence could satisfy selected Cholesky freshness. | Reuse and extend Sprint 199 normalizer regressions before any promotion. |
| Workflow upload broadens from exact selected files to broad comparison paths. | Artifact may include unreviewed rows and imply broader Windows report freshness. | Keep workflow and PowerShell guards in Day 8 validation. |
| Public docs compress guarded workflow evidence into broad Windows support. | Users may infer Windows parity, package support, or performance claims. | Maintain explicit non-goals and require docs calibration after the decision. |
| Local missing PowerShell is treated as failure or pass evidence incorrectly. | Validation status becomes misleading. | Preserve Sprint 199 distinction: hosted `--require-pwsh` owns parseability; local exit `2` is unavailable evidence. |

## Open Questions

1. What is the latest successful hosted Windows run that contains the selected
   Cholesky job after PR #230 landed?
2. Is the hosted artifact still available and does it contain exactly the six
   expected selected Cholesky files?
3. Do generated rows still contain `support_tier=local_only` and `no Windows
   report freshness`, or has generator wording changed since Sprint 199?
4. Should promotion require changing generated row support tier, manifest
   support tier, and docs together, or should Sprint 208 close as a stronger
   re-deferral if those semantics remain local-only?
5. Are current Windows path normalization regressions sufficient for manifest
   promotion, or do Day 9-Day 10 need additional stale/missing/artifact
   mismatch coverage?
6. Which exact docs become authoritative if Sprint 208 promotes the selected
   Windows Cholesky lane?
7. Which residual queue should receive any unearned Windows parity, package,
   ABI, performance, release, or state-of-the-art follow-up?

## Day 2 Hosted Artifact Inventory

| Field | Value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `35731703320` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35731703320` |
| Event | `push` |
| Display title | `Merge pull request #230 from jeffreyhorn/sprint-207` |
| Branch | `master` |
| Head SHA | `d75118349269c6805654070eb44c9c57608b3e47` |
| Created | `2026-09-22T13:10:04Z` |
| Updated | `2026-09-22T13:12:23Z` |
| Conclusion | `success` |

### Day 2 Job Inventory

| Job | Started | Completed | Conclusion |
| --- | --- | --- | --- |
| `Windows PowerShell validation ownership` | `2026-09-22T13:10:09Z` | `2026-09-22T13:10:20Z` | `success` |
| `Windows selected Cholesky comparison freshness (MSVC)` | `2026-09-22T13:10:09Z` | `2026-09-22T13:10:41Z` | `success` |
| `Windows reviewed CMake install/downstream validation path` | `2026-09-22T13:10:10Z` | `2026-09-22T13:11:36Z` | `success` |
| `Windows enforced reviewed CMake consumer subset (MSVC)` | `2026-09-22T13:10:09Z` | `2026-09-22T13:12:22Z` | `success` |

### Day 2 Artifact Metadata

| Field | Value |
| --- | --- |
| Artifact ID | `10696020870` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Size | `4598` bytes |
| Created | `2026-09-22T13:10:38Z` |
| Updated | `2026-09-22T13:10:38Z` |
| Expires | `2026-12-21T13:10:04Z` |
| Expired | `false` |
| Upload digest | `fab335a3b95e958111d500e74ce52b5630c85e8d3ae29b8967eeb75b73f72dda` |
| Local inspection directory | `/tmp/sprint208-day2-windows-artifact-check.compx4` |

### Day 2 Artifact File Set

| File | Day 2 status |
| --- | --- |
| `project_observations.tsv` | Present. |
| `baseline_observations.tsv` | Present. |
| `dependency_status.tsv` | Present. |
| `study.tsv` | Present. |
| `summary.md` | Present. |
| `manifest.tsv` | Present. |

### Day 2 Row Inventory

`study.tsv` contains six rows. All expected manifest row IDs are present in
the expected order, no unexpected row IDs were observed, and every row has
`status=pass`.

| Row ID | Metric | Status | Status reason |
| --- | --- | --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `project_status` | `pass` | `project_status_match` |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `baseline_status` | `pass` | `baseline_status_success` |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `residual_norm` | `pass` | `project_baseline_residual_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `solution_norm` | `pass` | `project_baseline_solution_norm_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `solution_values` | `pass` | `project_baseline_solution_values_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `pass` | `project_baseline_max_abs_delta_within_tolerance` |

### Day 2 Provenance And Metadata

| Field | Value |
| --- | --- |
| `platform` | `windows-amd64` |
| `compiler` | `cmake-probe:Visual Studio 17 2022:Release` |
| `source_commit` | `d75118349269c6805654070eb44c9c57608b3e47` |
| `source_branch` | `master` |
| `worktree_state` | `clean` |
| `generated_at_utc` | `2026-09-22T13:10:35+00:00` |
| `artifact_path` | `build\comparison\cholesky_spd_tridiag_5\study.tsv` |
| `support_tier` | `local_only` |
| `claim_scope` | `fixture-local Cholesky SPD tridiagonal solve comparison only` |

### Day 2 Manifest Comparison

| Check | Day 2 result |
| --- | --- |
| Manifest target ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Manifest target key | `cholesky-spd-tridiag-5` |
| Manifest workflow platforms | `linux;macos` |
| Manifest support tier | `local_only` |
| Manifest artifact pattern | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Artifact row count vs manifest expected rows | `6` of `6` |
| Artifact row IDs vs manifest expected row IDs | Match. |
| Artifact statuses | All `pass`. |
| Artifact platform | `windows-amd64`. |
| Artifact support tiers | `local_only`. |
| Artifact paths | Windows backslash path: `build\comparison\cholesky_spd_tridiag_5\study.tsv`. |
| Generated non-claims | Still include `no hosted CI proof` and `no Windows report freshness`. |

### Day 2 Blocker Classification

The latest hosted artifact is credible current evidence for the exact Windows
selected Cholesky workflow path at commit
`d75118349269c6805654070eb44c9c57608b3e47`. It is not sufficient by itself to
promote selected Windows Cholesky freshness because:

1. The selected manifest still lists only `linux;macos` for
   `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
2. Generated rows still carry `support_tier=local_only`.
3. Generated row and summary non-claims still include `no hosted CI proof` and
   `no Windows report freshness`.
4. Generated artifact paths use Windows backslashes while the manifest
   artifact pattern uses forward slashes.

Day 3 should trace these rows and paths across normalizer behavior and define
which of these blockers are semantic promotion blockers versus already guarded
path-format details.

## Day 3 Row And Path Traceability

| Trace surface | Current value | Day 3 classification |
| --- | --- | --- |
| Manifest target ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | Stable selected target identity. |
| Manifest target key | `cholesky-spd-tridiag-5` | Matches hosted generator and freshness commands. |
| Manifest family/subfamily | `comparison` / `cholesky_spd_tridiag_5` | Matches generated hosted rows. |
| Manifest artifact pattern | `build/comparison/cholesky_spd_tridiag_5/study.tsv` | Forward-slash source-controlled pattern. |
| Hosted artifact path | `build\comparison\cholesky_spd_tridiag_5\study.tsv` | Windows backslash generated path; covered by normalizer path normalization. |
| Manifest required files | `project_observations.tsv`; `baseline_observations.tsv`; `dependency_status.tsv`; `study.tsv`; `summary.md`; `manifest.tsv` | Matches hosted artifact file set. |
| Manifest expected rows | `6` | Matches hosted `study.tsv`. |
| Hosted row statuses | all `pass` | Evidence availability is positive for exact target. |
| Hosted row source commit | `d75118349269c6805654070eb44c9c57608b3e47` | Fresh to current `master` run and current branch base. |
| Hosted row platform | `windows-amd64` | Positive hosted Windows evidence. |
| Hosted row support tier | `local_only` | Claim-semantics blocker for promotion. |
| Hosted row non-claims | include `no hosted CI proof` and `no Windows report freshness` | Claim-semantics blocker for promotion. |
| Manifest workflow platforms | `linux;macos` | Manifest metadata blocker for promotion. |
| Manifest workflow artifact | Linux/macOS selected comparison artifacts only | Manifest metadata blocker until Windows artifact is added intentionally. |

### Day 3 Row Mapping

| Expected row ID | Hosted status | Artifact path form |
| --- | --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `pass` | Windows backslash path. |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `pass` | Windows backslash path. |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `pass` | Windows backslash path. |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `pass` | Windows backslash path. |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `pass` | Windows backslash path. |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `pass` | Windows backslash path. |

### Day 3 Normalizer Coverage Classification

| Risk | Current coverage | Day 3 disposition |
| --- | --- | --- |
| Windows backslash artifact paths fail selected filtering. | `selected_comparison_generated_rows()` normalizes `\` to `/`; `test_selected_comparison_generated_rows_match_windows_artifact_paths` covers backslash, mixed separator, and absolute Windows suffix paths. | Covered for Cholesky path matching. |
| Near-match artifacts are accepted. | `test_selected_comparison_generated_rows_reject_near_match_artifact_paths` rejects sibling/backup/suffix near matches. | Covered for Cholesky artifact filtering. |
| Stale selected rows pass silently. | `test_selected_comparison_target_freshness_rejects_windows_path_stale_rows` and stale/fail tests reject stale or failed rows with selected-target diagnostics. | Covered for Cholesky freshness diagnostics. |
| Wrong target rows satisfy Cholesky selected target. | `test_selected_comparison_target_freshness_rejects_wrong_target_rows` rejects QR rows for Cholesky selection with row-set mismatch diagnostics. | Covered for wrong-target evidence. |
| Workflow broad upload path includes unreviewed artifacts. | `test_windows_selected_cholesky_broad_upload_fails_clearly` and PowerShell guard reject broad comparison paths. | Covered for current workflow scope. |
| Future Windows manifest metadata points to wrong artifact. | `test_future_windows_metadata_rejects_wrong_artifact` rejects wrong Windows artifact on future allowlist state. | Covered for artifact-name drift; Day 4 should decide whether support tier and non-claim exactness also need stronger future-promotion assertions. |

### Day 3 Promotion Blocker Split

Evidence availability blockers:

- None identified for the current hosted run. The selected job succeeded, the
  artifact is available and unexpired, the file set matches the contract, all
  six rows are present, and all six rows pass.

Path-normalization blockers:

- No current blocker for basic selected Cholesky artifact matching. Backslash,
  mixed-separator, absolute Windows suffix, near-match, stale-row, and
  wrong-target coverage already exists. Day 9-Day 10 should still review
  whether artifact-mismatch and duplicate/extra-row cases need more direct
  Cholesky-specific promotion coverage.

Claim-semantics and manifest blockers:

- The selected manifest still omits `windows` from `workflow_platforms`.
- The selected manifest still names only Linux/macOS workflow artifacts.
- Generated hosted rows still say `support_tier=local_only`.
- Generated hosted rows and `summary.md` still say `no hosted CI proof` and
  `no Windows report freshness`.

Day 4 should turn this split into promotion criteria.

## Day 4 Promotion Criteria

### Promotion Criteria

Sprint 208 may promote selected Windows Cholesky freshness only if all of the
following are true at the same time:

1. A current hosted Windows CI run on `master` succeeds for the exact
   `Windows selected Cholesky comparison freshness (MSVC)` job.
2. The hosted artifact is present, unexpired, downloadable, and named
   `sprint190-windows-selected-comparison-cholesky`.
3. The artifact contains exactly the six selected Cholesky files:
   `project_observations.tsv`, `baseline_observations.tsv`,
   `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`.
4. `study.tsv` contains exactly the six expected
   `comparison_cholesky_spd_tridiag_5_*` row IDs and every selected row has
   `status=pass`.
5. Generated rows record the hosted Windows platform, current source commit,
   clean worktree, and expected MSVC/CMake compiler path.
6. Windows path normalization and selected target filtering reject stale,
   missing, wrong-target, near-match, duplicate, extra, and artifact-mismatch
   rows.
7. The selected manifest row lists the Windows workflow file, job, artifact,
   and platform in fields aligned by platform.
8. The selected manifest row's claim scope, support tier, and non-claims no
   longer contradict the hosted Windows claim.
9. Generated rows and `summary.md` no longer retain `no hosted CI proof` or
   `no Windows report freshness` for the promoted Windows selected Cholesky
   lane.
10. README, INSTALL, corpus docs, schema docs, maintainer guide, and planning
    docs all use the same narrow promoted vocabulary.
11. Broad Windows report freshness, Windows selected oracle freshness, Windows
    selected benchmark freshness, QR incompatible Windows comparison
    freshness, package/ABI support, performance, release, and state-of-the-art
    claims remain explicit non-claims.

### Re-Deferral Criteria

Sprint 208 must re-defer selected Windows Cholesky freshness if any promotion
criterion above is unmet. Known Day 4 re-deferral triggers include:

- no current hosted run is available or inspected;
- the selected Cholesky job fails, is cancelled, or is skipped;
- the artifact is missing, expired, not downloadable, or has unexpected
  membership;
- selected row IDs are missing, unexpected, stale, failed, duplicated, or
  produced for a different target;
- artifact paths cannot be matched without false positives;
- selected manifest workflow metadata cannot be updated exactly;
- generated rows or summary text still say `local_only`, `no hosted CI proof`,
  or `no Windows report freshness` in a way that contradicts promotion;
- public or maintainer docs cannot be made consistent with the selected
  manifest and generated evidence in the same branch.

### Decision-To-Change Map

| Decision | Required implementation changes |
| --- | --- |
| Promote selected Windows Cholesky freshness | Update `selected_report_targets.tsv` for exact Windows workflow metadata; update manifest/schema contract tests for exact support tier, claim scope, non-claims, workflow file/job/artifact/platform alignment, expected rows, and required files; update generator or generated-summary wording if needed; update normalizer tests for any uncovered path/row failure; update README, INSTALL, corpus docs, schema docs, maintainer guide, and planning docs; run full selected validation matrix. |
| Re-defer selected Windows Cholesky freshness | Keep selected manifest Windows platform absent; strengthen absence guards if any gap remains; update docs and planning evidence to say latest hosted evidence was reviewed but promotion remains blocked by exact criteria; add residual follow-up for generated support tier/non-claim or manifest blockers; run focused guard/docs validation. |

### Day 4 Required Validation By Outcome

| Validation | Promote | Re-defer |
| --- | --- | --- |
| Hosted Windows artifact fetch and inspection | Required | Required if available; blocker record if unavailable |
| `python3 tests/test_selected_report_targets_manifest.py` | Required | Required |
| `python3 tests/test_selected_comparison_workflow.py` | Required | Required |
| `python3 tests/test_normalize_report_index.py` | Required | Required if normalizer evidence is cited or changed |
| `python3 tests/test_run_external_comparison.py` | Required if generator or generated metadata changes | Required only if generator changes |
| `make windows-powershell-guard` | Required | Required |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Required after local generated evidence exists | Candidate validation; blocker if generated local evidence is unavailable |
| `make docs-check` | Required if docs change | Required if docs change |
| `make format && make lint && make test` | Required only if `.c` or `.h` changes | Required only if `.c` or `.h` changes |

Day 5 must apply these criteria to the Day 2-Day 3 evidence and select one
path.

## Day 5 Promotion Decision

Sprint 208 selects **continued re-deferral with stronger proof of absence** as
the implementation path.

### Criteria Evaluation

| Day 4 criterion group | Day 5 evaluation | Decision impact |
| --- | --- | --- |
| Current hosted Windows run inspected | Met. Run `35731703320` was inspected. | Supports future promotion input. |
| Exact selected job succeeded | Met. `Windows selected Cholesky comparison freshness (MSVC)` succeeded. | Supports future promotion input. |
| Artifact available and exact | Met. Artifact `10696020870` is unexpired, downloadable, and contains the six expected files. | Supports future promotion input. |
| Row identity and status clean | Met. Six expected selected Cholesky rows are present and pass. | Supports future promotion input. |
| Platform and compiler bounded | Met. Rows record `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release`. | Supports future promotion input. |
| Path handling guarded | Mostly met for path form, stale rows, and wrong-target rows. Day 9-Day 10 should still review direct duplicate, extra-row, and artifact-mismatch promotion coverage. | Not the primary blocker. |
| Manifest metadata positive and aligned | Not met. Manifest still lists Linux/macOS workflow metadata only. | Blocks promotion. |
| Support and claim fields agree | Not met. Manifest and generated rows remain `local_only`. | Blocks promotion. |
| Generated evidence wording agrees | Not met. Generated rows and summary still say `no hosted CI proof` and `no Windows report freshness`. | Blocks promotion. |
| Public and maintainer docs agree with promotion | Not met by design. Current docs correctly describe guarded workflow evidence and re-deferral. | Blocks promotion. |
| Broad non-claims stay explicit | Met. Non-claims remain explicit. | Must be preserved. |

### Decision Rationale

The latest hosted evidence proves one current, exact Windows MSVC/CMake
selected Cholesky workflow path for `cholesky-spd-tridiag-5`, but it does not
earn source-controlled selected Windows freshness promotion because the
manifest, generated metadata, and docs do not all promote Windows together.

The re-deferral is therefore evidence-backed, not evidence absence. Sprint 208
will preserve the hosted evidence ledger, keep `windows` absent from
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, and strengthen guards/docs around the exact
blockers rather than editing generator semantics and selected manifest claims
in this sprint.

### Rejected Stronger Claims

- promoted selected Windows Cholesky freshness;
- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- QR incompatible Windows comparison freshness;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library or dynamic ABI support;
- portable performance, release, external-library parity, or state-of-the-art
  claims.

### Selected Implementation Boundary

Allowed Day 6-Day 14 changes:

- keep the selected Cholesky manifest row Windows platform absent;
- strengthen manifest tests for exact current re-deferral and future promotion
  constraints, especially support tier, claim scope, non-claims, workflow
  metadata, and required files;
- strengthen workflow or PowerShell guards if any selected Cholesky drift is
  unguarded;
- add normalizer tests only for uncovered duplicate, extra-row, or artifact
  mismatch cases found during review;
- update README, INSTALL, corpus docs, schema docs, maintainer guide, and
  planning docs to say latest hosted evidence was reviewed but promotion
  remains re-deferred by generated support tier and non-claim blockers;
- record residual work for changing generated support tier and non-claims if a
  future sprint wants promotion.

Disallowed changes for this decision path:

- adding `windows` to `workflow_platforms`;
- listing `.github/workflows/windows-ci.yml` or
  `sprint190-windows-selected-comparison-cholesky` as positive selected
  manifest metadata;
- removing `no Windows report freshness` or `no hosted CI proof` from
  generated Cholesky rows without a promotion design;
- implying broad Windows report freshness, package/ABI support, performance,
  release, external-library parity, or state-of-the-art status.

## Day 6 Manifest Metadata Design

### Selected Metadata State

Sprint 208 keeps the source-controlled selected target row in a re-deferred
state:

| Manifest field | Required Sprint 208 state |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `family` / `subfamily` | `comparison` / `cholesky_spd_tridiag_5` |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `claim_scope` | Linux/macOS selected Cholesky freshness wording only; no Windows selected freshness promotion. |
| `non_claims` | Must retain `no Windows report freshness`, package/ABI non-claims, performance non-claim, and state-of-the-art non-claim. |

### Guard Design

| Guard owner | Day 6 requirement |
| --- | --- |
| `tests/test_selected_report_targets_manifest.py` | Add an exact current-state contract for the selected Cholesky row, not just a generic no-Windows scan. Assert support tier, claim scope, non-claims, workflow file/job/artifact/platform values, expected row count, required files, and expected row IDs. |
| `tests/test_selected_report_targets_manifest.py` future allowlist | Extend future Windows allowlist to reject promotion if support tier, claim scope, or non-claims still contain re-deferral wording that contradicts promotion. |
| `scripts/validate_windows_powershell.py` | Continue enforcing bounded workflow tokens, artifact name, exact six upload files, no broad comparison upload, and no Windows selected manifest platform. |
| `tests/test_selected_comparison_workflow.py` | Continue enforcing bounded Cholesky-only workflow and no broad selected report freshness lane. |
| `tests/test_normalize_report_index.py` | Day 9-Day 10 review owns duplicate/extra/artifact-mismatch gaps; Day 6 does not require immediate normalizer edits. |

### Documentation Wording Plan

Docs should say:

- latest hosted Windows evidence for exact `cholesky-spd-tridiag-5` was
  inspected and passed;
- selected Windows freshness remains re-deferred because generated evidence
  still says `local_only`, `no hosted CI proof`, and
  `no Windows report freshness`;
- the selected target manifest remains Linux/macOS-only for positive selected
  metadata;
- broad Windows report freshness, Windows oracle/benchmark freshness, QR
  incompatible Windows freshness, package/ABI support, performance, release,
  and state-of-the-art claims remain unclaimed.

### Day 7 Implementation Target

Day 7 should modify manifest tests and, only if needed, PowerShell/workflow
guards. It should not modify `selected_report_targets.tsv` unless the change
preserves re-deferral wording and does not add Windows positive metadata.

## Day 7 Manifest Guard Implementation

### Changed Files

- `tests/test_selected_report_targets_manifest.py`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day7-manifest-guard-implementation.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 7 implemented the Day 6 manifest test design without promoting Windows in
`tests/corpus/manifests/selected_report_targets.tsv`.

The selected Cholesky row is now guarded by an exact current-state contract for:

- row identity, family, subfamily, and target key;
- artifact pattern and generator command;
- `support_tier=local_only`;
- current Linux/macOS-only workflow file, job, artifact, and platform metadata;
- six expected row IDs and six required artifact files;
- current claim scope;
- the full current non-claim set, including `no Windows report freshness`.

The future Windows Cholesky allowlist was also tightened. If a later branch adds
`windows` to the selected Cholesky manifest metadata, the test guard now rejects
that state unless the row also stops using the `local_only` support tier, updates
claim scope to name Windows, and removes stale re-deferral wording such as
`no Windows report freshness` or `no hosted CI proof`.

### Manifest State

`selected_report_targets.tsv` remains unchanged for Day 7. The source-controlled
manifest still lists positive selected Cholesky workflow metadata only for
Linux/macOS:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `workflow_platforms=linux;macos`

This preserves the Day 5 re-deferral decision while making unsupported Windows
promotion drift fail loudly.

### Validation

Commands run:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 -m py_compile tests/test_selected_report_targets_manifest.py
python3 scripts/validate_corpus_schema.py
```

Results:

- `test-selected-report-targets-manifest: ok`
- Python compilation completed without diagnostics.
- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.3 has concrete manifest or guard changes. | Met. Manifest contract tests now enforce current re-deferral and future promotion prerequisites. |
| The selected target row cannot silently drift into unsupported Windows claims. | Met. Windows metadata, stale support tier, stale claim scope, and stale non-claims are rejected. |
| Focused manifest validation passes before broader normalizer work begins. | Met. Focused manifest and schema validation passed. |

## Day 8 Workflow And PowerShell Guard Alignment

### Changed Files

- `scripts/validate_windows_powershell.py`
- `tests/test_validate_windows_powershell.py`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day8-workflow-powershell-guard-alignment.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 8 aligned the Windows workflow-owner guard with the Day 5 re-deferral
decision and the Day 7 manifest contract.

`scripts/validate_windows_powershell.py` now independently verifies the current
selected Cholesky manifest re-deferral state:

- `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` must exist exactly once;
- `support_tier` must remain `local_only`;
- workflow metadata must remain the current Linux/macOS pair;
- the selected Windows artifact
  `sprint190-windows-selected-comparison-cholesky` must not appear as positive
  manifest metadata;
- `workflow_platforms` must remain `linux;macos`;
- the full current non-claim set must remain present, including
  `no Windows report freshness`.

The claim-boundary scanner now also rejects explicit positive wording that
promotes Windows selected Cholesky, comparison, or report freshness while the
manifest remains re-deferred.

### Regression Coverage

Added tests in `tests/test_validate_windows_powershell.py` for:

- direct Windows platform drift on the selected Cholesky manifest row;
- stale or misplaced Windows selected artifact metadata;
- removal of `no Windows report freshness`;
- unsupported public wording such as
  `Windows selected Cholesky freshness is promoted`.

Existing Windows workflow tests continue to cover:

- selected Cholesky target command and freshness command;
- selected artifact name;
- fail-closed upload configuration;
- exact six selected upload files;
- QR incompatible re-deferral;
- no broad selected report freshness lane outside the bounded Cholesky job.

### Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- Python compilation completed without diagnostics.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.3 covers workflow and PowerShell guard surfaces. | Met. The Windows PowerShell validator now owns exact Cholesky re-deferral metadata. |
| Promoted or re-deferred workflow state is enforced by tests. | Met. Tests reject Windows platform drift, stale artifact metadata, removed non-claims, and positive promotion wording. |
| Guard diagnostics identify the exact stale or unsupported field. | Met. Failures name the deferred manifest field or unsupported claim wording. |

## Day 9 Normalizer Regression Design

### Reviewed Surfaces

- `tests/test_normalize_report_index.py`
- `scripts/normalize_report_index.py`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day9-normalizer-regression-design.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Existing Normalizer Coverage

The current normalizer test suite already covers several Sprint 208 risks:

- selected Cholesky generated rows match forward-slash, backslash, mixed, and
  absolute Windows-style artifact paths;
- selected Cholesky near-match artifact paths are rejected;
- stale selected Cholesky rows are rejected under Windows backslash paths;
- stale and failed selected Cholesky rows are rejected for
  `--selected-target cholesky-spd-tridiag-5`;
- wrong-target selected rows produce Cholesky missing-row diagnostics;
- QR incompatible already has duplicate and unexpected Windows-path row
  regressions.

### Day 10 Targeted Gaps

Day 9 identified two high-value Cholesky-specific gaps for Day 10:

1. Duplicate selected Cholesky row IDs under Windows backslash artifact paths
   should fail with `duplicate normalized row_id`.
2. Unexpected selected Cholesky row IDs under Windows backslash artifact paths
   should fail with a `comparison_selected_rows` row-set mismatch that names
   both the missing expected Cholesky row and the unexpected row.

The remaining requested fixture classes already have adequate coverage or can
reuse existing tests unless Day 10 changes diagnostics:

- artifact-root absolute paths;
- missing selected rows;
- stale rows;
- wrong artifact names or near-match artifact paths.

### Expected Diagnostics

| Fixture | Required diagnostic details |
| --- | --- |
| Duplicate Cholesky Windows-path row | `duplicate normalized row_id` and `comparison_cholesky_spd_tridiag_5_project_status_v1`. |
| Unexpected Cholesky Windows-path row | `freshness: error:`, `comparison_selected_rows`, `row_set_mismatch`, `observed=6`, missing `comparison_cholesky_spd_tridiag_5_project_status_v1`, unexpected `comparison_cholesky_spd_tridiag_5_unexpected_metric_v1`, Cholesky artifact diagnostic, and `--selected-target cholesky-spd-tridiag-5`. |

### Helper Reuse

Day 10 should reuse:

- `write_selected_comparison_rows()`;
- `read_tsv()`;
- `COMPARISON_STUDY_FIELDS`;
- `run_command(..., expect_success=False)`;
- `SELECTED_CHOLESKY_ARTIFACT_DIAGNOSTIC`;
- the existing selected-target command shape for
  `cholesky-spd-tridiag-5`.

### Validation Plan

Planned Day 10 validation:

```sh
python3 tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_validate_windows_powershell.py
python3 -m py_compile tests/test_normalize_report_index.py
```

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.4 has an implementation-ready regression plan. | Met. Day 10 fixtures, helper reuse, and expected diagnostics are specified. |
| Every planned fixture ties to a selected Windows Cholesky risk. | Met. Duplicate and unexpected-row fixtures target the exact selected Cholesky artifact path. |
| Expected failure messages are defined before tests are written. | Met. Required stdout/stderr substrings are recorded above. |

## Day 10 Normalizer Regression Implementation

### Changed Files

- `tests/test_normalize_report_index.py`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day10-normalizer-regression-implementation.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 10 implemented the two Cholesky-specific normalizer regressions identified
on Day 9:

1. `test_cholesky_selected_freshness_rejects_duplicate_windows_path_rows`
   generates only `cholesky_spd_tridiag_5`, duplicates the first selected row,
   rewrites the artifact path to Windows backslash form, and verifies the
   normalizer fails with `duplicate normalized row_id` for
   `comparison_cholesky_spd_tridiag_5_project_status_v1`.
2. `test_cholesky_selected_freshness_rejects_unexpected_windows_path_rows`
   generates only `cholesky_spd_tridiag_5`, replaces the first row ID with
   `comparison_cholesky_spd_tridiag_5_unexpected_metric_v1`, rewrites artifact
   paths to Windows backslash form, and verifies selected freshness reports a
   row-set mismatch with the selected target ID, missing expected row,
   unexpected row, artifact diagnostic, and selected-target remediation.

These tests mirror the existing QR incompatible Windows-path duplicate and
unexpected-row coverage for the Sprint 208 selected Cholesky target.

### Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 -m py_compile tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
```

Results:

- `test-normalize-report-index: ok`
- Python compilation completed without diagnostics.
- `test-selected-report-targets-manifest: ok`
- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`

### Claim Boundary

No selected target manifest promotion was made. The source-controlled selected
Cholesky manifest row remains Linux/macOS-only and `local_only`, with
`no Windows report freshness` retained.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.4 has executable regression coverage. | Met. Two Cholesky Windows-path selected freshness regressions were added. |
| Windows-style selected Cholesky path and row failures are covered. | Met. Duplicate and unexpected-row failures are covered under backslash artifact paths. |
| Focused normalizer validation passes. | Met. The full `tests/test_normalize_report_index.py` standalone runner passed. |

## Day 11 Public Documentation Calibration

### Changed Files

- `README.md`
- `INSTALL.md`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day11-public-docs-calibration.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 11 updated user-facing support wording to match the Sprint 208 selected
Windows Cholesky decision.

`README.md` now says Sprint 208 reviewed the current bounded Windows Cholesky
hosted path for `cholesky-spd-tridiag-5`, strengthened manifest,
workflow/PowerShell, and normalizer guards, and kept selected Windows freshness
re-deferred. The selected comparison section retains the required marker that
the Sprint 190 artifact is evidence for that exact path and re-deferred
selected Windows freshness only.

`INSTALL.md` now updates the support/readiness matrix and platform table from
the older Sprint 199 wording to Sprint 208 wording. It still says support and
freshness promotion remain re-deferred until selected metadata, generated
support tier, and generated non-claim wording are promoted together.

### Retained Public Boundaries

The public docs continue to avoid claims for:

- broad Windows report freshness;
- Windows selected oracle or benchmark freshness;
- QR incompatible Windows selected freshness;
- Windows Makefile parity or Windows `pkg-config` execution parity;
- package-manager support;
- shared-library, dynamic ABI, or runtime-loader support;
- performance, release, external-library parity, or state-of-the-art evidence.

### Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
rg -n "Windows selected (Cholesky|comparison|report) freshness is promoted|Windows report freshness is supported|PowerShell validation proves Windows report freshness" README.md INSTALL.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- `test-selected-report-targets-manifest: ok`
- Forbidden broad-claim search returned no matches.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.5 has public documentation aligned with evidence. | Met. README and INSTALL now reference Sprint 208 evidence/guard state. |
| Users can identify what the selected Windows Cholesky lane does and does not prove. | Met. The public docs identify the exact bounded path and retained non-claims. |
| Public docs do not imply broad Windows, package, ABI, performance, or release support. | Met. Guard tests passed and broad-claim search returned no matches. |

## Day 12 Maintainer And Corpus Documentation

### Changed Files

- `docs/maintainer_guide.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `docs/planning/EPIC_19/PROJECT_PLAN.md`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day12-maintainer-corpus-docs.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 12 aligned maintainer-facing and corpus documentation with the Sprint 208
implementation state.

`docs/maintainer_guide.md` now treats Sprint 208 as the current review of the
bounded Sprint 190 Windows Cholesky hosted path and names the strengthened
manifest, workflow/PowerShell, and normalizer guard surfaces. It keeps selected
Windows freshness re-deferred until selected metadata, generated support tier,
generated non-claim wording, and claim contract are promoted together.

`tests/corpus/README.md` now records that Sprint 208 reviewed current hosted
evidence for the exact Cholesky path, strengthened guard surfaces, and
re-deferred selected Windows freshness promotion.

`tests/corpus/schemas/report_index_fields.md` now says Sprint 208 keeps
`windows` absent from selected target `workflow_platforms` as the
source-controlled authority and guards that absence with manifest, workflow,
and normalizer regressions.

`docs/planning/EPIC_19/PROJECT_PLAN.md` now marks Sprint 208 as in progress
with evidence through Day 12 rather than pending future execution. It does not
close Sprint 208 before integrated validation and closeout.

### Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
rg -n "Sprint 199 reviewed|Sprints 208 through 216|208-216 \\| Pending|Windows selected (Cholesky|comparison|report) freshness is promoted|PowerShell validation proves Windows report freshness" docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_19/PROJECT_PLAN.md README.md INSTALL.md
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`
- Stale/broad-claim search returned no matches.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.5 has maintainer-facing documentation aligned with implementation. | Met. Maintainer guide, corpus README, schema docs, and project-plan status now reference Sprint 208 state. |
| Maintainers know which commands protect the selected Cholesky lane. | Met. Maintainer guide keeps `make windows-powershell-guard`, `make windows-powershell-validate`, and selected normalizer command ownership visible. |
| Current-status wording does not overstate Sprint 208 outcomes. | Met. Project plan says Sprint 208 is in progress and re-deferred, not closed or promoted. |

## Day 13 Integrated Validation

### Changed Files

- `docs/planning/EPIC_19/SPRINT_208/artifacts/day13-integrated-validation.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 13 ran the integrated validation matrix for the Sprint 208 selected
Windows Cholesky re-deferral path and recorded the results in
`artifacts/day13-integrated-validation.md`.

The validation covered selected target manifest contracts, corpus schema,
workflow guard wiring, PowerShell claim boundaries, normalizer regressions,
comparison report-index freshness, documentation coverage, support quick
reference checks, Python syntax, and stale or overbroad claim searches.

No `.c` or `.h` files were modified by Sprint 208, so `make format`,
`make lint`, and `make test` were not required for Day 13.

### Validation

Commands run:

```sh
make windows-powershell-guard
python3 tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_validate_windows_powershell.py
python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py
make docs-check
make support-docs-guard
make report-index-comparison-freshness
rg -n "Sprint 199 reviewed|Sprints 208 through 216|208-216 \\| Pending|Windows selected (Cholesky|comparison|report) freshness is promoted|PowerShell validation proves Windows report freshness|hosted Windows report freshness is promoted|Windows Cholesky selected freshness promoted" docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_19/PROJECT_PLAN.md README.md INSTALL.md
```

Results:

- `make windows-powershell-guard` passed; local missing `pwsh` remains an
  expected unavailable-evidence path, not hosted pass evidence.
- `test-normalize-report-index: ok`
- `test-selected-report-targets-manifest: ok`
- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`
- `test-selected-comparison-workflow: ok`
- `test-validate-windows-powershell: ok`
- Python compilation completed without diagnostics.
- `make docs-check` passed with 18 checked-in public headers and 18 generated
  reference/source pages covered.
- `test-support-quick-reference-docs: ok`
- `make report-index-comparison-freshness` passed with freshness ok for 46
  rows.
- Stale and overbroad claim search returned no matches.

### Cleanup Audit

`git status --short` showed only expected Sprint 208 modified files plus the
untracked `docs/planning/EPIC_19/SPRINT_208/` directory.

`git status --ignored --short` showed ignored local/generated outputs:
`.claude/`, `.swp`, `archive/sparse_lu`, `build/`, `cmake-build/`,
`docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/`.

### Claim Boundary

Day 13 did not promote selected Windows Cholesky freshness. The selected
target manifest still keeps `windows` absent from `workflow_platforms` and
retains `no Windows report freshness` as a required non-claim.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.6 has current validation evidence. | Met. Integrated validation passed for the selected re-deferral path. |
| Required commands pass or blockers are documented with exact failing output. | Met. All required Day 13 commands passed; local missing `pwsh` remains explicitly documented by the guard suite. |
| No generated proof artifacts or unsupported claim changes remain untracked. | Met. Generated outputs are ignored, and untracked files are Sprint 208 planning artifacts. |

## Day 14 Closeout Review

### Changed Files

- `docs/planning/EPIC_19/PROJECT_PLAN.md`
- `docs/planning/EPIC_19/SPRINT_208/artifacts/day14-closeout-review.md`
- `docs/planning/EPIC_19/SPRINT_208/WORKING_NOTES.md`

### Implementation Summary

Day 14 closed Sprint 208 as continued selected Windows Cholesky freshness
re-deferral with stronger guard coverage. The closeout reconciled the item
checklist, evidence ledger, project-plan current-status row, residuals, and
retrospective inputs.

The final Sprint 208 state is not promotion. The latest hosted Windows
Cholesky run and artifact were accepted as bounded workflow evidence, while
generated `local_only`, `no hosted CI proof`, and `no Windows report
freshness` wording keep selected Windows freshness unearned.

### Final Item Status

| Item | Status |
| --- | --- |
| 208.1 | Complete with hosted run and artifact evidence inspected. |
| 208.2 | Complete as deliberate re-deferral. |
| 208.3 | Complete for re-deferral via strengthened manifest and PowerShell guards. |
| 208.4 | Complete with Cholesky Windows-path normalizer regressions. |
| 208.5 | Complete with public, maintainer, corpus, schema, and planning docs calibrated. |
| 208.6 | Complete with integrated validation and closeout evidence. |

### Validation

Commands run:

```sh
git diff --check
```

Results:

- `git diff --check` completed without diagnostics.

Day 14 relies on the Day 13 integrated matrix for the selected validation
evidence:

- `make windows-powershell-guard`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`
- Python compilation checks
- `make docs-check`
- `make support-docs-guard`
- `make report-index-comparison-freshness`
- stale and overbroad claim search

No `.c` or `.h` files were modified, so `make format`, `make lint`, and
`make test` were not required.

### Residuals

Future promotion must redesign generated support-tier and non-claim semantics,
then move manifest metadata, generated evidence, docs, and guards together.
Until then, selected Windows Cholesky freshness, broad Windows freshness,
package support, ABI support, performance, release readiness, external-library
parity, and state-of-the-art status remain unclaimed.

### Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.6 has closeout evidence for the final branch state. | Met. `day14-closeout-review.md` records final status and residuals. |
| Sprint 208 outcomes are traceable to artifacts, tests, and docs. | Met. The item checklist and closeout evidence ledger link every item to artifacts and guard surfaces. |
| Any stronger claims not earned by the branch remain explicitly residual or non-claims. | Met. Re-deferral blockers and stronger non-claims remain documented. |
