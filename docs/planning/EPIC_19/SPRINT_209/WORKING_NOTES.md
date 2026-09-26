# Sprint 209 Working Notes

## Sprint Goal

Add hosted Windows/MSVC proof for `qr-incompatible-ls` and promote selected
metadata only if the exact evidence supports it.

## Current Branch

- Branch: `sprint-209`
- Plan: `docs/planning/EPIC_19/SPRINT_209/PLAN.md`
- Epic source: `docs/planning/EPIC_19/PROJECT_PLAN.md`, Sprint 209

## Item Checklist

| Item | Name | Current disposition | Primary surfaces | Evidence |
| --- | --- | --- | --- | --- |
| 209.1 | MSVC Probe Design | Complete for Day 3: exact hosted MSVC/CMake QR incompatible proof command is designed, and latest hosted Windows evidence inspection confirms no QR artifact exists yet. | `scripts/run_external_comparison.py`, Windows CMake/MSVC workflow, generated comparison files, selected manifest row | Day 1 surface inventory, Day 2 probe design, Day 3 hosted evidence inventory |
| 209.2 | Workflow Implementation | Implemented for Day 5: one bounded Windows/MSVC QR incompatible selected comparison lane is wired with exact commands, exact fail-closed six-file upload, PowerShell ownership validation, and workflow regression coverage; hosted execution evidence remains pending Day 6. | `.github/workflows/windows-ci.yml`, `scripts/validate_windows_powershell.py`, workflow guard tests | Day 3 hosted evidence inventory, Day 4 workflow implementation design, Day 5 workflow implementation record |
| 209.3 | Artifact Inspection Tests | Complete for Day 6: normalizer freshness now checks selected comparison required sidecar files, QR missing-file and unrelated-artifact regressions pass, Windows-style QR path stale/duplicate/unexpected-row coverage remains active, and local QR generation produced the exact six-file bundle. | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, comparison artifacts | Day 1 validation matrix, Day 6 artifact inspection tests |
| 209.4 | Manifest Decision | Complete for Day 8 with re-deferral: no hosted `Windows CI` run exists for branch `sprint-209`, so `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`; explicit guards reject the Sprint 209 Windows QR workflow, job, artifact, and platform metadata while re-deferred. | `tests/corpus/manifests/selected_report_targets.tsv`, manifest contract tests, report-index schema docs | Day 1 manifest inventory, Day 7 manifest decision criteria, Day 8 manifest re-deferral decision |
| 209.5 | Docs And Claim Guards | Complete for Day 11: public README/INSTALL, maintainer guide, corpus README, report-index schema, and Epic 19 planning status now record Sprint 209 as a bounded QR evidence-collection lane while keeping selected Windows QR freshness re-deferred; public and maintainer/corpus/schema markers plus unsupported QR promotion wording are guarded. | README, INSTALL, `docs/maintainer_guide.md`, corpus docs, PowerShell claim guards | Day 1 claim-boundary inventory, Day 9 guard integration, Day 10 public docs calibration, Day 11 maintainer and corpus docs |
| 209.6 | Validation And Closeout | Complete for Day 14: focused and integrated validation passed for the non-C change set, Sprint 209 artifacts are reconciled, project-plan status is closed, and hosted Windows CI QR run/artifact absence remains the explicit residual that keeps selected Windows QR freshness re-deferred. | QR generator/freshness commands, normalizer tests, manifest tests, workflow tests, PowerShell guard, docs checks | Day 1 validation matrix, Day 12 focused validation, Day 13 integrated validation, Day 14 closeout review |

## Day 1 Evidence Map

| Evidence source | Day 1 interpretation |
| --- | --- |
| `docs/planning/EPIC_18/SPRINT_203/RETROSPECTIVE.md` | Sprint 203 closed as a re-deferral: local `qr-incompatible-ls` generator and selected freshness proof passed, but hosted Windows/MSVC proof and hosted artifact inspection were absent. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day2-msvc-probe-design.md` | Prior design identified the intended MSVC command, expected six QR incompatible rows, and exact artifact bundle needed for promotion. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day7-manifest-promotion-decision.md` | Manifest promotion was deliberately re-deferred because no hosted Windows/MSVC QR proof or artifact inspection existed. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day8-manifest-workflow-metadata.md` | Current QR incompatible row is guarded as Linux/macOS-only, `local_only`, with exact row IDs, required files, and non-claims. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day9-workflow-guard-integration.md` | Windows workflow guards reject accidental QR incompatible target commands, freshness commands, subfamily references, artifact names, and upload paths. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day12-integrated-validation.md` | Local QR generator and freshness commands passed, but hosted Windows/MSVC proof remained absent. |
| `docs/planning/EPIC_18/SPRINT_203/artifacts/day13-review-hardening.md` | Existing guard coverage protects re-deferral across manifest, workflow, normalizer diagnostics, and documentation markers. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Residual E18-RQ-006 names Windows QR incompatible comparison promotion as future work requiring MSVC/CMake proof, artifact inspection, and exact manifest metadata. |
| `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md` | Epic 19 closure track 3 directs Sprint 209 to add hosted MSVC/CMake proof, inspect artifacts, then promote or deliberately re-defer. |
| `docs/planning/EPIC_19/SPRINT_208/RETROSPECTIVE.md` | Sprint 208 retained QR incompatible selected freshness as a future Sprint 209 owner after closing bounded Cholesky evidence. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Current `SRT-COMP-QR-INCOMPATIBLE-LS` row lists Linux/macOS workflow metadata only, remains `support_tier=local_only`, and retains `no Windows report freshness`. |
| README, INSTALL, corpus README, maintainer guide | Current docs state that QR incompatible least-squares remains outside Windows selected freshness until hosted MSVC proof, selected artifact review, manifest metadata, generated support tier, and generated non-claim wording are promoted together. |

## Current QR Incompatible Claim Boundary

Earned or inherited evidence currently covers only:

- local selected QR incompatible generator proof for `qr-incompatible-ls`;
- local target-specific freshness proof for six QR incompatible comparison rows;
- Windows-style path and selected-row diagnostic regressions added in Sprint 203;
- manifest and workflow absence guards that prevent accidental Windows QR
  metadata or workflow drift;
- documentation markers that keep QR incompatible outside selected Windows
  freshness.

The following remain explicit non-claims until Sprint 209 records stronger
hosted evidence and updates all required surfaces together:

- promoted selected Windows QR incompatible freshness;
- broad Windows report freshness;
- broad QR parity;
- broad least-squares parity;
- raw QR basis identity;
- Q sign or orientation identity;
- global rank-threshold policy;
- broad rank-deficient solve support;
- NumPy, SciPy, LAPACK, SuiteSparse, or Eigen parity;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support or package-manager platform parity;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows parity;
- portable performance claims;
- release readiness;
- state-of-the-art status.

## Initial Surface Inventory

| Surface | Day 1 role |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Source of truth for selected QR incompatible metadata. Current row omits `windows`, remains `local_only`, and retains the full QR non-claim set. |
| `tests/corpus/schemas/report_index_fields.md` | Schema and support-tier wording for selected report metadata and retained Windows QR re-deferral. |
| `.github/workflows/windows-ci.yml` | Current Windows workflow contains bounded selected Cholesky proof only; no QR incompatible workflow lane is present. |
| `.github/workflows/ci.yml` and `.github/workflows/macos-ci.yml` | Existing Linux/macOS selected comparison freshness metadata for QR incompatible. |
| `scripts/run_external_comparison.py` | Generates the selected QR incompatible comparison rows locally and will be the command target for any hosted MSVC proof. |
| `scripts/normalize_report_index.py` | Freshness and selected target filtering implementation for generated comparison rows. |
| `scripts/validate_windows_powershell.py` | Windows workflow and claim-boundary validator; currently protects Cholesky ownership and QR re-deferral markers. |
| `tests/test_selected_report_targets_manifest.py` | Manifest contract coverage for QR incompatible re-deferral, row IDs, required files, support tier, non-claims, and absent Windows metadata. |
| `tests/test_selected_comparison_workflow.py` | Workflow guard coverage rejecting QR incompatible Windows commands, upload paths, and accidental selected freshness drift. |
| `tests/test_normalize_report_index.py` | Windows-style QR artifact path, row filtering, duplicate-row, unexpected-row, stale-row, and diagnostic coverage. |
| `tests/test_run_external_comparison.py` | Generator behavior coverage for comparison targets. |
| README | User-facing selected comparison and Windows QR non-claim surface. |
| INSTALL | Support/readiness matrix and Windows QR incompatible deferral wording. |
| `tests/corpus/README.md` | Corpus interpretation for selected comparison evidence and QR Windows deferral. |
| `docs/maintainer_guide.md` | Maintainer runbook for selected comparison freshness, Windows evidence, and claim boundaries. |
| Epic 18 Sprint 203 artifacts | Historical local proof, re-deferral rationale, and guard coverage. |
| Epic 19 project plan and reviews | Current sprint objective and closeout track. |

## Initial Validation Matrix

| Validation | Purpose | Day 1 status |
| --- | --- | --- |
| `gh run list` / `gh run view` / `gh run download` for Windows QR evidence | Fetch and inspect current hosted Windows run and artifact bundle if a QR lane exists or is added. | Candidate Day 3 command; not run on Day 1 intake. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Generate the local six-file selected QR incompatible comparison bundle. | Candidate Day 6 and Day 12 command. |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Prove generated QR incompatible rows are fresh to local `HEAD`. | Candidate Day 6, Day 12, and Day 13 command. |
| `python3 tests/test_selected_report_targets_manifest.py` | Enforce selected manifest contract, current Windows absence, and any future exact QR promotion contract. | Candidate Day 8, Day 9, Day 12, and Day 13 command. |
| `python3 tests/test_selected_comparison_workflow.py` | Enforce bounded Windows workflow behavior and QR incompatible absence or exact owned workflow path. | Candidate Day 5, Day 9, Day 12, and Day 13 command. |
| `python3 tests/test_normalize_report_index.py` | Validate selected freshness filtering, Windows paths, stale/missing/wrong rows, and diagnostics. | Candidate Day 6, Day 12, and Day 13 command. |
| `python3 tests/test_run_external_comparison.py` | Validate selected comparison generator behavior if QR generation semantics change. | Candidate Day 6, Day 12, and Day 13 command. |
| `make windows-powershell-guard` | Run source-controlled Windows workflow and PowerShell guard tests. | Candidate Day 5, Day 9, Day 12, and Day 13 command. |
| `make docs-check` | Validate docs and generated API coverage when public or maintainer docs change. | Candidate Day 10-Day 13 command. |
| `bash scripts/static_package_deferral_check.sh` | Protect adjacent package and ABI non-claims if Windows support docs change. | Candidate Day 13 command. |
| `make format && make lint && make test` | Full C quality gate. | Required only if Sprint 209 modifies `.c` or `.h` files. |
| `git diff --check` | Whitespace validation for all changed files. | Required before closeout. |

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sprint 203 local QR evidence is mistaken for hosted Windows promotion proof. | Manifest or docs could overstate selected Windows freshness. | Day 3 must fetch hosted Windows evidence or record current hosted-proof blockers. |
| Cholesky Windows proof is generalized to QR incompatible. | Selected Windows QR claims could be based on another target's evidence. | Day 2 and Day 7 require exact `qr-incompatible-ls` target identity, row IDs, and artifact paths. |
| Manifest adds `windows` without matching support tier, generated non-claims, workflow metadata, and docs. | Source-controlled claim surfaces become contradictory. | Day 7 criteria require all promotion surfaces to move together. |
| Workflow upload broadens from exact QR files to broad comparison paths. | Artifact may include unreviewed rows and imply broader Windows report freshness. | Day 4-Day 5 workflow design must use exact file uploads and fail-closed behavior. |
| Windows path normalization accepts near-match or wrong-target QR artifacts. | Wrong evidence could satisfy selected QR freshness. | Day 6 extends Windows-style path and selected-row diagnostics before any promotion. |
| Public docs compress selected QR evidence into broad QR, Windows, or external-library parity. | Users may infer support not earned by the sprint. | Day 10-Day 11 docs calibration must retain explicit non-goals and guard wording. |
| Local missing PowerShell is treated as hosted proof. | Validation status becomes misleading. | Preserve hosted `--require-pwsh` ownership; classify local unavailable PowerShell separately. |

## Open Questions

1. Does the current Windows workflow already contain any hidden or partial
   `qr-incompatible-ls` lane after Sprint 208, or must Sprint 209 add it from
   scratch?
2. What exact hosted Windows run and artifact will prove `qr-incompatible-ls`
   after the workflow path exists?
3. Does the hosted MSVC probe produce the same six QR incompatible row IDs as
   the local generator?
4. Do generated rows still carry `support_tier=local_only` and `no Windows
   report freshness`, or can Sprint 209 promote generated support-tier and
   non-claim wording together with manifest metadata?
5. Which tests need to distinguish Cholesky Windows evidence from QR
   incompatible Windows evidence to avoid cross-target leakage?
6. Should promotion require updating both selected manifest metadata and
   report-family/schema wording, or should Sprint 209 close as a stronger
   re-deferral if generator semantics remain local-only?
7. Which residual queue or later sprint should receive any unearned broad QR,
   broad Windows, package, ABI, external-library, performance, release, or
   state-of-the-art work?

## Day 2 MSVC Probe Design

Day 2 defines the exact hosted Windows/MSVC proof shape required before Sprint
209 can promote or re-defer the selected `qr-incompatible-ls` metadata. It does
not change workflow YAML, manifest metadata, generator behavior, or public docs.

### Selected Target Contract

| Field | Value |
| --- | --- |
| Target id | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Family | `comparison` |
| Subfamily | `qr_incompatible_ls` |
| Target key | `qr-incompatible-ls` |
| Fixture | `qr_overdetermined_incompatible_4x2` |
| Operation | Incompatible QR least-squares solve with expected nonzero residual. |
| Artifact directory | `build/comparison/qr_incompatible_ls/` |
| Artifact pattern | `build/comparison/qr_incompatible_ls/study.tsv` |
| Generator command | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Expected rows | `6` |
| Current platforms | `linux;macos` |
| Current support tier | `local_only` |
| Current Windows status | Re-deferred until hosted MSVC proof, artifact inspection, generated support tier, manifest metadata, and docs align. |

### Expected Row IDs

| Row | Meaning |
| --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | Project solve status. |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | Source-controlled dense baseline status. |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | Expected nonzero incompatible least-squares residual norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | Project solution norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | Project solution value vector. |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | Maximum project-vs-baseline solution delta. |

### Canonical Hosted Probe Sequence

```text
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target sparse_lu_ortho
python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

### Required Hosted Artifact Contract

| Required file | Promotion purpose |
| --- | --- |
| `build/comparison/qr_incompatible_ls/project_observations.tsv` | Confirms project status row source for the exact QR incompatible fixture. |
| `build/comparison/qr_incompatible_ls/baseline_observations.tsv` | Confirms dense reference helper status and keeps dependency failures separate from project failures. |
| `build/comparison/qr_incompatible_ls/dependency_status.tsv` | Records baseline availability and prevents missing helper state from being treated as project pass evidence. |
| `build/comparison/qr_incompatible_ls/study.tsv` | Holds the six selected comparison rows used by freshness and manifest contracts. |
| `build/comparison/qr_incompatible_ls/summary.md` | Human-readable selected proof summary for artifact review. |
| `build/comparison/qr_incompatible_ls/manifest.tsv` | Artifact membership and generation metadata for inspection. |

### CMake/MSVC Requirements

| Requirement | Rationale |
| --- | --- |
| Use `windows-2022` with `Visual Studio 17 2022` and `x64`. | Matches the reviewed Windows CMake/MSVC surface already used by CI. |
| Build `sparse_lu_ortho` in `Release`. | Produces the expected MSVC static library consumed by the external comparison probe. |
| Pass `--probe-build-system cmake`. | Exercises the downstream CMake consumer path rather than direct compiler mode. |
| Pass `--library build/Release/sparse_lu_ortho.lib`. | Avoids the Unix static-library default and binds the proof to the hosted MSVC artifact. |
| Run freshness with `--selected-target qr-incompatible-ls`. | Prevents unrelated comparison rows from satisfying the QR proof. |
| Upload only the six QR incompatible required files. | Prevents broad comparison artifact publication or accidental Windows report freshness claims. |
| Use fail-closed artifact upload behavior. | Missing selected files must block hosted proof instead of producing partial evidence. |

### Promotion Blockers To Preserve

| Blocker | Required treatment |
| --- | --- |
| CMake configure, build, or generated consumer failure | Block promotion until a repo-owned fix or explicit environment residual is recorded. |
| Missing or failing dense reference helper | Block promotion and keep dependency status separate from project status. |
| Missing required artifact file | Fail hosted proof and keep manifest metadata unpromoted. |
| Wrong, duplicate, stale, or incomplete selected row IDs | Fail freshness with QR-specific diagnostics. |
| Artifact path separator or root mismatch | Add or fix normalizer coverage before promotion. |
| Workflow upload includes broad comparison paths | Reject through workflow/PowerShell guards. |
| Generated support tier remains `local_only` or generated non-claims still say `no Windows report freshness` | Re-defer or coordinate generated metadata, manifest metadata, and docs together. |

### Day 2 Design Decision

Sprint 209 should treat the probe above as the minimum hosted proof contract for
`qr-incompatible-ls`. Day 3 should inspect current hosted Windows evidence if it
exists; if it does not, Day 4-Day 5 should add a bounded workflow path using
this exact command and artifact contract before any manifest promotion decision.

## Day 3 Hosted Evidence Inventory

Day 3 inspected the current hosted Windows workflow evidence after PR #231 was
merged to `master`.

### Windows Run Ledger

| Field | Value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `35768806616` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35768806616` |
| Event | `push` |
| Display title | `Merge pull request #231 from jeffreyhorn/sprint-208` |
| Branch | `master` |
| Head SHA | `a98c7b593d24c7514f01f7dbe443fbeeb541f5af` |
| Created | `2026-09-22T18:41:08Z` |
| Updated | `2026-09-22T18:44:14Z` |
| Conclusion | `success` |

### Day 3 Job Inventory

| Job | Started | Completed | Conclusion | QR relevance |
| --- | --- | --- | --- | --- |
| `Windows PowerShell validation ownership` | `2026-09-22T18:41:13Z` | `2026-09-22T18:41:26Z` | `success` | Validates workflow ownership and claim guards, but does not run QR generation. |
| `Windows selected Cholesky comparison freshness (MSVC)` | `2026-09-22T18:41:13Z` | `2026-09-22T18:41:55Z` | `success` | Bounded Cholesky-only selected comparison lane; not QR evidence. |
| `Windows reviewed CMake install/downstream validation path` | `2026-09-22T18:41:13Z` | `2026-09-22T18:42:38Z` | `success` | Static package/install proof; not generated QR comparison evidence. |
| `Windows enforced reviewed CMake consumer subset (MSVC)` | `2026-09-22T18:41:13Z` | `2026-09-22T18:44:13Z` | `success` | General CMake test lane; not selected QR comparison freshness. |

### Day 3 Artifact Metadata

| Field | Value |
| --- | --- |
| Artifact ID | `10712913903` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Size | `4598` bytes |
| Created | `2026-09-22T18:41:51Z` |
| Expired | `false` |
| Download command | `gh run download 35768806616 --name sprint190-windows-selected-comparison-cholesky --dir "$tmpdir"` |

### Downloaded Artifact Membership

The hosted artifact downloaded to a temporary local inspection directory and
contained exactly these files:

| File | QR relevance |
| --- | --- |
| `project_observations.tsv` | Cholesky project observations only. |
| `baseline_observations.tsv` | Cholesky dense-reference observations only. |
| `dependency_status.tsv` | Cholesky baseline dependency status only. |
| `study.tsv` | Six Cholesky rows only. |
| `summary.md` | Cholesky proof summary only. |
| `manifest.tsv` | Cholesky target metadata only. |

No downloaded file path, manifest field, summary field, or `study.tsv` row
mentions `qr-incompatible-ls`, `qr_incompatible_ls`, or
`comparison_qr_overdetermined_incompatible_4x2_*`.

### Artifact Row Evidence

The artifact records:

| Field | Observed value |
| --- | --- |
| Target | `cholesky-spd-tridiag-5` |
| Fixture key | `cholesky_spd_tridiag_5` |
| Platform | `windows-amd64` |
| Compiler | `cmake-probe:Visual Studio 17 2022:Release` |
| Source commit | `a98c7b593d24c7514f01f7dbe443fbeeb541f5af` |
| Worktree state | `clean` |
| Support tier in rows | `local_only` |
| Non-claim marker in rows | includes `no Windows report freshness` |

The `study.tsv` row set is the six selected Cholesky row IDs, not the six QR
incompatible row IDs required by Day 2.

### Day 3 Decision

Day 3 does not provide hosted QR incompatible promotion evidence. The latest
hosted Windows run is useful as proof that the Windows workflow is healthy after
Sprint 208, but it contains only the existing bounded Cholesky selected
comparison artifact. Sprint 209 must therefore add a bounded
`qr-incompatible-ls` workflow path before any manifest promotion can be
considered. Until that happens, `SRT-COMP-QR-INCOMPATIBLE-LS` must remain
Linux/macOS-only, `local_only`, and claim-bounded by `no Windows report
freshness`.

## Day 4 Workflow Implementation Design

Day 4 maps the Day 2 proof command and Day 3 hosted-evidence gap into a precise
workflow implementation plan. No workflow YAML, guard script, manifest, or public
claim text changes are made on Day 4.

### Existing Workflow Boundary

| Surface | Current Day 4 state |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Contains `build-and-test`, `powershell-validation`, `selected-comparison-freshness`, and `install-and-downstream`. |
| Existing selected comparison lane | `selected-comparison-freshness` is Cholesky-only and uploads `sprint190-windows-selected-comparison-cholesky`. |
| Current QR incompatible handling | QR target commands, freshness commands, artifact names, subfamily paths, and upload paths are rejected by workflow guard coverage. |
| Current claim boundary | Windows report freshness remains broad-deferred; QR incompatible remains outside selected Windows freshness. |

### Proposed Job Contract

Add one sibling job rather than overloading the Cholesky job:

| Field | Proposed value |
| --- | --- |
| Job id | `selected-qr-incompatible-comparison-freshness` |
| Job name | `Windows selected QR incompatible comparison freshness (MSVC)` |
| Runner | `windows-2022` |
| Timeout | `20` minutes |
| Checkout | `actions/checkout@v4` |
| Configure step | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` with `shell: pwsh` |
| Build step | `cmake --build build --config Release --target sparse_lu_ortho` with `shell: pwsh` |
| Generator step | Day 2 QR command with `shell: cmd` |
| Freshness step | Day 2 selected QR freshness command with `shell: cmd` |
| Artifact upload | `actions/upload-artifact@v4`, exact six-file path list, `if-no-files-found: error` |

The separate job keeps Cholesky and QR ownership independently reviewable and
prevents a future Cholesky edit from accidentally altering QR proof semantics.

### Exact Commands

```text
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target sparse_lu_ortho
python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

### Exact Upload Contract

| Field | Required value |
| --- | --- |
| Artifact name | `sprint209-windows-selected-comparison-qr-incompatible` |
| Failure behavior | `if-no-files-found: error` |
| Forbidden paths | `build/comparison/**`, `build/comparison/`, broad repository paths, or any non-QR comparison subfamily path. |

The upload path list must be exactly:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

### PowerShell Validator Update List

| Validator area | Required Day 5 update |
| --- | --- |
| Owned marker wording | Add a Sprint 209 marker that says exactly one bounded QR incompatible selected comparison lane is owned, while broad Windows report freshness remains deferred. |
| Forbidden freshness allowlist | Permit only the exact QR job id, artifact name, target command, freshness command, and six upload paths. |
| Job structure checks | Assert runner, timeout, configure/build/generator/freshness/upload steps, shell ownership, and fail-closed upload behavior. |
| QR non-claim markers | Continue requiring public and maintainer docs to say QR evidence is selected-target-only, not broad QR, broad least-squares, broad Windows, package, ABI, performance, release, or state-of-the-art evidence. |
| Drift rejection | Reject wrong QR target spellings, missing `--selected-target`, broad upload paths, missing required files, stale Sprint 203 artifact names, or any second unowned QR command. |

### Workflow Guard Regression List

Day 5 implementation should add or update
`tests/test_selected_comparison_workflow.py` coverage for:

| Regression | Expected failure |
| --- | --- |
| Missing QR job | Windows workflow contract fails with the missing job id. |
| Wrong QR target | Guard reports the exact missing `--target qr-incompatible-ls` contract. |
| Missing MSVC CMake probe options | Guard reports missing `--probe-build-system cmake`, generator, arch, config, or library path. |
| Missing selected freshness filter | Guard reports missing `--selected-target qr-incompatible-ls`. |
| Wrong artifact name | Guard reports the exact expected Sprint 209 QR artifact name. |
| Broad upload path | Guard rejects `build/comparison/**`, `build/comparison/`, and unrelated comparison paths. |
| Missing fail-closed upload | Guard reports missing `if-no-files-found: error`. |
| Missing required file | Guard reports the omitted QR artifact member by exact path. |
| Stale re-deferral-only guard | Guard suite should no longer reject the owned QR job but should still reject any unowned duplicate QR command outside that job. |

### Manifest And Documentation Sequencing

Day 5 should add the workflow and structural guards first. Manifest promotion
must wait until the hosted Windows run produces the exact QR artifact and Day 6
inspects its generated rows. Public documentation should remain conservative
until Day 7-Day 11 decide whether the evidence supports promotion or a renewed
re-deferral.

### Day 4 Decision

Item 209.2 now has an implementation-ready workflow design. The intended change
is one bounded Windows/MSVC selected QR incompatible comparison job with exact
commands and exact artifact upload membership. It does not claim broad Windows
report freshness, broad QR parity, broad least-squares parity, package-manager
support, ABI support, performance superiority, release readiness, or
state-of-the-art status.

## Day 5 Workflow Implementation

Day 5 implemented the bounded hosted Windows/MSVC QR incompatible selected
comparison proof path described on Day 4.

### Changed Files

| File | Day 5 change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added `selected-qr-incompatible-comparison-freshness` with VS 2022 configure/build, exact QR generator command, exact selected freshness command, and fail-closed six-file artifact upload. |
| `scripts/validate_windows_powershell.py` | Added Sprint 209 QR lane ownership constants, workflow comment marker, job runner/structure validation, exact command checks, exact artifact checks, and broad/stale path rejection. |
| `tests/test_selected_comparison_workflow.py` | Converted QR handling from blanket re-deferral rejection to one owned-lane allowance plus duplicate/unowned drift rejection; added QR missing-job, wrong-artifact, broad-upload, and missing-file regressions. |
| `tests/test_validate_windows_powershell.py` | Added QR lane PowerShell validator regressions for target drift, artifact drift, missing fail-closed upload, missing timeout, broad upload, and missing required artifact member. |
| `docs/planning/EPIC_19/SPRINT_209/WORKING_NOTES.md` | Recorded Day 5 implementation evidence and updated item 209.2 disposition. |
| `docs/planning/EPIC_19/SPRINT_209/artifacts/day5-workflow-implementation.md` | Added the Day 5 implementation artifact. |

### Implemented QR Workflow Contract

| Field | Implemented value |
| --- | --- |
| Job id | `selected-qr-incompatible-comparison-freshness` |
| Job name | `Windows selected QR incompatible comparison freshness (MSVC)` |
| Runner | `windows-2022` |
| Timeout | `20` minutes |
| Configure command | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` |
| Build command | `cmake --build build --config Release --target sparse_lu_ortho` |
| Generator command | `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib` |
| Freshness command | `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` |
| Artifact name | `sprint209-windows-selected-comparison-qr-incompatible` |
| Missing-file behavior | `if-no-files-found: error` |

### Implemented Upload Membership

The upload path list is exact and selected-target-only:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

### Guard Coverage Added

| Guard | Day 5 coverage |
| --- | --- |
| Workflow structural guard | Requires the QR job, runner, timeout, configure/build steps, generator command, freshness command, artifact name, fail-closed upload, and six exact paths. |
| Duplicate/unowned QR guard | Allows QR tokens only inside the owned QR job and rejects duplicate target commands, freshness commands, artifact names, subfamily paths, or file paths elsewhere. |
| Broad upload guard | Rejects `build/comparison/**` and broad comparison-root uploads in the QR job. |
| Stale artifact guard | Rejects the old Sprint 203 QR artifact name inside the owned QR job. |
| PowerShell ownership guard | Adds QR configure/build PowerShell snippets to the owned snippet set and keeps other PowerShell steps fail-closed. |
| Manifest boundary guard | Leaves selected manifest metadata unpromoted: no selected row lists `windows` yet. |

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `make windows-powershell-guard` | Passed. |

No `.c` or `.h` files changed, so the C compile/test quality path is not
required for Day 5.

### Remaining Day 5 Boundary

Day 5 wires the hosted proof path, but it does not yet promote
`SRT-COMP-QR-INCOMPATIBLE-LS` manifest metadata or public claim wording. Day 6
must inspect a hosted Windows run and verify the uploaded QR artifact before Day
7 can decide whether manifest promotion is earned or a narrower re-deferral is
needed.

## Day 6 Artifact Inspection Tests

Day 6 hardened selected comparison freshness so a hosted QR artifact cannot pass
with only `study.tsv`. The normalizer now verifies the full selected comparison
required-file bundle beside any generated selected `study.tsv` before treating
the selected row set as freshness evidence.

### Changed Files

| File | Day 6 change |
| --- | --- |
| `scripts/normalize_report_index.py` | Added selected comparison required-file diagnostics keyed by selected target, artifact pattern, and `required_files` manifest metadata. |
| `tests/test_normalize_report_index.py` | Updated synthetic selected comparison fixtures to emit all six required sidecar files and added QR missing-file and unrelated-artifact regressions. |
| `docs/planning/EPIC_19/SPRINT_209/WORKING_NOTES.md` | Recorded Day 6 implementation and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_209/artifacts/day6-artifact-inspection-tests.md` | Added the Day 6 artifact inspection evidence file. |

### Required-File Contract

The selected QR incompatible artifact bundle must contain exactly the files
listed by `SRT-COMP-QR-INCOMPATIBLE-LS`:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

If `study.tsv` exists but any sidecar is missing, selected freshness now emits a
`comparison_required_files: missing_artifact_file` error naming the exact
missing logical path and the selected target remediation command.

### Regression Coverage

| Regression | Evidence |
| --- | --- |
| Windows-style QR artifact paths match selected target filtering. | Existing `test_qr_incompatible_generated_rows_match_windows_artifact_paths()` remains active. |
| Near-match QR artifact paths are rejected. | Existing `test_qr_incompatible_generated_rows_reject_near_match_artifact_paths()` remains active. |
| Windows-style stale QR rows fail clearly. | Existing `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows()` remains active. |
| Duplicate Windows-style QR rows fail clearly. | Existing `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()` remains active. |
| Unexpected Windows-style QR row IDs fail clearly. | Existing `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()` remains active. |
| Missing QR sidecar artifact fails clearly. | Added `test_qr_incompatible_selected_freshness_rejects_missing_required_artifact_file()`. |
| Unrelated missing Cholesky sidecar does not broaden QR selected-target freshness. | Added `test_qr_incompatible_selected_freshness_ignores_unrelated_missing_artifacts()`. |

### Local QR Generation Evidence

| Command | Result |
| --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed and wrote all six files under `build/comparison/qr_incompatible_ls/`. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed with six QR incompatible generated rows fresh to current `HEAD`. |

The generated `study.tsv` row set was:

- `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`
- `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1`
- `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1`
- `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1`
- `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1`
- `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1`

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py` | Passed. |

### Day 6 Decision

Item 209.3 is complete for branch-local artifact inspection coverage. The repo
now has selected-target QR required-file diagnostics, Windows-style QR path
coverage, and local QR bundle proof. Manifest promotion still waits for hosted
Windows artifact evidence from the new Day 5 workflow lane.

## Day 7 Manifest Decision Criteria

Day 7 defines the objective criteria for Day 8. It does not edit
`tests/corpus/manifests/selected_report_targets.tsv`; the current QR row remains
Linux/macOS-only until hosted Windows evidence is available and inspected.

### Minimum Promotion Evidence

All promotion gates must pass before adding Windows metadata to
`SRT-COMP-QR-INCOMPATIBLE-LS`:

| Gate | Required evidence |
| --- | --- |
| Hosted workflow run | A GitHub Actions `Windows CI` run for the reviewed branch or reviewed merge commit completes successfully. |
| QR job identity | The run includes `selected-qr-incompatible-comparison-freshness` named `Windows selected QR incompatible comparison freshness (MSVC)`. |
| Generator command | The hosted job runs `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib`. |
| Freshness command | The hosted job runs `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`. |
| Artifact identity | The run uploads `sprint209-windows-selected-comparison-qr-incompatible`. |
| Artifact membership | The artifact contains only the exact six QR files under `build/comparison/qr_incompatible_ls/`. |
| Row set | `study.tsv` contains exactly the six `comparison_qr_overdetermined_incompatible_4x2_*` rows. |
| Row status | All six rows are `status=pass`, source commit matches the reviewed commit, worktree state is `clean`, and generated freshness reports no QR selected errors. |
| Support-tier alignment | Manifest, report-family/generated rows, and docs agree on selected hosted Windows evidence without implying broad Windows report freshness. |
| Guard alignment | Workflow, PowerShell, normalizer, manifest, and docs guards pass after the metadata change. |

### Promotion Manifest Contract

If all gates pass, Day 8 may update only the selected QR incompatible row with
these exact promoted values:

| Field | Promoted value |
| --- | --- |
| `target_id` | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| `family` | `comparison` |
| `subfamily` | `qr_incompatible_ls` |
| `target_key` | `qr-incompatible-ls` |
| `support_tier` | `hosted_selected` |
| `artifact_pattern` | `build/comparison/qr_incompatible_ls/study.tsv` |
| `required_files` | `project_observations.tsv;baseline_observations.tsv;dependency_status.tsv;study.tsv;summary.md;manifest.tsv` |
| `expected_rows` | `6` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml;.github/workflows/windows-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness;selected-qr-incompatible-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness;sprint209-windows-selected-comparison-qr-incompatible` |
| `workflow_platforms` | `linux;macos;windows` |
| `introduced_in` | Append `Sprint 209 Day 8` or a more precise Day 8 evidence marker. |

The promoted claim scope should say:

```text
Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.
```

The promoted non-claims should retain:

```text
no broad QR parity;no broad least-squares parity;no raw QR basis identity;no Q sign or orientation claim;no global rank-threshold policy;no broad rank-deficient solve claim;no NumPy parity;no SciPy parity;no LAPACK parity;no SuiteSparse parity;no Eigen parity;no broad Windows report freshness;no package-manager proof;no shared-library ABI proof;no performance superiority;no state-of-the-art claim
```

### Required Promotion Tests

If Day 8 promotes, it must add manifest-contract tests that assert:

- exact QR row identity, family, subfamily, target key, artifact pattern, row
  count, required files, and expected row IDs;
- exact Linux/macOS/Windows workflow metadata order;
- exact Sprint 209 QR artifact name;
- `support_tier=hosted_selected`;
- promoted claim scope includes Linux, macOS, and Windows hosted lanes;
- non-claims equal the full promoted tuple and contain
  `no broad Windows report freshness`, not `no Windows report freshness`;
- unrelated selected comparison rows still reject accidental Windows metadata.

### Re-Deferral Conditions

Day 8 must keep QR incompatible re-deferred if any of these conditions applies:

| Condition | Required outcome |
| --- | --- |
| No hosted Windows run exists for the reviewed QR workflow. | Keep manifest Linux/macOS-only and record hosted evidence as pending. |
| QR job missing, skipped, cancelled, timed out, or failed. | Keep manifest Linux/macOS-only and record the job conclusion. |
| Artifact missing or inaccessible. | Keep manifest Linux/macOS-only and record the missing artifact name. |
| Artifact has broad paths, extra subfamilies, missing required files, or stale Sprint 203 artifact naming. | Keep manifest Linux/macOS-only and harden workflow/guard coverage. |
| `study.tsv` row IDs are missing, duplicate, unexpected, stale, skipped, deferred, or failed. | Keep manifest Linux/macOS-only and record normalizer diagnostics. |
| Generated row support-tier or non-claim wording still contradicts selected Windows promotion. | Keep manifest Linux/macOS-only or coordinate generator/report-family/docs changes before promotion. |
| Docs or guards cannot be made consistent in the same branch. | Keep manifest Linux/macOS-only and add residual queue follow-up. |

### Re-Deferral Manifest Contract

If promotion is not earned, Day 8 should keep the current QR row exactly:

| Field | Retained value |
| --- | --- |
| `support_tier` | `local_only` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `non_claims` | Retain `no Windows report freshness` plus all current QR, external-library, package, ABI, performance, and state-of-the-art non-claims. |

Re-deferral guards should reject `.github/workflows/windows-ci.yml`,
`selected-qr-incompatible-comparison-freshness`, `sprint209-windows-selected-comparison-qr-incompatible`,
or `windows` in the QR manifest row until hosted evidence is explicitly
accepted.

### Decision Surface Map

| Surface | Promotion action | Re-deferral action |
| --- | --- | --- |
| Selected manifest | Add Windows metadata and promoted wording for QR row only. | Preserve Linux/macOS-only QR row and strengthen absence guards if needed. |
| Report-family/generated rows | Align generated support tier and non-claims with selected hosted evidence. | Preserve `local_only` and `no Windows report freshness`. |
| Workflow | Keep bounded QR job. | Keep bounded QR job as evidence-collection lane or disable/remove it only if it proves unsafe. |
| Normalizer | Keep required-file and row-set checks. | Keep required-file and row-set checks. |
| PowerShell/workflow guards | Permit only the owned QR lane and exact artifact. | Permit evidence collection but reject manifest/docs promotion unless explicitly chosen. |
| Public docs | State selected Windows QR freshness only if all gates pass. | State QR remains outside selected Windows freshness. |
| Residual queue | Remove or close the QR promotion residual. | Carry forward hosted evidence, artifact, or alignment blocker. |

### Day 7 Decision

Day 7 completes objective criteria for item 209.4. The current evidence is not
yet enough to promote because the new Day 5 hosted job has not produced an
inspected Windows artifact in this branch-local work. Day 8 must therefore fetch
hosted evidence if available, then either apply the exact promotion contract or
record a claim-safe re-deferral using the retained manifest contract above.

## Day 8 Manifest Decision

Day 8 applies the Day 7 criteria to the available evidence and keeps
`SRT-COMP-QR-INCOMPATIBLE-LS` re-deferred for Windows manifest metadata.

### Hosted Evidence Check

| Command | Result |
| --- | --- |
| `gh run list --workflow "Windows CI" --branch sprint-209 --limit 10 --json databaseId,displayTitle,event,headSha,createdAt,updatedAt,status,conclusion,url` | Returned `[]`; no hosted Windows run exists for this branch. |

Because no hosted run exists, there is no hosted
`selected-qr-incompatible-comparison-freshness` job conclusion, no uploaded
`sprint209-windows-selected-comparison-qr-incompatible` artifact, and no hosted
Windows QR row-set inspection evidence. This fails the first Day 7 promotion
gate.

### Decision

| Decision field | Day 8 result |
| --- | --- |
| Promotion decision | Re-defer selected Windows QR incompatible freshness. |
| Manifest edit | No edit to `tests/corpus/manifests/selected_report_targets.tsv`. |
| QR support tier | Retain `local_only`. |
| QR workflow metadata | Retain Linux/macOS-only workflow files, jobs, artifacts, and platforms. |
| QR non-claims | Retain `no Windows report freshness` and the full QR/external-library/package/ABI/performance/state-of-the-art boundary. |
| Day 5 workflow lane | Keep as evidence-collection lane for the future hosted run. |

### Added Absence Guards

`tests/test_selected_report_targets_manifest.py` now has explicit re-deferral
regressions that reject:

- `.github/workflows/windows-ci.yml` in the QR selected manifest row;
- `selected-qr-incompatible-comparison-freshness` in the QR selected manifest
  row;
- `sprint209-windows-selected-comparison-qr-incompatible` in the QR selected
  manifest row;
- `windows` in the QR selected manifest row.

These checks make the distinction clear: Sprint 209 may include an owned
workflow evidence-collection lane, but the source-of-truth selected manifest
cannot claim Windows QR freshness until hosted evidence is inspected and Day 7
promotion gates pass.

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

### Day 8 Decision Boundary

Item 209.4 is complete with one evidence-backed decision path: re-deferral.
The manifest remains claim-safe, and stronger selected Windows QR incompatible
freshness remains unclaimed until a future hosted Windows run produces the exact
Sprint 209 QR artifact and passes artifact inspection.

## Day 9 Guard Integration

Day 9 connects the Day 8 QR re-deferral decision to the central Windows
PowerShell guard so workflow evidence collection cannot accidentally promote
manifest evidence.

### Guard Updates

| Surface | Day 9 update |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added exact `SRT-COMP-QR-INCOMPATIBLE-LS` manifest constants for family, subfamily, target key, artifact pattern, generator command, support tier, workflow tuple, required files, row IDs, claim scope, and non-claims. |
| `scripts/validate_windows_powershell.py` | `validate_manifest_windows_deferral()` now rejects Sprint 209 QR workflow file, job, artifact, or `windows` platform metadata in the QR row while re-deferred. |
| `scripts/validate_windows_powershell.py` | Non-QR rows now fail if they reference the Sprint 209 QR Windows job or artifact while QR freshness remains re-deferred. |
| `tests/test_validate_windows_powershell.py` | Added negative regressions for QR workflow file, job, artifact, platform, identity, required-file, non-claim, and non-QR metadata leakage drift. |

### Integrated Decision Boundary

The Day 5 Windows workflow lane remains valid as an evidence-collection path.
The selected target manifest remains Linux/macOS-only until hosted Windows QR
evidence exists and all Day 7 promotion gates are satisfied. This prevents a
partial promotion where the workflow exists but the source-of-truth manifest,
support tier, and non-claims do not have matching hosted evidence.

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `make windows-powershell-guard` | Passed. |

### Day 9 Close

Item 209.5 is partially complete for guard integration. Public README,
INSTALL, maintainer-guide, corpus, and schema wording remain intentionally
deferred to Day 10 and Day 11 so those claim surfaces can be calibrated together
after the guard contract is stable.

## Day 10 Public Docs Calibration

Day 10 updates the public-facing README and INSTALL surfaces to match the Day 8
decision and Day 9 guard contract.

### Public Docs Updates

| Surface | Day 10 update |
| --- | --- |
| `README.md` report-index guidance | States that Sprint 209 adds a bounded Windows QR incompatible evidence-collection lane for `qr-incompatible-ls`, while the selected manifest remains Linux/macOS-only and `local_only` until hosted Windows run evidence and exact QR artifact inspection support promotion. |
| `README.md` selected comparison section | Keeps the QR incompatible least-squares target outside Windows selected freshness and lists the promotion prerequisites: hosted MSVC probe evidence, selected artifact review, selected-target manifest metadata, generated support tier, and generated non-claim wording. |
| `INSTALL.md` support/readiness matrix | Changes the Windows QR incompatible row from plain deferred to guarded-workflow deferred, naming Sprint 209 workflow/PowerShell and manifest guards as evidence owners without promoting support. |
| `INSTALL.md` platform table | Adds the Sprint 209 QR evidence-collection lane to the Windows row while preserving broad Windows, package-manager, shared-library, ABI, performance, release, and state-of-the-art non-claims. |

### Guard Updates

| Surface | Day 10 update |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Public-doc claim-boundary markers now require the Sprint 209 QR evidence-collection wording in README and INSTALL. |
| `scripts/validate_windows_powershell.py` | Unsupported-claim regex now rejects QR incompatible Windows selected freshness promotion wording. |
| `tests/test_validate_windows_powershell.py` | Added regressions for missing public QR markers and positive QR Windows selected freshness wording. |

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `make windows-powershell-guard` | Passed. |
| `make docs-check` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

### Day 10 Close

The public-doc claim surfaces now distinguish three states: existing Linux/macOS
selected comparison evidence, the bounded Windows Cholesky guarded path, and
the Sprint 209 QR evidence-collection lane that remains re-deferred for
selected Windows freshness.

## Day 11 Maintainer And Corpus Docs

Day 11 aligns maintainer-facing docs, corpus/schema interpretation, and Epic 19
planning status with the Day 8 re-deferral decision and Day 10 public wording.

### Documentation Updates

| Surface | Day 11 update |
| --- | --- |
| `docs/maintainer_guide.md` | Adds Sprint 209 to Windows/PowerShell ownership and selected comparison freshness rows; records the QR lane as evidence collection only while the selected manifest remains Linux/macOS-only and `local_only`. |
| `docs/maintainer_guide.md` | Updates selected comparison interpretation so maintainers treat the Sprint 209 QR lane like guarded workflow evidence, not promoted selected Windows freshness. |
| `tests/corpus/README.md` | Adds the Sprint 209 QR evidence-collection lane to corpus interpretation and keeps broad Windows report freshness, selected oracle/benchmark freshness, and local-only family claims out of scope. |
| `tests/corpus/schemas/report_index_fields.md` | Records that the QR incompatible selected row remains Linux/macOS-only and `local_only` until hosted run evidence, exact artifact inspection, manifest metadata, generated support tier, and non-claim wording move together. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Updates the then-current Day 11 status snapshot for Sprint 209 and leaves later sprints pending. |

### Guard Updates

| Surface | Day 11 update |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Claim-boundary markers now require Sprint 209 QR evidence-collection wording in maintainer guide, corpus README, and report-index schema docs. |
| `tests/test_validate_windows_powershell.py` | Added marker-removal regressions for maintainer guide, corpus README, and report-index schema wording. |

### Validation Run

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `make windows-powershell-guard` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `make docs-check` | Passed. |
| `make support-docs-guard` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

### Day 11 Close

Item 209.5 is complete for public and maintainer-facing claim calibration. The
Sprint 209 QR lane is documented as an evidence-collection path only, with
selected Windows QR freshness still re-deferred until hosted proof and all
manifest/generated claim surfaces promote together.

## Day 12 Focused Validation

Day 12 runs focused validation for the modified non-C surfaces and records the
hosted-only gap separately from local pass evidence.

### Pass Evidence

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed. | Wrote the six local QR incompatible comparison files and reported project-vs-baseline comparison passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed. | Reported freshness ok for `17` comparison rows, including all six generated QR incompatible rows fresh to current `HEAD`. |
| `python3 tests/test_normalize_report_index.py` | Passed. | Covers selected QR artifact inspection, required-file diagnostics, Windows-style paths, stale/missing/unexpected rows, and unrelated missing artifacts. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. | Confirms QR manifest re-deferral and selected target contract remain intact. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. | Confirms the workflow owns exactly the bounded QR lane and rejects broad upload or stale metadata drift. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. | Confirms workflow/PowerShell ownership, QR manifest re-deferral, and public/maintainer/corpus/schema claim markers. |
| `make windows-powershell-guard` | Passed. | Runs the Windows PowerShell guard regression suite. |
| `make docs-check` | Passed. | Regenerated local Doxygen HTML and passed API docs coverage. |
| `make support-docs-guard` | Passed. | Support/readiness quick-reference docs remain coherent. |
| `python3 tests/test_run_external_comparison.py` | Passed. | External comparison generator regressions remain green. |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_comparison_workflow.py tests/test_selected_report_targets_manifest.py tests/test_validate_windows_powershell.py tests/test_run_external_comparison.py` | Passed. | Python syntax check for modified scripts/tests and adjacent generator regression file. |
| `git diff --check` | Passed. | No whitespace errors in the current diff. |

### Unavailable Hosted Evidence

| Evidence | Status | Interpretation |
| --- | --- | --- |
| Hosted Windows CI run for branch `sprint-209` with QR artifact `sprint209-windows-selected-comparison-qr-incompatible` | Unavailable in current branch evidence. | This remains the reason selected Windows QR freshness is re-deferred; local validation and workflow structure are not counted as hosted promotion proof. |

### Day 12 Close

Focused validation is complete for all modified non-C surfaces. No `.c` or
`.h` files were modified, so the full C quality gate is deferred unless later
Sprint 209 work changes C sources or headers.

## Day 13 Integrated Validation

Day 13 broadens validation beyond the focused QR path and reviews adjacent
claim-boundary surfaces before closeout.

### Integrated Validation Results

| Command | Result | Notes |
| --- | --- | --- |
| `make package-manager-deferral-guard` | Passed. | Package-manager public non-claims and selected Homebrew local-proof boundary remain intact. |
| `bash scripts/static_package_deferral_check.sh` | Passed. | Static-first package, no shared export/ABI metadata, and Windows package non-claim checks passed. |
| `make api-docs-freshness` | Passed. | API docs coverage, local-only generated HTML checks, workflow non-publication checks, and API routing checks passed. |
| `make report-index-comparison-freshness` | Passed. | Regenerated all selected local comparison outputs and reported freshness ok for `46` comparison rows. |
| `make windows-powershell-guard` | Passed. | Windows workflow/PowerShell ownership, QR re-deferral, and claim-boundary guard suite passed. |
| `make support-docs-guard` | Passed. | Support/readiness quick-reference docs remain coherent. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. | Manifest contract remains re-deferred for QR incompatible and guarded for future promotion drift. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. | Workflow guard confirms the Sprint 209 QR lane remains exact and bounded. |
| `git diff --name-only -- '*.c' '*.h'` | Passed. | Returned no changed C sources or headers; full C quality gate is not required by the sprint rule. |
| Stale-status and overclaim wording scan | Passed. | No obsolete Sprint 209 pending-range or positive QR/Windows overclaim wording was found; remaining state-of-the-art mentions are retained non-claims or future blueprint planning. |
| `git diff --check` | Passed. | No whitespace errors. |

### Review Hardening Notes

| Surface | Day 13 review result |
| --- | --- |
| Selected QR manifest state | `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`; no Windows workflow metadata is present in the manifest row. |
| Windows workflow state | The Sprint 209 QR job exists only as a bounded evidence-collection lane with exact generator/freshness commands and exact six-file artifact upload. |
| Docs and schema state | README, INSTALL, maintainer guide, corpus README, report-index schema, and Epic 19 planning status consistently describe QR selected Windows freshness as re-deferred. |
| Adjacent package/API claims | Package-manager, static/shared ABI, generated API publication, performance, release, external-library parity, and state-of-the-art claims remain unearned and guarded. |

### Day 13 Close

Integrated validation is complete for the current non-C change set. No blocking
local validation failures remain. The only remaining evidence gap is hosted
Windows QR run/artifact inspection, which is deliberately recorded as the
reason for selected Windows QR re-deferral.

## Day 14 Closeout Review

Day 14 reconciles Sprint 209 items, artifacts, implementation changes,
validation, residuals, and PR-ready handoff notes.

### Final Item Disposition

| Item | Final disposition |
| --- | --- |
| 209.1 MSVC Probe Design | Complete. The hosted MSVC/CMake `qr-incompatible-ls` command, expected six-file artifact layout, and row contract are recorded in Day 2, and Day 3 records that no hosted Windows QR artifact existed for promotion. |
| 209.2 Workflow Implementation | Complete. `.github/workflows/windows-ci.yml` now has one bounded QR incompatible evidence-collection lane with exact generator/freshness commands and exact fail-closed upload membership. |
| 209.3 Artifact Inspection Tests | Complete. `normalize_report_index.py` checks selected comparison required sidecar files, and normalizer regressions cover missing QR required artifacts and unrelated missing artifacts. |
| 209.4 Manifest Decision | Complete with re-deferral. `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`; guards reject Sprint 209 QR Windows workflow/job/artifact/platform metadata while re-deferred. |
| 209.5 Docs And Claim Guards | Complete. README, INSTALL, maintainer guide, corpus README, report-index schema, and Epic 19 planning status now describe Sprint 209 as evidence collection only, with guarded non-claims. |
| 209.6 Validation And Closeout | Complete. Focused and integrated validation passed for the non-C change set, and hosted Windows QR artifact absence is recorded as residual evidence, not pass evidence. |

### Final Changed Surfaces

| Area | Files |
| --- | --- |
| Windows workflow | `.github/workflows/windows-ci.yml` |
| Normalizer and Windows guards | `scripts/normalize_report_index.py`, `scripts/validate_windows_powershell.py` |
| Guard and regression tests | `tests/test_normalize_report_index.py`, `tests/test_selected_comparison_workflow.py`, `tests/test_selected_report_targets_manifest.py`, `tests/test_validate_windows_powershell.py` |
| User and maintainer docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md` |
| Planning docs | `docs/planning/EPIC_19/PROJECT_PLAN.md`, `docs/planning/EPIC_19/SPRINT_209/PLAN.md`, `docs/planning/EPIC_19/SPRINT_209/WORKING_NOTES.md`, Day 1-Day 14 artifacts |

### Final Validation Ledger

| Command | Result |
| --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_run_external_comparison.py` | Passed. |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_comparison_workflow.py tests/test_selected_report_targets_manifest.py tests/test_validate_windows_powershell.py tests/test_run_external_comparison.py` | Passed. |
| `make windows-powershell-guard` | Passed. |
| `make docs-check` | Passed. |
| `make support-docs-guard` | Passed. |
| `make package-manager-deferral-guard` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `make api-docs-freshness` | Passed. |
| `make report-index-comparison-freshness` | Passed. |
| `git diff --name-only -- '*.c' '*.h'` | Passed with no changed C sources or headers. |
| Stale/overclaim `rg` scan | Passed with no obsolete Sprint 209 pending-range wording or positive QR/Windows freshness promotion wording. |
| `git diff --check` | Passed. |

### Residuals

| Residual | Handoff |
| --- | --- |
| Hosted Windows QR run/artifact inspection is unavailable for branch `sprint-209`. | Keep selected Windows QR freshness re-deferred until a hosted Windows run produces the exact `sprint209-windows-selected-comparison-qr-incompatible` artifact and the six required QR files are inspected. |
| `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`. | Any future promotion must update manifest workflow metadata, support tier, generated non-claim wording, public and maintainer docs, schema/corpus wording, and guards together. |
| Broad QR, broad least-squares, broad Windows report freshness, package-manager, shared-library ABI, performance, release, external-library parity, and state-of-the-art claims remain unearned. | Preserve the retained non-claims until separate evidence and approval exist. |

### PR-Ready Summary

Sprint 209 adds a bounded Windows/MSVC QR incompatible comparison
evidence-collection lane for `qr-incompatible-ls`, strengthens selected artifact
inspection and manifest re-deferral guards, and calibrates public and maintainer
claim surfaces. Because no hosted Windows QR run artifact is available for
inspection, selected Windows QR freshness remains re-deferred and the selected
target manifest remains Linux/macOS-only and `local_only`.

### Day 14 Close

Sprint 209 is ready for retrospective creation, final commit, and PR review.
No local validation blockers remain for the current non-C change set.
