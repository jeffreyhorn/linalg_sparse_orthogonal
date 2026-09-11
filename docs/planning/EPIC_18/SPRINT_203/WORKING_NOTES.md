# Sprint 203 Working Notes: Windows QR Incompatible Comparison Promotion

## Sprint Goal

Promote the QR incompatible comparison target on Windows only if MSVC/CMake
generation, artifacts, manifest metadata, and docs support it.

## Day 1: Windows QR Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial artifact |
| --- | --- | --- |
| 203.1 MSVC Probe | Run or reproduce exactly `qr-incompatible-ls` under the Windows MSVC/CMake path and record proof output or scoped failures. | MSVC probe design and execution ledger. |
| 203.2 Generator Fixes | Fix only selected Windows generator, path, or CMake issues found by the probe; preserve Linux/macOS selected comparison behavior. | Generator/CMake fix design artifact. |
| 203.3 Manifest Promotion | Promote Windows metadata only if hosted evidence proves the exact QR incompatible target and selected artifact set. | Manifest promotion or re-deferral decision record. |
| 203.4 Normalizer And Workflow Tests | Add Windows QR path, row-filtering, dependency-status, stale-output, and workflow artifact-drift tests. | Normalizer/workflow regression matrix. |
| 203.5 Docs Calibration | Update public, corpus, install, and maintainer docs with exact selected Windows QR boundaries. | Claim-surface checklist and docs guard updates. |
| 203.6 Validation | Run selected comparison, normalizer, manifest, workflow, QR-focused, Windows guard, docs, and full C gate if required. | Integrated validation matrix. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 203 is allocated 168 hours to decide whether Windows can promote the existing QR incompatible comparison target after MSVC/CMake proof, generator/path fixes, manifest decision, tests, docs, and validation. |
| `docs/planning/EPIC_18/SPRINT_203/PLAN.md` | Day 1 is intake only; the MSVC probe is designed on Day 2 and run or reproduced on Day 3 before implementation or manifest promotion. |
| `docs/planning/EPIC_17/SPRINT_191/RETROSPECTIVE.md` | Sprint 191 added exactly one bounded selected comparison family, `qr-incompatible-ls`, and explicitly left Windows selected comparison metadata unchanged because no MSVC proof was added. |
| `docs/planning/EPIC_18/SPRINT_199/RETROSPECTIVE.md` | Sprint 199 reviewed hosted Windows Cholesky evidence, hardened Windows path handling, but re-deferred selected manifest promotion because generated support tier and claim semantics still remained local-only. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | E18-RQ-006 remains open for Windows QR incompatible comparison promotion and requires MSVC/CMake proof, artifact inspection, manifest metadata alignment, and retained QR/Windows non-claims. |
| `tests/corpus/manifests/selected_report_targets.tsv` | `SRT-COMP-QR-INCOMPATIBLE-LS` exists for Linux/macOS selected comparison freshness only; the non-claims include no Windows report freshness, no package-manager proof, no shared-library ABI proof, no performance superiority, and no state-of-the-art claim. |
| `tests/corpus/manifests/report_families.tsv` | The `qr_incompatible_ls` report family remains generated-local, local-only evidence with no hosted CI proof from generated-local row metadata and no broad platform portability proof. |
| `.github/workflows/ci.yml` and `.github/workflows/macos-ci.yml` | Linux and macOS selected comparison workflows already generate/upload `qr-incompatible-ls` artifacts as part of selected comparison freshness. |
| `.github/workflows/windows-ci.yml` | Windows selected comparison workflow currently owns the bounded `cholesky-spd-tridiag-5` path and artifact only, not QR incompatible evidence. |
| `scripts/run_external_comparison.py` | The selected target key `qr-incompatible-ls` is the generator entry point that must run or fail diagnostically under the Windows MSVC/CMake probe path. |
| `scripts/normalize_report_index.py` | Target-specific selected comparison freshness and artifact-path normalization are existing owner surfaces that must not silently drop Windows-style QR artifacts. |
| `tests/test_run_external_comparison.py` | Runner tests already cover QR incompatible target behavior, malformed output, reference failures, and generated target metadata on the local path. |
| `tests/test_selected_comparison_workflow.py` | Workflow guard tests already enforce Linux/macOS selected comparison artifacts and Windows Cholesky workflow boundaries; Sprint 203 may extend this only for exact QR evidence. |

### Current QR Incompatible Comparison Inventory

| Surface | Current Day 1 state |
| --- | --- |
| Selected target id | `SRT-COMP-QR-INCOMPATIBLE-LS`. |
| Target key | `qr-incompatible-ls`. |
| Report family | `comparison` / `qr_incompatible_ls`. |
| Fixture | `qr_overdetermined_incompatible_4x2`. |
| Generator command | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls`. |
| Artifact directory | `build/comparison/qr_incompatible_ls/`. |
| Primary artifact | `build/comparison/qr_incompatible_ls/study.tsv`. |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| Expected selected rows | Six rows: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Current selected platforms | Linux and macOS only. |
| Current Windows status | Not promoted; no Windows metadata for this selected target. |
| Current support tier | `local_only` selected comparison evidence. |
| Claim boundary | Fixture-local QR incompatible least-squares comparison only; no broad QR parity, broad least-squares parity, external-library ecosystem parity, Windows report freshness, package-manager proof, shared-library ABI proof, performance superiority, or state-of-the-art claim. |

### Item-To-Evidence Map

| Item | Evidence needed before closeout | Initial owner surfaces |
| --- | --- | --- |
| 203.1 | MSVC/CMake probe command, output, generated artifact bundle, and proof or failure classification. | `scripts/run_external_comparison.py`, CMake probe template, Windows workflow logs/artifacts. |
| 203.2 | Minimal generator/path/CMake fixes with preservation notes for Linux/macOS behavior. | Runner script, generated CMake project text, path handling helpers, runner tests. |
| 203.3 | Manifest promotion or explicit re-deferral backed by hosted Windows evidence and exact non-claims. | `tests/corpus/manifests/selected_report_targets.tsv`, manifest tests, residual queue, project plan. |
| 203.4 | Windows QR artifact path, selected filtering, stale output, dependency status, and workflow drift regressions. | Normalizer tests, selected comparison workflow tests, external comparison runner tests. |
| 203.5 | Public and maintainer claim surfaces calibrated to the final disposition. | README, INSTALL, corpus docs, schema docs, maintainer guide, docs guard tests. |
| 203.6 | Focused validation commands pass; full C gate passes if `.c` or `.h` files changed. | Python tests, selected freshness commands, Windows PowerShell guard, QR focused tests, docs checks. |

### Owner Surface Inventory

| Surface | Sprint 203 relevance |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Candidate hosted Windows selected QR comparison lane; currently only promotes the Cholesky workflow path. |
| `.github/workflows/ci.yml` | Existing Linux selected comparison workflow must remain unchanged except for compatibility-preservation expectations. |
| `.github/workflows/macos-ci.yml` | Existing macOS selected comparison workflow must remain unchanged except for compatibility-preservation expectations. |
| `scripts/run_external_comparison.py` | Generates selected QR incompatible artifacts and probe CMake project. |
| `scripts/normalize_report_index.py` | Owns selected-target freshness filtering, artifact-path matching, and stale diagnostics. |
| `scripts/validate_windows_powershell.py` | Owns Windows workflow/PowerShell validation boundaries and may need exact QR guard additions if promotion proceeds. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Source of truth for selected target platform metadata, support tier, claim scope, and non-claims. |
| `tests/corpus/manifests/report_families.tsv` | Source of truth for generated-local report family boundaries. |
| `tests/test_run_external_comparison.py` | Runner and target-specific comparison regression owner. |
| `tests/test_normalize_report_index.py` | Freshness, selected-target filtering, and path diagnostic regression owner. |
| `tests/test_selected_report_targets_manifest.py` | Manifest promotion and non-claim regression owner. |
| `tests/test_selected_comparison_workflow.py` | Workflow job, artifact upload, and selected target drift guard owner. |
| `tests/test_validate_windows_powershell.py` | Windows PowerShell guard regression owner. |
| `tests/test_qr.c` and QR helper tests | Focused QR behavior preservation candidates if implementation touches solver or fixture behavior. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer claim-surface owners. |
| `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md` | Corpus and report-index interpretation owners. |

### Initial Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Day 1 changes planning documentation only. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Deferred. | Required before or during MSVC probe comparison; local non-Windows run is not Windows proof. |
| MSVC/CMake selected probe command | Deferred. | Day 2 defines command and Day 3 runs or records hosted evidence. |
| `python3 tests/test_run_external_comparison.py` | Deferred. | Required once generator or comparison fixtures change. |
| `python3 tests/test_normalize_report_index.py` | Deferred. | Required once selected freshness/path diagnostics change. |
| `python3 tests/test_selected_report_targets_manifest.py` | Deferred. | Required if selected manifest metadata changes or promotion/re-deferral tests are added. |
| `python3 tests/test_selected_comparison_workflow.py` | Deferred. | Required if workflow metadata or artifact upload guard behavior changes. |
| `make windows-powershell-guard` | Deferred. | Required if Windows workflow or PowerShell validation ownership changes. |
| Focused QR tests | Deferred. | Required if solver, QR fixture, or target behavior changes. |
| `make docs-check` or docs guards | Deferred. | Required once public or maintainer docs change. |
| Hosted Windows CI evidence review | Deferred. | Required for any promoted Windows selected QR freshness claim. |
| `make format && make lint && make test` | Not required for Day 1. | Required later if `.c` or `.h` files change. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Treating workflow existence as promotion | Sprint 199 showed that a green Windows workflow path is insufficient if manifest support tier and generated claim semantics remain local-only. | Require hosted artifact review, manifest metadata, generated support semantics, docs, and guards to agree before promotion. |
| Windows path separators drop selected rows | Backslashes or absolute drive-letter paths can cause selected artifact filtering to skip rows. | Reuse and extend path-normalization tests for the exact QR incompatible artifact path. |
| QR evidence broadens into QR parity | The fixture proves one incompatible least-squares comparison, not broad QR correctness or ecosystem parity. | Keep selected fixture id, six row ids, and explicit QR/ecosystem non-claims adjacent to all docs and manifest wording. |
| Generator fix broadens comparison family | Changing the runner for Windows can accidentally affect all targets. | Map fixes to target-scoped tests and run existing Linux/macOS comparison regressions. |
| Dependency status hides build failure | A missing reference helper and a project build failure need different diagnostics. | Preserve separate dependency, project status, baseline status, and study row checks. |
| Hosted evidence is unavailable during branch work | Local simulation cannot prove hosted Windows runner behavior. | Carry promotion as blocked/re-deferred until hosted Windows artifact evidence is reviewed. |
| Cholesky Windows metadata drift | Existing Windows Cholesky guard evidence should not be disturbed by QR work. | Add preservation checks for existing Cholesky workflow/artifact names and selected target metadata. |
| C gate omitted after implementation changes | Later generator or test work could touch `.c` or `.h` files. | Re-check changed file types before validation; run full `make format && make lint && make test` if any C/header file changed. |

### Explicit Non-Goals

- No broad Windows report freshness claim.
- No Windows selected oracle freshness claim.
- No Windows selected benchmark freshness claim.
- No broad QR parity, broad least-squares parity, or external-library ecosystem
  parity claim.
- No raw QR basis identity, Q sign/orientation, or global rank-threshold
  policy claim.
- No package-manager, shared-library ABI, release proof, performance
  superiority, platform parity, or state-of-the-art claim.
- No promotion of unselected comparison families or broad report-index
  freshness.

### Day 1 Open Questions For Day 2

1. Which exact MSVC/CMake command form should be considered the canonical
   probe for `qr-incompatible-ls`?
2. Does the existing runner render all include, library, and build paths in a
   Windows-safe form for a generated CMake project?
3. Should the Windows workflow add a second selected comparison target or keep
   one selected-comparison job with multiple explicitly guarded selected
   targets?
4. Which artifact name should be reserved for Windows QR incompatible evidence
   if promotion proceeds?
5. What exact generated support tier and non-claim wording must change before
   manifest promotion is allowed?

### Day 1 Completion Notes

- Every Sprint 203 item has an initial evidence path and owner surface.
- Prior Sprint 191 and Sprint 199 decisions are recorded before
  implementation starts.
- Broad Windows and broad QR comparison claims remain explicitly out of scope.

## Day 2: MSVC Probe Design

### Selected Target File And Row Inventory

| Field | Day 2 probe design value |
| --- | --- |
| Target id | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Target key | `qr-incompatible-ls` |
| Subfamily | `qr_incompatible_ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Operation | `least_squares_solve` |
| Generator command | `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib` |
| Artifact directory | `build/comparison/qr_incompatible_ls/` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Expected row count | 6 |
| Expected row ids | `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`; `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1`; `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1`; `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1`; `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1`; `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` |
| Dependency helper | `tests/qr_external_dense_reference.py` |
| Expected success message | `external-comparison: qr-incompatible-ls project-vs-baseline comparison passed` |

### Canonical Windows Probe Command

The Day 2 canonical hosted Windows probe should mirror the existing Sprint 190
Cholesky workflow shape while changing only the selected target and artifact
family:

```text
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target sparse_lu_ortho
python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

The probe is intentionally target-specific. It does not authorize adding broad
Windows comparison freshness, Windows oracle freshness, Windows benchmark
freshness, or any unselected comparison family to the selected manifest.

### CMake Probe Mapping

| Probe input | Source or expected representation |
| --- | --- |
| Temporary project source | Generated by `scripts/run_external_comparison.py` for the selected target probe. |
| Probe build system | `--probe-build-system cmake`, not direct compiler mode. |
| CMake generator | `Visual Studio 17 2022`. |
| CMake architecture | `x64`. |
| CMake configuration | `Release`. |
| Library path | `build/Release/sparse_lu_ortho.lib` after building target `sparse_lu_ortho`. |
| Include paths | Source `include/` and generated `build/include/`, rendered through `cmake_path_literal()`. |
| Imported target | `sparse_lu_ortho` as `STATIC IMPORTED GLOBAL` with `IMPORTED_LOCATION` pointing at the MSVC `.lib`. |
| Math library | Omitted on MSVC through the generated `if(NOT MSVC)` guard. |
| Probe binary candidates | `cmake-build/Release/<binary>.exe`, `cmake-build/Release/<binary>`, `cmake-build/<binary>.exe`, or `cmake-build/<binary>`. |
| Compiler metadata | `cmake-probe:Visual Studio 17 2022:Release`. |

### Windows Path And Configuration Checklist

| Windows concern | Required Day 3 check |
| --- | --- |
| Backslash escaping | Generated CMake include and library literals must use forward slashes so paths such as `D:\a\...` do not produce invalid `\a` escapes. |
| Drive-letter paths | Absolute drive-letter paths must remain valid CMake string literals and valid artifact suffix matches. |
| Mixed separators | Normalizer selected-artifact matching must accept Windows backslashes and mixed separators for `build/comparison/qr_incompatible_ls/study.tsv`. |
| Release configuration | The probe must build and run the `Release` executable, matching `build/Release/sparse_lu_ortho.lib`. |
| Visual Studio architecture | The generator must include `-A x64`; omitting architecture is not equivalent evidence. |
| Shell boundary | Workflow command can run under `cmd` for parity with the existing selected Cholesky probe, while configure/build steps can remain PowerShell. |
| Artifact paths | Uploaded paths, if added later, must be the exact six QR incompatible required files and must not include unrelated comparison artifacts. |
| Target filter | Freshness validation must run with `--selected-target qr-incompatible-ls`. |

### Evidence Source Decision

| Evidence source | Day 2 disposition |
| --- | --- |
| Hosted Windows CI | Required for promotion. Only a successful hosted Windows run with inspected artifacts can support manifest metadata promotion. |
| Local non-Windows simulation | Useful for runner/test preservation, but not Windows proof. |
| Local Windows developer run | Useful as preliminary evidence if available, but not a substitute for hosted PR workflow evidence unless explicitly accepted in closeout. |
| Existing Sprint 190 Cholesky workflow | Reusable command pattern and cautionary example only; it does not prove QR incompatible behavior. |

### Expected Success Outputs

- `project_observations.tsv` includes project status and observed QR
  incompatible least-squares values from the generated C probe.
- `baseline_observations.tsv` includes source-controlled dense reference
  helper output for the same fixture.
- `dependency_status.tsv` records the required helper status separately from
  project build status.
- `study.tsv` contains exactly six selected row ids for
  `qr_overdetermined_incompatible_4x2`.
- `summary.md` states fixture-local QR incompatible comparison scope and
  retained non-claims.
- `manifest.tsv` records target key, artifact paths, source commit, command,
  support tier, and claim boundary consistently with the selected target row.
- The normalizer freshness command passes for `--selected-target
  qr-incompatible-ls` and reports the selected rows fresh to the current
  commit.

### Failure Diagnostic Classes

| Failure class | Meaning | Required disposition |
| --- | --- | --- |
| `project_build_failed` | CMake configure/build or generated probe execution failed. | Fix selected generator/CMake/path issue if repo-owned; record environment residual if runner-owned. |
| Dependency helper failure | Dense reference helper missing, malformed, or exited nonzero. | Keep separate from project build failures and do not promote. |
| Missing required file | One of the six required artifacts is absent. | Fail freshness/workflow guard and block promotion. |
| Wrong selected row set | `study.tsv` row ids do not match the six QR incompatible expected ids. | Fail target-specific freshness and block promotion. |
| Stale generated row | Row commit/timestamp does not match current freshness contract. | Regenerate or block promotion. |
| Artifact path mismatch | Windows separator or absolute path form fails selected artifact matching. | Fix normalizer/path tests before promotion. |
| Claim-boundary mismatch | Generated support tier, claim scope, or non-claims imply broader support. | Re-defer or update all metadata/docs together after evidence supports it. |

### Day 2 Completion Notes

- The canonical Day 3 probe command is scoped to exactly
  `qr-incompatible-ls`.
- Expected files, row ids, dependency helper, success output, and diagnostic
  classes are recorded before any generator or workflow edits.
- Windows-specific path, generator, architecture, configuration, library, and
  shell assumptions are explicit.

## Day 3: Probe Execution And Failure Record

### Commands Run

| Command | Result | Evidence |
| --- | --- | --- |
| `cmake --version` | Passed | Local CMake is available as version `4.4.3`. |
| `python3 --version` | Passed | Local Python is `3.14.5`. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed | Generated the six required QR incompatible comparison artifacts through the default local probe path. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-config Release --keep-temp` | Passed | Generated the same selected artifact bundle through local CMake probe mode. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Reported all six QR incompatible rows fresh to current `HEAD`; normalized comparison freshness remained `46` rows overall. |
| `uname -a` | Passed | Execution environment is Darwin x86_64, not Windows/MSVC. |

### Generated Artifact Capture

| Artifact | Day 3 state |
| --- | --- |
| `build/comparison/qr_incompatible_ls/project_observations.tsv` | Generated by direct and local CMake probe runs. |
| `build/comparison/qr_incompatible_ls/baseline_observations.tsv` | Generated by the source-controlled dense QR reference helper. |
| `build/comparison/qr_incompatible_ls/dependency_status.tsv` | Generated; `python3` and `tests/qr_external_dense_reference.py` passed, while optional NumPy and SciPy baselines remained deferred. |
| `build/comparison/qr_incompatible_ls/study.tsv` | Generated with exactly six selected QR incompatible rows, each passing the expected status or tolerance check. |
| `build/comparison/qr_incompatible_ls/summary.md` | Generated with fixture-local QR incompatible scope. |
| `build/comparison/qr_incompatible_ls/manifest.tsv` | Generated with `target=qr-incompatible-ls`, `platform=darwin-x86_64`, `compiler=cmake-probe:default:Release`, `support_tier=local_only`, and `source_commit=0660e84324dcf879d4f957ca7bad335297fe1019`. |

### Selected Row Results

| Row id | Day 3 result |
| --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | `SPARSE_SUCCESS`, status match. |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | Baseline helper status `success`. |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | Project and baseline both reported `1.7320508075688772`, delta `0`. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | Project and baseline both reported `2.2360679774997894`, delta `0`. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | Project and baseline matched within tolerance, max component delta `2.2204460492503131e-16`. |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | Max absolute delta `2.2204460492503131e-16`, below `1e-10`. |

### Failure And Proof Classification

| Area | Day 3 classification | Promotion impact |
| --- | --- | --- |
| Local generator | Passed. | Confirms the selected target still generates locally. |
| Local CMake probe | Passed with default local CMake generator. | Confirms CMake probe mode works locally, but does not prove MSVC/Windows behavior. |
| Dependency status | Passed for required Python/helper dependencies; optional NumPy/SciPy rows deferred. | Required helper is available; optional package deferrals remain non-evidence. |
| Selected freshness | Passed for `--selected-target qr-incompatible-ls`. | Confirms local selected freshness, not Windows promotion. |
| Windows/MSVC hosted evidence | Not available in Day 3 local environment. | Blocks manifest promotion. |
| Generated support tier | Still `local_only`. | Blocks Windows selected metadata promotion until support tier and claim surfaces are intentionally updated with hosted proof. |
| Worktree provenance | Generated rows record `worktree_state=dirty` because Sprint 203 planning docs are uncommitted. | Acceptable for local Day 3 provenance; not promotion evidence. |

### Initial Promotion Recommendation

Sprint 203 should not promote Windows QR incompatible comparison metadata based
on Day 3 evidence alone. The selected comparison target, local artifact set,
local CMake probe path, dependency rows, selected study rows, and freshness
checker all passed, but the evidence is Darwin/local and generated-local. Item
203.1 therefore has concrete local proof plus a remaining hosted Windows/MSVC
evidence gap.

Day 4 should design any needed generator/path/CMake fixes by comparing this
local proof with the canonical Windows command from Day 2. If no repo-owned
fix is evident before hosted execution, the workflow decision should remain
blocked on hosted Windows CI rather than changing manifest metadata
preemptively.

### Validation Matrix Update

| Validation | Day 3 status | Notes |
| --- | --- | --- |
| Direct `qr-incompatible-ls` generator | Passed | Generated all six required files locally. |
| Local CMake probe for `qr-incompatible-ls` | Passed | Used `--probe-build-system cmake --cmake-config Release --keep-temp`; local generator was not Visual Studio/MSVC. |
| Target-specific QR incompatible freshness | Passed | Six selected rows fresh to `0660e84324dcf879d4f957ca7bad335297fe1019`. |
| Required dependency helper | Passed | `tests/qr_external_dense_reference.py` available and selected. |
| Optional NumPy/SciPy baselines | Deferred | Explicit non-evidence. |
| Hosted Windows MSVC probe | Pending | Required for promotion. |
| Manifest metadata promotion | Blocked | No Windows hosted proof and generated support tier remains `local_only`. |
| Full C gate | Not required | Day 3 changed planning documentation only. |

### Day 3 Completion Notes

- Item 203.1 now has concrete local generator, local CMake probe, artifact,
  dependency, and freshness evidence.
- The only promotion-blocking failure class observed on Day 3 is missing
  hosted Windows/MSVC evidence, not local target generation.
- No manifest, workflow, public docs, C source, or header promotion was made.

## Day 4: Generator And CMake Fix Design

### Day 3 Evidence Review

| Surface | Day 3 observation | Day 4 design conclusion |
| --- | --- | --- |
| Direct generator | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` passed locally. | No target-logic fix is justified before Windows evidence. |
| Local CMake probe | `--probe-build-system cmake --cmake-config Release --keep-temp` passed locally. | The CMake probe template is structurally viable on local CMake. |
| Path literal handling | `cmake_path_literal()` converts backslashes to forward slashes and has a Windows-path regression. | Keep this helper as the CMake string-literal owner; add target-specific QR coverage only if Day 5 changes it. |
| Normalizer artifact matching | Selected comparison generated-row filtering normalizes backslashes before matching selected artifact suffixes. | Existing path model should cover Windows separator forms; Day 6 should add QR-specific coverage if promotion work touches filtering. |
| Dependency status | Required Python and `tests/qr_external_dense_reference.py` passed; NumPy/SciPy remain deferred. | Preserve required-helper status separately from optional package deferrals and project build status. |
| Generated support tier | Generated rows remain `local_only`. | Manifest promotion is blocked until hosted Windows evidence and generated metadata semantics are updated together. |
| Windows workflow | Current `selected-comparison-freshness` job is Cholesky-specific. | Do not add QR to the workflow until the Day 7 promotion decision; any later workflow edit must update guards at the same time. |

### Minimal Fix Boundary

Day 4 does not identify a repo-owned generator or CMake defect from local
evidence. The likely implementation path is therefore evidence-gated:

1. If hosted Windows/MSVC probe fails because the generated CMake project
   cannot configure, build, or locate the probe executable, make the smallest
   change in `scripts/run_external_comparison.py` and add focused runner tests.
2. If hosted Windows artifacts are generated but freshness filtering drops
   them due to path shape, make the smallest change in
   `scripts/normalize_report_index.py` and add QR-specific Windows path tests.
3. If hosted evidence passes and generated metadata remains intentionally
   `local_only`, choose between re-deferral or a coordinated metadata/docs
   promotion rather than changing only the manifest.
4. If no hosted evidence is available, keep promotion blocked and record the
   hosted evidence gap instead of adding speculative workflow or manifest
   metadata.

### Compatibility Requirements

| Requirement | Preservation check |
| --- | --- |
| Existing Linux/macOS selected comparison workflows continue to upload the six selected comparison target families. | `python3 tests/test_selected_comparison_workflow.py`. |
| Existing Windows Cholesky selected workflow remains guarded with the same target, artifact, required files, and fail-closed upload. | `python3 tests/test_selected_comparison_workflow.py`; `make windows-powershell-guard` if workflow or PowerShell guard changes. |
| `qr-incompatible-ls` local generator output stays six rows with fixture-local non-claims. | `python3 tests/test_run_external_comparison.py`; target-specific freshness command. |
| Selected artifact matching continues to accept Windows-style Cholesky paths and should gain QR-specific coverage if the QR path is promoted. | `python3 tests/test_normalize_report_index.py`. |
| Selected manifest keeps Windows metadata absent unless promotion evidence supports the exact target. | `python3 tests/test_selected_report_targets_manifest.py`. |
| No C solver behavior changes are introduced by comparison promotion work. | Full C gate only if `.c` or `.h` files change. |

### Planned Fix-To-Test Map

| Potential fix | Trigger | Required tests |
| --- | --- | --- |
| CMake string-literal or include/library path fix in `run_external_comparison.py` | Hosted configure failure, invalid escape, wrong `.lib`, or missing generated include path. | `python3 tests/test_run_external_comparison.py`; add a focused Windows path literal or generated CMakeLists fixture. |
| Probe executable discovery fix | Hosted build succeeds but runner cannot find the `Release` `.exe`. | Runner test for candidate executable layout; hosted workflow evidence. |
| Dependency-status classification fix | Hosted output collapses required helper failure into project build failure or treats optional baselines as proof. | Runner tests for dependency status and malformed reference helper behavior. |
| QR artifact path normalization fix | Hosted artifact path uses backslashes or absolute paths and selected filtering drops QR rows. | `python3 tests/test_normalize_report_index.py` with `qr_incompatible_ls` Windows path cases. |
| Windows QR workflow lane/update | Hosted or Day 7 decision supports adding QR to the selected Windows path. | `python3 tests/test_selected_comparison_workflow.py`; `make windows-powershell-guard`; exact upload-path negative fixture. |
| Manifest promotion metadata | Hosted Windows evidence proves exact target and docs/metadata are ready. | `python3 tests/test_selected_report_targets_manifest.py`; docs guard tests; target-specific freshness command. |

### Unchanged Surfaces For Day 4

- No `.c` or `.h` implementation files.
- No QR solver behavior or numerical tolerance changes.
- No selected target manifest promotion.
- No Windows workflow YAML edits.
- No README, INSTALL, corpus, schema, or maintainer guide claim updates.
- No package-manager, ABI, performance, release, or state-of-the-art support
  claims.

### Item 203.2 Implementation Boundary

Item 203.2 should be implemented only after a concrete hosted Windows/MSVC
failure or proof gap identifies a repo-owned surface. Local evidence currently
supports preserving the runner and normalizer behavior rather than editing
them preemptively. The Day 5 implementation pass should therefore start with a
hosted-evidence check or a targeted simulation fixture and avoid broad
comparison-family changes.

### Day 4 Completion Notes

- The narrow implementation boundary is recorded before generator or workflow
  edits.
- Existing Linux/macOS selected comparison behavior and Windows Cholesky guard
  behavior have explicit preservation checks.
- No new target family, broad comparison workflow, or manifest promotion was
  introduced.

## Day 5: Selected Generator Fixes

### Implementation Decision

Day 5 did not apply generator, CMake template, normalizer, workflow, or
manifest code changes. Day 3 local proof and Day 4 design found no repo-owned
defect that can be fixed safely without hosted Windows/MSVC evidence. The
current implementation already has these relevant Windows-safe behaviors:

- `scripts/run_external_comparison.py` supports `--probe-build-system cmake`,
  `--cmake-generator`, `--cmake-arch`, `--cmake-config`, and explicit
  `--library` paths.
- `cmake_path_literal()` converts backslashes to forward slashes before
  rendering include and library paths into generated CMake code.
- The generated CMake probe links `m` only inside `if(NOT MSVC)`.
- `scripts/normalize_report_index.py` normalizes backslashes before selected
  artifact matching.
- The selected QR incompatible target already has local runner, row, dependency
  status, and target-specific freshness coverage.

Speculative edits would increase review surface without proving the Windows
promotion condition. The next implementation change should be driven by hosted
Windows evidence or a focused failing fixture.

### Commands Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_run_external_comparison.py` | Passed | Runner regressions, including QR incompatible fixture, dependency, project probe, malformed output, and Windows path literal coverage, passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | All six selected QR incompatible generated rows were fresh to current `HEAD`; normalized comparison freshness remained `46` rows. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-config Release --keep-temp` | Passed | Local CMake probe regenerated the six-file QR incompatible artifact bundle and emitted the selected success message. |

### Generated Day 5 Evidence

| Evidence | Day 5 result |
| --- | --- |
| Selected generator behavior | Passed locally with direct runner tests and local CMake probe. |
| Path literal coverage | Existing runner test verifies a Windows path such as `D:\a\...` renders as `D:/a/...` and avoids `\a` escape sequences. |
| Dependency status separation | Existing runner coverage and generated artifacts keep required helper status separate from optional NumPy/SciPy deferrals. |
| Target-specific freshness | Passed for `--selected-target qr-incompatible-ls`. |
| Windows/MSVC hosted proof | Still pending; required for promotion. |
| Manifest promotion | Still blocked. |

### Preservation Record

| Surface | Day 5 disposition |
| --- | --- |
| Linux selected comparison workflow | Unchanged. |
| macOS selected comparison workflow | Unchanged. |
| Windows selected Cholesky workflow | Unchanged. |
| Selected target manifest | Unchanged. |
| Public and maintainer docs | Unchanged. |
| C source and headers | Unchanged. |
| Generated local `build/` artifacts | Regenerated as ignored local evidence only. |

### Item 203.2 Status

Item 203.2 remains evidence-gated rather than code-complete. The repo-owned
runner and CMake probe surfaces passed the available local validation, so no
minimal fix is currently justified. If Day 6 or hosted Windows evidence finds
a QR-specific path, artifact, dependency, or CMake failure, the fix should be
targeted to that failure and paired with the mapped tests from Day 4.

### Day 5 Completion Notes

- The selected QR incompatible comparison can generate locally through direct
  and CMake probe paths.
- Existing comparison targets were not promoted, broadened, or rewritten.
- Focused runner and selected freshness tests passed without code changes.
- Windows promotion remains blocked on hosted MSVC evidence.

## Day 6: Artifact Path And Row Filtering Tests

### Implementation Summary

Day 6 added QR incompatible-specific normalizer coverage for Windows artifact
path matching. Existing tests already covered QR incompatible target-specific
freshness, Windows-style stale row diagnostics, dependency-only row rejection,
and wrong-target row-set mismatch behavior. The missing gap was raw generated
row matching for the QR artifact itself.

Changed file:

- `tests/test_normalize_report_index.py`

New tests:

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_generated_rows_match_windows_artifact_paths()` | Verifies selected generated rows for `build/comparison/qr_incompatible_ls/study.tsv` match forward-slash, backslash, mixed-separator, and absolute drive-letter artifact paths. |
| `test_qr_incompatible_generated_rows_reject_near_match_artifact_paths()` | Verifies near-match QR paths such as `qr_incompatible_ls_extra`, `not_qr_incompatible_ls`, pluralized directory names, `.bak` suffixes, and absolute near-match paths do not satisfy selected artifact filtering. |

Both tests are invoked from the standalone `main()` runner so `python3
tests/test_normalize_report_index.py` enforces the new coverage.

### Existing QR Diagnostic Coverage Confirmed

| Existing test | Confirmed Day 6 role |
| --- | --- |
| `test_selected_comparison_target_freshness_accepts_qr_incompatible_subset()` | Confirms selected-target filtering accepts only the QR incompatible six-row subset. |
| `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows()` | Confirms Windows-style QR artifact paths still report stale rows clearly. |
| `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows()` | Confirms dependency-only QR rows cannot satisfy the selected row set. |
| `test_selected_comparison_target_freshness_rejects_wrong_target_rows()` | Confirms wrong-target rows fail selected freshness with row-set mismatch diagnostics. |
| `test_selected_comparison_required_freshness_rejects_duplicate_rows()` | Confirms duplicate generated rows fail the selected comparison freshness set. |

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed | Full normalizer regression suite passed with the new QR artifact-path fixtures. |
| `python3 -m py_compile tests/test_normalize_report_index.py` | Passed | Syntax compilation passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Six selected QR incompatible rows remained fresh to current `HEAD`; normalized comparison freshness remained `46` rows. |

### Item 203.4 Evidence

| Requirement | Day 6 status |
| --- | --- |
| Backslash, forward-slash, absolute, and mixed artifact paths | Covered for QR incompatible generated rows. |
| Selected-target filtering keeps only QR incompatible rows | Covered by existing QR subset freshness test and Day 6 validation. |
| Stale-output diagnostics | Covered by existing Windows-style QR stale row test. |
| Missing/wrong target diagnostics | Covered by existing selected row-set mismatch diagnostics. |
| Dependency-status diagnostics | Covered by existing dependency-only QR row rejection. |
| Near-match rejection | Covered by new QR-specific near-match path fixture. |

### Day 6 Completion Notes

- Windows path separator differences cannot silently drop selected QR
  incompatible generated rows.
- Near-match QR artifact paths cannot satisfy selected artifact filtering.
- Tests remain selected-target scoped and do not promote Windows manifest
  metadata.
- No `.c` or `.h` files changed.

## Day 7: Manifest Promotion Decision

### Decision

Windows QR incompatible selected comparison freshness is re-deferred on Day 7.
The branch has local generator proof, local CMake probe proof, QR-specific
Windows path matching tests, target-specific freshness, selected manifest
validation, and workflow guard validation. It does not have hosted
Windows/MSVC execution or inspected Windows artifact evidence. The selected
target manifest must therefore keep `SRT-COMP-QR-INCOMPATIBLE-LS` scoped to
Linux and macOS.

### Promotion Gate Comparison

| Promotion requirement | Day 7 evidence | Decision |
| --- | --- | --- |
| Exact selected target evidence for `qr-incompatible-ls` | Local direct generator and local CMake probe passed. | Helpful but insufficient for Windows promotion. |
| Hosted Windows/MSVC configure, build, and probe pass | Not available on branch-local Day 7 evidence. | Blocks promotion. |
| Uploaded Windows artifact bundle inspected | Not available. | Blocks promotion. |
| Six expected QR incompatible rows present and passing | Local generated `study.tsv` has all six rows passing. | Satisfied locally only. |
| Windows artifact path normalization | Day 6 QR-specific backslash, mixed-separator, absolute-path, and near-match tests passed. | Satisfied as guard coverage. |
| Manifest workflow metadata for Windows | Current manifest has no Windows entry for QR incompatible. | Preserve absent metadata. |
| Generated support tier and claim scope promote Windows together | Generated artifacts still record `support_tier=local_only`. | Blocks promotion. |
| Docs can be calibrated to exact Windows QR evidence | No hosted Windows evidence to document as promoted. | Defer docs promotion. |

### Selected Metadata Checklist

| Manifest field | Current value for `SRT-COMP-QR-INCOMPATIBLE-LS` | Day 7 disposition |
| --- | --- | --- |
| `target_id` | `SRT-COMP-QR-INCOMPATIBLE-LS` | Keep. |
| `target_key` | `qr-incompatible-ls` | Keep. |
| `artifact_pattern` | `build/comparison/qr_incompatible_ls/study.tsv` | Keep. |
| `required_files` | `project_observations.tsv`; `baseline_observations.tsv`; `dependency_status.tsv`; `study.tsv`; `summary.md`; `manifest.tsv` | Keep. |
| `expected_rows` | `6` | Keep. |
| `workflow_file` | `.github/workflows/ci.yml`; `.github/workflows/macos-ci.yml` | Do not add `.github/workflows/windows-ci.yml` yet. |
| `workflow_job` | `generated-report-freshness`; `selected-comparison-freshness` | Do not add Windows job metadata yet. |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness`; `sprint175-macos-selected-comparison-freshness` | Do not add Windows artifact metadata yet. |
| `workflow_platforms` | `linux`; `macos` | Do not add `windows` yet. |
| `support_tier` | `local_only` | Keep until generated and source metadata are promoted together. |

### Retained Non-Claims

Day 7 keeps these boundaries active for QR incompatible comparison work:

- no broad QR parity;
- no broad least-squares parity;
- no raw QR basis identity;
- no Q sign or orientation claim;
- no global rank-threshold policy;
- no broad rank-deficient solve claim;
- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- no Windows report freshness;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Current selected manifest remains valid with QR incompatible Windows metadata absent. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow guards remain valid; Windows selected workflow is still Cholesky-specific. |
| `python3 tests/test_normalize_report_index.py` | Passed | Includes Day 6 QR-specific Windows artifact-path coverage. |

### Hosted Evidence Residual

| Residual | Required closure evidence |
| --- | --- |
| Hosted Windows/MSVC probe | Run the canonical Day 2 Visual Studio 2022/x64/Release command sequence for `qr-incompatible-ls`. |
| Hosted artifact inspection | Verify the Windows upload contains only the six required QR incompatible files and no unrelated comparison artifacts. |
| Generated support semantics | Decide whether generated rows should remain `local_only` with re-deferral or move with coordinated support-tier/non-claim updates. |
| Manifest/docs promotion | Add Windows metadata and public/maintainer docs only after hosted evidence and generated semantics agree. |

### Risk Register Update

| Risk | Day 7 state | Mitigation |
| --- | --- | --- |
| Premature Windows QR promotion | Active. | Re-defer manifest promotion until hosted Windows artifacts exist. |
| Windows path filtering false negative | Reduced. | Day 6 QR-specific Windows path tests passed. |
| Cholesky Windows workflow drift | Reduced. | Selected comparison workflow guard passed. |
| Generated local evidence overstated as hosted proof | Active. | Day 7 records local-only evidence and leaves Windows metadata absent. |

### Day 7 Completion Notes

- Item 203.3 has a recorded re-deferral decision before manifest edits.
- Promotion remains explicitly dependent on exact hosted Windows/MSVC evidence.
- Deferral is tied to concrete missing evidence, not an ambiguous blocker.

## Day 8: Manifest And Workflow Metadata

### Metadata State

Day 8 kept `SRT-COMP-QR-INCOMPATIBLE-LS` re-deferred for Windows, matching the
Day 7 decision. No Windows workflow metadata was added to the selected target
manifest because hosted Windows/MSVC proof and artifact review are still
absent.

Changed file:

- `tests/test_selected_report_targets_manifest.py`

Unchanged source-of-truth files:

- `tests/corpus/manifests/selected_report_targets.tsv`;
- `.github/workflows/windows-ci.yml`;
- `.github/workflows/ci.yml`;
- `.github/workflows/macos-ci.yml`.

### Manifest Contract Added

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_manifest_remains_redeferred_for_windows()` | Verifies the QR incompatible selected target keeps six expected rows, the six required artifact files, exact expected row ids, no `windows` platform, no Windows workflow file, no reused Cholesky Windows artifact, `support_tier=local_only`, and retained non-claims. |

The test is invoked from the standalone `main()` runner, so
`python3 tests/test_selected_report_targets_manifest.py` enforces the Day 8
metadata decision.

### Current QR Incompatible Manifest State

| Manifest field | Day 8 expected value |
| --- | --- |
| `target_id` | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| `target_key` | `qr-incompatible-ls` |
| `expected_rows` | `6` |
| `required_files` | `project_observations.tsv`; `baseline_observations.tsv`; `dependency_status.tsv`; `study.tsv`; `summary.md`; `manifest.tsv` |
| `workflow_file` | `.github/workflows/ci.yml`; `.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness`; `selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness`; `sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux`; `macos` |
| `support_tier` | `local_only` |

### Retained Non-Claims Guarded

The new contract explicitly guards these non-claims for the selected QR
incompatible row:

- no broad QR parity;
- no broad least-squares parity;
- no Windows report freshness;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Includes the new QR incompatible re-deferral contract. |
| `python3 -m py_compile tests/test_selected_report_targets_manifest.py` | Passed | Syntax compilation passed. |

### Day 8 Completion Notes

- Manifest metadata matches the Day 7 re-deferral decision.
- Windows QR promotion cannot silently reuse the Cholesky Windows artifact or
  drift row/file/non-claim metadata.
- Existing selected targets retain their prior support boundaries.
- No `.c` or `.h` files changed.

## Day 9: Workflow Guard Integration

### Workflow Guard State

Day 9 kept the Windows QR incompatible comparison re-deferred and hardened the
workflow guard against accidental partial promotion. The selected Windows
comparison workflow remains Cholesky-specific until hosted Windows/MSVC QR
evidence exists.

Changed file:

- `tests/test_selected_comparison_workflow.py`

Unchanged source-of-truth files:

- `.github/workflows/windows-ci.yml`;
- `tests/corpus/manifests/selected_report_targets.tsv`.

### Guard Coverage Added

| Guard | Coverage |
| --- | --- |
| `assert_windows_qr_incompatible_remains_redeferred()` | Rejects accidental Windows workflow references to the raw `qr-incompatible-ls` target key, `qr_incompatible_ls` subfamily, spaced or equals-form target flags, selected freshness flags, the proposed Sprint 203 QR artifact name, or any of the six QR incompatible artifact paths. |

### Negative Fixtures Added

| Test | Drift detected |
| --- | --- |
| `test_windows_qr_incompatible_target_drift_fails_clearly()` | Generator command promotion without evidence. |
| `test_windows_qr_incompatible_freshness_target_drift_fails_clearly()` | Freshness checker target promotion without evidence. |
| `test_windows_qr_incompatible_equals_target_drift_fails_clearly()` | Equals-form generator command promotion without evidence. |
| `test_windows_qr_incompatible_subfamily_drift_fails_clearly()` | QR incompatible subfamily/path promotion without evidence. |
| `test_windows_qr_incompatible_artifact_upload_drift_fails_clearly()` | QR incompatible alternate artifact upload path promotion without evidence. |

The tests are invoked by the standalone
`tests/test_selected_comparison_workflow.py` runner.

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Includes Day 9 QR incompatible workflow re-deferral fixtures. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Confirms Day 8 QR incompatible manifest re-deferral still holds. |
| `python3 -m py_compile tests/test_selected_comparison_workflow.py` | Passed | Syntax compilation passed. |
| `git diff --check -- docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed | No whitespace errors in Sprint 203 docs or changed Python tests. |

### Day 9 Completion Notes

- Item 203.4 now has workflow-level drift protection for the Windows QR
  incompatible re-deferral state.
- Day 9 did not add Windows QR workflow metadata, upload paths, or selected
  freshness commands.
- Promotion remains blocked on hosted Windows/MSVC execution and artifact
  inspection.
- No `.c` or `.h` files changed.

## Day 10: Normalizer And Freshness Diagnostics

### Diagnostic State

Day 10 added QR incompatible selected-target diagnostics for the remaining
Windows-style row-set failure cases. The normalizer still treats Windows QR
incompatible comparison freshness as selected-target/local evidence only; broad
Windows report-index freshness remains unpromoted.

Changed file:

- `tests/test_normalize_report_index.py`

Unchanged source-of-truth files:

- `scripts/normalize_report_index.py`;
- `tests/corpus/manifests/selected_report_targets.tsv`;
- `.github/workflows/windows-ci.yml`.

### Coverage Matrix

| Failure mode | Day 10 coverage |
| --- | --- |
| Missing selected QR artifact | Existing QR subset freshness test verifies the selected missing-artifact diagnostic before generated rows exist. |
| Stale selected QR rows with Windows path | Existing QR Windows-path stale test verifies stale `source_commit` diagnostics. |
| Dependency-only selected QR rows | Existing QR dependency-only test verifies incomplete selected row-set diagnostics. |
| Duplicate selected QR row with Windows path | Added `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()`. |
| Unexpected selected QR row with Windows path | Added `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()`. |
| Wrong selected target | Existing wrong-target test verifies selected-target mismatch diagnostics remain scoped. |

### Diagnostic Assertions Added

| Test | Assertion |
| --- | --- |
| `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()` | Confirms duplicate QR incompatible rows fail through the normalizer duplicate row-id guard and name `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`. |
| `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()` | Confirms unexpected QR incompatible rows fail with `comparison_selected_rows`, `row_set_mismatch`, the missing expected QR row, the unexpected row id, the selected QR artifact diagnostic, and `--selected-target qr-incompatible-ls`. |

Both fixtures rewrite artifact paths to
`build\comparison\qr_incompatible_ls\study.tsv` before running the selected
freshness check.

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed | Includes Day 10 duplicate and unexpected QR Windows-path diagnostics. |
| `python3 -m py_compile tests/test_normalize_report_index.py` | Passed | Syntax compilation passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Reported freshness ok for 46 rows and the six QR incompatible generated rows as fresh. |
| `git diff --check -- docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed | No whitespace errors in Sprint 203 docs or changed Python tests. |

### Day 10 Completion Notes

- Selected Windows-style QR freshness diagnostics now cover missing, stale,
  dependency-only, duplicate, unexpected, and wrong-target paths.
- The new tests are invoked by the standalone normalizer test runner.
- No normalizer production code change was needed.
- No `.c` or `.h` files changed.

## Day 11: Documentation Calibration

### Claim-Surface State

Day 11 updated public, install, maintainer, corpus, and schema documentation
to state one consistent Sprint 203 outcome: the QR incompatible least-squares
target remains outside Windows selected freshness until hosted MSVC probe
evidence, selected artifact review, selected-target manifest metadata,
generated support tier, and generated non-claim wording are promoted together.

Changed files:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`;
- `tests/corpus/schemas/report_index_fields.md`;
- `scripts/validate_windows_powershell.py`.

Unchanged source-of-truth files:

- `tests/corpus/manifests/selected_report_targets.tsv`;
- `.github/workflows/windows-ci.yml`;
- `scripts/run_external_comparison.py`;
- `scripts/normalize_report_index.py`.

### Documentation Updates

| Surface | Day 11 wording |
| --- | --- |
| `README.md` | Adds the QR incompatible Windows selected freshness re-deferral marker near normalized report-index and selected comparison claim boundaries. |
| `INSTALL.md` | Adds a deferred Windows QR incompatible least-squares support-readiness row and repeats the boundary under platform support. |
| `docs/maintainer_guide.md` | Distinguishes the guarded Sprint 190 Windows Cholesky path from the still-deferred QR incompatible Windows selected freshness path. |
| `tests/corpus/README.md` | Adds the QR incompatible Windows re-deferral marker to selected comparison corpus docs. |
| `tests/corpus/schemas/report_index_fields.md` | Adds the same boundary to report-index field semantics. |

### Guard Marker Added

`scripts/validate_windows_powershell.py` now requires the QR incompatible
Windows selected freshness re-deferral marker in:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`.

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Includes current-doc claim-boundary validation with the Day 11 QR marker. |
| `python3 scripts/validate_windows_powershell.py` | Structural pass, exit `2` | Claim boundaries passed; local `pwsh` is unavailable, which remains environment residual evidence rather than pass evidence. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed | Syntax compilation passed. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Existing selected-performance docs markers remain intact. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest re-deferral contract remains intact. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow re-deferral guard remains intact. |

### Day 11 Completion Notes

- Item 203.5 now has an enforced documentation marker for QR incompatible
  Windows selected freshness re-deferral.
- Docs do not claim broad Windows report freshness, selected Windows QR
  freshness, package-manager proof, shared-library ABI proof, performance
  superiority, or state-of-the-art status.
- No selected target manifest, workflow YAML, generated artifact, `.c`, or
  `.h` files changed.

## Day 12: Integrated Validation

### Validation Scope

Day 12 ran the focused validation matrix across generator, normalizer,
manifest, workflow, documentation, Windows claim-boundary, and QR runtime
surfaces. The validation confirms the current re-deferred Windows QR
incompatible state; it does not promote Windows selected freshness.

### Command Results

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed | Regenerated the six-file QR incompatible local artifact bundle and reported project-vs-baseline comparison passed. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison regression suite passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Reported freshness ok for 46 rows and the six QR incompatible generated rows as fresh. |
| `python3 tests/test_normalize_report_index.py` | Passed | Includes selected QR Windows-path and row-set diagnostics. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest re-deferral contract remains intact. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow QR re-deferral and selected Cholesky guard behavior remain intact. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Existing selected performance doc markers remain intact. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Windows claim-boundary and workflow structure regressions passed. |
| `python3 scripts/validate_windows_powershell.py` | Structural pass, exit `2` | Claim boundaries passed; local `pwsh` is unavailable and remains residual evidence. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_run_external_comparison.py` | Passed | Python syntax compilation passed. |
| `build/test_qr_corpus` | Passed | QR corpus proof owner passed 14 tests with zero failures. |
| `build/test_qr_solve` | Passed | QR solve proof owner passed 19 tests, including the incompatible 4x2 dense-reference comparison. |

### Corrected Focused QR Command

`make test_qr_corpus` was attempted first, but it is not a Makefile target.
The existing built proof owner `build/test_qr_corpus` was then run directly
and passed. `build/test_qr_solve` was also run directly because it owns the
incompatible 4x2 dense-reference solve check.

### Full C Gate Decision

No `.c` or `.h` files were modified through Day 12. The full
`make format && make lint && make test` gate was therefore not required by the
sprint instruction.

### Residual Checklist

| Residual | Day 12 disposition |
| --- | --- |
| Hosted Windows/MSVC `qr-incompatible-ls` proof | Still absent; blocks Windows QR selected freshness promotion. |
| Hosted Windows QR artifact inspection | Still absent; blocks selected target manifest promotion. |
| Local PowerShell availability | `pwsh` unavailable locally; structural checks pass and this remains environment residual evidence. |
| Broad Windows report-index freshness | Not promoted. |

### Day 12 Completion Notes

- Item 203.6 has a focused validation record for the current branch state.
- All required focused validation passed or produced the expected local
  environment residual.
- Windows QR incompatible selected freshness remains re-deferred.
- No `.c` or `.h` files changed.

## Day 13: Review Hardening

### Review Scope

Day 13 audited the Sprint 203 diff for unrelated churn, stale wording,
accidental metadata/workflow promotion, documentation guard completeness, and
reviewer-facing evidence clarity.

### Hardening Change

The review found one documentation guard gap: Day 11 added the QR incompatible
Windows selected freshness boundary to
`tests/corpus/schemas/report_index_fields.md`, but
`scripts/validate_windows_powershell.py` did not enforce that schema marker.
Day 13 added the report-index schema doc to `CLAIM_BOUNDARY_MARKERS`, so the
validator now reports Windows/PowerShell claim boundaries across five files.

Changed file:

- `scripts/validate_windows_powershell.py`.

### Changed-File Inventory

| Category | Files |
| --- | --- |
| Public/user docs | `README.md`; `INSTALL.md` |
| Maintainer docs | `docs/maintainer_guide.md` |
| Corpus/schema docs | `tests/corpus/README.md`; `tests/corpus/schemas/report_index_fields.md` |
| Windows claim-boundary guard | `scripts/validate_windows_powershell.py` |
| Normalizer tests | `tests/test_normalize_report_index.py` |
| Workflow guard tests | `tests/test_selected_comparison_workflow.py` |
| Manifest guard tests | `tests/test_selected_report_targets_manifest.py` |
| Sprint planning/evidence | `docs/planning/EPIC_18/SPRINT_203/PLAN.md`; `docs/planning/EPIC_18/SPRINT_203/WORKING_NOTES.md`; Day 1 through Day 13 artifacts |

### Unchanged Promotion Surfaces

| Surface | Day 13 status |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Unchanged; no QR incompatible Windows workflow lane or upload path was added. |
| `.github/workflows/ci.yml` | Unchanged. |
| `.github/workflows/macos-ci.yml` | Unchanged. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Unchanged; QR incompatible remains Linux/macOS-only and `local_only`. |
| `scripts/run_external_comparison.py` | Unchanged. |
| `scripts/normalize_report_index.py` | Unchanged. |

### Guard Coverage Checklist

| Guard | Day 13 coverage state |
| --- | --- |
| Manifest guard | QR incompatible row ids, required files, retained non-claims, absent Windows platform, absent Windows workflow file, and no reused Cholesky artifact. |
| Workflow guard | Accidental Windows `qr-incompatible-ls` generator command, equals-form target command, selected freshness command, artifact name, QR subfamily token, and all QR upload paths are rejected. |
| Normalizer guard | Windows-style QR artifact paths, near-match rejection, stale rows, dependency-only rows, duplicate rows, unexpected rows, and wrong-target diagnostics. |
| Documentation guard | QR incompatible Windows selected freshness re-deferral markers are enforced across README, INSTALL, maintainer guide, corpus README, and report-index schema docs. |

### Validation Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Claim-boundary coverage now reports five files. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed | Syntax compilation passed. |
| `python3 scripts/validate_windows_powershell.py` | Structural pass, exit `2` | Claim boundaries passed; local `pwsh` is unavailable and remains residual evidence. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest re-deferral contract remains intact. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow QR re-deferral guard remains intact. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalizer selected QR diagnostics remain intact. |
| `git diff --check -- README.md INSTALL.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/validate_windows_powershell.py docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed | No whitespace errors in changed Sprint 203 docs or guard/test files. |
| `git diff -- .github/workflows tests/corpus/manifests` | Passed | Empty diff; no workflow or selected manifest promotion occurred. |
| `git diff --name-only \| sort` | Passed | Reviewed changed-file inventory. |

### Reviewer-Facing Summary

Sprint 203 provides local selected QR incompatible generator proof, local
selected freshness proof, normalizer path/diagnostic regression coverage,
manifest/workflow re-deferral guards, and claim-safe documentation. It does
not provide hosted Windows/MSVC QR incompatible proof or inspected hosted
Windows QR artifacts. The correct interpretation is re-deferral, not
promotion.

### Day 13 Completion Notes

- Review-surface inventory now matches the actual changed files.
- The schema documentation boundary is now enforced by the Windows
  claim-boundary validator.
- Workflows and selected target manifest metadata remain unchanged.
- No `.c` or `.h` files changed.

## Day 14: Closeout And Retrospective Inputs

Day 14 finalized Sprint 203 as a re-deferral of Windows QR incompatible
selected comparison freshness promotion. The sprint produced local selected
generator evidence, generated-row freshness evidence, guard coverage,
documentation calibration, and planning/residual alignment, but it did not
produce hosted Windows/MSVC proof or inspect hosted Windows QR artifacts.

### Closeout Artifact

- `docs/planning/EPIC_18/SPRINT_203/artifacts/day14-closeout-review.md`

### Final Disposition

| Topic | Closeout disposition |
| --- | --- |
| Windows QR incompatible selected freshness promotion | Re-deferred. |
| Local QR incompatible generator evidence | Delivered. |
| Local selected generated-row freshness evidence | Delivered with explicit `--include-generated`. |
| Hosted Windows/MSVC proof | Still absent. |
| Hosted Windows QR artifact inspection | Still absent. |
| Workflow promotion | Not performed. |
| Selected manifest Windows metadata promotion | Not performed. |

### Planning Updates

| File | Day 14 update |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 203 is now recorded as closed with Windows QR incompatible promotion re-deferred. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | E18-RQ-006 now records the delivered local evidence and the remaining hosted Windows/MSVC proof path. |

### Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed | Regenerated local QR incompatible comparison artifacts and reported project-vs-baseline comparison passed. |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Reported freshness ok for `46` rows and all six selected QR incompatible generated rows fresh to current `HEAD`. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison generator regressions passed. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalizer selected-target, Windows-path, and diagnostic regressions passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest re-deferral contract passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow QR re-deferral guard passed. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Selected performance documentation guards remained clean. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Windows/PowerShell workflow and claim-boundary regressions passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_run_external_comparison.py` | Passed | Python syntax compilation passed. |
| `build/test_qr_corpus` | Passed | QR corpus proof passed. |
| `build/test_qr_solve` | Passed | QR solve scenario proof passed. |
| `git diff -- .github/workflows tests/corpus/manifests` | Passed | Empty diff; no workflow or selected manifest promotion occurred. |

### Retrospective Inputs

- Sprint 203 should be reported as a claim-safe re-deferral with stronger
  future-promotion infrastructure.
- The delivered proof is local selected QR incompatible evidence plus guard
  coverage, not hosted Windows selected freshness.
- Future promotion still requires hosted Windows/MSVC generator proof, hosted
  artifact inspection, selected manifest metadata promotion, and Windows
  workflow upload path changes in one coherent evidence update.
- No `.c` or `.h` files changed; focused validation was sufficient for Day 14.

### Day 14 Completion Notes

- Day 14 closeout artifact created.
- Project-plan and residual-queue dispositions aligned with the re-deferral.
- Final focused validation passed.
- Workflow and selected target manifest promotion surfaces remain unchanged.
