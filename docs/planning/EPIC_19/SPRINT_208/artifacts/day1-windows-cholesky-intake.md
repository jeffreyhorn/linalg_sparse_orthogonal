# Sprint 208 Day 1: Windows Cholesky Intake

## Purpose

Establish Sprint 208 scope, inherited Windows Cholesky evidence, selected
freshness boundaries, and validation owners before changing manifest,
workflow, guard, or documentation surfaces.

## Sprint 208 Scope

Sprint 208 is the Epic 19 selected Windows Cholesky freshness promotion sprint.
Its goal is to fully promote or deliberately re-defer the selected Windows
Cholesky freshness lane using hosted evidence, manifest metadata,
documentation, and guards.

The six project-plan items are:

| Item | Day 1 interpretation |
| --- | --- |
| 208.1 Hosted Artifact Intake | Fetch and inspect current hosted Windows `cholesky-spd-tridiag-5` evidence, row IDs, paths, and artifact membership. |
| 208.2 Manifest Promotion Decision | Decide whether `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` earns Windows selected metadata or remains re-deferred. |
| 208.3 Metadata And Guard Implementation | Update selected manifest, workflow metadata, PowerShell guards, or absence guards according to the decision. |
| 208.4 Normalizer Regression Coverage | Add or confirm tests for Windows path normalization, selected target filtering, missing rows, stale rows, and artifact mismatch. |
| 208.5 Documentation Calibration | Update README, INSTALL, corpus docs, and maintainer guide with promoted or re-deferred wording. |
| 208.6 Validation And Closeout | Run selected manifest, workflow, PowerShell, normalizer, freshness, docs, and applicable C quality gates. |

## Inherited Evidence

| Source | Current evidence |
| --- | --- |
| Sprint 199 retrospective | Hosted Windows evidence for exact `cholesky-spd-tridiag-5` was reviewed, but selected Windows freshness promotion was re-deferred. |
| Sprint 199 Day 2 hosted artifact inventory | Prior run `34269219871` on `master` uploaded `sprint190-windows-selected-comparison-cholesky`, containing six expected selected Cholesky files and six passing rows. |
| Sprint 199 Day 4 manifest decision | Manifest promotion was blocked by Windows path separator concerns, generated `support_tier=local_only`, generated non-claim wording, and incomplete diagnostics. |
| Sprint 199 Day 14 closeout | Windows Cholesky remained guarded workflow evidence only; broad Windows, package, ABI, performance, release, and state-of-the-art claims stayed unearned. |
| Epic 18 closeout | Sprint 199 is recorded as closed re-deferral, not selected Windows freshness promotion. |
| Epic 19 todo review | Closure track requires latest hosted artifact fetch, target identity verification, artifact membership/path/support-tier validation, manifest contract review, and either promotion or stronger re-deferral. |

## Current Source-Controlled Surfaces

| Surface | Day 1 role |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Source of truth for selected target metadata. Current Cholesky row lists Linux/macOS only and retains `no Windows report freshness`. |
| `.github/workflows/windows-ci.yml` | Contains bounded Windows selected Cholesky job using MSVC/CMake, target-specific generation, target-specific freshness, and exact six-file artifact upload. |
| `scripts/validate_windows_powershell.py` | Validates Windows PowerShell workflow ownership, selected Cholesky target tokens, artifact name, upload paths, and Windows non-claim markers. |
| `tests/test_selected_report_targets_manifest.py` | Enforces current no-Windows manifest state and defines a future exact Cholesky metadata allowlist if promotion is earned. |
| `tests/test_selected_comparison_workflow.py` | Guards Windows workflow scope and rejects broad selected report freshness lanes. |
| `tests/test_normalize_report_index.py` | Holds selected comparison freshness and Windows path regression coverage. |
| README, INSTALL, corpus README, maintainer guide | Public and maintainer claim surfaces currently describe guarded workflow evidence and retained re-deferral. |

## Current Claim Boundary

Sprint 208 starts from this bounded state:

- one Windows hosted workflow path exists for `cholesky-spd-tridiag-5`;
- the workflow generates and checks only the selected Cholesky comparison;
- artifact upload scope is limited to the six selected Cholesky files;
- prior hosted evidence was reviewed in Sprint 199;
- selected manifest metadata still omits `windows`;
- public and maintainer docs still describe the lane as guarded workflow
  evidence, not promoted selected Windows freshness.

Sprint 208 must not infer the following without explicit evidence and matching
metadata/docs/guards:

- promoted selected Windows Cholesky freshness;
- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- QR incompatible Windows comparison freshness;
- unselected Windows comparison families;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library or dynamic ABI support;
- runtime-loader behavior;
- broad Windows parity;
- performance superiority;
- release readiness;
- external-library ecosystem parity;
- state-of-the-art status.

## Initial Validation Matrix

| Command | Purpose | Day 1 disposition |
| --- | --- | --- |
| `gh run list` / `gh run view` / `gh run download` for the Windows selected Cholesky job | Fetch current hosted evidence and artifact membership. | Candidate Day 2 commands; not run on intake day. |
| `python3 tests/test_selected_report_targets_manifest.py` | Validate selected manifest current absence or promoted exact metadata. | Required after manifest changes and during integrated validation. |
| `python3 tests/test_selected_comparison_workflow.py` | Validate workflow scope, selected artifact, and Windows non-claim boundaries. | Required after workflow or guard changes and during integrated validation. |
| `python3 tests/test_normalize_report_index.py` | Validate selected freshness, Windows path handling, stale/missing rows, and diagnostics. | Required after normalizer or selected freshness changes. |
| `python3 tests/test_run_external_comparison.py` | Validate generator behavior if Cholesky comparison generation changes. | Required if generator behavior changes. |
| `make windows-powershell-guard` | Run Windows workflow and PowerShell guard tests. | Required after Windows workflow or guard changes. |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Validate local selected Cholesky generated rows are fresh when generated evidence exists. | Candidate Day 13 validation. |
| `make docs-check` | Validate documentation surfaces if public or maintainer docs change. | Required after docs calibration. |
| `make format && make lint && make test` | Full C quality gate. | Required only if `.c` or `.h` files change. |
| `git diff --check` | Whitespace validation. | Required before closeout. |

## Initial Risk Register

| Risk | Day 1 mitigation |
| --- | --- |
| Stale Sprint 199 hosted evidence is reused as current promotion proof. | Day 2 must fetch latest hosted Windows evidence or document current blockers. |
| Manifest metadata is promoted without generated support tier and non-claim alignment. | Day 4 criteria will require manifest, generated metadata, docs, and guards to agree. |
| Windows artifact paths pass because of broad or near-match filtering. | Day 9-Day 10 will reassess Windows path and artifact mismatch regressions. |
| Workflow upload expands beyond exact selected files. | Day 8 guard work keeps upload scope exact and fail-closed. |
| Public docs imply broad Windows or package support. | Non-goals and claim boundaries are recorded before docs are edited. |
| Local missing PowerShell is misreported as pass evidence. | Sprint 208 will preserve hosted `--require-pwsh` as the authoritative PowerShell owner and classify local missing `pwsh` as unavailable evidence. |

## Day 1 Completion Criteria

| Criterion | Status |
| --- | --- |
| Every Sprint 208 item has an initial evidence path or artifact category. | Complete in `WORKING_NOTES.md`. |
| Existing Windows Cholesky promotion and re-deferral evidence is identified before edits. | Complete in this artifact and working notes. |
| Unsupported Windows, package, ABI, performance, release, and state-of-the-art claims remain explicitly out of scope. | Complete via the claim-boundary and non-goal records. |

## Day 1 Outcome

Day 1 is complete. Sprint 208 starts from a reviewed-but-re-deferred Windows
Cholesky state: the bounded Windows workflow path exists and prior hosted
evidence was inspected, but selected manifest metadata still omits `windows`
and current docs preserve the non-claim boundary. The next sprint step is to
fetch and inspect current hosted Windows artifact evidence before deciding
whether promotion is earned.

