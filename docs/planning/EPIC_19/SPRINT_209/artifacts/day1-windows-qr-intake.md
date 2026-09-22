# Sprint 209 Day 1: Windows QR Incompatible Intake

## Purpose

Establish Sprint 209 scope, inherited QR incompatible evidence, selected
freshness boundaries, and validation owners before changing workflow, manifest,
guard, or documentation surfaces.

## Sprint 209 Scope

Sprint 209 is the Epic 19 Windows QR incompatible promotion-decision sprint. Its
goal is to add hosted Windows/MSVC proof for `qr-incompatible-ls` and promote
selected metadata only if the exact evidence supports it.

The six project-plan items are:

| Item | Day 1 interpretation |
| --- | --- |
| 209.1 MSVC Probe Design | Define the hosted MSVC/CMake proof command, exact artifact layout, and expected QR incompatible rows. |
| 209.2 Workflow Implementation | Add or update hosted Windows QR proof steps without broad Windows promotion. |
| 209.3 Artifact Inspection Tests | Add tests for Windows-style QR artifact paths, row filtering, generated rows, stale artifacts, and missing files. |
| 209.4 Manifest Decision | Promote or re-defer `SRT-COMP-QR-INCOMPATIBLE-LS` Windows metadata based on hosted proof. |
| 209.5 Docs And Claim Guards | Update README, INSTALL, corpus docs, maintainer guide, and Windows/manifest guards for the chosen decision. |
| 209.6 Validation And Closeout | Run QR generator/freshness, normalizer, manifest, workflow, PowerShell, docs, and applicable C gates. |

## Inherited Evidence

| Source | Current evidence |
| --- | --- |
| Sprint 203 retrospective | Local `qr-incompatible-ls` generator and selected freshness proof passed, but hosted Windows/MSVC QR proof and hosted artifact inspection were absent. |
| Sprint 203 Day 2 MSVC probe design | The intended Windows command is target-specific MSVC/CMake generation plus target-specific freshness for `qr-incompatible-ls`. |
| Sprint 203 Day 7 manifest decision | Selected Windows QR promotion was re-deferred because no hosted proof or artifact review existed. |
| Sprint 203 Day 8 manifest metadata | QR incompatible selected manifest row remains Linux/macOS-only, `local_only`, and protected by exact row/required-file/non-claim checks. |
| Sprint 203 Day 9 workflow guard integration | Windows workflow guards reject accidental QR incompatible commands, selected freshness commands, artifact names, subfamily tokens, and upload paths. |
| Sprint 203 Day 12 integrated validation | Local QR generator and freshness passed; hosted Windows/MSVC proof remained absent. |
| Sprint 203 Day 13 review hardening | Guard coverage protects re-deferral across manifest, workflow, normalizer diagnostics, and documentation markers. |
| Epic 18 residual queue | Windows QR incompatible comparison promotion remains future work requiring MSVC/CMake proof, artifact inspection, and exact manifest metadata. |
| Epic 19 todo review | Closure track 3 asks for hosted MSVC/CMake proof, artifact inspection, and promotion only if metadata and claims are evidence-backed. |
| Sprint 208 retrospective | QR incompatible Windows selected freshness remains a future Sprint 209 owner after bounded Cholesky evidence work. |

## Current Source-Controlled Surfaces

| Surface | Day 1 role |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Source of truth for selected target metadata. Current QR incompatible row lists Linux/macOS only and retains `no Windows report freshness`. |
| `.github/workflows/windows-ci.yml` | Current Windows workflow owns bounded Cholesky selected comparison proof only; no QR incompatible lane is present. |
| `.github/workflows/ci.yml` and `.github/workflows/macos-ci.yml` | Current Linux/macOS selected comparison freshness lanes include QR incompatible. |
| `scripts/run_external_comparison.py` | Generates the QR incompatible comparison rows locally and is the candidate hosted MSVC proof command owner. |
| `scripts/normalize_report_index.py` | Performs selected target filtering and freshness diagnostics. |
| `scripts/validate_windows_powershell.py` | Owns Windows workflow and claim-boundary validation, including QR incompatible re-deferral markers. |
| `tests/test_selected_report_targets_manifest.py` | Guards QR incompatible row identity, required files, expected row IDs, non-claims, and absent Windows metadata. |
| `tests/test_selected_comparison_workflow.py` | Rejects accidental QR incompatible Windows workflow commands or artifact uploads. |
| `tests/test_normalize_report_index.py` | Holds QR incompatible Windows-style path and selected row-set diagnostics. |
| README, INSTALL, corpus README, maintainer guide | Public and maintainer claim surfaces currently keep QR incompatible outside Windows selected freshness. |

## Current Claim Boundary

Sprint 209 starts from this bounded state:

- local QR incompatible generator proof exists;
- local selected QR incompatible freshness proof exists;
- Windows-style QR artifact path and row-set diagnostics exist;
- no hosted Windows/MSVC QR incompatible proof is source-controlled as pass
  evidence;
- no hosted Windows QR artifact inspection has been recorded for promotion;
- selected manifest metadata still omits `windows`;
- public and maintainer docs still describe QR incompatible Windows selected
  freshness as deferred.

Sprint 209 must not infer the following without explicit evidence and matching
metadata/docs/guards:

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
- package-manager support;
- shared-library or dynamic ABI support;
- runtime-loader behavior;
- broad Windows parity;
- performance superiority;
- release readiness;
- state-of-the-art status.

## Initial Validation Matrix

| Command | Purpose | Day 1 disposition |
| --- | --- | --- |
| `gh run list` / `gh run view` / `gh run download` for Windows QR evidence | Fetch current hosted evidence and artifact membership after a QR lane exists. | Candidate Day 3 commands; not run on intake day. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Regenerate local QR incompatible comparison artifacts. | Required before local freshness validation. |
| `python3 scripts/normalize_report_index.py --family comparison --include-generated --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Validate local selected QR incompatible generated rows are fresh. | Candidate Day 6 and Day 12 validation. |
| `python3 tests/test_selected_report_targets_manifest.py` | Validate current re-deferral or future exact promotion metadata. | Required after manifest changes and during integrated validation. |
| `python3 tests/test_selected_comparison_workflow.py` | Validate workflow scope, QR absence guards, or QR owned workflow path. | Required after workflow or guard changes. |
| `python3 tests/test_normalize_report_index.py` | Validate selected freshness, Windows path handling, stale/missing rows, and diagnostics. | Required after normalizer or selected freshness changes. |
| `python3 tests/test_run_external_comparison.py` | Validate comparison generator behavior if QR generation changes. | Required if generator behavior changes. |
| `make windows-powershell-guard` | Run Windows workflow and PowerShell guard tests. | Required after Windows workflow or guard changes. |
| `make docs-check` | Validate documentation surfaces if public or maintainer docs change. | Required after docs calibration. |
| `make format && make lint && make test` | Full C quality gate. | Required only if `.c` or `.h` files change. |
| `git diff --check` | Whitespace validation. | Required before closeout. |

## Initial Risk Register

| Risk | Day 1 mitigation |
| --- | --- |
| Local Sprint 203 QR proof is reused as hosted Windows proof. | Day 3 must fetch hosted Windows evidence or document current blockers. |
| Bounded Windows Cholesky evidence is generalized to QR incompatible. | Day 2-Day 8 must require exact QR target identity, row IDs, artifacts, and manifest fields. |
| Manifest metadata is promoted without generated support tier and non-claim alignment. | Day 7 criteria require manifest, generated metadata, docs, and guards to agree. |
| Workflow upload expands beyond exact selected QR files. | Day 4-Day 5 workflow design keeps uploads target-specific and fail-closed. |
| Windows artifact paths pass because of broad or near-match filtering. | Day 6 revisits Windows path and row-set diagnostics. |
| Public docs imply broad Windows, QR, package, ABI, or external-library support. | Non-goals and claim boundaries are recorded before docs are edited. |
| Local missing PowerShell is misreported as pass evidence. | Hosted `--require-pwsh` remains the authoritative PowerShell owner; local missing `pwsh` is unavailable evidence only. |

## Day 1 Completion Criteria

| Criterion | Status |
| --- | --- |
| Every Sprint 209 item has an initial evidence path or artifact category. | Complete in `WORKING_NOTES.md`. |
| Existing QR incompatible promotion and re-deferral evidence is identified before edits. | Complete in this artifact and working notes. |
| Unsupported Windows, package, ABI, performance, release, external-library, and state-of-the-art claims remain explicitly out of scope. | Complete via the claim-boundary and non-goal records. |

## Day 1 Outcome

Day 1 is complete. Sprint 209 starts from a local-proof-but-re-deferred QR
incompatible state: local generator/freshness evidence and guard coverage exist,
but hosted Windows/MSVC QR proof and hosted artifact inspection remain unproven
for promotion. The next sprint step is to design the exact MSVC probe and
artifact contract before adding or changing hosted workflow metadata.
