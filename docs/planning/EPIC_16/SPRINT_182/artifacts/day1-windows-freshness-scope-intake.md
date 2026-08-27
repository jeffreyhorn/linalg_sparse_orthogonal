# Sprint 182 Day 1: Windows Freshness Scope Intake

## Purpose

Day 1 establishes Sprint 182 scope, inherited selected-target authority,
current Windows CI boundaries, and reusable decision criteria for the Windows
report freshness promotion-or-deferral decision.

Sprint 182 implements the Epic 16 project-plan section "Sprint 182: Windows
Report Freshness Decision". The sprint goal is to promote one Windows-safe
generated report freshness path or close Windows report freshness as an
explicit product deferral with guard coverage.

## Project-Plan Scope

| Item | Day 1 intake position |
| --- | --- |
| 182.1 Windows Report Audit | Start the audit by fixing the evidence sources and compatibility fields each candidate must satisfy. |
| 182.2 Candidate Selection | Defer selection until selected report commands, workflow shells, generated paths, dependencies, and runtime costs are audited. |
| 182.3 CI or Deferral Implementation | Preserve the implementation fork: either add a Windows hosted freshness lane or implement a formal deferral artifact and guard. |
| 182.4 Manifest Integration | Treat `tests/corpus/manifests/selected_report_targets.tsv` as the selected target authority any Windows status must update. |
| 182.5 Documentation Alignment | Track README, INSTALL, maintainer guide, workflow comments, and report-index language as claim surfaces. |
| 182.6 Validation | Use workflow guard tests, selected report checks, feasible CMake/PowerShell review, and whitespace review as the validation baseline. |

## Inherited Sprint 181 Authority

Sprint 181 leaves `tests/corpus/manifests/selected_report_targets.tsv` as the
selected report target authority. It owns selected target identifiers,
generator commands, artifact patterns, required files, expected rows,
workflow files, workflow jobs, upload artifact names, workflow platforms,
support tiers, freshness policies, claim scopes, non-claims, owners, and
provenance.

| Authority surface | Sprint 182 starting point |
| --- | --- |
| Selected oracle | `SRT-ORACLE-QR-PSVD-LOCAL` uses `make report-index-oracle-freshness` and is hosted on Linux only. |
| Selected comparison | Four selected comparison rows use `python3 scripts/run_external_comparison.py --target ...` and are hosted on Linux/macOS only. |
| Selected benchmark | `SRT-BENCH-REFACTOR-CSC-NOS4` uses `make bench-canonical-report-freshness` and is hosted on Linux only. |
| Workflow platforms | Current selected rows list `linux` or `linux;macos`; none list `windows`. |
| Non-claims | Current selected rows preserve `no Windows report freshness` and avoid broad platform, package-manager, shared-library ABI, performance superiority, and state-of-the-art claims. |

## Current Windows Boundary

`.github/workflows/windows-ci.yml` currently proves a CMake-first Windows
surface:

- MSVC configure and build through CMake;
- `ctest -N` count validation and full `ctest` execution;
- static-first CMake install and installed downstream consumer validation;
- installed static library, headers, CMake package metadata, and metadata-only
  `sparse.pc` inspection;
- explicit non-claims for Makefile parity, pkg-config execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, and broad Windows parity.

It does not run selected report freshness commands or upload selected report
freshness artifacts.

## Existing Guard Boundary

`tests/test_selected_comparison_workflow.py` includes
`test_windows_report_freshness_remains_unselected()`, which reads the Windows
workflow and rejects selected report freshness command names and selected
upload artifact names. The forbidden Windows workflow strings include:

- `report-index-oracle-freshness`
- `report-index-comparison-freshness`
- `bench-canonical-report-freshness`
- `check_bench_canonical_freshness.py`
- `sprint159-oracle-freshness`
- `sprint175-linux-selected-comparison-freshness`
- `sprint175-macos-selected-comparison-freshness`
- `sprint168-selected-performance-freshness`

Any Sprint 182 promotion must deliberately replace or narrow this guard with
manifest-backed Windows metadata. Any Sprint 182 deferral should preserve and
document the guard as the product boundary.

## Windows Freshness Candidate Evaluation Fields

| Field | Evaluation question |
| --- | --- |
| Shell compatibility | Does the command avoid POSIX-only shell syntax, tools, glob behavior, and Makefile assumptions under the reviewed Windows shell? |
| Path semantics | Are source, build, generated, and upload paths stable with Windows quoting, drive roots, separators, spaces, and temporary directories? |
| Newline behavior | Are generated files and freshness checks stable under CRLF/LF handling and text-mode I/O? |
| Executable availability | Are CMake, compiler tools, Python, scripts, and helper executables available in the reviewed hosted Windows lane? |
| Python dependency availability | Does the target depend only on source-controlled Python helpers and modules available on hosted Windows without package-manager setup? |
| Runtime cost | Can the target run inside the Windows CI budget without excessive latency or flake risk? |
| Artifact scope | Can the workflow upload exact selected artifacts and fail closed when required files are missing? |
| Support tier | What selected support tier would the Windows result create, and does it match documentation language? |
| Claim boundary | What narrow freshness claim would be allowed, and which broad report, package, ABI, parity, and performance claims remain unsupported? |
| Guardability | Can tests detect workflow/manifest drift and accidental unsupported Windows freshness claims clearly? |

## Day 1 Decisions

- Start Sprint 182 from the Sprint 181 selected target manifest, not from
  duplicated workflow prose.
- Keep the current Windows report freshness non-claim in force until a
  specific selected candidate passes the audit and receives manifest/workflow
  metadata.
- Evaluate every candidate against compatibility, artifact, support-tier,
  claim-boundary, and guardability criteria before deciding promotion or
  deferral.
- Treat formal deferral as a valid product outcome if no candidate can meet
  the reviewed Windows lane constraints without unsupported setup or broad
  claims.

## Day 2 Handoff

Day 2 should inspect `.github/workflows/windows-ci.yml`,
`.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, selected report
commands, and current workflow guard logic to map the exact Windows shell and
toolchain assumptions each candidate would inherit.

## Validation

Day 1 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 182 scope is tied to the Epic 16 project plan. | Complete | Project-plan scope table references items 182.1 through 182.6. |
| Inherited Windows non-claims and selected-target authority are explicit. | Complete | Inherited Sprint 181 authority, current Windows boundary, and existing guard boundary sections. |
| Audit work starts from concrete compatibility and claim-boundary criteria. | Complete | Windows freshness candidate evaluation field table and Day 2 handoff. |
