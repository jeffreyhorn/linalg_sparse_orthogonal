# Sprint 190 Day 1: Windows Freshness Intake

## Purpose

Establish the Sprint 190 decision baseline for Windows selected report
freshness, including prior deferral evidence, accepted Windows support,
candidate lanes, owner surfaces, decision criteria, and Day 2 audit questions.

## Sprint 190 Decision Frame

Sprint 190 must end in exactly one of two states:

1. Promote one Windows-safe selected report freshness lane with source,
   manifest, workflow, artifact, guard, documentation, and hosted evidence.
2. Renew the formal Windows report freshness deferral with sharper blockers,
   stronger guards, and exact revisit criteria.

Partial promotion is not acceptable. Broad Windows report freshness is not in
scope.

## Prior Evidence Baseline

| Source | Relevant Day 1 conclusion |
| --- | --- |
| Sprint 182 deferral decision | Windows report freshness remains formally deferred because selected generators still lack reviewed Windows-safe command, CMake/MSVC probe, `.lib` link, `.exe` execution, and artifact-upload paths. |
| Sprint 187 Windows acceptance gates | Sprint 190 must promote exactly one lane or renew the deferral; local-only source checks are sufficient only for deferral and guard work. |
| Sprint 189 closeout | PowerShell validation ownership is complete, but it intentionally does not run report generators or publish selected report artifacts. |
| Selected report target manifest | Current selected rows list `linux` or `linux;macos`, never `windows`. |
| Windows CI workflow | Windows currently proves CMake/MSVC build/test, CMake install/downstream package validation, and PowerShell validation ownership only. |

## Current Accepted Windows Evidence

- Hosted CMake configure/build/test on `windows-2022`.
- Hosted `ctest -N` registration count and full `ctest`.
- Hosted CMake install/downstream validation for the maintained static-first
  package surface.
- Static `.lib`, headers, CMake package metadata, exact-version and
  mismatch-version checks, and metadata-only `sparse.pc` inspection.
- Hosted PowerShell validation ownership for selected workflow snippets.

## Current Non-Evidence

- Selected oracle freshness on Windows.
- Selected comparison freshness on Windows.
- Selected benchmark freshness on Windows.
- Broad generated report freshness on Windows.
- Windows Makefile parity.
- Windows `pkg-config` execution parity.
- Package-manager support.
- Shared-library, dynamic ABI, DLL/import-library, or runtime-loader support.
- Portable performance, performance superiority, or state-of-the-art evidence.

## Candidate Inventory

| Candidate | Manifest row | Generator command | Expected rows | Current platforms | Day 1 disposition |
| --- | --- | --- | ---: | --- | --- |
| Cholesky selected comparison | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` | 6 | `linux;macos` | First-ranked candidate because it is small, recent, and claim-narrow. |
| LU selected comparison | `SRT-COMP-LU-NONSYM-SQUARE-5` | `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5` | 6 | `linux;macos` | Second-ranked candidate with small row count and bounded square-solve claim. |
| QR minimum-norm selected comparison | `SRT-COMP-QR-MINNORM` | `python3 scripts/run_external_comparison.py --target qr-minnorm` | 6 | `linux;macos` | Viable but carries broader QR/minimum-norm wording risk. |
| QR compatible LS selected comparison | `SRT-COMP-QR-COMPATIBLE-LS` | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | 6 | `linux;macos` | Viable but not clearly safer than Cholesky or LU. |
| Partial-SVD selected comparison | `SRT-COMP-PSVD-DIAG6-K2` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | 10 | `linux;macos` | Larger diagnostic surface; keep behind smaller comparison candidates. |
| Canonical benchmark selected freshness | `SRT-BENCH-REFACTOR-CSC-NOS4` | `make bench-canonical-report-freshness` | 1 | `linux` | Row count is attractive, but Bash/performance-adjacent risk makes it a poor first promotion. |
| Selected oracle QR/partial-SVD freshness | `SRT-ORACLE-QR-PSVD-LOCAL` | `make report-index-oracle-freshness` | 52 | `linux` | Too broad and Makefile-dependent for first Windows promotion. |

## Initial Ranking

1. `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`
2. `SRT-COMP-LU-NONSYM-SQUARE-5`
3. `SRT-COMP-QR-MINNORM`
4. `SRT-COMP-QR-COMPATIBLE-LS`
5. `SRT-COMP-PSVD-DIAG6-K2`
6. `SRT-BENCH-REFACTOR-CSC-NOS4`
7. `SRT-ORACLE-QR-PSVD-LOCAL`

The Day 1 preference is to investigate the selected comparison family first,
with Cholesky and LU as the strongest initial candidates. Oracle and benchmark
promotion should be rejected early unless Day 2 finds comparison promotion is
blocked and a narrower alternative exists.

## Promotion Acceptance Criteria

A promoted lane must include:

- one exact selected manifest row;
- one exact generator command;
- Windows-safe execution that avoids Makefile, Bash, Unix archive/link, Unix
  `-lm`, extensionless executable, and POSIX-only path assumptions;
- CMake/MSVC probe build/link support where compiled probes are involved;
- `.exe`-aware probe execution;
- exact workflow file, job, artifact name, required files, expected row count,
  and expected row IDs;
- hard failure on missing hosted artifacts;
- guard updates that allow only the promoted Windows lane;
- claim-calibrated README, INSTALL, maintainer, and corpus/report docs;
- hosted Windows evidence before the promotion is considered complete.

## Renewed Deferral Acceptance Criteria

A renewed deferral must include:

- a refreshed decision record;
- exact remaining command, runtime, manifest, artifact, and claim blockers;
- guard evidence that Windows CI still does not run selected report freshness
  commands or upload selected report artifacts;
- manifest evidence that selected rows do not list `windows`;
- concrete future revisit criteria for one candidate lane;
- documentation that keeps Windows report freshness unclaimed.

## Owner Surfaces

| Surface | Why it matters |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Owns any future Windows report freshness job or refreshed no-promotion guard. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Structured source of truth for selected row, platform, workflow, artifact, and expected-row metadata. |
| `scripts/run_external_comparison.py` | Likely generator implementation surface for a selected comparison promotion. |
| `scripts/normalize_report_index.py` | Existing report freshness validation and normalized-index owner. |
| `scripts/validate_corpus_schema.py` | Manifest schema and metadata coherence owner. |
| `scripts/validate_windows_powershell.py` | Existing Windows workflow non-promotion, artifact, and claim-boundary guard. |
| `tests/test_selected_report_targets_manifest.py` | Current deferral/platform guard test owner. |
| `tests/test_selected_comparison_workflow.py` | Current workflow command/artifact guard test owner. |
| `tests/test_validate_windows_powershell.py` | Current Windows workflow and claim-boundary regression owner. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | Public and maintainer claim-calibration surfaces. |

## Day 2 Audit Questions

1. Which selected comparison target has the least generator/probe/link work to
   make Windows-safe?
2. Can the generator reuse the reviewed CMake/MSVC build output, or does it
   need a separate CMake probe mode?
3. What exact hosted artifact should one promoted Windows comparison lane
   upload?
4. Which schema checks currently assume Windows is always deferred, and how
   would they become exact allowlists?
5. If promotion remains blocked, what stronger evidence should the refreshed
   deferral record contain?

## Validation

Day 1 performed read-only/source discovery only. No `.c` or `.h` files were
modified, and no behavior changed, so `make format && make lint && make test`
is not required.

`git diff --check` should be run after the Day 1 artifact and working notes are
added.
