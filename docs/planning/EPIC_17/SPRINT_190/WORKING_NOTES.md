# Sprint 190 Working Notes: Windows Selected Report Freshness Decision

## Sprint Goal

Promote one Windows-safe selected report freshness lane or renew the formal
deferral with stronger guard evidence.

## Day 1: Windows Freshness Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 190.1 Freshness Candidate Selection | Build the candidate ledger from selected report manifest rows, current workflow jobs, generator commands, expected row counts, and Sprint 182 blockers. |
| 190.2 Workflow Implementation | Identify which workflow surfaces would need change if promotion is selected, or which guard surfaces must stay no-op if deferral is renewed. |
| 190.3 Manifest And Schema Updates | Identify current manifest fields that omit `windows` and the schema tests that enforce the active deferral. |
| 190.4 Freshness Guard | Identify existing guard owners that forbid Windows selected freshness commands, artifact uploads, and `windows` platform metadata. |
| 190.5 Docs And Claim Calibration | Identify public and maintainer docs that currently state Windows report freshness is deferred. |
| 190.6 Validation | Define the minimum Day 1 source validation and future hosted-evidence requirements. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 190 has one accepted goal: promote one Windows-safe selected report freshness lane or renew the formal deferral. |
| `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` | Windows report freshness remains formally deferred because selected generators lack reviewed Windows-safe Makefile-free/CMake-MSVC probe and artifact paths. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day8-windows-acceptance-gates.md` | Sprint 190 must finish in one of two states: exactly one selected Windows lane promoted with evidence, or a renewed deferral with stronger blockers and guards. |
| `docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md` | Sprint 189 completed PowerShell validation ownership but intentionally did not promote Windows report generation or artifact upload. |
| `tests/corpus/manifests/selected_report_targets.tsv` | All selected report rows currently omit `windows` from `workflow_platforms`. |
| `.github/workflows/windows-ci.yml` | Windows CI currently runs CMake/MSVC build/test, CMake install/downstream validation, and PowerShell validation ownership only. |
| `scripts/validate_windows_powershell.py` | Current guard forbids selected report generation commands and selected report artifact uploads in Windows CI while the Sprint 182 deferral remains active. |

### Current Windows Evidence Boundary

Accepted Windows evidence at Day 1:

- hosted `windows-2022` CMake configure/build/test through MSVC;
- hosted `ctest -N` test-surface registration and full `ctest`;
- hosted CMake install/downstream validation for the maintained static-first
  package surface;
- installed static `.lib`, headers, CMake package metadata, exact-version and
  mismatch-version checks, and metadata-only `sparse.pc` inspection;
- hosted PowerShell validation ownership for selected Windows workflow
  snippets.

Current non-evidence:

- selected oracle freshness on Windows;
- selected comparison freshness on Windows;
- selected benchmark freshness on Windows;
- broad generated report freshness on Windows;
- Windows Makefile parity;
- Windows `pkg-config` command execution parity;
- package-manager support, shared-library support, dynamic ABI support,
  runtime-loader support, portable performance, or state-of-the-art claims.

### Candidate Ledger

| Candidate | Manifest target | Generator | Expected rows | Current hosted platforms | Initial Day 1 risk |
| --- | --- | --- | ---: | --- | --- |
| Selected comparison: QR minimum-norm | `SRT-COMP-QR-MINNORM` | `python3 scripts/run_external_comparison.py --target qr-minnorm` | 6 | `linux;macos` | Best-sized comparison candidate, but still needs Windows-safe CMake/MSVC probe/link and `.exe` execution path. |
| Selected comparison: QR compatible LS | `SRT-COMP-QR-COMPATIBLE-LS` | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | 6 | `linux;macos` | Similar to QR minimum-norm; small but probe/link assumptions must be proven. |
| Selected comparison: LU nonsymmetric square | `SRT-COMP-LU-NONSYM-SQUARE-5` | `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5` | 6 | `linux;macos` | Small row count, but direct probe compilation/link behavior must be made Windows-safe. |
| Selected comparison: Cholesky SPD tridiagonal | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` | 6 | `linux;macos` | Smallest recent family candidate; same Windows probe/link risk as other comparison rows. |
| Selected comparison: partial-SVD diag6 k2 | `SRT-COMP-PSVD-DIAG6-K2` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | 10 | `linux;macos` | Larger comparison row set and partial-SVD numerical diagnostics; less attractive for first Windows promotion. |
| Selected oracle QR/partial-SVD | `SRT-ORACLE-QR-PSVD-LOCAL` | `make report-index-oracle-freshness` | 52 | `linux` | Broad selected oracle surface and Makefile wrapper dependency; not a first-choice Sprint 190 promotion. |
| Selected benchmark canonical nos4 | `SRT-BENCH-REFACTOR-CSC-NOS4` | `make bench-canonical-report-freshness` | 1 | `linux` | One row but benchmark/performance-adjacent and Bash/Unix report-generator assumptions make claim risk high. |

### Day 1 Initial Ranking

| Rank | Candidate | Reason |
| ---: | --- | --- |
| 1 | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | Six rows, focused SPD fixture, newest selected comparison family, and relatively narrow claim surface. |
| 2 | `SRT-COMP-LU-NONSYM-SQUARE-5` | Six rows and simple square solve claim, but LU probe/link behavior still needs Windows proof. |
| 3 | `SRT-COMP-QR-MINNORM` | Six rows and strong existing selected comparison precedent, but QR minimum-norm claim wording is easier to overbroaden. |
| 4 | `SRT-COMP-QR-COMPATIBLE-LS` | Six rows, but overlaps QR wording/claim risk and is not clearly safer than Cholesky or LU. |
| 5 | `SRT-COMP-PSVD-DIAG6-K2` | Ten rows and more numerical diagnostics; keep as fallback only if comparison infrastructure becomes easy. |
| 6 | `SRT-BENCH-REFACTOR-CSC-NOS4` | One row but performance-adjacent, Bash-based, and higher claim-risk. |
| 7 | `SRT-ORACLE-QR-PSVD-LOCAL` | Too broad for the first Windows freshness promotion. |

### Decision Criteria

Promotion is acceptable only if the selected lane has:

- one exact manifest row and one exact generator command;
- a Windows-safe generator path that avoids Makefile, Bash, Unix archive/link,
  extensionless executable, and `-lm` assumptions;
- CMake/MSVC project probe build/link support against the reviewed static
  package shape if a compiled probe is needed;
- `.exe`-aware temporary executable handling;
- exact hosted workflow job, artifact name, required files, expected rows, and
  row IDs;
- artifact upload with hard failure on missing files;
- guard updates that allow only the selected Windows lane and reject broader
  Windows selected freshness;
- docs that retain all unsupported Windows/package/performance non-claims.

Renewed deferral is acceptable only if it has:

- a refreshed decision record;
- an exact blocker list grounded in Day 2/Day 3 evidence;
- guard evidence proving Windows CI still does not run selected report
  freshness commands or upload selected report artifacts;
- selected target manifest rows without `windows`;
- concrete revisit criteria for one future lane.

### Owner Surfaces

| Surface | Day 1 owner role |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Promotion or deferral workflow owner; currently must stay free of selected report generation and selected report artifact uploads. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Manifest authority for selected report target metadata and future Windows platform inclusion. |
| `scripts/validate_windows_powershell.py` | Current Windows workflow/report non-promotion and claim-boundary guard. |
| `scripts/validate_corpus_schema.py` | Manifest schema and selected report metadata validation owner. |
| `scripts/normalize_report_index.py` | Selected report freshness and normalized report index validation owner. |
| `scripts/run_external_comparison.py` | Likely implementation owner for any selected comparison promotion. |
| `scripts/check_bench_canonical_freshness.py` | Candidate owner only if benchmark promotion is selected. |
| `tests/test_selected_report_targets_manifest.py` | Deferral or exact-allowlist test owner for manifest Windows platform metadata. |
| `tests/test_selected_comparison_workflow.py` | Workflow guard owner for selected freshness command/artifact allowlist behavior. |
| `tests/test_validate_windows_powershell.py` | Windows PowerShell validation and non-promotion guard test owner. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | Claim calibration and maintainer evidence interpretation surfaces. |

### Initial Risks

| Risk | Why it matters | Day 2 question |
| --- | --- | --- |
| Generator uses Unix compile/link assumptions | Windows promotion cannot depend on `cc`, `.a`, `-lm`, or extensionless probe execution. | Which selected comparison probes can be made CMake/MSVC-safe with the least change? |
| Manifest currently encodes non-Windows status | Promotion needs structured metadata, not prose-only documentation. | Which row can be updated without implying broader family promotion? |
| Workflow artifact policy is currently no-upload | Promotion requires a precise artifact name and upload scope. | What exact required files should a Windows artifact contain? |
| Benchmark lane has performance-adjacent wording | A one-row benchmark is tempting but could imply portable performance. | Should benchmark remain out of Sprint 190 first-promotion scope? |
| Oracle lane is broad | A 52-row oracle lane is unlikely to be the smallest credible Windows path. | Should Day 2 reject oracle early unless comparison is impossible? |
| Hosted evidence is required for promotion | Local source checks alone cannot prove Windows report freshness. | Can Sprint 190 trigger or rely on hosted Windows CI evidence after implementation? |

### Day 1 Validation

Read-only/source checks:

- `git status --short --branch`
- `sed -n '133,165p' docs/planning/EPIC_17/PROJECT_PLAN.md`
- targeted reads of Sprint 182, Sprint 187, and Sprint 189 Windows artifacts
- selected report manifest inspection with `column -t -s $'\t'
  tests/corpus/manifests/selected_report_targets.tsv`

No code files were changed on Day 1, so `make format && make lint &&
make test` is not required.

### Day 2 Questions

1. Which six-row selected comparison candidate has the least Windows probe and
   claim risk after reading `scripts/run_external_comparison.py`?
2. Can a Windows-safe generator path reuse the reviewed CMake install/build
   outputs, or does it need a new probe mode?
3. What exact artifact files would be uploaded for one selected Windows
   comparison lane?
4. Should benchmark and oracle candidates be rejected early to keep Sprint 190
   focused?

## Day 2: Candidate Lane Audit

### Generator Findings

`scripts/run_external_comparison.py` keeps the selected comparison targets in a
single Python generator, but the project-probe path is currently Unix-shaped:

- `compiler_argv()` defaults to `CC` or `cc`;
- `ensure_library()` builds a missing library by running `make <target>`;
- `run_project_probe()` links the generated C probe against
  `build/libsparse_lu_ortho.a`;
- the compile command adds Unix-style `-std=c99`, include flags, static
  archive path, `-lm`, and `-o`;
- the temporary executable path is extensionless;
- the probe runs by executing `str(binary)` directly.

Those assumptions apply to all selected comparison candidates, including the
six-row Cholesky, LU, QR minimum-norm, and QR compatible least-squares lanes.
No selected comparison lane is already Windows-safe in the current generator.

### Candidate Audit Matrix

| Candidate | Runtime/generator risk | Artifact risk | Claim risk | Day 2 disposition |
| --- | --- | --- | --- | --- |
| `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | Medium: same C probe/link work as other comparison lanes, but small fixture and no external dependency beyond source-controlled dense helper. | Low-medium: existing required files are the standard six comparison files under one subdirectory. | Low: fixture-local SPD tridiagonal solve is narrow and already has explicit non-claims. | Preferred promotion candidate. |
| `SRT-COMP-LU-NONSYM-SQUARE-5` | Medium: same C probe/link work; direct linked-list LU solve has simple expected solution rows. | Low-medium: standard comparison artifact shape. | Medium-low: square-solve language is narrow, but LU pivoting/non-symmetric wording needs careful non-claims. | First fallback promotion candidate. |
| `SRT-COMP-QR-MINNORM` | Medium: same C probe/link work; QR minimum-norm probe uses generated fixture metadata. | Low-medium: standard comparison artifact shape. | Medium: minimum-norm wording is easy to overread as broad QR or SVD-pseudoinverse parity. | Second fallback promotion candidate. |
| `SRT-COMP-QR-COMPATIBLE-LS` | Medium: same C probe/link work; overdetermined solve path requires QR factor/solve cleanup. | Low-medium: standard comparison artifact shape. | Medium: compatible least-squares wording must avoid broad least-squares/QR claims. | Third fallback promotion candidate. |
| `SRT-COMP-PSVD-DIAG6-K2` | Medium-high: same C probe/link work plus partial-SVD diagnostics and ten expected rows. | Low-medium: standard comparison artifact shape. | Medium-high: singular-vector orientation and partial-SVD convergence non-claims are more extensive. | Keep as late fallback only. |
| `SRT-BENCH-REFACTOR-CSC-NOS4` | High: Makefile and Bash-oriented freshness command, benchmark binary behavior, and performance-adjacent environment sensitivity. | Medium: one selected CSV plus contextual files, but benchmark artifacts are easy to overinterpret. | High: benchmark freshness can be mistaken for portable performance. | Reject for first Windows promotion. |
| `SRT-ORACLE-QR-PSVD-LOCAL` | High: Makefile wrapper and broad 52-row selected oracle surface. | Medium-high: globbed oracle TSV output and corpus report files are broader than one lane. | Medium-high: broader QR/partial-SVD oracle wording. | Reject for first Windows promotion. |

### Selected Candidate

Day 2 selects `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` as the primary Sprint 190
promotion candidate.

Reasons:

- expected output is bounded to six selected comparison rows;
- artifact scope is one subdirectory,
  `build/comparison/cholesky_spd_tridiag_5/`;
- required files already match the standard comparison bundle:
  `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`;
- claim scope is narrow: fixture-local Cholesky SPD tridiagonal solve
  comparison only;
- non-claims already exclude broad Cholesky correctness, SPD coverage,
  reordering, fill superiority, Windows report freshness, package-manager
  proof, shared-library ABI proof, performance superiority, and
  state-of-the-art status.

### Required Promotion Work

The candidate can proceed only if Day 3 proves or designs:

- a Windows-safe project probe path that does not call `make`;
- a CMake/MSVC-compatible way to build the generated probe;
- linkage against the reviewed static `.lib` shape or another reviewed
  Windows build artifact;
- `.exe`-aware temporary probe execution;
- hosted workflow command and artifact upload scoped to the Cholesky
  comparison directory only;
- manifest metadata that adds `windows` only for
  `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`;
- guard updates that allow only the Cholesky Windows lane and keep every other
  selected Windows freshness lane blocked.

### Guard Surfaces To Convert If Promotion Proceeds

| Guard surface | Current behavior | Promotion update needed |
| --- | --- | --- |
| `scripts/validate_windows_powershell.py` | Forbids selected report freshness commands and selected artifact names in Windows CI. | Replace broad forbiddance with an allowlist for one exact Windows Cholesky command/job/artifact while continuing to reject other selected lanes. |
| `tests/test_validate_windows_powershell.py` | Tests the current no-promotion and claim-boundary behavior. | Add positive Cholesky allowlist coverage and negative drift coverage for other targets/artifacts. |
| `tests/test_selected_comparison_workflow.py` | Requires Windows report freshness to remain formally deferred and rejects selected Windows commands/artifacts. | Split into selected Cholesky Windows allowlist checks plus retained no-promotion checks for unselected lanes. |
| `tests/test_selected_report_targets_manifest.py` | Asserts no selected manifest row lists `windows` while the Sprint 182 deferral is active. | Allow exactly the selected Cholesky row to list `windows` only after a new Sprint 190 decision record supersedes the broad deferral. |
| `scripts/validate_corpus_schema.py` | Validates hosted metadata consistency but does not currently express a one-row Windows promotion allowlist. | Add or retain schema checks so Windows platform metadata cannot be added broadly or with mismatched artifact names. |

### Fallback Deferral Path

If Day 3 cannot produce a credible Windows-safe CMake/MSVC probe plan, Sprint
190 should renew the deferral rather than partially promote. The refreshed
deferral should preserve current no-Windows manifest metadata and add sharper
blockers:

- comparison generator still depends on Unix-style `cc`, `.a`, `-lm`, and
  extensionless probe execution;
- no reviewed CMake/MSVC generated-probe helper exists yet;
- no exact Windows artifact upload path has been proven against hosted output;
- current guards intentionally reject Windows selected freshness and would
  require allowlist conversion before promotion.

### Day 2 Validation

Commands run:

- `git status --short --branch`
- `rg -n "def .*comparison|target|compile|subprocess|\\.exe|cc|CMake|library|solver_library|run_external_comparison|cholesky-spd-tridiag-5|lu-nonsym-square-5|qr-minnorm|qr-compatible-ls|partial-svd-diag6-k2" scripts/run_external_comparison.py`
- `sed -n '70,250p' scripts/run_external_comparison.py`
- `sed -n '380,880p' scripts/run_external_comparison.py`
- `sed -n '1,220p' tests/test_selected_report_targets_manifest.py`
- `sed -n '1,470p' tests/test_selected_comparison_workflow.py`

Day 2 changed only planning documentation. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

### Day 3 Questions

1. Can `run_external_comparison.py` add a Windows/CMake probe mode without
   disrupting Linux and macOS selected comparison freshness?
2. Should the Windows Cholesky candidate use the already-built workflow tree or
   perform a separate install/downstream-style probe build?
3. What should the exact Windows artifact name be if promotion proceeds?
4. Which existing deferral text must be superseded by a Sprint 190 decision
   record before any manifest row can list `windows`?

## Day 3: Feasibility Probe

### Probe Command

Day 3 ran the selected Cholesky comparison generator locally:

```sh
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5
```

Result: pass on the local Darwin host. The command wrote the expected six-file
bundle under `build/comparison/cholesky_spd_tridiag_5/`:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

The generated `study.tsv` has six selected rows plus the header, matching the
manifest row for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.

### Feasibility Findings

| Finding | Evidence | Impact |
| --- | --- | --- |
| Local Cholesky comparison generation succeeds | Generator exited `0` and wrote all six expected files. | Candidate remains viable from a report-shape and row-count standpoint. |
| Current project probe is Unix-shaped | Generated manifest records `cc -std=c99 ... build/libsparse_lu_ortho.a -lm -o ... && .../cholesky_spd_tridiag_5_probe`. | Cannot promote Windows freshness without a Windows-safe probe mode. |
| Local output is not promotion evidence | Manifest records `platform=darwin-x86_64`, `source_branch=sprint-190`, and `worktree_state=dirty`. | Output is feasibility evidence only, not hosted Windows freshness evidence. |
| Required artifact shape is precise | Required files are the standard comparison bundle and `artifact_pattern` is `build/comparison/cholesky_spd_tridiag_5/study.tsv`. | A future Windows upload can be narrow and reviewable. |
| Claim surface remains narrow | Study rows retain `fixture-local Cholesky SPD tridiagonal solve comparison only` and broad non-claims. | Candidate remains the lowest claim-risk promotion target. |

### Windows-Safe Command Sequence Sketch

Promotion should use a Windows-specific generator/probe path rather than the
current Unix compile command. The likely command sequence is:

1. Configure and build the library through CMake/MSVC on `windows-2022`.
2. Invoke the comparison generator for only the Cholesky target with explicit
   Windows build metadata, not the default `make` fallback.
3. Build the generated Cholesky probe through CMake/MSVC or a reviewed MSVC
   command helper.
4. Link against the reviewed static `.lib` artifact from the CMake build.
5. Execute the generated probe with `.exe`-aware path handling.
6. Validate the six selected study rows and upload only the Cholesky comparison
   bundle with `if-no-files-found: error`.

### Runtime Estimate

Local generator runtime was small, approximately three seconds after existing
build artifacts were available. Hosted Windows runtime risk is dominated by
CMake/MSVC configure/build and generated-probe compilation, not by the
comparison row writer. A Sprint 190 hosted lane should set a conservative
timeout around the selected job, with the first implementation targeting a
single Cholesky comparison command and no broad report regeneration.

### Day 3 Decision Checkpoint

Continue toward promotion, but only through a Windows-safe Cholesky comparison
probe mode. Day 3 does not promote Windows report freshness.

The selected candidate remains:

- `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`
- `python3 scripts/run_external_comparison.py --target
  cholesky-spd-tridiag-5`
- expected rows: `6`
- expected report directory: `build/comparison/cholesky_spd_tridiag_5/`
- likely Windows artifact name: `sprint190-windows-selected-comparison-cholesky`

### Fallback Decision

If Day 4/Day 5 cannot define a reviewable Windows-safe CMake/MSVC probe
contract, renew the formal deferral. The refreshed deferral should cite the
same concrete blockers confirmed by Day 3:

- default `cc` compiler selection;
- `make` fallback for missing static library;
- Unix archive path `build/libsparse_lu_ortho.a`;
- Unix `-lm` link flag;
- extensionless probe executable;
- lack of hosted Windows output/artifact proof.

### Day 3 Validation

Commands run:

- `git status --short --branch`
- `python3 scripts/run_external_comparison.py --target
  cholesky-spd-tridiag-5`
- `sed -n '1,80p'
  build/comparison/cholesky_spd_tridiag_5/manifest.tsv`
- `sed -n '1,20p'
  build/comparison/cholesky_spd_tridiag_5/study.tsv`
- `wc -l build/comparison/cholesky_spd_tridiag_5/*.tsv
  build/comparison/cholesky_spd_tridiag_5/summary.md`

Day 3 generated ignored local report output under `build/` and changed only
planning documentation. No `.c` or `.h` files were modified, so `make format &&
make lint && make test` is not required.

### Day 4 Questions

1. Should the Sprint 190 decision record select promotion conditioned on
   implementing a Windows-safe Cholesky probe, or state promotion is still
   provisional until hosted evidence exists?
2. What exact manifest fields must change for
   `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` if the Windows lane is implemented?
3. Which no-Windows guard should own the transition from broad deferral to
   exact Cholesky allowlist?
4. Should the refreshed deferral artifact be drafted now as an explicit
   fallback even if promotion continues?

## Day 4: Decision Record Draft

### Draft Decision

Sprint 190 selects a provisional promotion path for exactly one selected
Windows report freshness lane:
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.

The decision is conditional. It becomes an accepted promotion only after the
branch implements a Windows-safe Cholesky comparison probe, updates manifest
metadata, converts current no-Windows guards into exact allowlists, aligns
docs, and obtains hosted Windows evidence. Until then, the Sprint 182 deferral
remains active.

If those implementation gates fail, Sprint 190 will renew the formal deferral
with the Day 2 and Day 3 blockers.

### Selected Promotion Contract

| Field | Contract |
| --- | --- |
| Manifest row | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Target key | `cholesky-spd-tridiag-5` |
| Family | `comparison` |
| Subfamily | `cholesky_spd_tridiag_5` |
| Generator command | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` plus a reviewed Windows-safe probe mode or equivalent explicit Windows arguments |
| Workflow file | `.github/workflows/windows-ci.yml` |
| Workflow job | `selected-comparison-freshness` |
| Workflow platform | `windows` added only for the Cholesky row |
| Workflow artifact | `sprint190-windows-selected-comparison-cholesky` |
| Expected rows | `6` |
| Artifact directory | `build/comparison/cholesky_spd_tridiag_5/` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Upload failure mode | `if-no-files-found: error` |
| Timeout target | One bounded hosted Windows job, initially capped at 20 minutes or less unless implementation evidence requires a lower value |
| Claim scope | Fixture-local Cholesky SPD tridiagonal solve comparison freshness on hosted Windows only |

### Required Manifest Changes If Promotion Lands

For `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` only:

- append `.github/workflows/windows-ci.yml` to `workflow_file`;
- append `selected-comparison-freshness` to `workflow_job`;
- append `sprint190-windows-selected-comparison-cholesky` to
  `workflow_artifact`;
- append `windows` to `workflow_platforms`;
- update `selection_scope` or `support_tier` only if existing schema semantics
  require a separate Windows-hosted distinction;
- keep `expected_rows=6`;
- keep the exact expected row IDs unchanged unless the generator contract
  changes;
- keep non-claims narrow and explicit.

No other selected report row may gain `windows` during Sprint 190.

### Guard Update Checklist

| Guard | Required behavior |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Allow the exact Cholesky Windows freshness job/artifact only after the decision record supersedes the broad Sprint 182 no-promotion state; continue to reject oracle, benchmark, QR, LU, partial-SVD, broad comparison, and broad artifact uploads. |
| `tests/test_validate_windows_powershell.py` | Add positive coverage for the allowed Cholesky path and negative coverage for unowned Windows selected report freshness. |
| `tests/test_selected_comparison_workflow.py` | Replace the current all-Windows-deferral assertion with an exact allowlist for the Cholesky Windows job, artifact, files, target, row count, and non-claims. |
| `tests/test_selected_report_targets_manifest.py` | Permit `windows` only for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` when a Sprint 190 decision record is present; reject all other Windows selected rows. |
| `scripts/validate_corpus_schema.py` | Preserve hosted metadata cardinality and reject mismatched per-platform artifact lists. Add a one-row Windows allowlist only if tests need schema-level enforcement. |
| `scripts/normalize_report_index.py` | Confirm selected Cholesky row identity, freshness, required files, and current source commit when Windows generated output is available. |

### Failure Behavior

The promoted path must fail clearly when:

- the Windows workflow omits the selected job;
- the job uses the wrong runner, shell, timeout, command, target, or artifact
  name;
- artifact upload is broad or lacks `if-no-files-found: error`;
- any required Cholesky report file is missing;
- `study.tsv` does not contain exactly six expected row IDs;
- generated `source_commit` does not match the checked-out commit when
  freshness is required;
- the manifest adds `windows` to any unselected row;
- docs claim broad Windows report freshness or broader Windows parity.

### Superseded Deferral Boundary

The Sprint 182 deferral remains active until the implementation is complete.
If promotion succeeds, Sprint 190 should not delete the Sprint 182 record; it
should add a new decision artifact that supersedes it only for the single
Cholesky selected comparison lane. All other Windows report freshness remains
deferred.

### Day 4 Validation

Day 4 changed only Sprint 190 planning documentation. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required.

### Day 5 Questions

1. Can the workflow scaffold be added without running any report generation
   until the Windows-safe probe mode exists?
2. Should the first implementation add a guarded placeholder job or wait until
   the generator can actually run on Windows?
3. Which tests should change first to express the exact Cholesky allowlist?
4. What is the minimal source change needed to make the Cholesky probe build
   through CMake/MSVC?

## Day 5: Workflow Scaffold

### Scaffold Decision

Day 5 does not add a live Windows freshness job yet. The current branch still
lacks the Windows-safe generated-probe mode required by Day 3 and Day 4, so a
workflow job that appears to run selected report freshness would overstate the
evidence.

Instead, Day 5 records the exact workflow scaffold contract that should be
implemented once the generator can build and execute the Cholesky probe on
Windows.

### Current Workflow State

`.github/workflows/windows-ci.yml` currently owns:

- `build-and-test`: reviewed CMake/MSVC configure, build, `ctest -N`, and
  full `ctest`;
- `powershell-validation`: hosted PowerShell validation ownership through
  `python scripts/validate_windows_powershell.py --require-pwsh`;
- `install-and-downstream`: reviewed CMake install/downstream package
  validation.

The workflow still intentionally does not run selected report generation
commands, upload selected report artifacts, or list a selected report freshness
job.

### Proposed Workflow Job

When the Windows-safe Cholesky probe mode exists, add one job:

```yaml
  selected-comparison-freshness:
    name: Windows selected Cholesky comparison freshness
    runs-on: windows-2022
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4

      - name: Configure reviewed Windows CMake build
        run: cmake -S . -B build -G "Visual Studio 17 2022" -A x64
        shell: pwsh

      - name: Build reviewed Windows CMake library
        run: cmake --build build --config Release
        shell: pwsh

      - name: Run selected Cholesky comparison freshness
        run: >
          python scripts/run_external_comparison.py
          --target cholesky-spd-tridiag-5
          --windows-cmake-build build
          --windows-config Release
        shell: cmd

      - name: Validate selected Cholesky comparison rows
        run: >
          python scripts/normalize_report_index.py
          --family comparison
          --require-generated comparison
          --check-freshness
        shell: cmd

      - name: Upload selected Windows Cholesky comparison freshness
        uses: actions/upload-artifact@v4
        with:
          name: sprint190-windows-selected-comparison-cholesky
          if-no-files-found: error
          path: |
            build/comparison/cholesky_spd_tridiag_5/project_observations.tsv
            build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv
            build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv
            build/comparison/cholesky_spd_tridiag_5/study.tsv
            build/comparison/cholesky_spd_tridiag_5/summary.md
            build/comparison/cholesky_spd_tridiag_5/manifest.tsv
```

The command flags above are placeholders for the Day 6/Day 7 implementation
contract. They must not be added to the live workflow until the generator
accepts and tests them.

### Artifact Contract

| Field | Value |
| --- | --- |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Artifact root | `build/comparison/cholesky_spd_tridiag_5/` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Upload failure mode | `if-no-files-found: error` |
| Broad paths forbidden | `build/comparison/**`, `build/**`, and any unselected comparison directory |

### Drift Risks

| Risk | Guard response needed |
| --- | --- |
| Job lands before generator supports Windows CMake/MSVC probe flags | Workflow tests should fail because the command is not executable evidence. |
| Job runs `make report-index-comparison-freshness` | Windows guard should reject broad selected comparison freshness. |
| Job uploads the Linux/macOS selected comparison artifact name | Windows guard should reject reused selected artifact names. |
| Job uploads `build/comparison/**` | Workflow guard should reject broad comparison upload paths. |
| Manifest adds `windows` to more than Cholesky | Manifest tests should reject every unselected Windows row. |
| Docs remove retained non-claims | Claim-boundary tests should fail. |

### Day 5 Validation

Commands run:

- `git status --short --branch`
- `sed -n '166,198p' docs/planning/EPIC_17/SPRINT_190/PLAN.md`
- `sed -n '1,130p' .github/workflows/windows-ci.yml`
- `python3 tests/test_selected_comparison_workflow.py`

Day 5 changed only Sprint 190 planning documentation. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required.

### Day 6 Questions

1. Which manifest-field change should be implemented first in tests: exact
   Cholesky allowlist or continued deferral guard?
2. Should Windows metadata use a per-platform artifact list on the Cholesky row
   or a separate Windows-only selected row?
3. Does `normalize_report_index.py` need a target-specific comparison
   freshness mode to avoid requiring all comparison targets on Windows?
4. Should `run_external_comparison.py` grow explicit Windows CMake probe flags
   before any manifest update is attempted?

## Day 6: Manifest Metadata

### Manifest Decision

Day 6 keeps `tests/corpus/manifests/selected_report_targets.tsv` unchanged.

Reason: the Sprint 190 branch still lacks a live Windows-safe Cholesky probe
path and hosted workflow job. Adding `windows` to the selected manifest now
would create structured freshness metadata without executable Windows
evidence, violating the Day 4 decision record.

The correct Day 6 outcome is an exact manifest mutation contract and guard-test
plan.

### Current Cholesky Row

| Field | Current value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `support_tier` | `local_only` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `required_files` | standard six-file comparison bundle |
| `expected_rows` | `6` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |

### Future Manifest Mutation Contract

When promotion is implemented, mutate only the Cholesky row:

| Field | Future value |
| --- | --- |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml;.github/workflows/windows-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness;sprint190-windows-selected-comparison-cholesky` |
| `workflow_platforms` | `linux;macos;windows` |

Do not change `expected_rows`, `expected_row_ids`, `artifact_pattern`, or
`required_files` unless the generator contract changes and all dependent tests
are updated together.

### Support-Tier Decision

Keep `support_tier=local_only` until hosted Windows output exists and the
report-index schema has a reviewed distinction for one hosted Windows selected
comparison row. The existing support-tier vocabulary does not need to change on
Day 6.

### Schema Guard Findings

`scripts/validate_corpus_schema.py` already validates:

- selected rows require artifact patterns;
- countable selected rows require positive `expected_rows` and exact
  `expected_row_ids`;
- generated selected rows require generator commands and required files;
- hosted metadata must include workflow file, job, artifact, and platform
  fields together;
- `workflow_artifact` may be one shared artifact or one artifact per platform;
- duplicate workflow artifact keys may not cross report families.

Those checks are sufficient for general metadata coherence, but they do not
encode the Sprint 190 one-row Windows allowlist. That policy currently lives in
tests and Windows-specific guards.

### Test Guard Decision

The first test update after generator implementation should be an exact
allowlist helper:

- permit `windows` only on `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`;
- require the paired Windows workflow file/job/artifact/platform entries;
- reject `windows` on QR, LU, partial-SVD, oracle, and benchmark selected
  rows;
- reject mismatched per-platform artifact counts;
- reject a Windows artifact name that reuses Linux or macOS selected artifact
  names;
- require a Sprint 190 decision record marker before the allowlist activates.

Until then, `test_windows_report_freshness_deferral_keeps_manifest_unselected`
and `test_windows_platform_drift_fails_clearly` remain correct and should keep
failing any premature manifest promotion.

### Day 6 Validation

Commands run:

- `git status --short --branch`
- `sed -n '199,234p' docs/planning/EPIC_17/SPRINT_190/PLAN.md`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`
- `awk -F '\t' 'NR==1 || $1=="SRT-COMP-CHOLESKY-SPD-TRIDIAG-5" {print}' tests/corpus/manifests/selected_report_targets.tsv`
- `sed -n '220,340p' tests/test_selected_report_targets_manifest.py`
- `sed -n '640,735p' scripts/validate_corpus_schema.py`

Day 6 changed only Sprint 190 planning documentation. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required.

### Day 7 Questions

1. Should the freshness guard first add a target-specific comparison freshness
   check for Cholesky before any Windows workflow job is introduced?
2. Should Windows selected comparison freshness use the existing
   `normalize_report_index.py --family comparison --require-generated
   comparison --check-freshness` command, or should it gain a narrower
   target-specific mode?
3. Which guard should own stale `source_commit` diagnostics for the future
   Windows Cholesky artifact?
4. Can artifact-name guards be converted to a one-row allowlist without
   weakening no-promotion checks for the other selected rows?

## Day 7: Freshness Guard

### Guard Decision

Day 7 adds the narrow selected comparison freshness guard before any Windows
workflow or manifest promotion.

The new command is:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

This uses the existing selected report target manifest, filters to
`target_key=cholesky-spd-tridiag-5`, and validates only the Cholesky selected
comparison artifact:

```text
build/comparison/cholesky_spd_tridiag_5/study.tsv
```

### Implemented Behavior

- Target-specific selected comparison contracts are resolved from
  `selected_report_targets.tsv`.
- Expected rows, expected row IDs, artifact diagnostics, and policy diagnostics
  are filtered to the requested target.
- Generated rows are filtered by selected artifact path and handle both
  repo-relative paths and temporary build-root paths.
- Freshness diagnostics skip non-selected comparison subfamilies when
  `--selected-target` is present.
- Missing, stale, failed, duplicate, unexpected, and row-count mismatch errors
  point at the selected Cholesky artifact and exact target-specific
  remediation command.

### Test Coverage

Added normalize-report-index tests for:

- Cholesky-only selected comparison freshness success;
- missing Cholesky generated output with no QR artifact noise;
- stale Cholesky `source_commit` diagnostics;
- failed Cholesky generated-row diagnostics;
- preservation of broad selected comparison row-set mismatch checks.

The selected comparison synthetic writer now supports an `only_subfamilies`
filter so tests can model the future Windows Cholesky-only lane without
weakening the existing all-selected comparison freshness gate.

### Non-Promotion Boundary

Day 7 intentionally does not add Windows manifest metadata, workflow jobs, or
artifact uploads. The branch still does not claim hosted Windows selected
report freshness.

The Day 7 result is an executable guard command that Day 8 can wire into
hosted CI if the workflow contract remains bounded.

### Day 7 Validation

Commands run:

- `python3 tests/test_normalize_report_index.py`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused validation commands passed.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required.

### Day 8 Questions

1. Should the hosted Windows workflow invoke the target-specific guard directly
   after generating the Cholesky comparison artifact?
2. Which timeout should own the Windows Cholesky freshness step: job-level,
   step-level, or both?
3. Should artifact upload be introduced only after hosted proof lands, or in
   the same Day 8 workflow patch?
4. Should the Windows PowerShell guard require the exact
   `--selected-target cholesky-spd-tridiag-5` command before manifest metadata
   can gain `windows`?

## Day 8: Hosted Integration

### Hosted Lane Added

Day 8 adds one hosted Windows selected report freshness job:

```text
selected-comparison-freshness
```

The job runs on `windows-2022`, has `timeout-minutes: 20`, builds the static
library through CMake/MSVC, generates only the selected Cholesky comparison
report, checks only the Cholesky target-specific freshness guard, and uploads
only the approved Cholesky artifact bundle.

### Generator Command

```sh
python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
```

`scripts/run_external_comparison.py` now supports an explicit CMake probe build
path. The Linux/macOS direct compiler path remains the default behavior for
existing callers unless `--probe-build-system cmake` is requested or `auto`
detects a Windows `.lib`/Windows environment.

### Freshness Command

```sh
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

This reuses the Day 7 target-specific freshness guard and avoids requiring QR,
LU, partial-SVD, or benchmark artifacts in the Windows lane.

### Artifact Upload Policy

The workflow uploads one artifact:

```text
sprint190-windows-selected-comparison-cholesky
```

Allowed files:

- `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv`
- `build/comparison/cholesky_spd_tridiag_5/study.tsv`
- `build/comparison/cholesky_spd_tridiag_5/summary.md`
- `build/comparison/cholesky_spd_tridiag_5/manifest.tsv`

The upload uses `if-no-files-found: error` and does not permit
`build/comparison/**`.

### Guard Updates

`scripts/validate_windows_powershell.py` now allows only the exact selected
Cholesky lane and still rejects selected oracle freshness, broad selected
comparison freshness, selected benchmark freshness, reused Linux/macOS selected
artifact names, unexpected uploads outside the Cholesky job, and unowned
PowerShell steps.

`tests/test_selected_comparison_workflow.py` now validates the Windows
Cholesky job contract and keeps the no-Windows-manifest-platform check active.

### Remaining Metadata Boundary

`tests/corpus/manifests/selected_report_targets.tsv` still does not list
`windows`. Day 8 introduces the hosted execution path, but manifest promotion
and documentation claim calibration remain future work. Local CMake-probe
success is not hosted Windows evidence.

### Day 8 Validation

Commands run:

- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`
- `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`

All focused validation commands passed.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required.

### Day 9 Questions

1. Should the selected target manifest gain `windows` for only
   `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` now that the hosted job exists?
2. Should deterministic tests require manifest metadata to match the Day 8
   workflow before documentation claims are updated?
3. Should the no-Windows-platform guard become a one-row allowlist rather than
   a blanket deferral assertion?
4. Which stale, missing, broad artifact, and reused-artifact drift cases should
   be promoted from workflow tests into schema-level tests?

## Day 9: Deterministic Tests

### Manifest Decision

Day 9 keeps `tests/corpus/manifests/selected_report_targets.tsv` unchanged.
The hosted workflow lane exists, but public docs and claim-boundary markers
still need Day 10 calibration before source metadata should list `windows`.

To make that future change deterministic, `tests/test_selected_report_targets_manifest.py`
now has a simulated allowlist helper for the exact Windows Cholesky promotion
contract.

### Manifest Regression Surface

The simulated allowlist accepts only:

- `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`;
- `.github/workflows/windows-ci.yml`;
- `selected-comparison-freshness`;
- `sprint190-windows-selected-comparison-cholesky`;
- `workflow_platforms=windows` in the aligned metadata position;
- `expected_rows=6`;
- `project_observations.tsv`, `baseline_observations.tsv`,
  `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`.

Added negative manifest tests reject:

- `windows` on an unselected comparison row;
- wrong or reused Windows artifact names;
- Cholesky row-count drift;
- missing required Cholesky artifact files.

### Workflow Regression Surface

`tests/test_selected_comparison_workflow.py` now has Windows-specific drift
tests for:

- missing `timeout-minutes: 20`;
- wrong generator target;
- missing `--selected-target cholesky-spd-tridiag-5`;
- wrong Windows selected artifact name;
- broad `build/comparison/**` upload paths;
- missing required Cholesky upload files.

### Validator Regression Surface

`tests/test_validate_windows_powershell.py` mirrors the hosted-lane drift
coverage so `python3 scripts/validate_windows_powershell.py` can reject the
same problems without running GitHub Actions.

The fake PowerShell tests still isolate hosted-only behavior. Local lack of
`pwsh` remains unavailable evidence, not a pass.

### Day 9 Validation

Commands run:

- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused Day 9 validation commands passed.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required.

### Day 10 Questions

1. Should Day 10 update public docs to say one Windows selected Cholesky
   freshness lane is wired but still pending hosted pass evidence?
2. Should the selected target manifest be promoted in the same change as docs,
   or only after hosted CI confirms the Day 8 job on the branch?
3. Which existing Sprint 182 deferral markers should become "all other Windows
   report freshness remains deferred" markers?
4. Should claim-boundary tests reject "Windows report freshness" unless the
   sentence names `cholesky-spd-tridiag-5` and excludes oracle, benchmark, and
   broad comparison freshness?

## Day 10: Claim Calibration

### Claim Split

Day 10 updates documentation to distinguish the one Sprint 190 workflow path
from broader unsupported Windows report freshness.

Accepted wording:

- one bounded Windows selected Cholesky comparison freshness workflow exists
  for `cholesky-spd-tridiag-5`;
- the workflow uses the target-specific freshness command and the exact
  artifact `sprint190-windows-selected-comparison-cholesky`;
- broad Windows report freshness remains unsupported;
- Windows selected oracle freshness and selected benchmark freshness remain
  unsupported;
- selected target manifest metadata still omits `windows` until metadata,
  support tier, and claim wording are reviewed together.

### Updated Files

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `scripts/validate_windows_powershell.py`
- `tests/test_validate_windows_powershell.py`

### Guard Updates

`scripts/validate_windows_powershell.py` now requires the new bounded-Cholesky
claim markers and continues to reject unsupported wording that implies broad
Windows report freshness has been supported, promoted, completed, or closed.

The guard also preserves these boundaries:

- PowerShell validation ownership is not report freshness evidence;
- local unavailable PowerShell is not pass evidence;
- the Sprint 182 deferral remains active for all other Windows report
  freshness surfaces.

### Day 10 Validation

Commands run:

- `python3 tests/test_validate_windows_powershell.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`

All focused Day 10 validation commands passed.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required.

### Day 11 Questions

1. Should Day 11 regenerate Cholesky report evidence with the CMake probe path
   and capture the normalized freshness output in a Sprint artifact?
2. Should the selected target manifest remain unpromoted until hosted Windows
   CI has run, or should Day 11 stage the exact Cholesky metadata patch?
3. Should Day 11 add a report-index evidence artifact that explicitly separates
   local CMake-probe output from hosted Windows evidence?
4. Which generated files should be inspected for source commit, platform,
   compiler, and artifact-path fields before closeout?

## Day 11: Report Evidence

### Evidence Regenerated

Day 11 regenerated the selected Cholesky comparison report locally through the
CMake probe path:

```sh
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake
```

Generated files:

- `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv`
- `build/comparison/cholesky_spd_tridiag_5/study.tsv`
- `build/comparison/cholesky_spd_tridiag_5/summary.md`
- `build/comparison/cholesky_spd_tridiag_5/manifest.tsv`

### Freshness Result

The target-specific freshness guard passed:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

The command reported six generated Cholesky rows as fresh and did not require
unselected comparison artifacts.

### Metadata Inspection

| Field | Observed value |
| --- | --- |
| `source_commit` | `4155eee320cea528513603130da41bf887de6d7b` |
| `source_branch` | `sprint-190` |
| `worktree_state` | `dirty` |
| `platform` | `darwin-x86_64` |
| `compiler` | `cmake-probe:default:Release` |
| `study_path` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |

The dirty worktree state is expected for the in-progress sprint branch and is
recorded as local provenance. It is not hosted Windows evidence.

### Row Evidence

`study.tsv` contains exactly six rows and all are `pass`:

- `comparison_cholesky_spd_tridiag_5_project_status_v1`
- `comparison_cholesky_spd_tridiag_5_baseline_status_v1`
- `comparison_cholesky_spd_tridiag_5_residual_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_values_v1`
- `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1`

### Dependency Evidence

Required dependencies:

- `python3`: `pass`
- `tests/chol_external_dense_reference.py`: `pass`

Optional dependencies remain deferred:

- `numpy`: `defer`
- `scipy`: `defer`

### Residual Risks

- Local CMake-probe success on macOS is not hosted Windows pass evidence.
- `selected_report_targets.tsv` still omits `windows`; manifest promotion is a
  later review surface.
- The generated `build/` evidence is ignored local output and was not added to
  source control.
- Hosted `windows-2022` must pass before docs can cite reviewed Windows
  selected Cholesky freshness evidence.

### Day 11 Validation

Commands run:

- `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused Day 11 validation commands passed.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required.

### Day 12 Questions

1. Which integrated validation checks should Day 12 run beyond the focused Day
   11 set?
2. Should Day 12 run `make windows-powershell-validate` in addition to the
   direct Python validator?
3. Should documentation whitespace and schema validation be included in the
   integrated validation log?
4. Should Day 12 explicitly inspect that generated `build/` evidence remains
   ignored and uncommitted?

## Day 12: Integrated Validation

### Validation Scope

Day 12 ran the integrated Sprint 190 validation surface across selected
workflow guards, selected target manifest checks, corpus schema validation,
Windows PowerShell validation ownership, report generation, report freshness,
documentation claim boundaries, whitespace hygiene, and generated-output
hygiene.

### Command Results

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Windows workflow remains bounded to the selected Cholesky comparison freshness lane. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Selected manifest rows remain coherent and still omit `windows` platform metadata. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus schema and report-index field documentation validate. |
| `python3 tests/test_validate_windows_powershell.py` | Pass | PowerShell wiring, parse checks, claim boundaries, unavailable semantics, and `--require-pwsh` failure behavior validate. |
| `make windows-powershell-validate` | Unavailable | Structural checks passed; local `pwsh` is unavailable, so the Make target exited 2 by design. |
| `python3 tests/test_normalize_report_index.py` | Pass | Target-specific freshness, stale-output, row-set, and advisory handling validate. |
| `python3 tests/test_run_external_comparison.py` | Pass | External comparison generation and CMake probe metadata validate. |
| `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake` | Pass | Regenerated the selected Cholesky comparison bundle locally. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Pass | Six generated selected Cholesky rows are fresh against current `HEAD`. |
| `git diff --check` | Pass | No whitespace errors found in changed files. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No C or header files changed. |
| `git status --short --ignored build/comparison/cholesky_spd_tridiag_5` | Pass | Generated report evidence remains ignored under `build/`. |

Because no `.c` or `.h` files changed, the full `make format && make lint &&
make test` C gate is not required for Day 12.

### Integrated Findings

- The `selected-comparison-freshness` workflow job is present on
  `windows-2022`.
- The hosted workflow command is scoped to
  `--target cholesky-spd-tridiag-5 --probe-build-system cmake`.
- The target-specific freshness guard accepts the selected Cholesky row set
  without requiring unselected comparison outputs.
- The selected source manifest still omits `windows`, preserving the staged
  manifest-promotion boundary.
- Claim-boundary checks continue to reject broad Windows report freshness,
  selected oracle freshness, and selected benchmark freshness.
- Local PowerShell unavailability is explicit and cannot be mistaken for hosted
  Windows pass evidence.

### Residual Risks

- Hosted `windows-2022` evidence is still required before this can be described
  as reviewed Windows selected Cholesky freshness support.
- Generated local report evidence is ignored `build/` output and remains
  macOS CMake-probe evidence.
- Manifest promotion remains a separate review surface because
  `selected_report_targets.tsv` still does not list `windows`.
- Day 13 must audit the Day 4 decision record against workflow, tests, docs,
  and residual language.

### Day 13 Questions

1. Should the final audit close Sprint 190 as "bounded workflow path wired,
   manifest promotion pending hosted evidence"?
2. Should the Day 13 audit compare the Day 4 decision record against workflow,
   tests, and docs line by line?
3. Should the residual record remain open until hosted CI confirms the
   `windows-2022` selected Cholesky freshness job?
4. Should source docs continue to avoid saying Windows selected Cholesky
   freshness is reviewed until PR CI passes?

## Day 13: Final Claim and Residual Audit

### Decision Record Audit

Day 13 compared the Day 4 provisional promotion contract against the current
workflow, selected manifest, guards, docs, and residual queue.

| Contract Area | Status | Notes |
| --- | --- | --- |
| Selected Cholesky workflow job | Implemented | `.github/workflows/windows-ci.yml` contains `selected-comparison-freshness` on `windows-2022` with `timeout-minutes: 20`. |
| Windows CMake probe path | Implemented | The hosted job configures/builds through CMake/MSVC and passes the `.lib` path to `run_external_comparison.py`. |
| Generator scope | Implemented | The workflow runs only `--target cholesky-spd-tridiag-5`. |
| Freshness scope | Implemented | The workflow uses `--selected-target cholesky-spd-tridiag-5`, so unselected selected comparison outputs are not required. |
| Artifact policy | Implemented | Upload uses `sprint190-windows-selected-comparison-cholesky`, `if-no-files-found: error`, and explicit required files only. |
| Guard coverage | Implemented | Workflow, manifest, normalizer, generator, and PowerShell validation tests cover exact positive and negative drift cases. |
| Source manifest `windows` metadata | Staged | `tests/corpus/manifests/selected_report_targets.tsv` still omits `windows`; future exact Cholesky-only metadata is tested but not promoted. |
| Hosted Windows evidence | Pending | Local validation cannot observe the hosted `windows-2022` job. |

### Claim Audit

README, INSTALL, maintainer guide, corpus docs, report-index schema docs, and
Sprint 190 artifacts all describe the same boundary:

- one bounded Windows selected Cholesky comparison freshness workflow path
  exists for `cholesky-spd-tridiag-5`;
- broad Windows report freshness remains unsupported;
- Windows selected oracle freshness and selected benchmark freshness remain
  unsupported;
- selected target source metadata still does not list `windows`;
- PowerShell validation ownership is not generated report freshness evidence;
- package-manager, shared-library, dynamic ABI, runtime-loader, broad platform,
  performance-superiority, and state-of-the-art claims remain out of scope.

### Residual Decision

`R186-WIN-REPORT-FRESHNESS` is renewed and narrowed rather than closed.

Sprint 190 wired the smallest credible hosted Windows workflow path and local
CMake-probe generator path, but the residual remains open until hosted
`windows-2022` evidence and selected manifest promotion are reviewed together.
The Epic 16 residual queue now records this narrower closure target.

### Validation

Commands run:

- `python3 tests/test_selected_comparison_workflow.py`
- `python3 scripts/validate_windows_powershell.py`
- `rg` audits across workflow, selected manifest, tests, docs, schema docs,
  and Sprint 190 artifacts

Results:

- `python3 tests/test_selected_comparison_workflow.py` passed.
- `python3 scripts/validate_windows_powershell.py` completed structural checks
  and reported local `pwsh` unavailable with exit 2, as expected.

No `.c` or `.h` files were modified during Day 13, so the full
`make format && make lint && make test` C gate is not required.

### Day 14 Closeout Checklist

1. Create the final Sprint 190 closeout artifact.
2. Re-run the Day 12 validation set.
3. Confirm local `pwsh` unavailable semantics or hosted evidence status.
4. Confirm no `.c` or `.h` files changed before deciding on the full C gate.
5. Confirm generated `build/` comparison evidence remains ignored.
6. Confirm `R186-WIN-REPORT-FRESHNESS` remains renewed and narrowed.
7. Prepare retrospective inputs for commit, push, and pull request.

### Day 14 Questions

1. Should Day 14 repeat the entire Day 12 validation set or only the focused
   checks affected by the residual queue update?
2. Should the closeout artifact call the sprint outcome "bounded workflow path
   implemented, residual narrowed" rather than "promotion complete"?
3. Should the retrospective carry hosted Windows evidence as the sole remaining
   blocker for manifest promotion?

## Day 14: Sprint Closeout

### Final Outcome

Sprint 190 closes as **bounded workflow path implemented, residual narrowed**.

The sprint selected `cholesky-spd-tridiag-5` as the smallest credible Windows
selected report freshness lane, added the hosted Windows workflow path, added
target-specific freshness validation, hardened workflow/manifest/PowerShell
guards, and calibrated public and maintainer docs. It does not claim broad
Windows report freshness and does not promote source manifest `windows`
metadata before hosted evidence is reviewed.

### Final Validation Results

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Workflow contract and selected-lane drift checks pass. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Manifest invariants and future exact Windows Cholesky allowlist tests pass. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus schema and report-index field docs validate. |
| `python3 tests/test_validate_windows_powershell.py` | Pass | PowerShell validation unit coverage passes. |
| `python3 tests/test_normalize_report_index.py` | Pass | Normalizer and target-specific freshness regression coverage passes. |
| `python3 tests/test_run_external_comparison.py` | Pass | External comparison and CMake probe regression coverage passes. |
| `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake` | Pass | Local CMake-probe selected Cholesky comparison bundle regenerated. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Pass | Six generated selected Cholesky rows are fresh against current `HEAD`. |
| `make windows-powershell-validate` | Unavailable | Structural checks passed; local `pwsh` is unavailable, so the target exited 2 by design. |
| `git diff --check` | Pass | No whitespace errors found. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No C or header files changed. |
| `git status --short --ignored build/comparison/cholesky_spd_tridiag_5` | Pass | Generated report output remains ignored under `build/`. |

No `.c` or `.h` files changed during Sprint 190, so the full
`make format && make lint && make test` C gate is not required.

### Changed Surfaces

- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md`
- `docs/planning/EPIC_17/SPRINT_190/`
- `scripts/normalize_report_index.py`
- `scripts/run_external_comparison.py`
- `scripts/validate_windows_powershell.py`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `tests/test_normalize_report_index.py`
- `tests/test_selected_comparison_workflow.py`
- `tests/test_selected_report_targets_manifest.py`
- `tests/test_validate_windows_powershell.py`

### Residual and Handoff

`R186-WIN-REPORT-FRESHNESS` remains open as a narrowed residual. Sprint 190
wired the bounded hosted workflow path and local CMake-probe validation, but
hosted `windows-2022` evidence and selected manifest promotion remain a
review-time decision.

Retrospective should carry:

- bounded selected Cholesky workflow path implemented;
- local generated report evidence is ignored `build/` output;
- local PowerShell Make wrapper unavailable because `pwsh` is not installed;
- manifest source metadata still omits `windows`;
- broad Windows report freshness, Windows selected oracle freshness, and
  Windows selected benchmark freshness remain non-claims.
