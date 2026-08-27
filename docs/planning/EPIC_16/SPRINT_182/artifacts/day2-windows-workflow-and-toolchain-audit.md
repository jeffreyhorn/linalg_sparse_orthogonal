# Sprint 182 Day 2: Windows Workflow And Toolchain Audit

## Purpose

Day 2 audits the current Windows workflow jobs, shells, toolchain assumptions,
and artifact behavior before Sprint 182 evaluates any selected report
freshness candidate for Windows promotion.

The key Day 2 conclusion is that Windows currently has a reviewed CMake/MSVC
lane, not a reviewed Makefile or generated-report lane.

## Windows Workflow Job Inventory

| Job | Runner | Shell | Steps | Current evidence claim |
| --- | --- | --- | --- | --- |
| `build-and-test` | `windows-2022` | `pwsh` | checkout; CMake configure; CMake build; `ctest -N` count inspection; full `ctest`. | Reviewed CMake consumer subset with promoted portable tests. |
| `install-and-downstream` | `windows-2022` | `pwsh` | checkout; CMake install/build/install; package metadata inspection; generated downstream consumers; installed example; exact-version proof; mismatch-version rejection. | Reviewed static-first CMake install/downstream validation path. |

Current Windows workflow comments preserve explicit non-claims for Makefile
parity, pkg-config execution parity, package-manager support, shared-library
support, dynamic ABI support, runtime-loader behavior, and broad Windows
parity.

## Windows Toolchain And Shell Assumptions

| Capability | Current status | Notes for report freshness |
| --- | --- | --- |
| `windows-2022` runner | Proven | Both jobs pin the runner because the reviewed lane depends on the VS 2022 generator. |
| PowerShell (`pwsh`) | Proven | Every Windows command step uses `shell: pwsh`. Promotion work should either use this shell or explicitly justify another shell. |
| CMake | Proven | Configure, build, install, downstream, and package-version checks all use CMake directly. |
| MSVC x64 | Proven | Commands use `-G "Visual Studio 17 2022" -A x64`. |
| `ctest` | Proven | The workflow validates test registration and execution with `ctest`. |
| Python | Implied but unproven in current Windows workflow | Existing report lanes use `python3`, but Windows does not currently prove `python3` or `python` availability. |
| Makefile targets | Unproven and explicitly unclaimed | Current selected report freshness commands are Makefile targets. Windows comments state no Makefile parity claim. |
| POSIX shell scripts | Unproven | The canonical benchmark report path calls `scripts/bench_canonical_report.sh`, which assumes Bash and POSIX shell behavior. |
| Unix metadata commands | Platform-specific | Linux performance metadata uses `/proc/cpuinfo`; Windows would need a PowerShell-native equivalent or no promoted claim that depends on it. |
| pkg-config execution | Unproven and explicitly unclaimed | Windows validates `sparse.pc` as installed metadata only. |
| Package-manager setup | Unsupported | A Windows freshness lane should not require package-manager-installed tools unless Sprint 182 changes the product boundary. |

## Linux And macOS Selected Freshness Patterns

| Platform | Lane | Selected commands | Guard/upload shape |
| --- | --- | --- | --- |
| Linux | `generated-report-freshness` | `make report-index-oracle-freshness`; `make report-index-comparison-freshness`. | Inline Python summaries validate row counts and metadata; exact upload blocks use `if-no-files-found: error`. |
| Linux | `hosted-performance-freshness` | `make bench-canonical-report`; `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`. | Metadata environment variables define selected support tier and claim boundary; exact selected benchmark artifacts are uploaded. |
| macOS | `selected-comparison-freshness` | `make report-index-comparison-freshness`. | Inline Python summary mirrors Linux selected comparison checks; exact selected comparison artifacts are uploaded. |

The transferable parts are the fail-closed artifact upload pattern, explicit
summary checks, selected-target manifest alignment, and narrow claim wording.
The non-transferable parts are the unreviewed Windows Makefile assumption,
POSIX shell scripts, and Linux/macOS-specific shell details.

## Selected Report Command Exposure

Current selected freshness commands from
`tests/corpus/manifests/selected_report_targets.tsv` are:

| Target family | Selected command | Current workflow platforms | Day 2 Windows exposure |
| --- | --- | --- | --- |
| oracle | `make report-index-oracle-freshness` | `linux` | Requires Makefile target and Python report scripts; not reviewed on Windows. |
| comparison | `python3 scripts/run_external_comparison.py --target ...` through `make report-index-comparison-freshness` | `linux;macos` | Underlying Python targets may be candidates, but current hosted command is the Makefile wrapper. |
| benchmark | `make bench-canonical-report-freshness` | `linux` | Depends on Makefile, compiled benchmark binaries, and Bash script `scripts/bench_canonical_report.sh`; not a Day 2 Windows-safe copy candidate. |

## Report-Lane Constraints

Any Windows promotion must satisfy these lane constraints:

- use a shell and executable names proven in the Windows workflow;
- avoid unsupported Makefile, Bash, pkg-config execution, package-manager, and
  shared-library assumptions;
- generate artifacts under deterministic paths that work with Windows path
  semantics and GitHub artifact upload matching;
- upload exact selected artifacts with `if-no-files-found: error`;
- check required files, expected rows, platform metadata, support tier, and
  claim boundary before upload;
- update selected target manifest workflow metadata deliberately;
- keep all non-promoted report families and broad Windows parity claims as
  explicit non-claims.

## Existing Guard Boundary

`tests/test_selected_comparison_workflow.py` currently keeps Windows
fail-closed by rejecting selected report freshness command and artifact names
in `.github/workflows/windows-ci.yml`.

Forbidden Windows workflow strings currently include:

- `report-index-oracle-freshness`
- `report-index-comparison-freshness`
- `bench-canonical-report-freshness`
- `check_bench_canonical_freshness.py`
- `sprint159-oracle-freshness`
- `sprint175-linux-selected-comparison-freshness`
- `sprint175-macos-selected-comparison-freshness`
- `sprint168-selected-performance-freshness`

If Sprint 182 promotes a Windows candidate, the guard should change from a
blanket rejection to a manifest-backed allowlist for the one selected Windows
path. If Sprint 182 defers Windows freshness, this guard remains the product
boundary.

## Day 2 Decisions

- Treat Windows report freshness as unpromoted until a candidate can run
  without relying on unreviewed Makefile or POSIX shell behavior.
- Use the Linux/macOS exact upload and summary-check patterns as design
  constraints, not as commands to copy directly.
- Require Day 3 to inspect selected command internals, especially whether the
  comparison Python targets can be invoked directly under Windows without the
  Makefile wrapper.
- Treat benchmark freshness as higher risk for Windows because the current
  selected benchmark path is Bash-script based and includes platform-specific
  metadata assumptions.

## Day 3 Handoff

Day 3 should audit selected command internals:

- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `python3 scripts/run_external_comparison.py --target ...`
- `make bench-canonical-report-freshness`
- `scripts/bench_canonical_report.sh`
- `scripts/check_bench_canonical_freshness.py`

The audit should classify each command by shell assumptions, executable
suffix assumptions, path handling, newline handling, dependency availability,
and generated artifact scope.

## Validation

Day 2 is documentation-only, but the workflow guard was run to confirm the
documented Windows fail-closed boundary still passes:

- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Windows workflow capabilities and limitations are explicit. | Complete | Windows job inventory and toolchain assumption tables. |
| Candidate freshness paths can be evaluated against actual Windows CI state. | Complete | Cross-platform freshness pattern table and report-lane constraints. |
| Current non-claim guard boundaries are documented before selection. | Complete | Existing guard boundary section and validation plan. |
