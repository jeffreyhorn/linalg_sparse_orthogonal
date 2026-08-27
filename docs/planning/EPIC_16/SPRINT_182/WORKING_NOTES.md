# Sprint 182 Working Notes

**Sprint:** 182 - Windows Report Freshness Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_182/`
**Status:** In progress

## Source Artifact Note

The Sprint 182 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 182: Windows Report
Freshness Decision". Sprint 182 artifacts in this directory follow the Epic 16
scope.

## Sprint Goal

Promote one Windows-safe generated report freshness path or close Windows
report freshness as an explicit product deferral with guard coverage.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_182/PLAN.md`
- `docs/planning/EPIC_16/SPRINT_181/RETROSPECTIVE.md`
- `docs/planning/EPIC_16/SPRINT_181/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_181/artifacts/day14-closeout-and-handoff.md`
- `tests/corpus/manifests/selected_report_targets.tsv`
- `tests/test_selected_comparison_workflow.py`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`

## Starting Branch Snapshot

- Branch: `sprint-182`
- Starting commit: `2a937585c2f457b80095cd54bd2deede12509b02`
- Recent base context:
  - `2a937585` Merge pull request #201 from `sprint-181`
  - `7ed432ec` Address PR #201 review comments
  - `e9efe1b5` Complete Sprint 181 retrospective and PR artifacts

## Sprint 182 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 182.1 | Windows Report Audit | Complete | Day 1 establishes scope, inherited selected-target authority, current Windows boundaries, and candidate evaluation fields. Day 2 inventories Windows workflow/toolchain constraints and Linux/macOS selected freshness lane patterns. Day 3 audits selected command internals and classifies preliminary Windows promotion risk. Day 4 audits generated artifacts, data semantics, and upload-scope requirements. |
| 182.2 | Candidate Selection | Complete | Day 5 selects formal Windows report freshness deferral as the path. Day 6 designs the decision-record, manifest, guard, documentation, and future-promotion contract. Selected comparison remains the strongest future promotion candidate after CMake/MSVC probe work. |
| 182.3 | CI or Deferral Implementation | Complete | Day 7 adds the formal Windows report freshness deferral record and strengthens workflow guard coverage. Day 8 adds selected-target manifest regression coverage for Windows non-selection while preserving Linux/macOS selected report behavior. |
| 182.4 | Manifest Integration | Complete | Day 9 aligns selected target manifest semantics and corpus/report-index support-tier docs with formal Windows deferral. The manifest remains positive selected-target authority and does not gain a fake Windows deferral row. |
| 182.5 | Documentation Alignment | Complete | Day 10 updates README, maintainer guide, and Windows workflow comments so user-facing and maintainer surfaces cite the formal Sprint 182 Windows report freshness deferral. Day 11 hardens workflow and upload diagnostics for missing Windows jobs, missing deferral wording, wrong selected commands, wrong artifacts, broad uploads, and missing required files. Day 13 reconciles docs, guards, manifests, validation records, and residual risks. INSTALL and benchmark docs remain unchanged because package/install and selected-performance semantics did not change. |
| 182.6 | Validation | Complete | Day 12 runs the feasible local validation sweep for the formal Windows report freshness deferral path. Day 14 repeats final closeout validation and confirms no C/header files or generated build/report artifacts are in the changed surface. Local PowerShell parsing is unavailable because `pwsh` is not installed in this environment; workflow text guards cover the Windows workflow contract locally. |

## Inherited Sprint 181 Authority

Sprint 181 made
`tests/corpus/manifests/selected_report_targets.tsv` the selected report target
authority for selected generator commands, artifact patterns, required files,
expected rows, workflow files, workflow jobs, uploaded artifact names, workflow
platforms, support tiers, freshness policies, claim scopes, non-claims,
owners, and provenance.

| Inherited boundary | Day 1 baseline |
| --- | --- |
| Selected oracle freshness | `make report-index-oracle-freshness` is selected and hosted on Linux only. The manifest preserves `no macOS oracle freshness` and `no Windows report freshness`. |
| Selected comparison freshness | `make report-index-comparison-freshness` is selected and hosted on Linux/macOS only for QR minimum-norm, QR compatible least-squares, partial-SVD diagonal top-k, and LU nonsymmetric square solve targets. |
| Selected benchmark freshness | `make bench-canonical-report-freshness` is selected and hosted on Linux only for `bench_refactor_csc`. |
| Windows workflow | `.github/workflows/windows-ci.yml` remains CMake-first and install/downstream scoped. It does not run selected oracle, comparison, or benchmark report freshness commands. |
| Guard coverage | `tests/test_selected_comparison_workflow.py` rejects selected report freshness command names and selected upload artifact names in the Windows workflow. |
| Documentation claims | Public docs preserve Windows report freshness as a non-claim alongside package-manager, shared-library ABI, broad platform parity, and broad external-library parity non-claims. |

## Current Windows Boundary

Windows currently proves a reviewed MSVC/CMake surface:

- configure, build, `ctest -N`, and full `ctest` for the reviewed CMake test
  subset;
- static-first CMake install/downstream consumer validation;
- installed static `.lib`, headers, CMake package metadata, and
  metadata-only `sparse.pc` inspection;
- absence of unsupported DLL/shared imported metadata, loader behavior,
  static/shared selector metadata, package-manager support, Makefile parity,
  and pkg-config execution parity.

It does not currently prove generated report freshness on Windows.

## Windows Freshness Candidate Evaluation Fields

| Field | Question for each candidate |
| --- | --- |
| Shell compatibility | Can the command run under the reviewed Windows shell without relying on POSIX-only syntax, tools, glob expansion, or Makefile behavior? |
| Path semantics | Do generated and checked paths work with Windows drive roots, backslashes, forward slashes, quoting, spaces, and temporary directories? |
| Newline behavior | Are generated files stable under CRLF/LF handling, text-mode I/O, and exact freshness comparisons? |
| Executable availability | Are `cmake`, compiler tools, Python, scripts, and any helper executables available in the reviewed Windows lane? |
| Python dependency availability | Does the target require only source-controlled helpers and Python modules available on hosted Windows without extra package-manager setup? |
| Runtime cost | Can the target fit inside the Windows hosted CI budget without making the lane slow or flaky? |
| Artifact scope | Can the lane upload only selected, exact artifacts with fail-closed missing-file behavior? |
| Support tier | Would promotion create a local-only, reviewed hosted, or cross-platform selected support tier, and is that tier documented? |
| Claim boundary | What positive Windows freshness claim would be allowed, and which broad report, package, ABI, parity, and performance claims remain unsupported? |
| Guardability | Can tests fail clearly if the Windows workflow drifts into unsupported report freshness or if promoted metadata and workflow scope diverge? |

## Daily Log

### Day 1: Windows Freshness Scope Intake

- Re-read the Sprint 182 project-plan section and Day 1 plan.
- Reviewed Sprint 181 closeout and retrospective handoff notes.
- Confirmed the selected target manifest has Linux/macOS selected report
  workflow platforms and no Windows selected report freshness row.
- Confirmed the Windows workflow remains CMake-first and install/downstream
  scoped with explicit non-claims.
- Confirmed the workflow guard rejects selected report freshness commands and
  selected upload artifact names in the Windows workflow.
- Defined Windows freshness candidate evaluation fields for Days 2-6.
- Added Day 1 scope intake artifact.

### Day 2: Windows Workflow And Toolchain Audit

- Inspected `.github/workflows/windows-ci.yml` job layout, runner image, shell
  choices, environment variables, generated paths, and artifact behavior.
- Compared Linux and macOS selected report freshness jobs against the Windows
  CMake-first lane.
- Confirmed Windows currently proves CMake, MSVC, `ctest`, PowerShell, and
  static install/downstream behavior.
- Confirmed Makefile execution, POSIX shell scripts, Linux/macOS artifact
  upload scopes, and selected report freshness commands remain unproven on
  Windows.
- Recorded the current guard boundary that rejects selected report freshness
  commands and selected upload artifact names in the Windows workflow.
- Added Day 2 workflow and toolchain audit artifact.

### Day 3: Report Command Compatibility Audit

- Audited `make report-index-oracle-freshness` and
  `scripts/run_corpus_oracle.py` for Windows shell, compiler, library,
  executable, path, and newline assumptions.
- Audited `make report-index-comparison-freshness` and
  `scripts/run_external_comparison.py` for direct Python candidate potential
  and current Unix archive/linker assumptions.
- Audited `make bench-canonical-report-freshness`,
  `scripts/bench_canonical_report.sh`, and
  `scripts/check_bench_canonical_freshness.py` for Bash, benchmark binary,
  metadata, and runner assumptions.
- Reviewed `tests/corpus/manifests/report_families.tsv` to keep advisory and
  unselected report families separate from Windows promotion candidates.
- Classified selected commands as possible with refactor, deferral candidate,
  or out-of-scope for the Sprint 182 Windows decision.
- Added Day 3 report command compatibility audit artifact.

### Day 4: Artifact And Data Semantics Audit

- Inspected selected report artifact patterns, required files, expected rows,
  expected row IDs, workflow artifact names, and workflow platforms in
  `tests/corpus/manifests/selected_report_targets.tsv`.
- Audited TSV/CSV writers, manifest readers, path display, generated artifact
  glob expansion, and freshness diagnostics in report scripts.
- Confirmed Python-generated TSV semantics are mostly portable: `pathlib`
  paths, explicit LF TSV output, and deterministic selected output roots.
- Confirmed Windows promotion risk is dominated by command/runtime and exact
  artifact ownership, not by TSV delimiter or newline behavior.
- Recorded Windows upload scope requirements and deferral blockers.
- Added Day 4 artifact and data semantics audit artifact.

### Day 5: Candidate Decision Matrix

- Defined candidate options for selected comparison, selected oracle, selected
  benchmark, and formal Windows report freshness deferral.
- Scored each option against Windows CI feasibility, shell portability,
  artifact stability, dependency requirements, runtime cost, maintenance cost,
  user value, and claim risk.
- Selected formal Windows report freshness deferral as the Day 6 decision
  target.
- Preserved selected comparison direct Python invocation as the strongest
  future promotion candidate after Windows-safe CMake/MSVC probe work.
- Recorded concrete blockers for rejected promotion options.
- Added Day 5 candidate decision matrix artifact.

### Day 6: Decision Record Design

- Designed the formal Windows report freshness deferral decision record
  structure for Sprint 182 implementation.
- Defined supported Windows claims, unsupported report freshness scope,
  blocker taxonomy, revisit criteria, and pass/fail contract.
- Chose a separate Sprint 182 decision artifact as the explicit deferral
  record, with manifest/docs/guard references preserving the selected target
  manifest as positive selected-target authority.
- Mapped required guard behavior for the Windows workflow and selected target
  manifest platform metadata.
- Mapped required documentation and workflow comment wording for Days 12-13.
- Added Day 6 decision-record design artifact.

### Day 7: Implementation Batch 1

- Added the formal Sprint 182 Windows report freshness deferral record.
- Updated `tests/test_selected_comparison_workflow.py` to verify the deferral
  record and ensure selected target manifest rows do not list `windows` in
  `workflow_platforms`.
- Added focused Windows drift tests for accidental selected freshness command
  strings and selected freshness artifact names.
- Added a focused deferral-record drift test for missing blocker wording.
- Preserved existing Linux/macOS selected report freshness guard behavior.
- Added Day 7 implementation batch 1 artifact.

### Day 8: Implementation Batch 2

- Updated `tests/test_selected_report_targets_manifest.py` to enforce that
  selected target rows do not list `windows` while the formal Sprint 182
  deferral record is active.
- Added a manifest drift regression that appends `windows` to a selected
  comparison row and requires a clear failure naming the row.
- Linked manifest regression coverage to the formal Windows report freshness
  deferral record.
- Preserved existing Linux/macOS selected report manifest and workflow
  behavior.
- Added Day 8 implementation batch 2 artifact.

### Day 9: Manifest And Support-Tier Alignment

- Kept `tests/corpus/manifests/selected_report_targets.tsv` unchanged as
  positive selected-target authority.
- Documented in `tests/corpus/README.md` that Windows report freshness is
  formally deferred by the Sprint 182 decision record and that selected rows
  must not list `windows` while deferral is active.
- Documented in `tests/corpus/schemas/report_index_fields.md` that the
  selected-target manifest does not use fake selected rows for deferrals and
  that any future Windows promotion must add exact workflow metadata.
- Confirmed Day 8 manifest and workflow regressions are the executable
  protection against accidental Windows selected freshness drift.
- Added Day 9 manifest and support-tier alignment artifact.

### Day 10: Documentation Alignment

- Updated `README.md` CI and report-index sections to cite the formal Sprint
  182 Windows report freshness deferral record.
- Documented in `README.md` that Windows report freshness promotion requires a
  Windows-safe generator path, selected manifest metadata, exact artifact upload
  scope, and guard updates.
- Updated `docs/maintainer_guide.md` selected comparison/report-index guidance
  to keep Windows out of selected target `workflow_platforms` while the
  deferral is active.
- Updated `.github/workflows/windows-ci.yml` comments so the reviewed Windows
  lane explicitly excludes generated report freshness.
- Left `INSTALL.md` and `benchmarks/README.md` unchanged because the Day 10
  decision did not alter package/install support or selected performance
  freshness semantics.
- Added Day 10 documentation-alignment artifact.

### Day 11: Guard And Failure Diagnostics Hardening

- Added a Windows workflow contract helper that requires the reviewed
  `build-and-test` and `install-and-downstream` jobs and the Sprint 182 formal
  deferral wording.
- Added drift coverage for a missing reviewed Windows workflow job with a
  diagnostic naming the missing job.
- Added drift coverage for missing Windows report freshness deferral wording in
  `.github/workflows/windows-ci.yml`.
- Added selected comparison upload drift coverage for a missing required upload
  file with a diagnostic naming the exact path.
- Preserved existing diagnostics for accidental Windows selected commands,
  selected upload artifact names, missing deferral blockers, wrong upload
  artifact names, missing `if-no-files-found: error`, and broad selected upload
  paths.
- Added Day 11 guard-and-failure-diagnostics-hardening artifact.

### Day 12: Validation Sweep

- Ran Python compile checks for the touched schema, normalizer, and focused test
  files.
- Ran selected-target schema validation and selected manifest regression tests.
- Ran Linux/macOS/Windows workflow guard regression tests, including the Day 11
  Windows deferral diagnostics.
- Ran report-index normalizer regression coverage and non-required freshness
  diagnostics for oracle, comparison, coverage, dead-code, and package rows.
- Ran static package and package-manager deferral guards because Day 10 touched
  support-tier and package-adjacent wording.
- Confirmed local `pwsh` is unavailable, so no PowerShell parse check was run.
- Added Day 12 validation-sweep artifact.

### Day 13: Decision Reconciliation

- Reconciled Sprint 182 project-plan items against the produced artifacts and
  changed files.
- Confirmed the decision record, selected target manifest, Windows workflow,
  README, maintainer guide, corpus docs, report-index schema docs, and guard
  tests all describe the same formal Windows report freshness deferral.
- Confirmed `tests/corpus/manifests/selected_report_targets.tsv` remains
  unchanged and does not list `windows` in selected `workflow_platforms`.
- Recorded retained Windows non-claims for selected oracle freshness, selected
  comparison freshness, selected benchmark freshness, broad generated report
  freshness, Makefile parity, package-manager support, shared-library/dynamic
  ABI/runtime-loader behavior, broad Windows parity, portable performance, and
  state-of-the-art status.
- Prepared retrospective inputs and Sprint 183 handoff notes around the
  Windows-safe selected comparison promotion prerequisites.
- Added Day 13 decision-reconciliation artifact.

### Day 14: Closeout And Handoff

- Re-read Sprint 182 artifacts and working notes for consistency.
- Finalized the Sprint 182 decision summary: Windows report freshness remains
  formally deferred while Windows CMake build/test and static
  install/downstream claims remain intact.
- Confirmed the changed file surface contains no C or header files.
- Confirmed no generated build/report artifact paths are staged.
- Recorded final validation commands and the local `pwsh` availability gap.
- Prepared Sprint 183 handoff notes for a future Windows-safe selected
  comparison promotion path.
- Added Day 14 closeout-and-handoff artifact.

## Day 2 Windows Workflow Inventory

| Job | Runner | Shell | Current proof | Report freshness status |
| --- | --- | --- | --- | --- |
| `build-and-test` | `windows-2022` | `pwsh` | CMake configure, CMake build, `ctest -N` count check, full `ctest`. | No selected report command, report summary, or report artifact upload. |
| `install-and-downstream` | `windows-2022` | `pwsh` | CMake install/export, static `.lib`, installed headers, CMake package files, metadata-only `sparse.pc`, downstream consumers, exact-version and mismatch-version behavior, unsupported shared/loader metadata absence. | No selected report command, report summary, or report artifact upload. |

## Day 2 Cross-Platform Freshness Lane Comparison

| Platform | Selected freshness lane | Shell/tool pattern | Artifact pattern | Transfer note for Windows |
| --- | --- | --- | --- | --- |
| Linux | `generated-report-freshness` | `make report-index-oracle-freshness`, `make report-index-comparison-freshness`, inline `python3` summaries. | Exact `actions/upload-artifact@v4` paths for oracle and selected comparison outputs with `if-no-files-found: error`. | Pattern is structurally useful, but direct Makefile use is not reviewed on Windows. |
| Linux | `hosted-performance-freshness` | Makefile benchmark target plus `python3 scripts/check_bench_canonical_freshness.py`; Linux shell uses `/proc/cpuinfo`. | Exact canonical benchmark files uploaded with `if-no-files-found: error`. | Least transferable as-is because it depends on POSIX shell script execution and Linux CPU metadata collection. |
| macOS | `selected-comparison-freshness` | `make report-index-comparison-freshness`, inline `python3` summaries. | Exact selected comparison files uploaded with `if-no-files-found: error`. | Useful comparison shape, but still assumes Unix Makefile and shell behavior. |
| Windows | none | `pwsh`, CMake, MSVC, `ctest`, generated PowerShell downstream consumer checks. | No selected report upload. | Any promotion needs deliberate PowerShell/CMake-compatible command and exact artifact upload scope. |

## Day 2 Windows Toolchain Assumptions

| Assumption | Current status | Evidence |
| --- | --- | --- |
| `windows-2022` runner | Proven | Windows workflow pins both jobs to `windows-2022` for the VS 2022 generator. |
| PowerShell execution | Proven | All current Windows run steps use `shell: pwsh`. |
| CMake configure/build/install | Proven | Both Windows jobs invoke CMake directly. |
| MSVC x64 generator | Proven | Windows commands use `-G "Visual Studio 17 2022" -A x64`. |
| `ctest` execution | Proven | `build-and-test` runs `ctest -N` and full `ctest`. |
| Python availability | Implied, not used by current Windows jobs | GitHub Windows images normally include Python, but the current workflow does not execute `python3` or `python`. A promoted freshness lane should prove the exact executable name. |
| Makefile execution | Unproven and explicitly unclaimed | Windows workflow comments state no Makefile parity claim. |
| POSIX shell scripts | Unproven | Current Windows jobs do not run `bash`, `.sh` scripts, POSIX redirection-heavy report scripts, or Unix command substitutions. |
| `pkg-config` execution | Unproven and explicitly unclaimed | Windows only inspects `sparse.pc` metadata. |
| Package-manager setup | Unsupported | Windows workflow comments and docs preserve no package-manager support. |
| Shared-library/runtime loader behavior | Unsupported | Windows install checks reject unsupported shared/loader metadata. |

## Day 2 Report-Lane Constraints

- A Windows promotion cannot simply copy the Linux/macOS `make
  report-index-*` or `make bench-canonical-*` steps without first proving
  Makefile and shell compatibility.
- A promoted Windows lane should use `pwsh` or another explicitly reviewed
  shell, set failure behavior explicitly, and avoid relying on implicit
  POSIX glob expansion.
- Artifact uploads must remain exact selected paths with
  `if-no-files-found: error`; broad `build/**` or report-family-wide uploads
  would widen the selected claim boundary.
- Candidate summaries should check required files, expected rows, platform
  metadata, and support-tier fields before upload.
- If Windows remains deferred, the existing guard should continue rejecting
  selected report freshness command names and selected artifact upload names
  in `.github/workflows/windows-ci.yml`.

## Day 3 Selected Command Compatibility Matrix

| Command surface | Current selected role | Windows risk | Preliminary classification |
| --- | --- | --- | --- |
| `make report-index-oracle-freshness` | Linux selected oracle freshness for QR/partial-SVD oracle rows. | Makefile wrapper; default `cc`; Unix static archive `build/libsparse_lu_ortho.a`; `-lm`; extensionless temp executable; requires report normalizer freshness check. | Possible with refactor, but not a direct Windows promotion candidate. |
| `scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Underlying oracle generator. | Python/pathlib and LF TSV output are portable, but solver probes assume Unix compiler/linker/library conventions. | Possible with CMake/MSVC-aware probe refactor. |
| `make report-index-comparison-freshness` | Linux/macOS selected comparison freshness wrapper. | Makefile wrapper is unreviewed on Windows; underlying targets are Python commands. | Possible with refactor; direct Python target invocation is the leading promotion candidate for deeper evaluation. |
| `scripts/run_external_comparison.py --target ...` | Underlying selected comparison generator. | Uses source-controlled Python helpers and LF TSV output, but project probe assumes `cc`, Unix `.a`, `-lm`, extensionless executable, and fallback `make` library build. | Leading possible-with-refactor candidate. |
| `make bench-canonical-report-freshness` | Linux selected benchmark freshness wrapper. | Makefile wrapper, Bash report script, compiled Unix benchmark paths, Linux/macOS metadata commands, and benchmark runtime cost. | Deferral candidate unless Windows-native benchmark report generation is built. |
| `scripts/bench_canonical_report.sh` | Canonical benchmark report generator. | Requires Bash, POSIX redirection, `date`, `git`, `uname`, `${CC:-cc} --version`, `head`, `otool`/`ldd`, `grep`, `basename`, here-docs, and extensionless benchmark binaries. | Deferral candidate for Sprint 182 promotion. |
| `scripts/check_bench_canonical_freshness.py` | Benchmark report freshness checker. | Python/pathlib checker is relatively portable after report generation exists, but it cannot compensate for Bash report generation and benchmark-binary assumptions. | Support tool only; not a standalone promotion candidate. |

## Day 3 Advisory And Unselected Surface Classification

| Surface | Windows relevance | Classification |
| --- | --- | --- |
| report-index missing/generated rows | Reads generated report roots and selected target contracts. | Guard/support surface, not a promotion candidate by itself. |
| coverage | Local generated advisory report. | Out of scope for Windows report freshness promotion. |
| dead-code report | Local generated advisory report. | Out of scope; existing Windows workflow does not run the dead-code flow. |
| sentinel and guardrail reports | Local generated runtime/large-matrix reports. | Out of scope for selected Windows report freshness unless later selected explicitly. |
| package/static install rows | Source-controlled/reviewed package evidence. | Separate Windows CMake install proof, not generated report freshness. |
| CI and documentation rows | Source-controlled hosted/docs surfaces. | Claim-boundary surfaces only, not promotion candidates. |

## Day 3 Candidate Direction

- The selected comparison Python targets are the most plausible promotion path
  for deeper evaluation because they already avoid optional external packages
  and use source-controlled dense-reference helpers.
- The comparison path still needs refactor or Windows-specific invocation
  because project probes currently assume Unix compiler/linker behavior and
  the default selected hosted command is a Makefile wrapper.
- Oracle freshness has similar probe/linker risks and a larger selected row
  scope, so it is behind comparison as a promotion candidate.
- Benchmark freshness is the clearest deferral candidate because the selected
  generator is Bash-based and embeds Unix metadata commands plus benchmark
  runtime concerns.

## Day 4 Artifact Semantics Findings

| Surface | Data semantics | Windows implication |
| --- | --- | --- |
| Selected comparison artifacts | Four exact `build/comparison/<target>/study.tsv` patterns with six required files per target and expected row counts of 6, 6, 10, and 6. | Best data-shape candidate. A Windows lane can upload exact files if generation succeeds and manifest metadata gains Windows workflow status. |
| Selected oracle artifacts | `build/corpus/oracle/*.tsv` plus required `build/corpus-reports/manifest.txt` and `build/report-index/normalized-index.tsv`; expected total is 52 selected rows. | Data shape is valid but broader and uses a selected glob, so Windows upload scope would need explicit fail-closed file inventory or a justified selected glob. |
| Selected benchmark artifacts | `build/bench-reports/canonical/bench_refactor_csc.csv` plus `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`; Linux also uploads unselected context CSVs. | Data checker is portable, but generator/runtime and performance claim sensitivity keep this behind comparison. |
| Manifest workflow metadata | Current selected rows list `linux` or `linux;macos`; no `windows` platform exists. | Any promotion must update `workflow_file`, `workflow_job`, `workflow_artifact`, and `workflow_platforms` deliberately. |
| Normalized index paths | `display_path()` emits POSIX-style repo-relative paths with `as_posix()` when paths are under the repository. | Favorable for cross-platform docs/artifact comparison, but Windows generated paths still need validation in hosted CI before claiming freshness. |
| Freshness checks | Selected comparison/oracle diagnostics validate expected row IDs, row counts, pass status, and `source_commit` freshness against current HEAD. | Data semantics can support Windows, but remediation text currently names Makefile commands and would need Windows-aware wording if promoted. |

## Day 4 Windows Upload Scope Requirements

- Use exact selected artifact paths, not broad `build/**`,
  `build/comparison/**`, or `build/bench-reports/**` globs.
- Preserve `actions/upload-artifact@v4` with `if-no-files-found: error`.
- Add a Windows-specific artifact name instead of reusing Linux or macOS
  selected artifact names.
- Ensure summaries check required files, expected row counts, selected row
  IDs, pass status, `source_commit`, `source_branch`, and `platform` before
  upload.
- Keep unselected advisory context files out of a Windows selected freshness
  upload unless the manifest explicitly records why they are required
  context.
- If deferring Windows freshness, keep the Windows workflow free of selected
  report commands and selected artifact names.

## Day 4 Data-Format Versus Workflow Blockers

| Blocker type | Finding |
| --- | --- |
| Data-format blocker | No major TSV newline/delimiter/path-display blocker found for Python-generated comparison or oracle artifacts. |
| Manifest blocker | Windows selected workflow metadata is absent by design and must be added only if a candidate is promoted. |
| Upload-scope blocker | A promoted lane needs exact selected artifact paths and guard coverage against broad uploads. |
| Runtime blocker | Current selected command wrappers and probe builds still assume Makefile/Unix compiler behavior. |
| Documentation blocker | Current non-claims remain correct until promotion or formal deferral changes claim wording. |

## Day 5 Candidate Decision Summary

| Candidate | Feasibility | User value | Maintenance cost | Claim risk | Day 5 position |
| --- | --- | --- | --- | --- | --- |
| Selected comparison direct Python promotion | Medium after refactor | Medium to high | Medium | Medium | Strongest future promotion candidate, but not the Day 6 implementation target because Windows-safe probe/link behavior is not implemented. |
| Selected oracle promotion | Low to medium | Medium | Medium to high | Medium to high | Rejected for Sprint 182 implementation target due to broad 52-row scope, selected glob upload handling, and Unix probe assumptions. |
| Selected benchmark promotion | Low | Low to medium | High | High | Rejected for Sprint 182 implementation target due to Bash generator, benchmark runtime, and performance-claim sensitivity. |
| Formal Windows report freshness deferral | High | Medium | Low | Low | Selected Day 6 decision target; closes the product decision while preserving guards and exact future promotion blockers. |

## Day 5 Selected Path

Sprint 182 should proceed with **formal Windows report freshness deferral**.

This does not mean Windows can never gain selected report freshness. It means
Sprint 182 should close the current product question by recording that no
current selected report freshness path is Windows-safe under the reviewed
Windows lane without new CMake/MSVC probe work or Windows-native report
generation.

## Day 5 Rejected Path Blockers

| Rejected path | Concrete blockers |
| --- | --- |
| Selected comparison promotion now | Needs Windows-safe project probe build/link path; exact Python executable proof; `.lib`/MSVC link handling; extension-aware temp executable invocation; Windows-specific manifest/workflow metadata; Windows-aware remediation text. |
| Selected oracle promotion now | Needs the comparison blockers plus a larger selected 52-row freshness surface, oracle selected glob upload policy, and broader report-index generated artifact handling. |
| Selected benchmark promotion now | Needs Windows-native replacement for Bash report generation, benchmark executable path handling, Windows metadata collection, runtime-cost proof, and stricter claim controls for performance-adjacent artifacts. |

## Day 6 Target

Day 6 should design a formal deferral decision record with:

- exact unsupported Windows report freshness scope;
- preserved Windows CMake/install claims;
- blocker list for future promotion;
- selected comparison as the first future candidate only after Windows-safe
  probe work exists;
- required guard behavior to keep `.github/workflows/windows-ci.yml` free of
  selected report freshness commands and artifact names;
- manifest/support-tier status needed to represent explicit Windows deferral.

## Day 6 Deferral Contract Summary

| Area | Day 6 contract |
| --- | --- |
| Decision record location | Add a Sprint 182 formal deferral record under `docs/planning/EPIC_16/SPRINT_182/artifacts/` during implementation. |
| Selected target manifest | Keep `windows` absent from `workflow_platforms` for selected freshness rows; add explicit Windows deferral/status metadata only if schema/docs support it without pretending deferral is a selected target. |
| Windows workflow | Keep `.github/workflows/windows-ci.yml` CMake/install scoped and free of selected report freshness commands and selected upload artifact names. |
| Workflow guards | Preserve fail-closed rejection of selected report freshness commands/artifacts in Windows; add explicit checks for any formal deferral marker chosen during implementation. |
| Documentation | State that Windows report freshness is formally deferred; keep Windows CMake build/test and static install/downstream claims intact. |
| Future promotion gate | Require Windows-safe CMake/MSVC probe support, exact Python executable proof, `.lib`/`.exe` handling, manifest workflow metadata, exact upload scope, guard allowlist, and docs update. |

## Day 6 Required Implementation Plan

Days 7-13 should implement the deferral path in this order:

1. Add the formal Windows report freshness deferral record.
2. Strengthen workflow/manifest guards so Windows remains unselected and any
   future promotion must be manifest-backed.
3. Add manifest/support-tier representation only if it improves clarity
   without turning deferral into a selected target row.
4. Update README, INSTALL, maintainer guide, workflow comments, and report
   index language to cite the formal deferral.
5. Validate selected target manifest checks, workflow guard checks,
   normalizer/report checks, and whitespace.

## Validation Log

| Day | Validation | Status |
| --- | --- | --- |
| 1 | `git diff --check` | Pass |
| 2 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 2 | `git diff --check` | Pass |
| 3 | `python3 -m py_compile scripts/run_corpus_oracle.py scripts/run_external_comparison.py scripts/normalize_report_index.py scripts/check_bench_canonical_freshness.py` | Pass |
| 3 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 3 | `git diff --check` | Pass |
| 4 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 4 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 4 | `git diff --check` | Pass |
| 5 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 5 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 5 | `git diff --check` | Pass |
| 6 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 6 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 6 | `git diff --check` | Pass |
| 7 | `python3 -m py_compile tests/test_selected_comparison_workflow.py` | Pass |
| 7 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 7 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 7 | `git diff --check` | Pass |
| 8 | `python3 -m py_compile tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Pass |
| 8 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 8 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 8 | `git diff --check` | Pass |
| 9 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 9 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 9 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 9 | `git diff --check` | Pass |
| 10 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 10 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 10 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 10 | `git diff --check` | Pass |
| 11 | `python3 -m py_compile tests/test_selected_comparison_workflow.py` | Pass |
| 11 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 11 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 11 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 11 | `git diff --check` | Pass |
| 12 | `python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py` | Pass |
| 12 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 12 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 12 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 12 | `python3 tests/test_normalize_report_index.py` | Pass |
| 12 | `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` | Pass |
| 12 | `python3 scripts/normalize_report_index.py --family oracle --check-freshness` | Pass, with expected stale local oracle warnings |
| 12 | `python3 scripts/normalize_report_index.py --family comparison --check-freshness` | Pass, with advisory local comparison freshness diagnostics |
| 12 | `python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness` | Pass, with advisory absent local report diagnostics |
| 12 | `bash scripts/static_package_deferral_check.sh` | Pass |
| 12 | `bash scripts/package_manager_deferral_check.sh` | Pass |
| 12 | `command -v pwsh` | Not available locally |
| 12 | `git diff --check` | Pass |
| 13 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 13 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 13 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 13 | `python3 tests/test_normalize_report_index.py` | Pass |
| 13 | `git diff --check` | Pass |
| 14 | `python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py` | Pass |
| 14 | `python3 scripts/validate_corpus_schema.py` | Pass |
| 14 | `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| 14 | `python3 tests/test_selected_comparison_workflow.py` | Pass |
| 14 | `python3 tests/test_normalize_report_index.py` | Pass |
| 14 | `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` | Pass |
| 14 | `python3 scripts/normalize_report_index.py --family oracle --check-freshness` | Pass, with expected stale local oracle warnings |
| 14 | `python3 scripts/normalize_report_index.py --family comparison --check-freshness` | Pass, with advisory local comparison freshness diagnostics |
| 14 | `python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness` | Pass, with advisory absent local report diagnostics |
| 14 | `bash scripts/static_package_deferral_check.sh` | Pass |
| 14 | `bash scripts/package_manager_deferral_check.sh` | Pass |
| 14 | `git diff --check` | Pass |
