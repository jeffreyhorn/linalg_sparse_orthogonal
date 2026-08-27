# Sprint 182 Day 4: Artifact And Data Semantics Audit

## Purpose

Day 4 audits generated files, row counts, newline behavior, path semantics,
normalized-index behavior, and upload scopes independently from command
runtime. This lets Sprint 182 distinguish data-format blockers from workflow
or toolchain blockers before choosing a Windows promotion or deferral path.

## Selected Manifest Artifact Risk Table

| Target | Artifact pattern | Required files | Expected rows | Current platforms | Day 4 risk |
| --- | --- | --- | ---: | --- | --- |
| `SRT-ORACLE-QR-PSVD-LOCAL` | `build/corpus/oracle/*.tsv` | `build/corpus-reports/manifest.txt`; `build/report-index/normalized-index.tsv` | 52 | `linux` | Broader selected row surface and selected glob require careful Windows upload design. |
| `SRT-COMP-QR-MINNORM` | `build/comparison/qr_minnorm/study.tsv` | six files in `build/comparison/qr_minnorm/` | 6 | `linux;macos` | Best candidate data shape: exact directory, exact files, small row count. |
| `SRT-COMP-QR-COMPATIBLE-LS` | `build/comparison/qr_compatible_ls/study.tsv` | six files in `build/comparison/qr_compatible_ls/` | 6 | `linux;macos` | Same favorable artifact shape as QR minimum-norm. |
| `SRT-COMP-PSVD-DIAG6-K2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | six files in `build/comparison/partial_svd_diag6_k2/` | 10 | `linux;macos` | Exact artifacts are favorable; numerical/vector semantics are broader than QR. |
| `SRT-COMP-LU-NONSYM-SQUARE-5` | `build/comparison/lu_nonsym_square_5/study.tsv` | six files in `build/comparison/lu_nonsym_square_5/` | 6 | `linux;macos` | Exact artifacts are favorable; LU solver/probe portability remains runtime risk. |
| `SRT-BENCH-REFACTOR-CSC-NOS4` | `build/bench-reports/canonical/bench_refactor_csc.csv` | `bench_refactor_csc.csv`; `index.tsv`; `manifest.txt` | 1 | `linux` | Checker data shape is manageable, but generator/runtime and performance claim semantics are high risk. |

## Generated File Semantics

| Component | Semantics | Windows assessment |
| --- | --- | --- |
| `scripts/run_external_comparison.py` TSV outputs | Writes `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, and `manifest.tsv` with `encoding="utf-8"`, `newline=""`, tab delimiter, and `lineterminator="\n"`. | Favorable; generated comparison files should remain LF-stable if the command can run. |
| `scripts/run_corpus_oracle.py` TSV outputs | Writes oracle/report TSV files with `newline=""`, tab delimiter, and `lineterminator="\n"`. | Favorable; runtime probe assumptions are the bigger blocker. |
| `scripts/normalize_report_index.py` normalized index | Reads TSV with `newline=""`; writes normalized output with tab delimiter and LF line endings; emits repo-relative display paths with `as_posix()`. | Favorable for cross-platform display and comparison. |
| `scripts/check_bench_canonical_freshness.py` | Reads `index.tsv` and `manifest.txt`, checks required artifact names, selected row identity, claim boundary, and manifest field matches. | Data checker is likely portable after artifacts exist. |
| `scripts/bench_canonical_report.sh` | Writes CSV/TSV/manifest through Bash redirection and here-docs after running benchmark binaries. | Data outputs are simple, but generation is not Windows-native. |

## Normalized-Index Windows Semantics

`scripts/normalize_report_index.py` already has several useful
cross-platform behaviors:

- `pattern_to_paths()` resolves `build/...` artifact patterns relative to the
  configured build root and uses `Path.glob()` for globbed generated artifacts.
- `display_path()` converts repository-relative paths to POSIX-style display
  paths with `as_posix()`.
- selected oracle diagnostics validate expected total rows, selected solver
  family counts, and selected fixture keys from
  `selected_report_targets.tsv`.
- selected comparison diagnostics validate expected row IDs, total expected
  rows, duplicate rows, unexpected rows, non-pass rows, and deferred rows.
- freshness diagnostics compare generated `source_commit` values with current
  `HEAD` for strict generated policies.

These are data-semantics strengths. The Windows gap is that diagnostic
remediation strings still name Makefile commands, and no Windows selected
workflow metadata exists yet.

## Artifact Upload Requirements For Windows Promotion

A promoted Windows selected freshness lane must:

- use a new Windows-specific workflow artifact name;
- list exact selected artifact paths;
- include `if-no-files-found: error`;
- avoid broad globs such as `build/**`, `build/comparison/**`, or
  `build/bench-reports/**`;
- run a summary/check step before upload that verifies required files,
  expected rows, selected row IDs, pass status, `source_commit`,
  `source_branch`, and `platform`;
- align uploaded files with `tests/corpus/manifests/selected_report_targets.tsv`;
- keep unselected report families and advisory artifacts outside the selected
  Windows upload unless the manifest explicitly marks them as required
  context.

## Deferral Requirements

If Sprint 182 chooses formal deferral, the data-format audit supports this
blocker statement:

- Python-generated comparison and oracle TSV formats are not the primary
  blocker.
- Windows workflow/toolchain execution is the primary blocker because current
  selected commands rely on Makefile wrappers, Unix probe compiler/linker
  conventions, or Bash benchmark generation.
- Manifest metadata should continue to omit `windows` from selected workflow
  platforms.
- Guard tests should continue rejecting selected report freshness commands and
  selected upload artifact names in the Windows workflow.

## Data-Format Versus Workflow Blockers

| Area | Blocker status | Evidence |
| --- | --- | --- |
| TSV delimiters/newlines | Not a major blocker | Python writers use tab delimiter and LF line endings. |
| Repo-relative paths | Not a major blocker | Normalizer emits POSIX-style display paths for paths under the repo. |
| Exact comparison artifacts | Favorable | Selected comparison rows have exact `build/comparison/<target>/...` required files and expected row counts. |
| Oracle glob artifacts | Moderate data-scope risk | Oracle uses `build/corpus/oracle/*.tsv`, so a Windows upload would need exact selected file handling or a narrow justified glob. |
| Benchmark artifacts | Runtime/claim risk | Data checker is manageable, but generator and benchmark metadata are Unix/Bash-centered. |
| Manifest Windows metadata | Intentional absence | No selected rows currently list `windows`; promotion must add metadata deliberately. |
| Remediation text | Follow-up risk | Freshness diagnostics currently point to Makefile commands, which would be misleading for a Windows-native promoted path. |

## Day 4 Decisions

- Keep selected comparison as the leading data-shape candidate for Day 5.
- Treat command/runtime compatibility, not TSV data format, as the main
  blocker for selected comparison promotion.
- Do not promote oracle freshness until upload scope is narrowed or explicitly
  justified for its selected glob and 52-row surface.
- Keep benchmark freshness as a deferral candidate because the data checker is
  not enough to overcome Bash/runtime/performance claim risk.
- Require any Windows promotion to update manifest workflow metadata and
  guard tests before docs claim Windows freshness.

## Day 5 Handoff

Day 5 should build a decision matrix that compares:

- selected comparison direct Python promotion with a Windows-specific
  CMake/MSVC probe path;
- selected oracle promotion with broader artifact and row-scope handling;
- selected benchmark promotion with Windows-native report generation;
- formal Windows report freshness deferral with explicit blockers and guard
  coverage.

The matrix should score feasibility, user value, maintenance cost, claim
clarity, CI cost, and guardability.

## Validation

Day 4 is documentation-only. Validation:

- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Generated file semantics are evaluated independently from command runtime. | Complete | Generated file semantics and data-format versus workflow blocker tables. |
| Windows promotion cannot rely on broad artifact globs. | Complete | Artifact upload requirements require exact selected paths and reject broad globs. |
| Deferral blockers distinguish data-format issues from workflow issues. | Complete | Deferral requirements identify workflow/toolchain execution as the primary blocker. |
