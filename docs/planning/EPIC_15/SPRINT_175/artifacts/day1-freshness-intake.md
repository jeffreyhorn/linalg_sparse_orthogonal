# Day 1: Sprint Intake And Report Freshness Boundary

## Purpose

Establish Sprint 175 scope, inherited generated-output boundaries, current
report freshness commands, platform status, and retained non-claims before
selecting any cross-platform report freshness promotion lane.

## Source Artifact Decision

The Day 1 source of truth is the active Epic 15 project plan:

```text
docs/planning/EPIC_15/PROJECT_PLAN.md
Sprint 175: Cross-Platform Report Freshness Promotion
```

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but that path is
not the active merged Sprint 175 planning source. Sprint 175 records this
mismatch and proceeds from Epic 15.

## Inherited Sprint 173 Boundary

Sprint 173 closed generated API HTML publication as guarded local-only output:

- command: `make api-docs-freshness`;
- generated path: `docs/api/`;
- staging policy: ignored, untracked, unstaged local output;
- support claim: maintainers can regenerate local Doxygen HTML and validate
  expected pages;
- non-claim: no hosted generated API HTML, committed generated API HTML,
  artifact publication, release evidence, package support, ABI support,
  platform parity, external-library parity, performance, or state-of-the-art
  evidence.

Sprint 175 must not use generated API HTML freshness as report freshness
promotion evidence.

## Inherited Sprint 174 Boundary

Sprint 174 closed one additional selected comparison family:

- command: `make report-index-comparison-freshness`;
- selected comparison families:
  - `qr_minnorm`;
  - `qr_compatible_ls`;
  - `partial_svd_diag6_k2`;
  - `lu_nonsym_square_5`;
- generated paths:
  - `build/comparison/qr_minnorm/study.tsv`;
  - `build/comparison/qr_compatible_ls/study.tsv`;
  - `build/comparison/partial_svd_diag6_k2/study.tsv`;
  - `build/comparison/lu_nonsym_square_5/study.tsv`;
- support claim: selected local generated comparison rows can be regenerated
  and checked for freshness against the current source commit;
- non-claim: no broad solver correctness, external-library parity, hosted
  comparison publication, package support, ABI support, platform portability,
  performance, release, or state-of-the-art evidence.

Sprint 175 can evaluate whether one selected report freshness path is ready for
macOS or Windows promotion, but it must not infer that promotion from the local
Sprint 174 command alone.

## Current Freshness Commands

| Command | Generated outputs | Current boundary |
| --- | --- | --- |
| `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | selected local oracle freshness; reviewed Linux hosted mirror only for selected artifacts |
| `make report-index-comparison-freshness` | `build/comparison/*/study.tsv` for selected QR, partial-SVD, and LU families | selected local comparison freshness; reviewed Linux hosted mirror only for selected artifacts |
| `make bench-canonical-report-freshness` | selected canonical benchmark report rows | selected local benchmark freshness; reviewed Linux hosted selected-performance mirror |
| `make api-docs-freshness` | `docs/api/` | generated API HTML local-only staging and coverage guard |

## Current Manifest Families

| Family | Subfamily | Generator | Artifact | Support tier |
| --- | --- | --- | --- | --- |
| oracle | generated_reference | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | `local_only` |
| oracle | solver_backed | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | `local_only` |
| coverage | src | `make coverage` | `coverage/coverage-src.info` | `local_only` |
| report_index | missing_generated | `python3 scripts/normalize_report_index.py` | `build/report-index/normalized-index.tsv` | `local_only` |
| comparison | qr_minnorm | `python3 scripts/run_external_comparison.py --target qr-minnorm` | `build/comparison/qr_minnorm/study.tsv` | `local_only` |
| comparison | qr_compatible_ls | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | `build/comparison/qr_compatible_ls/study.tsv` | `local_only` |
| comparison | partial_svd_diag6_k2 | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | `local_only` |
| comparison | lu_nonsym_square_5 | `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5` | `build/comparison/lu_nonsym_square_5/study.tsv` | `local_only` |
| ci | reviewed_lanes | `GitHub Actions` | `.github/workflows/*.yml` | `reviewed_cross_platform` |

## Platform Boundary At Intake

| Platform | Current report freshness status |
| --- | --- |
| Linux | Reviewed hosted source of truth for selected report freshness lanes that have explicit hosted jobs and uploaded selected artifacts. |
| macOS | Reviewed package/install evidence exists, but selected generated report freshness is not yet promoted as a macOS report-freshness lane. |
| Windows | Reviewed CMake-first build/test and CMake install/downstream evidence exists, but selected generated report freshness is not yet promoted as a Windows report-freshness lane. |

## Day 1 Non-Claims

Sprint 175 does not yet claim:

- broad cross-platform report freshness;
- macOS selected report freshness;
- Windows selected report freshness;
- hosted publication of all generated reports;
- hosted generated API HTML;
- package-manager availability;
- shared-library ABI support;
- runtime-loader behavior;
- release evidence;
- external-library ecosystem parity;
- performance superiority beyond selected benchmark rows;
- state-of-the-art sparse linear algebra status.

## Day 1 Completion Record

- Sprint 175 scope is tied to the active Epic 15 project-plan section.
- Current report freshness paths are visible before lane selection.
- Generated API, comparison, oracle, coverage, report-index, and benchmark
  freshness boundaries are separated.
- Unsupported platform, hosted, package, ABI, performance, release, and
  state-of-the-art claims remain protected.
