# Sprint 166 Day 6: Supplemental Validation Sweep

## Purpose

Day 6 runs the supplemental generated documentation, report-index, package,
benchmark, performance-sentinel, and claim-boundary checks selected by the
Day 4 validation design. The result complements the Day 5 local baseline and
keeps generated/local evidence separate from hosted CI proof.

## Scope

This sweep validated existing generated-output and package boundaries. It did
not edit source, headers, public docs, scripts, workflows, package metadata, or
generated report content. Hosted Windows and Linux CI results remain separate
evidence and are not inferred from this local macOS run.

## Command Results

| Command | Result | Evidence class | Notes |
| --- | --- | --- | --- |
| `make docs-check` | Pass | generated API documentation check | Generated Doxygen output locally and passed API docs coverage for 18 checked-in public headers, 18 reference pages, and 18 source pages. `sparse_version.h` remains a separate installed-header policy row rather than a generated reference page. |
| `make report-index-oracle-freshness` | Pass | local generated oracle report freshness | Wrote local oracle/report-index outputs and reported freshness ok for 54 rows. |
| `make report-index-comparison-freshness` | Pass | local generated comparison freshness | Regenerated selected QR min-norm, QR compatible least-squares, and partial-SVD diag6 k2 comparison rows; reported freshness ok for 25 rows. |
| `python3 scripts/normalize_report_index.py --check` | Pass | normalized report-index schema/check | Reported `normalize-report-index: 176 rows ok`. |
| `python3 scripts/normalize_report_index.py --family package --check` | Pass | package report-index check | Reported `normalize-report-index: 6 rows ok`. |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Pass | package report-index freshness | Verified 6 source-controlled advisory/package rows as fresh. |
| `bash scripts/static_package_deferral_check.sh` | Pass | static-first package boundary | Confirmed shared-library rejection, static install metadata, no shared export/ABI metadata, no package selector, retained support deferral wording, Windows package non-claim wording, and no unselected Windows package execution. |
| `bash tests/test_install.sh` | Pass | Make install and pkg-config package proof | Passed 23 install validation checks with 0 failures. |
| `bash tests/test_cmake_install.sh` | Pass | CMake install/export package proof | Passed 27 CMake install validation checks with 0 failures and 0 skips. |
| `make bench-canonical-report` | Pass | local generated benchmark publication | Wrote local canonical benchmark reports under `build/bench-reports/canonical`. |
| `make performance-sentinels` | Pass | local performance sentinel check | Wrote local sentinel artifacts under `build/bench-reports/sentinels` and passed the maintained local wall-check surface. |
| Targeted public/workflow claim scans | Pass with expected matches | claim-boundary hygiene | Matches were non-claim, deferred-scope, methodology-bound, or validation-owner wording; no new unsupported claim wording was found. |
| `git diff --check` | Pass | whitespace hygiene | No whitespace errors reported. |

## Generated Output Boundaries

| Output area | Boundary |
| --- | --- |
| `docs/api/html/` | Local generated Doxygen output only; not hosted proof and not source-controlled evidence. |
| `build/corpus/oracle/` and `build/corpus-reports/` | Local generated oracle/report-index outputs only. |
| `build/comparison/` | Local generated comparison outputs only, even when rows are selected for maintained freshness checks. |
| `build/bench-reports/` | Local benchmark and sentinel outputs only; not portable performance proof. |
| Temporary install prefixes | Created and cleaned by install-validation scripts; not committed evidence. |

## Claim Scan Summary

The targeted scans covered current public and workflow claim surfaces:

- `README.md`
- `INSTALL.md`
- `docs/api_reference.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `docs/algorithm.md`
- `docs/algorithm_history.md`
- `docs/solver_selection.md`
- `docs/maintainer_guide.md`
- `.github/workflows/*.yml`
- `CMakeLists.txt`

Observed matches were expected boundary language, including static-first
package wording, explicit shared-library and ABI deferrals, Windows staged-lane
non-claims, fixture-scoped solver comparison wording, local-only generated
report wording, and methodology-bound performance wording.

The Day 4 scan plan included `docs/benchmarking.md`; that path is stale in the
current repository. There is no top-level `docs/benchmarking.md`; benchmark and
performance claim wording currently lives in the public docs and maintainer
guide listed above. Later scan plans should use those current paths instead of
the stale file name.

## Remaining Reconciliation Item

The hosted comparison workflow still needs Day 7 review. The maintained local
comparison freshness target now covers QR min-norm, QR compatible least-squares,
and partial-SVD diag6 k2 selected comparisons, while the hosted comparison
workflow artifact naming and summary wording remain QR-minnorm-oriented. Day 6
does not change that workflow; it records the mismatch for Day 7 reconciliation.

## Non-Claims Preserved

Day 6 evidence does not add or imply:

- hosted Linux or Windows CI proof;
- source-controlled or hosted generated API HTML proof;
- broad external-library parity;
- broad QR, SVD, partial-SVD, eigensolver, graph, or direct-solver parity;
- portable performance, backend superiority, or state-of-the-art status;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager distribution;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected supplemental checks pass or blockers are recorded. | Complete | Generated docs, report freshness, package boundary, install/export, benchmark, sentinel, claim-scan, and whitespace checks passed. |
| Generated/report/package evidence remains bounded. | Complete | Generated outputs are recorded as local-only; static-first package proof remains separate from shared-library, ABI, package-manager, and Windows Make/`pkg-config` claims. |
| Unsupported claim wording is not introduced by validation artifacts. | Complete | Claim scans found expected non-claim/deferred wording only; this artifact preserves the same boundaries. |
| Follow-on reconciliation inputs are explicit. | Complete | Hosted comparison artifact-scope mismatch is retained as a Day 7 item. |
