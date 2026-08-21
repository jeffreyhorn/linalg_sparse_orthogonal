# Day 11: Cross-Platform Claim Review

## Purpose

Review maintained docs, manifests, workflow comments, and focused tests after
the Sprint 175 selected macOS comparison freshness promotion to ensure the
claim boundary remains precise.

## Scope Reviewed

Reviewed these maintained surfaces:

- `README.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`;
- `benchmarks/README.md`;
- `tests/corpus/manifests/report_families.tsv`;
- `.github/workflows/ci.yml`;
- `.github/workflows/macos-ci.yml`;
- `.github/workflows/windows-ci.yml`;
- `tests/test_selected_comparison_workflow.py`;
- `tests/test_normalize_report_index.py`.

## Claim Scan Findings

### Selected Comparison Freshness

The selected comparison lane is consistently described as:

- local Make target freshness for four selected comparison families;
- reviewed Linux hosted selected-artifact evidence;
- reviewed macOS hosted selected-artifact evidence;
- not source-controlled generated output publication.

The selected comparison lane does not claim:

- broad report-index freshness;
- hosted publication of all generated reports;
- unselected comparison family freshness;
- broad QR, partial-SVD, LU, or external-library parity;
- Windows report freshness;
- broad platform parity;
- package-manager support;
- shared-library ABI support;
- runtime-loader support;
- release evidence;
- performance superiority;
- state-of-the-art status.

### Selected Oracle Freshness

Selected oracle freshness remains Linux-hosted only. The scan found Linux-only
oracle wording in `README.md` and the oracle generated-reference manifest row;
both are intentional and not stale Sprint 175 comparison wording.

Sprint 175 does not promote selected oracle freshness on macOS.

### CI Metadata

The CI reviewed-lanes manifest row now names:

- Linux selected oracle/comparison freshness;
- macOS selected comparison freshness;
- macOS static-first install/export proof;
- Windows CMake subset and CMake install/downstream validation.

It still says CI metadata alone is not local report freshness proof and keeps
Windows report freshness as a non-claim.

### Generated Row Metadata

Selected comparison generated rows remain:

- `row_origin=generated_local`;
- `support_tier=local_only`;
- `freshness_policy=generated_compare_inputs`.

That is intentional. Hosted evidence lives in Linux/macOS workflow artifacts,
not in generated-local TSV row support tiers.

## Stale Wording Fix List

No additional stale claim wording required code or documentation edits on Day
11. The Day 9 and Day 10 updates already reconciled the stale Linux-only
selected comparison wording and generated-local manifest non-claims.

## Remaining Deferrals

The following remain explicitly deferred:

- Windows report freshness;
- selected oracle freshness on macOS;
- hosted publication of all generated report families;
- hosted generated API HTML publication;
- broad report-index freshness;
- unselected comparison family freshness;
- package-manager provider support;
- shared-library ABI support;
- runtime-loader support;
- release evidence;
- performance superiority;
- external-library parity;
- state-of-the-art sparse linear algebra status.

## Validation Results

| Check | Result |
| --- | --- |
| targeted stale selected-comparison Linux-only scan | Passed with intentional oracle-only exceptions. |
| targeted selected oracle/macOS and Windows report freshness scan | Passed; no promoted claims found. |
| targeted package/ABI/performance/release/state-of-the-art scan | Passed; non-claims remain visible. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 11.
