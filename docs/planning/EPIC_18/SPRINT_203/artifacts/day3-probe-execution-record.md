# Day 3: Probe Execution And Failure Record

## Purpose

Day 3 ran the selected `qr-incompatible-ls` comparison through the available
local generator and local CMake probe paths, captured generated artifact facts,
and classified whether the evidence is sufficient for Windows promotion.

## Environment

| Field | Value |
| --- | --- |
| Host | Darwin x86_64 |
| Kernel | `Darwin Kernel Version 24.6.0` |
| CMake | `4.4.3` |
| Python | `3.14.5` |
| Git commit | `0660e84324dcf879d4f957ca7bad335297fe1019` |
| Branch | `sprint-203` |

This is not a Windows/MSVC host. Day 3 therefore records local CMake probe
evidence plus a promotion-blocking hosted Windows evidence gap.

## Commands And Results

| Command | Result |
| --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed; wrote `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-config Release --keep-temp` | Passed; generated the same selected bundle through local CMake probe mode. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed; all six QR incompatible rows were fresh to current `HEAD`. |

## Generated Manifest Facts

| Manifest key | Observed value |
| --- | --- |
| `target` | `qr-incompatible-ls` |
| `platform` | `darwin-x86_64` |
| `compiler` | `cmake-probe:default:Release` |
| `configuration` | `stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only` |
| `source_commit` | `0660e84324dcf879d4f957ca7bad335297fe1019` |
| `source_branch` | `sprint-203` |
| `worktree_state` | `dirty` |
| `study_path` | `build/comparison/qr_incompatible_ls/study.tsv` |

The `dirty` worktree state reflects uncommitted Sprint 203 planning docs. It
does not change the conclusion that the generated output remains local-only
evidence.

## Artifact Set

| Artifact | Status |
| --- | --- |
| `project_observations.tsv` | Present. |
| `baseline_observations.tsv` | Present. |
| `dependency_status.tsv` | Present; required Python and dense QR helper passed. |
| `study.tsv` | Present; six selected QR incompatible rows passed. |
| `summary.md` | Present. |
| `manifest.tsv` | Present. |

## Selected Row Outcome

| Row | Outcome |
| --- | --- |
| Project status | `SPARSE_SUCCESS`. |
| Baseline status | `success`. |
| Residual norm | Project and baseline both `1.7320508075688772`. |
| Solution norm | Project and baseline both `2.2360679774997894`. |
| Solution values | Matched within tolerance with max component delta `2.2204460492503131e-16`. |
| Project vs baseline max absolute delta | `2.2204460492503131e-16`, below `1e-10`. |

## Failure Classification

| Category | Day 3 result | Disposition |
| --- | --- | --- |
| Generator failure | Not observed locally. | No Day 3 generator fix required. |
| Local CMake probe failure | Not observed. | No Day 3 local CMake fix required. |
| Dependency failure | Not observed for required helper. | Optional NumPy/SciPy deferrals remain non-evidence. |
| Artifact packaging failure | Not observed locally. | Hosted upload still unproven. |
| Selected freshness failure | Not observed locally. | Freshness passed for the selected target. |
| Windows/MSVC evidence gap | Observed. | Blocks manifest promotion. |

## Promotion Recommendation

Do not promote `SRT-COMP-QR-INCOMPATIBLE-LS` Windows metadata based on Day 3.
The selected target has good local generator and local CMake evidence, but it
does not have hosted Windows/MSVC proof, Windows artifact upload review, or
promoted generated support-tier semantics. Day 4 should design any needed
generator/path/CMake fixes conservatively and keep manifest promotion blocked
until hosted Windows evidence exists.
