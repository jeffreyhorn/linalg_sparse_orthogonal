# Sprint 212 Day 3: Runner And Variance Inventory

## Summary

Day 3 inspects the selected benchmark runner, compiler, repeat, warmup,
variance, and hosted artifact evidence. The current selected benchmark
freshness evidence is useful and recent, but it is not stable enough to
support a normal hosted timing threshold. The strongest supported direction is
still threshold-free hosted selected freshness, unless Sprint 212 deliberately
chooses a much narrower local smoke-ceiling policy.

## Evidence Commands

Day 3 used GitHub Actions metadata and retained artifacts:

```sh
gh run list --limit 20 --json databaseId,workflowName,displayTitle,headBranch,headSha,status,conclusion,createdAt,updatedAt,event
gh run view 36798099138 --json jobs,conclusion,workflowName,headSha,createdAt
gh run view 36798099076 --json jobs,conclusion,workflowName,headSha,createdAt
gh run download 36798099138 --name sprint168-selected-performance-freshness --dir <tmpdir>
gh run download 36798099076 --name sprint202-macos-selected-performance-freshness --dir <tmpdir>
python3 scripts/check_bench_canonical_freshness.py --report-dir <linux_tmpdir> --mode hosted
python3 scripts/check_bench_canonical_freshness.py --report-dir <macos_tmpdir> --mode hosted
```

The latest completed PR-hosted Linux and macOS selected performance artifacts
were downloadable and passed the hosted freshness checker. Three earlier
retained attempts for each platform were also downloadable for timing-sample
inspection.

## Runner And Methodology Matrix

| Lane | Runner class | OS/compiler evidence | CPU disclosure | Command | Repeat/warmup/variance | Artifact retention | Claim boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local baseline | Local workstation | Darwin, Apple clang 17.0.0 | `unknown` | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1`, `none_configured`, `not_computed_single_sample` | Ignored local `build/` output | `local_threshold_free` |
| Linux hosted selected | GitHub Actions hosted Linux | Ubuntu/Linux, GCC 13.3.0 through `cc` | Captured from `/proc/cpuinfo`; retained samples include AMD EPYC 9V74, AMD EPYC 7763, and Intel Xeon Platinum 8573C. | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1`, `none_configured`, `not_computed_single_sample` | 7-day workflow artifact retention | `hosted_selected_threshold_free` |
| macOS hosted selected | GitHub Actions hosted macOS | macOS, Apple clang 21.0.0 | Captured from `sysctl`; retained samples show `Apple M1 (Virtual)`. | `tests/data/suitesparse/nos4.mtx --repeat 1` | `configured_repeat_1`, `none_configured`, `not_computed_single_sample` | 7-day workflow artifact retention | `hosted_selected_threshold_free` |

## Retained Hosted Samples

These values are useful for assessing comparability risk. They are not a
controlled variance model because the retained samples span different commits
and runner assignments.

| Platform | Run id | Commit | CPU model | `refactor_csc_ms` | `speedup_refactor` |
| --- | --- | --- | --- | ---: | ---: |
| Linux | `36798099138` | `1e39d4c` | AMD EPYC 9V74 80-Core Processor | 0.058 | 1.14 |
| Linux | `36793801250` | `1d3af74` | AMD EPYC 7763 64-Core Processor | 0.047 | 1.78 |
| Linux | `36791438422` | `4105b9c` | AMD EPYC 7763 64-Core Processor | 0.047 | 1.79 |
| Linux | `36790897256` | `0eb0e97` | Intel Xeon Platinum 8573C | 0.061 | 1.35 |
| macOS | `36798099076` | `1e39d4c` | Apple M1 (Virtual) | 0.032 | 1.72 |
| macOS | `36793801179` | `1d3af74` | Apple M1 (Virtual) | 0.030 | 1.80 |
| macOS | `36791438432` | `4105b9c` | Apple M1 (Virtual) | 0.037 | 2.11 |
| macOS | `36790897338` | `0eb0e97` | Apple M1 (Virtual) | 0.091 | 1.48 |

## Variance And Comparability Findings

- Linux hosted samples changed CPU model across retained runs while keeping the
  same runner label, which blocks treating `ubuntu-latest` as one timing
  class.
- Linux `refactor_csc_ms` ranged from `0.047` to `0.061` ms in retained
  artifacts.
- macOS retained samples used the same `Apple M1 (Virtual)` CPU label, but
  `refactor_csc_ms` still ranged from `0.030` to `0.091` ms.
- Every retained selected row remains a single-repeat measurement with no
  warmup and no computed variance.
- The hosted artifacts retain enough metadata to audit freshness and context,
  but not enough to justify a thresholded hosted performance gate.
- Artifact retention is seven days, so historical timing review is short-lived
  unless Sprint 212 adds a longer-lived source-controlled baseline or another
  retained evidence path.

## Minimum Threshold Metadata

| Required metadata | Current status |
| --- | --- |
| Exact selected target, command, artifact, and row count | Present and guarded. |
| Runner class or same-machine comparison policy | Missing for hosted thresholding; runner labels are not stable timing classes. |
| Compiler, build flags, build mode, and thread count | Present as context. |
| Repeat/sample count sufficient for thresholding | Missing; selected row is `configured_repeat_1`. |
| Warmup policy | Missing; current value is `none_configured`. |
| Variance or outlier rule | Missing; current value is `not_computed_single_sample`. |
| Baseline provenance | Missing for canonical selected freshness; current value is `n/a`. |
| Threshold and allowed-regression policy | Missing for canonical selected freshness; current value is `n/a`. |
| Long enough artifact/baseline retention | Weak; hosted artifacts retain for seven days. |
| Claim-boundary wording and tests | Present for threshold-free policy; would need exact updates for thresholded policy. |

## Candidate Assessment

| Candidate | Day 3 decision input |
| --- | --- |
| Hosted Linux selected threshold | Blocked by CPU variability, single sample, no warmup, no variance, no baseline, and no threshold. |
| Hosted macOS selected threshold | Blocked by timing outlier, single sample, no warmup, no variance, no baseline, and no threshold. |
| Cross-platform hosted threshold | Rejected: Linux/macOS timing comparability is explicitly unclaimed. |
| Existing S6 local selected smoke ceiling | Plausible only as a local smoke ceiling. It is separate from hosted selected canonical freshness and cannot support hosted publication or portable performance. |
| Stronger threshold-free hosted selected policy | Best-supported candidate for Day 4 criteria and Day 5 decision. |

## Day 3 Outcome

The sprint can now explain why the current selected hosted benchmark evidence
is eligible for freshness/methodology validation but blocked for a normal
timing threshold. Day 4 should convert this inventory into explicit decision
criteria for either a deliberately local smoke gate or stronger threshold-free
deferral.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
