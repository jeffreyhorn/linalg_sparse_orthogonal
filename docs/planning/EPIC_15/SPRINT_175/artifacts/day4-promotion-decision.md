# Day 4: Promotion Or Deferral Decision

## Purpose

Select one Sprint 175 report freshness path for complete closure, define its
platform support scope, preserve non-claims, and establish the implementation
and validation checklist before editing workflows, scripts, Make targets, or
documentation.

## Decision

Sprint 175 selects **macOS selected comparison freshness promotion**.

The selected lane is:

| Field | Decision |
| --- | --- |
| Platform | macOS hosted CI |
| Workflow target | `.github/workflows/macos-ci.yml` |
| Report family | selected `comparison` report freshness |
| Command | `make report-index-comparison-freshness` |
| Selected targets | `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, `lu-nonsym-square-5` |
| Generated artifact root | `build/comparison/` |
| Required selected study artifacts | `build/comparison/qr_minnorm/study.tsv`, `build/comparison/qr_compatible_ls/study.tsv`, `build/comparison/partial_svd_diag6_k2/study.tsv`, `build/comparison/lu_nonsym_square_5/study.tsv` |
| Support tier after successful implementation | reviewed macOS selected comparison freshness |
| Support tier before implementation lands | staged/local-only |
| Claim scope | The selected comparison freshness gate runs on hosted macOS for the four selected fixture-local comparison families and uploads selected generated artifacts for reviewer inspection. |

This is a report freshness promotion, not a package, ABI, solver-correctness,
performance, release, or state-of-the-art promotion.

## Why This Lane

Day 3 identified three plausible closure paths:

1. Linux hosted selected comparison reconciliation for the Sprint 174 LU target.
2. macOS selected comparison freshness promotion.
3. Windows selected comparison freshness formal deferral.

The macOS selected comparison freshness lane is selected because:

- it directly satisfies Sprint 175's "beyond Linux" goal;
- macOS already has reviewed static-first package/install and CMake
  install/export lanes, so the hosted runner is already part of the maintained
  platform contract;
- the selected comparison freshness command is already source-controlled and
  complete locally;
- the lane can be bounded to one Make target and one generated artifact tree;
- it avoids Windows Make/POSIX shell and `.exe` probe complexity for this
  sprint;
- it can be validated through hosted CI plus source-controlled artifact lists
  and support-tier documentation.

## Required Implementation Work

### Workflow Work

Add a macOS hosted comparison freshness job to `.github/workflows/macos-ci.yml`
or an equivalent reviewed macOS workflow lane.

The job should:

1. run on `macos-latest`;
2. check out the repository;
3. install any required reviewed-path tools if the selected command needs them;
4. run `make report-index-comparison-freshness`;
5. summarize all four selected comparison targets:
   - `qr-minnorm`;
   - `qr-compatible-ls`;
   - `partial-svd-diag6-k2`;
   - `lu-nonsym-square-5`;
6. upload selected comparison artifacts for all four target directories;
7. use failure behavior that fails if selected artifacts are missing.

### Path And Execution Work

Audit and normalize, if needed:

- Make target behavior under hosted macOS;
- Python helper invocation under hosted macOS;
- temporary C project probe compile/run behavior;
- generated `build/comparison/*` path creation and cleanup;
- TSV parsing and newline handling;
- artifact upload paths and retention policy;
- branch/source commit metadata in generated manifests.

### Documentation Work

Update maintained docs so they state:

- selected comparison freshness is local and reviewed Linux hosted today;
- Sprint 175 promotes a macOS hosted selected comparison freshness lane after
  the new job passes;
- the macOS lane covers only the four selected comparison families and selected
  uploaded artifacts;
- it does not promote broad report-index freshness, unselected comparison
  families, generated API HTML, package support, shared-library ABI support,
  Windows report freshness, performance, release, or state-of-the-art claims.

### Linux Reconciliation Work

Day 2 and Day 3 found that the existing Linux hosted selected comparison
summary/upload inventory omits the Sprint 174 LU target even though
`make report-index-comparison-freshness` now includes it.

This is not the primary Sprint 175 promotion decision, but it must be
reconciled while implementing the selected lane so Linux and macOS hosted
selected comparison inventories describe the same four selected targets.

## Validation Checklist

Local feasible checks:

- `make report-index-comparison-freshness`
- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 scripts/run_external_comparison.py --self-check`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`
- workflow syntax inspection for `.github/workflows/macos-ci.yml`
- selected comparison artifact path scan for all four targets
- stale hosted-comparison wording scan
- broad-claim/non-claim wording scan
- `bash scripts/package_manager_deferral_check.sh` if public/platform/package wording changes
- `bash scripts/static_package_deferral_check.sh` if package/ABI/static/shared wording changes
- `git diff --check`

Hosted validation:

- macOS selected comparison freshness job passes on GitHub Actions.
- The job uploads artifacts for all four selected comparison target
  directories.
- The summary reports four selected targets and 28 generated selected rows.

If `.c` or `.h` files are modified during implementation, run:

```sh
make format && make lint && make test
```

## Rejected Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| Linux hosted selected comparison LU reconciliation only | rejected as primary lane | It is a real consistency gap and should be fixed, but it does not promote report freshness beyond Linux. |
| macOS selected oracle freshness | deferred | Feasible, but selected comparison has a more immediate post-Sprint-174 inventory gap and stronger closure value. |
| Windows selected comparison CMake-first promotion | deferred | Valuable but too risky before designing CMake-native or PowerShell-native report freshness execution and temporary probe handling. |
| Windows selected comparison formal deferral | deferred | Useful if implementation proves blocked, but Day 4 selects a positive macOS promotion path first. |
| selected canonical benchmark freshness on macOS/Windows | deferred | Timing metadata and executable behavior risk accidental portable performance claims. |
| generated API freshness lane | rejected for Sprint 175 | Sprint 173 already closed generated API HTML as guarded local-only, and this sprint is report freshness promotion. |
| coverage/deadcode/sentinel/guardrail platform promotion | deferred | These are advisory/local or tool-heavy reports and are not the highest-value complete closure for Sprint 175. |

## Non-Claims

The selected macOS comparison freshness lane does not claim:

- broad macOS platform parity;
- Windows report freshness;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- hosted publication of all generated reports;
- hosted generated API HTML;
- broad report-index freshness;
- unselected comparison families;
- broad QR, SVD, partial-SVD, LU, or sparse-direct solver correctness;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or broad external-library parity;
- package-manager support;
- shared-library ABI support;
- runtime-loader behavior;
- release evidence;
- performance superiority;
- state-of-the-art sparse linear algebra status.

## Day 4 Completion Record

- Exactly one path is selected for complete closure: macOS selected comparison
  freshness promotion.
- The selected platform, command, artifacts, support tier, claim scope, and
  non-claims are documented before implementation.
- Linux hosted selected comparison LU reconciliation is recorded as an
  implementation consistency requirement, not the primary promotion decision.
- Rejected alternatives and validation expectations are documented.
