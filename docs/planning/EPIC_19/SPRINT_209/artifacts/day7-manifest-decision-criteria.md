# Sprint 209 Day 7: Manifest Decision Criteria

## Purpose

Define the objective rules Day 8 must use when deciding whether
`SRT-COMP-QR-INCOMPATIBLE-LS` can be promoted to selected Windows hosted
freshness or must remain re-deferred.

## Promotion Gates

Promotion requires all gates below:

| Gate | Required evidence |
| --- | --- |
| Hosted run | Successful `Windows CI` run on the reviewed branch or reviewed merge commit. |
| QR job | Successful `selected-qr-incompatible-comparison-freshness` job. |
| Generator | Exact hosted MSVC/CMake command for `--target qr-incompatible-ls`. |
| Freshness | Exact selected freshness command for `--selected-target qr-incompatible-ls`. |
| Artifact | Uploaded `sprint209-windows-selected-comparison-qr-incompatible`. |
| Files | Exactly the six required QR files under `build/comparison/qr_incompatible_ls/`. |
| Rows | Exactly the six `comparison_qr_overdetermined_incompatible_4x2_*` rows. |
| Status | All six rows pass, source commit matches reviewed commit, worktree is clean, and freshness reports no selected QR errors. |
| Alignment | Manifest, generated metadata, docs, workflow guards, PowerShell guards, and normalizer checks agree on selected-only Windows QR evidence. |

## Promoted Manifest Values

If all gates pass, Day 8 may promote only the QR incompatible row with:

| Field | Required value |
| --- | --- |
| `support_tier` | `hosted_selected` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml;.github/workflows/windows-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness;selected-qr-incompatible-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness;sprint209-windows-selected-comparison-qr-incompatible` |
| `workflow_platforms` | `linux;macos;windows` |

The promoted claim scope must be:

```text
Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.
```

The promoted non-claims must be:

```text
no broad QR parity;no broad least-squares parity;no raw QR basis identity;no Q sign or orientation claim;no global rank-threshold policy;no broad rank-deficient solve claim;no NumPy parity;no SciPy parity;no LAPACK parity;no SuiteSparse parity;no Eigen parity;no broad Windows report freshness;no package-manager proof;no shared-library ABI proof;no performance superiority;no state-of-the-art claim
```

## Required Promotion Tests

Promotion must add exact manifest-contract coverage for:

- QR identity fields, row count, required files, and expected row IDs;
- Linux/macOS/Windows workflow metadata in exact order;
- Sprint 209 Windows QR artifact name;
- `support_tier=hosted_selected`;
- promoted claim scope text;
- full promoted non-claim tuple;
- rejection of Windows metadata on unrelated selected targets.

## Re-Deferral Conditions

Day 8 must retain re-deferral if any of these are true:

- hosted Windows run is absent, inaccessible, cancelled, skipped, failed, or not
  tied to the reviewed branch or commit;
- QR job is missing or failed;
- artifact is missing, inaccessible, stale, broadly scoped, or incompletely
  uploaded;
- `study.tsv` has missing, duplicate, unexpected, stale, skipped, deferred, or
  failed QR rows;
- generated row support-tier or non-claim wording still says local-only Windows
  freshness is unearned;
- docs and guards cannot be made consistent with promotion in the same branch.

## Retained Re-Deferral Values

If re-deferred, keep:

| Field | Required value |
| --- | --- |
| `support_tier` | `local_only` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `non_claims` | Current full QR non-claim set including `no Windows report freshness`. |

## Day 7 Disposition

The branch has local QR generation, selected freshness, workflow wiring, and
artifact-bundle validation, but no inspected hosted Windows QR artifact yet.
Day 8 must fetch hosted evidence if available before editing manifest metadata.
