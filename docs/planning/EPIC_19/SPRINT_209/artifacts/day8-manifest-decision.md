# Sprint 209 Day 8: Manifest Decision

## Purpose

Apply the Day 7 promotion criteria to the evidence currently available for
`SRT-COMP-QR-INCOMPATIBLE-LS`.

## Hosted Evidence Check

Command:

```text
gh run list --workflow "Windows CI" --branch sprint-209 --limit 10 --json databaseId,displayTitle,event,headSha,createdAt,updatedAt,status,conclusion,url
```

Result:

```text
[]
```

No hosted Windows workflow run exists yet for branch `sprint-209`, so no hosted
QR incompatible artifact can be inspected.

## Decision

`SRT-COMP-QR-INCOMPATIBLE-LS` remains re-deferred for selected Windows freshness.

| Field | Retained value |
| --- | --- |
| `support_tier` | `local_only` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `non_claims` | Current full QR boundary including `no Windows report freshness`. |

The Day 5 Windows QR workflow lane remains an evidence-collection path only. It
does not promote the source-of-truth selected target manifest without a hosted
run and inspected artifact.

## Absence Guard Updates

`tests/test_selected_report_targets_manifest.py` now rejects these values in the
QR incompatible selected target row while re-deferred:

- `.github/workflows/windows-ci.yml`;
- `selected-qr-incompatible-comparison-freshness`;
- `sprint209-windows-selected-comparison-qr-incompatible`;
- `windows`.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

## Day 8 Close

Manifest promotion is not earned yet. Day 9 should integrate this re-deferral
with workflow, PowerShell, normalizer, schema, and docs guards so the branch can
collect hosted evidence without overstating selected Windows QR freshness.
