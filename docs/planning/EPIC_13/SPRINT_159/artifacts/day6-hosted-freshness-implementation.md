# Day 6 Hosted Freshness Implementation

## Scope

Day 6 implements the hosted CI execution surface designed on Day 5. The
implementation promotes only selected oracle/comparison generated freshness
rows to Linux reviewed hosted evidence. It does not promote broad report-index
freshness, macOS/Windows report parity, generated API HTML, package/ABI,
performance, or broad external-library parity claims.

## Changed Files

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | Added selected hosted oracle/comparison freshness to the Linux reviewed baseline comment and added a serialized Linux hosted freshness job. |
| `docs/planning/EPIC_13/SPRINT_159/WORKING_NOTES.md` | Recorded Day 6 implementation notes and Day 7 handoff. |
| `docs/planning/EPIC_13/SPRINT_159/artifacts/day6-hosted-freshness-implementation.md` | Captured this implementation artifact. |

No C or public-header files were modified.

## Workflow Implementation

Added one Linux job:

```yaml
generated-report-freshness:
  name: Linux reviewed hosted oracle/comparison freshness
  runs-on: ubuntu-latest
  timeout-minutes: 15
```

The job runs selected gates serially:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
```

Serial execution is intentional because both commands write under ignored
`build/` report paths and Day 4 measured combined runtime well below the
15-minute timeout.

## Artifact Upload

The job uploads one scoped artifact on success or failure:

```yaml
name: sprint159-hosted-oracle-comparison-freshness
retention-days: 7
```

Uploaded paths:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`
- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`

The artifact is intentionally scoped to selected oracle/comparison outputs.
It does not upload broad `build/report-index/normalized-index.tsv`.

## Selected Evidence Scope

| Selected evidence | Hosted command | Reviewed-hosted interpretation after CI pass |
| --- | --- | --- |
| QR oracle solver-backed rows | `make report-index-oracle-freshness` | Fixture-local QR rank/nullity/nullspace and minimum-norm oracle freshness evidence. |
| Partial-SVD oracle solver-backed rows | `make report-index-oracle-freshness` | Fixture-local partial-SVD top-k, rank, projector, residual, orthogonality, sparse low-rank, fail-closed, and recovery oracle freshness evidence. |
| QR minimum-norm comparison rows | `make report-index-comparison-freshness` | Fixture-local comparison for `qr_underdetermined_minnorm_2x4` against the selected source-controlled dense reference helper. |
| Oracle generated-reference rows | `make report-index-oracle-freshness` | Supplemental hosted context for selected solver-backed rows, not primary public claim evidence. |

## Non-Promoted Families

The implementation does not promote:

- broad report-index `--check-freshness`;
- benchmark, sentinel, guardrail, coverage, dead-code, package, CI,
  documentation, runtime-backend, or corpus metadata families;
- optional NumPy/SciPy comparison baselines;
- generated API HTML from Sprint 158;
- macOS or Windows report-index execution parity.

Workflow comments explicitly state that non-selected generated report families
remain local-only, supplemental, or advisory unless a later sprint promotes
them with their own runtime, artifact, and claim policy.

## Failure Semantics

The hosted job fails if either selected Make target exits nonzero. The
maintained Make targets already run the strict selected normalizer checks:

- `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`

Failure states include stale, missing, failing, partial, row-count mismatch,
missing-solver-family, missing-fixture-key, selected comparison row-set
mismatch, or non-pass selected comparison row diagnostics.

Optional NumPy/SciPy package baselines remain deferred and are not treated as
pass evidence.

## Support-Tier Boundaries Preserved

- Linux is the reviewed source-of-truth for this hosted generated report lane.
- macOS and Windows workflows were not changed.
- Windows still does not claim Makefile parity or `pkg-config` execution
  parity from Sprint 159.
- The static-first package contract remains separate from generated report
  freshness.
- QR and partial-SVD evidence remains fixture-local.
- The QR comparison lane remains a single fixture-local minimum-norm
  comparison, not broad QR parity.
- Package, ABI, shared-library, dynamic-loader, package-manager, performance,
  broad platform, broad external-library parity, and state-of-the-art claims
  remain non-claims.

## Day 7 Handoff

Day 7 should refine artifact publication policy:

- decide whether the combined artifact should remain combined or be split into
  `sprint159-oracle-freshness` and `sprint159-comparison-qr-minnorm`;
- define expected passing summary text for hosted logs;
- define failure-artifact expectations if the oracle step fails before
  comparison output exists;
- decide whether `if-no-files-found` should stay at the default warning
  behavior or become stricter after step-level artifact split;
- document how reviewers should interpret hosted artifacts versus advisory
  local-only families.

## Completion Check

- Selected rows are executable from hosted CI configuration.
- Non-selected generated families are not accidentally promoted.
- Failure semantics match Day 5 design through maintained strict Make targets.
- Artifact upload is scoped to selected oracle/comparison outputs.
