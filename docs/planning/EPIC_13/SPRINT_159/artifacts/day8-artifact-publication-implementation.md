# Day 8 Artifact Publication Implementation

## Scope

Day 8 implements the artifact publication policy designed on Day 7. The
implementation keeps hosted evidence inspectable by splitting oracle and QR
minimum-norm comparison artifacts, adding deterministic summaries, and making
missing selected artifact files fail clearly.

No C or public-header files were modified.

## Changed Files

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | Split selected artifact uploads, added oracle/comparison summary steps, and made missing selected files fail with `if-no-files-found: error`. |
| `docs/planning/EPIC_13/SPRINT_159/WORKING_NOTES.md` | Recorded Day 8 implementation notes and Day 9 handoff. |
| `docs/planning/EPIC_13/SPRINT_159/artifacts/day8-artifact-publication-implementation.md` | Captured this implementation artifact. |

## Workflow Implementation

The hosted job still runs the selected commands serially:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
```

Day 8 added summary steps immediately after each selected command and before
the corresponding artifact upload.

## Oracle Summary

The oracle summary step reads:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/manifest.txt`

It prints:

```text
sprint159-oracle-summary: total_rows=<n> qr_rows=<n> partial_svd_rows=<n> generated_reference_rows=<n> pass_rows=<n>
sprint159-oracle-summary: source_commit=<commit> source_branch=<branch> support_tier=<tier>
```

Expected current selected values are:

- `total_rows=52`
- `qr_rows=23`
- `partial_svd_rows=26`
- `generated_reference_rows=3`
- `pass_rows=52`

The summary intentionally reports the generated manifest support tier rather
than rewriting generated-file provenance during the CI step.

## Oracle Artifact Upload

Artifact:

```yaml
name: sprint159-oracle-freshness
retention-days: 7
if-no-files-found: error
```

Paths:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

Interpretation:

- reviewed hosted evidence only for selected QR and partial-SVD oracle rows
  after the hosted job passes;
- generated-reference rows remain supplemental hosted context;
- corpus metadata and broad report-index families remain unpromoted.

## Comparison Summary

The comparison summary step reads:

- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/manifest.tsv`

It prints:

```text
sprint159-comparison-summary: fixture=<fixture> selected_rows=<n> pass_rows=<n>
sprint159-comparison-summary: dependency_pass=<n> dependency_defer=<n> deferred_optional=<names>
sprint159-comparison-summary: source_commit=<commit> source_branch=<branch>
```

Expected current selected values are:

- `fixture=qr_underdetermined_minnorm_2x4`
- `selected_rows=6`
- `pass_rows=6`
- `dependency_pass=2`
- `dependency_defer=2`
- `deferred_optional=numpy,scipy`

Deferred optional dependencies are printed for review context only. They are
not pass evidence and are not required packages.

## Comparison Artifact Upload

Artifact:

```yaml
name: sprint159-comparison-qr-minnorm
retention-days: 7
if-no-files-found: error
```

Paths:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`

Interpretation:

- reviewed hosted evidence only for the selected
  `qr_underdetermined_minnorm_2x4` minimum-norm comparison after the hosted
  job passes;
- not broad QR parity or broad external-library parity.

## Failure Diagnostics

| Failure point | Expected hosted behavior |
| --- | --- |
| Oracle command fails before writing files | Job fails; oracle upload also fails with missing selected files. |
| Oracle command writes partial files then fails | Job fails; oracle upload publishes available selected files if all listed paths exist, otherwise upload reports missing files. |
| Oracle summary fails | Job fails before comparison command; oracle upload still runs because `if: always()` is set. |
| Comparison command fails before writing files | Job fails; comparison upload also fails with missing selected files. |
| Comparison command writes partial files then fails | Job fails; comparison upload publishes available selected files if all listed paths exist, otherwise upload reports missing files. |
| Optional NumPy/SciPy dependency is deferred | Job can pass if selected comparison rows pass; summary prints the defers as context. |

Day 9/10 should decide whether the artifact policy needs partial-output path
lists or separate failure-only diagnostic uploads. Day 8 deliberately keeps
strict missing-file handling for selected files so absent selected evidence is
visible.

## Local Dry-Run Notes

The summary steps use inline Python with only the standard library:

- `csv`
- `pathlib`

They read the same generated TSV and manifest files produced by the selected
Make targets. They do not require NumPy, SciPy, PyYAML, or package-manager
setup.

The workflow was syntax-checked locally with Ruby YAML parsing, and the
workflow/documentation diff passed whitespace checks.

## Boundaries Preserved

- Selected commands are unchanged.
- Broad normalized report-index output is not uploaded.
- macOS and Windows workflows are unchanged.
- Generated API HTML policy is unchanged.
- Optional NumPy/SciPy baselines remain deferred context only.
- Package, ABI, shared-library, dynamic-loader, package-manager, performance,
  broad platform, broad external-library parity, and state-of-the-art claims
  remain non-claims.

## Completion Check

- Promoted rows produce inspectable hosted evidence.
- Outputs are deterministic enough for hosted review and rerun comparison.
- Failure diagnostics do not require reproducing the full run locally when
  selected generated files are available.
- Missing selected files fail loudly.
