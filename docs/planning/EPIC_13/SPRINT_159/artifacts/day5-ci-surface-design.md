# Day 5 CI Surface Design

## Scope

Day 5 designs the hosted CI surface for Sprint 159 selected oracle and
comparison freshness rows. It does not edit workflow files. Day 6 owns the
actual implementation.

The design must promote only selected generated oracle/comparison rows to
reviewed hosted evidence while preserving existing reviewed, supplemental,
advisory, staged, and local-only support-tier boundaries.

## Workflow Review

| Workflow | Current role | Sprint 159 design decision |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Linux enforced source-of-truth reviewed baseline plus Linux supplemental runtime, sanitizer, benchmark, TSan, coverage, dead-code, and static-first package lanes. | Use this workflow for Sprint 159 hosted selected oracle/comparison freshness. Linux is the right reviewed source-of-truth for Makefile/report-index commands. |
| `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang path, supplemental Homebrew GCC, wall/sanitize, and reviewed static-first package install/export proof. | Do not add Sprint 159 hosted report promotion here. macOS package/static-first lanes should remain package/platform proof, not generated report promotion. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake-first subset and CMake install/downstream package validation. | Do not add Sprint 159 hosted report promotion here. Windows does not claim Makefile parity or Unix report-index execution parity. |

## Placement Decision

Add one new serialized Linux job to `.github/workflows/ci.yml`:

```yaml
generated-report-freshness:
  name: Linux reviewed hosted oracle/comparison freshness
  runs-on: ubuntu-latest
  timeout-minutes: 15
```

Rationale:

- Linux is already documented as the strongest reviewed source-of-truth
  baseline.
- The selected commands are Makefile/report-index commands and fit the Linux
  reviewed topology.
- Day 4 measured cold oracle plus comparison runtime at about 48 seconds when
  duplicate rebuild cost is included; a 15-minute timeout leaves hosted
  margin without making failures slow.
- One serialized job keeps artifact paths truthful and avoids accidental
  concurrent writes under `build/corpus/`, `build/corpus-reports/`,
  `build/comparison/`, and `build/report-index/`.
- A separate job gives clear failure attribution without conflating generated
  report freshness with compile-quality, package, coverage, or benchmark lanes.

## Proposed Day 6 Step Shape

Recommended job steps:

```yaml
- uses: actions/checkout@v4

- name: Run reviewed hosted oracle freshness
  run: make report-index-oracle-freshness

- name: Run reviewed hosted QR minimum-norm comparison freshness
  run: make report-index-comparison-freshness

- name: Upload reviewed oracle/comparison freshness artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: sprint159-hosted-oracle-comparison-freshness
    retention-days: 7
    path: |
      build/corpus/oracle/corpus.oracle.tsv
      build/corpus-reports/index.tsv
      build/corpus-reports/skips.tsv
      build/corpus-reports/manifest.txt
      build/comparison/qr_minnorm/project_observations.tsv
      build/comparison/qr_minnorm/baseline_observations.tsv
      build/comparison/qr_minnorm/dependency_status.tsv
      build/comparison/qr_minnorm/study.tsv
      build/comparison/qr_minnorm/summary.md
      build/comparison/qr_minnorm/manifest.tsv
```

Day 6 may split artifact upload into two uploads if review clarity is better:

- `sprint159-oracle-freshness`
- `sprint159-comparison-qr-minnorm`

Either shape is acceptable if artifact names stay scoped to selected hosted
oracle/comparison rows.

## Required Command And Environment List

| Purpose | Command or environment | Requirement |
| --- | --- | --- |
| Oracle selected freshness | `make report-index-oracle-freshness` | Required hosted reviewed step. |
| Comparison selected freshness | `make report-index-comparison-freshness` | Required hosted reviewed step. |
| Oracle generator | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Run indirectly through Makefile target unless debugging. |
| Oracle normalizer | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | Run indirectly through Makefile target unless debugging. |
| Comparison generator | `python3 scripts/run_external_comparison.py --target qr-minnorm` | Run indirectly through Makefile target unless debugging. |
| Comparison normalizer | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Run indirectly through Makefile target unless debugging. |
| Compiler | Ubuntu default `cc` | Accept default reviewed Linux Makefile compiler unless later evidence requires pinning. |
| Python | Ubuntu default `python3` | Required for generator/normalizer scripts and source-controlled QR reference helper. |
| Optional NumPy/SciPy | none | Do not install; optional package baselines remain deferred and are not pass evidence. |
| Timeout | `timeout-minutes: 15` | Required if implemented as one combined serialized job. |
| Artifact retention | `retention-days: 7` | Required for selected generated outputs. |

No new package installation is required by the Day 5 design.

## Artifact Map

| Artifact group | Paths | Support-tier interpretation |
| --- | --- | --- |
| Selected oracle generated rows | `build/corpus/oracle/corpus.oracle.tsv` | Reviewed hosted evidence only for selected QR and partial-SVD oracle rows after the hosted job passes. |
| Oracle report context | `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | Hosted context for selected oracle row interpretation. Generated-reference rows remain supplemental context. |
| Selected QR minimum-norm comparison rows | `build/comparison/qr_minnorm/study.tsv` | Reviewed hosted evidence only for the `qr_underdetermined_minnorm_2x4` fixture-local comparison after the hosted job passes. |
| Comparison diagnostics | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `summary.md`, `manifest.tsv` | Hosted context for selected comparison row interpretation and failure triage. |

Do not upload broad `build/report-index/normalized-index.tsv` as a product
artifact in Day 6 unless the implementation explicitly limits it to selected
families. Broad report-index freshness remains advisory/local until selected
by a later sprint.

## PR Failure Semantics

| State | CI result | Interpretation |
| --- | --- | --- |
| Selected oracle command exits nonzero | fail | Product failure for selected hosted oracle freshness. |
| Selected comparison command exits nonzero | fail | Product failure for selected hosted comparison freshness. |
| Missing selected oracle artifact | fail | Missing generated selected rows are not pass evidence. |
| Missing selected comparison artifact | fail | Missing generated selected rows are not pass evidence. |
| Stale selected row | fail | Source-commit or freshness mismatch blocks hosted evidence. |
| Selected oracle row `comparison_status != pass` | fail | Solver-backed selected oracle evidence failed. |
| Selected comparison row `status != pass` | fail | Fixture-local comparison evidence failed. |
| Oracle selected row-count mismatch | fail | Expected QR, partial-SVD, generated-reference, or total row counts changed without review. |
| Comparison selected row-set mismatch | fail | Expected QR minimum-norm comparison row set changed without review. |
| Optional NumPy/SciPy row is deferred | pass with context | Optional package baselines are not selected and not pass evidence. |
| GitHub Actions service outage before row generation | rerunnable infrastructure failure | Rerun is acceptable only when no selected generated row result exists. |
| Artifact upload fails after selected commands pass | fail unless logs are sufficient and artifact failure is known infrastructure | Reviewed hosted evidence requires inspectable artifacts or deterministic summaries. |

## Support-Tier Wording To Preserve

Day 6 implementation must preserve these boundaries:

- Linux is the reviewed source-of-truth for this hosted generated report lane.
- macOS and Windows do not gain report-index parity from Sprint 159.
- Windows still does not claim Makefile parity or `pkg-config` execution
  parity from Sprint 159.
- Generated API HTML remains local-only under the Sprint 158 decision.
- QR and partial-SVD evidence remains fixture-local.
- QR comparison remains the single `qr_underdetermined_minnorm_2x4`
  minimum-norm comparison, not broad QR parity.
- Optional NumPy/SciPy comparison baselines remain deferred and are not pass
  evidence.
- Package, ABI, shared-library, dynamic-loader, package-manager, performance,
  platform, broad external-library parity, and state-of-the-art claims remain
  non-claims.

## Day 6 Implementation Checklist

1. Edit only `.github/workflows/ci.yml` unless implementation discovers a
   narrow docs wording dependency.
2. Add the new Linux job with a clear reviewed hosted freshness name.
3. Run selected Makefile gates serially.
4. Add selected artifact upload with 7-day retention and scoped artifact names.
5. Do not install optional NumPy/SciPy dependencies.
6. Do not add macOS or Windows lanes.
7. Run documentation/workflow hygiene checks after editing.
8. If workflow YAML is touched only, do not run C quality gates unless code or
   headers are also modified.

## Completion Check

- CI lane placement is justified by existing support-tier boundaries.
- Selected checks have concrete commands and output paths.
- PR failure semantics distinguish product failures from optional/deferred
  context.
- Existing reviewed, supplemental, staged, advisory, and local-only wording
  remains coherent.
