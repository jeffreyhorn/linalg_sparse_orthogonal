# Day 13 Hosted Readiness Review

## Scope

Day 13 reviews the Sprint 159 hosted report-freshness work as one reviewer
path: workflow, selected commands, generated artifacts, normalizer semantics,
documentation wording, local validation, and Sprint 160 handoff.

No C or public-header files were modified.

## Hosted Readiness Checklist

| Check | Status | Evidence |
| --- | --- | --- |
| Workflow is syntactically valid | Pass | `.github/workflows/ci.yml` parsed with Ruby YAML. |
| Hosted job is Linux-only and reviewed | Pass | Job `generated-report-freshness`, name `Linux reviewed hosted oracle/comparison freshness`. |
| Timeout is bounded | Pass | `timeout-minutes: 15`, matching Day 4 runtime budget. |
| Selected oracle command is maintained | Pass | Hosted step runs `make report-index-oracle-freshness`. |
| Selected comparison command is maintained | Pass | Hosted step runs `make report-index-comparison-freshness`. |
| Oracle summary is deterministic | Pass | Inline Python summarizes total, QR, partial-SVD, generated-reference, pass count, source commit, branch, and support tier from generated TSV/manifest. |
| Comparison summary is deterministic | Pass | Inline Python summarizes fixture, selected/pass row counts, dependency pass/defer counts, deferred optional names, source commit, and branch. |
| Artifact uploads are split and named | Pass | `sprint159-oracle-freshness` and `sprint159-comparison-qr-minnorm`. |
| Artifact retention is bounded | Pass | `retention-days: 7` for both uploads. |
| Missing selected files fail visibly | Pass | Both uploads use `if-no-files-found: error`. |
| Broad report-index output is not uploaded | Pass | No upload path includes `build/report-index/normalized-index.tsv`. |
| macOS/Windows report-index parity is not implied | Pass | Only `.github/workflows/ci.yml` changed; macOS/Windows workflows remain out of scope. |

## Selected-Row Evidence Map

| Promoted surface | Hosted command | Hosted artifacts | Summary signal | Failure behavior |
| --- | --- | --- | --- | --- |
| QR and partial-SVD selected oracle rows | `make report-index-oracle-freshness` | `build/corpus/oracle/corpus.oracle.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | `total_rows`, `qr_rows`, `partial_svd_rows`, `generated_reference_rows`, `pass_rows`, commit, branch, support tier | Missing, stale, failing, incomplete, missing-solver-family, or missing-fixture-key selected rows fail the command before artifact publication. Missing listed files fail upload. |
| QR minimum-norm selected comparison rows | `make report-index-comparison-freshness` | `build/comparison/qr_minnorm/project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` | selected row count, pass row count, dependency pass/defer counts, deferred optional dependency names, commit, branch | Missing, stale, failed, skipped, deferred, duplicate, unexpected, or incomplete selected rows fail normalizer tests and required freshness semantics. Missing listed files fail upload. |

## Local-Only Boundary Confirmation

Sprint 159 does not change generated row support-tier metadata by itself.
Generated oracle and comparison rows remain fixture-local generated report
rows under ignored `build/` paths. The new evidence is reviewed Linux hosted
execution of the selected gates plus split artifact upload.

These surfaces remain out of reviewed hosted claims:

- broad `python3 scripts/normalize_report_index.py --check-freshness`;
- broad report-index normalized TSV output;
- benchmark, coverage, dead-code, package, CI metadata, documentation, runtime
  backend, sentinel, and guardrail rows not selected by Sprint 159;
- optional NumPy/SciPy dependency defers;
- macOS/Windows report-index parity;
- package-manager, shared-library, ABI, dynamic-loader, and install proof;
- performance superiority;
- broad QR, broad partial-SVD, external-library parity, or state-of-the-art
  sparse linear algebra claims.

## Rerun Expectations

| Failure class | Expected response |
| --- | --- |
| GitHub runner service outage before checkout or action resolution | Rerun is acceptable; selected evidence was not produced. |
| Build or selected generator command fails | Treat as product failure until logs show an infrastructure-only cause. |
| Normalizer reports stale, missing, failing, skipped, deferred, duplicate, unexpected, row-count, solver-family, or fixture-key error | Treat as product failure; do not rerun as flake without a concrete root cause. |
| Artifact upload reports missing selected file | Treat as product failure unless the selected command already failed for an understood upstream reason. |
| Optional NumPy/SciPy rows defer | Do not fail solely for optional defers; they remain context and not pass evidence. |
| Summary step fails after selected command passes | Treat as product failure in Sprint 159 because reviewer traceability depends on deterministic summaries. |

## Residual Risks

| Risk | Status | Mitigation |
| --- | --- | --- |
| Hosted Ubuntu environment may expose a generator/runtime difference not seen locally. | Accepted for hosted readiness. | The selected job is the promotion vehicle; Day 12 local validation passed, and CI must pass before merge. |
| Uploads use `if: always()`, so a command failure may still attempt artifact upload. | Accepted. | Missing-file errors are visible, and available selected files are preserved when they exist. |
| Generated row metadata still says `local_only`. | Accepted and documented. | README, maintainer guide, corpus README, and solver-selection docs explain the distinction between row metadata and reviewed Linux hosted execution. |
| The maintainer evidence table still carries broad historical non-claim phrases. | Accepted. | New gate sections and public docs provide the authoritative Sprint 159 interpretation without broadening claims. |

## Final Sprint 160 QR Comparison Handoff

Sprint 160 should extend comparison evidence only by closing one named family
end to end. Start from the Sprint 159 pattern:

1. Select one QR comparison fixture family and define its exact row IDs before
   implementation.
2. Prefer a source-controlled dense-reference helper unless the chosen
   comparison truly requires an optional external dependency.
3. Define artifact paths, summary fields, support tier, claim scope, and
   non-claims before editing CI.
4. Add normalizer tests for complete, missing, stale, duplicate, unexpected,
   failed, skipped, and deferred selected rows.
5. Measure cold/warm runtime locally and set a hosted timeout before promotion.
6. Add one reviewed Linux hosted job or step only after local runtime and
   semantics are stable.
7. Upload split artifacts with a family-specific name.
8. Update docs only after local and hosted evidence passes.

Recommended first candidate: a single overdetermined compatible QR
least-squares fixture with residual and solution checks against the
source-controlled dense helper. It is narrower than broad external-library
parity and complements the current underdetermined minimum-norm comparison.

## Validation Performed

Day 13 did not rerun the Day 12 command suite. It reviewed readiness using the
passing Day 12 validation record and added workflow-oriented checks:

```sh
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml"); puts "ci.yml YAML parse ok"'
```

Local generated artifact presence from Day 12 was confirmed for every hosted
upload path.

## Completion Check

- A reviewer can trace each promoted claim to a hosted command, summary, and
  artifact group.
- Hosted artifacts and summaries are inspectable and bounded.
- Advisory/local-only families remain out of reviewed hosted claims.
- Sprint 160 has a concrete QR comparison handoff.
