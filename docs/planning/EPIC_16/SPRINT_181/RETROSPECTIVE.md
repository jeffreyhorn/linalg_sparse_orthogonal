# Sprint 181 Retrospective

**Sprint:** 181 - Selected Report Target Manifest
**Duration:** 14 days (Days 1-14 landed on branch `sprint-181`)
**Status:** Complete

## Source Artifact Note

Sprint 181 was executed from the Epic 16 project-plan section for Sprint 181
and lives under `docs/planning/EPIC_16/SPRINT_181/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 181 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Inventoried selected oracle, comparison, benchmark, workflow, docs,
      artifact, expected-row, support-tier, and freshness-policy duplication.
- [x] Designed and added `tests/corpus/manifests/selected_report_targets.tsv`
      as the source-controlled selected target authority.
- [x] Added selected-target manifest validation with duplicate detection,
      enum checks, required generated-file checks, expected-row checks,
      workflow metadata checks, and row-specific diagnostics.
- [x] Refactored selected oracle and comparison normalizer freshness checks to
      consume manifest-owned selected target expectations.
- [x] Refactored selected benchmark freshness checks to consume
      manifest-owned artifact, required-file, support-tier, and selected-row
      identity metadata.
- [x] Refactored workflow guard tests to consume manifest-owned workflow file,
      job, platform, and upload artifact metadata while keeping exact YAML
      block checks.
- [x] Updated README, maintainer guide, benchmark docs, corpus docs, and
      report-index schema docs to name the selected-target manifest as
      authority.
- [x] Preserved non-claims for broad report-index freshness, macOS selected
      oracle freshness, Windows report freshness, unselected report families,
      package-manager support, shared-library ABI support, broad platform
      proof, performance superiority, release proof, and state-of-the-art
      status.
- [x] Ran schema, manifest, workflow, normalizer, selected freshness, benchmark
      freshness, package deferral, Python compile, and whitespace validation.

## What Went Well

1. **The selected target authority became concrete.** Sprint 181 moved from a
   planned manifest to a reviewed TSV with six selected rows covering oracle,
   comparison, and benchmark freshness.

2. **The migration was incremental.** Normalizer checks moved first, benchmark
   checks followed, workflow scope checks came next, and documentation was
   aligned after guard behavior existed.

3. **Workflow guards stayed exact.** The workflow test now reads manifest
   expectations, but still checks exact job blocks, upload blocks,
   `actions/upload-artifact@v4`, `if-no-files-found: error`, and broad upload
   rejection.

4. **Diagnostics became more actionable.** Selected-target validation now
   reports `target_id` for unsupported policies, support tiers, expected-row
   failures, required-file failures, hosted metadata failures, and artifact
   pattern failures.

5. **Documentation now points maintainers at one authority.** README,
   maintainer guide, benchmark docs, corpus docs, and report-index schema docs
   now tell maintainers to update `selected_report_targets.tsv` instead of
   copying target lists into prose.

6. **Windows stayed intentionally unselected.** Sprint 181 did not accidentally
   promote Windows report freshness while centralizing Linux/macOS selected
   report lanes.

## What Didn't Go Well

1. **Some duplication remains by design.** Make target names, readable
   high-level non-claims, generated artifact inspection examples, workflow
   YAML structure checks, benchmark methodology fields, oracle solver-family
   bucket checks, and generated row-name summaries still live outside the
   manifest.

2. **Benchmark methodology is still checker-owned.** The current manifest
   schema does not model workload command, matrix size, repeat semantics,
   warmup, variance, baseline, threshold, or methodology notes as typed
   fields.

3. **Oracle solver-family bucket counts are still compatibility logic.** The
   selected-target manifest owns the total selected row count and fixture-key
   set, not per-solver-family bucket counts.

4. **The benchmark validation surface has shared output.** `make
   bench-canonical-report-freshness` and
   `python3 tests/test_bench_canonical_freshness.py` both write
   `build/bench-reports/canonical/` and should be run sequentially.

5. **The selected manifest is now a high-leverage file.** Future target changes
   must update validation, workflow expectations, docs, and claim boundaries
   deliberately or the manifest can become a bottleneck.

## Final Metrics

### Validation

| Metric | Sprint 181 close state |
| --- | --- |
| corpus schema and selected-target manifest validation | passed: `python3 scripts/validate_corpus_schema.py` |
| selected-target malformed-row regressions | passed: `python3 tests/test_selected_report_targets_manifest.py` |
| selected workflow guard and drift tests | passed: `python3 tests/test_selected_comparison_workflow.py` |
| normalizer manifest/freshness regressions | passed: `python3 tests/test_normalize_report_index.py` |
| selected oracle freshness | passed: `make report-index-oracle-freshness` |
| selected comparison freshness | passed: `make report-index-comparison-freshness` |
| selected benchmark freshness | passed: `make bench-canonical-report-freshness` |
| benchmark freshness regressions | passed: `python3 tests/test_bench_canonical_freshness.py` |
| static package/support deferral guard | passed: `bash scripts/static_package_deferral_check.sh` |
| Python compile checks | passed for changed Python tooling and tests |
| documentation whitespace hygiene | passed: `git diff --check` |
| C source/header quality gate | not required: no `*.c` or `*.h` files changed |

### Changed Surface

| Metric | Sprint 181 close state |
| --- | ---: |
| selected target manifests added | 1 |
| selected manifest rows | 6 |
| selected oracle rows | 1 |
| selected comparison rows | 4 |
| selected benchmark rows | 1 |
| Python scripts changed | 3 |
| Python tests added | 1 |
| Python tests changed | 3 |
| public/maintainer/report docs changed | 5 |
| daily artifacts | 14 |
| closeout artifacts | 1 |
| retrospective files | 1 |
| project-plan items completed | 6 |
| C source files changed | 0 |
| public header files changed | 0 |

### Claim Governance

| Metric | Sprint 181 close state |
| --- | ---: |
| selected target authority files | 1 |
| selected Linux oracle workflow rows | 1 |
| selected Linux comparison workflow rows | 4 |
| selected macOS comparison workflow rows | 4 |
| selected Linux benchmark workflow rows | 1 |
| selected Windows report freshness rows | 0 |
| broad report-index freshness claims added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI claims added | 0 |
| portable performance claims added | 0 |
| external-library parity claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 181 closes this Epic 16 selected report target manifest claim:

Selected oracle, comparison, and benchmark report target metadata now has a
single source-controlled selected-target authority at
`tests/corpus/manifests/selected_report_targets.tsv`. Schema validation,
normalizer freshness checks, benchmark freshness checks, workflow guard tests,
and documentation now read or validate against that authority instead of
maintaining independent selected target lists.

This does not claim broad report-index freshness, selected oracle freshness on
macOS, Windows report freshness, unselected report-family freshness, package
manager support, shared-library ABI support, release proof, broad platform
parity, broad external-library parity, performance superiority, or
state-of-the-art status.

## Follow-Up Risks

1. **Windows report freshness needs a product decision.** Sprint 182 should
   either promote one Windows-safe selected report freshness path with explicit
   manifest workflow metadata or formally close Windows report freshness as a
   guarded deferral.

2. **Benchmark methodology may need typed manifest fields.** If future sprints
   need benchmark methodology in the selected-target authority, add typed
   workload, matrix-size, repeat, warmup, variance, baseline, threshold, and
   methodology-note fields instead of overloading prose fields.

3. **Oracle bucket counts may need manifest support.** If solver-family row
   counts continue to matter as selected target metadata, add explicit
   per-bucket schema rather than keeping them as normalizer compatibility
   constants.

4. **Manifest drift now carries more blast radius.** Selected target changes
   should run schema, manifest tests, normalizer tests, workflow guard tests,
   selected freshness targets, and docs checks together.

5. **Benchmark freshness checks should not run in parallel against the same
   report directory.** Keep `make bench-canonical-report-freshness` and
   `python3 tests/test_bench_canonical_freshness.py` sequential unless one of
   them gets an isolated output directory.

## Sprint 182 Readiness

Sprint 182 should begin from the Epic 16 project-plan section
`Sprint 182: Windows Report Freshness Decision`.

Sprint 181 leaves these inputs ready:

- selected target manifest exists and has no Windows workflow platform row;
- Windows workflow remains CMake-first and package/static install scoped;
- workflow guard tests reject selected report freshness commands and selected
  upload artifacts in the Windows workflow;
- docs preserve Windows report freshness as a non-claim;
- Linux selected oracle/comparison and benchmark lanes plus macOS selected
  comparison lanes remain manifest-owned.

The highest-value next action is to audit whether any selected report
freshness path is Windows-safe enough to promote. If not, Sprint 182 should
formalize Windows report freshness as a product deferral with guard coverage.
