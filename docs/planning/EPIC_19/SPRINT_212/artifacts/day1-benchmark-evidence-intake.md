# Sprint 212 Day 1: Benchmark Evidence Intake

## Summary

Day 1 establishes the Sprint 212 benchmark methodology evidence surface before
any threshold or threshold-free implementation changes. The current selected
hosted benchmark policy is threshold-free: it validates selected
`bench_refactor_csc` report freshness and methodology metadata for Linux and
macOS hosted lanes, but it does not compare timing values, set a portable
threshold, or claim performance superiority.

## Sprint Scope Mapping

| Item | Day 1 owner interpretation |
| --- | --- |
| 212.1 Benchmark Evidence Inventory | Inventory selected hosted benchmark lanes, report fields, runner metadata, freshness scripts, selected manifest rows, and non-claims. |
| 212.2 Threshold Decision | Preserve both decision paths until criteria and baseline evidence are complete: one selected threshold gate or stronger threshold-free deferral. |
| 212.3 Methodology Implementation | Defer implementation until the Day 5 policy decision. |
| 212.4 Regression Tests | Identify existing freshness, manifest, report-index, and docs guard owners that can be extended later. |
| 212.5 Documentation Calibration | Identify benchmark, top-level, install, maintainer, corpus, and schema docs that constrain performance wording. |
| 212.6 Validation And Closeout | Use documentation-only validation on Day 1; reserve benchmark/tooling validation for later changed surfaces. |

## Current Evidence Sources

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 212 is a 166-hour sprint to decide and implement one selected benchmark methodology policy while preserving no portable performance overclaim. |
| `tests/corpus/manifests/selected_report_targets.tsv` | The selected benchmark target is `SRT-BENCH-REFACTOR-CSC-NOS4`, covering `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1` for Linux/macOS hosted selected lanes. |
| `scripts/check_bench_canonical_freshness.py` | The checker validates selected row identity, artifact presence, manifest agreement, hosted metadata, and threshold-free methodology fields; it intentionally does not compare timing values. |
| `scripts/bench_canonical_report.sh` | The canonical report generator records platform, compiler, runner context, build flags, CPU model, build mode, thread count, support tier, claim boundary, baseline, threshold, warmup, variance, and methodology notes. |
| `scripts/performance_sentinels.sh` | Local sentinel reports already contain narrow thresholded behavior for S5/S6 and threshold-free backend context for S2/S3, but this is separate from hosted canonical selected freshness. |
| `Makefile` | `bench-canonical-report`, `bench-canonical-report-freshness`, `bench-canonical-report-freshness-tests`, and `performance-sentinels` are the central benchmark methodology targets. |
| `benchmarks/README.md` | Main benchmark methodology guide. It documents canonical report fields, selected hosted threshold-free freshness, local sentinel interpretation, and report-index handoff. |
| `README.md` | High-level route says hosted selected-performance lanes check freshness and methodology only, without timing thresholds or portable performance claims. |
| `INSTALL.md` | Support/readiness matrix classifies Linux/macOS selected performance freshness as hosted evidence with no portable performance, timing threshold, release benchmark, platform parity, package/ABI proof, broad package-manager distribution, or state-of-the-art claim. |
| `docs/maintainer_guide.md` | Maintainer owner for selected performance repair workflow, hosted lane metadata, threshold-free fields, and validation commands. |
| `tests/test_selected_performance_docs.py` | Existing docs guard for required selected performance markers and forbidden overclaims. |
| `tests/test_bench_canonical_freshness.py` | Existing regression suite for selected canonical benchmark freshness. |
| `.github/workflows/ci.yml` and `.github/workflows/macos-ci.yml` | Hosted selected-performance lanes upload the Linux and macOS reviewed selected performance freshness artifacts. |

## Selected Benchmark Boundary

| Boundary field | Current evidence |
| --- | --- |
| Selected target | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Artifact | `bench_refactor_csc` |
| Generated CSV | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Command | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Repeat semantics | `configured_repeat_1` |
| Warmup | `none_configured` |
| Variance | `not_computed_single_sample` |
| Baseline | `n/a` |
| Threshold | `n/a` |
| Local claim boundary | `local_threshold_free` |
| Hosted claim boundary | `hosted_selected_threshold_free` |
| Hosted platforms | Linux and macOS |
| Non-claims | No portable performance, release benchmark, algorithmic superiority, platform parity, state-of-the-art, package/ABI, package-manager, or Windows selected benchmark freshness claim. |

## Threshold Readiness Intake

The current selected hosted benchmark evidence is strong enough for freshness
and methodology validation, but Day 1 does not establish that it is strong
enough for a timing threshold:

- hosted rows identify platform/compiler/runner context, but GitHub-hosted CPU
  assignment is explicitly variable;
- selected canonical evidence uses `--repeat 1`;
- warmup is `none_configured`;
- variance is `not_computed_single_sample`;
- canonical selected `baseline` and `threshold` are both `n/a`;
- public and maintainer docs currently state that hosted selected performance
  freshness is not a timing gate or portable performance claim.

That means the thresholded path remains possible only if Days 2-5 produce a
bounded smoke-ceiling or threshold methodology that is explicitly narrower than
portable performance. Otherwise Sprint 212 should strengthen threshold-free
deferral.

## Initial Test And Guard Owners

| Guard owner | Likely Sprint 212 use |
| --- | --- |
| `tests/test_bench_canonical_freshness.py` | Positive/negative fixtures for missing methodology fields, stale baseline/threshold values, unsupported threshold metadata, or hosted/local mode mismatches. |
| `tests/test_selected_performance_docs.py` | Required wording and forbidden overclaim coverage after the final policy decision. |
| `tests/test_selected_report_targets_manifest.py` | Exact selected target manifest fields, non-claims, workflow platforms, and artifact metadata. |
| `tests/test_normalize_report_index.py` | Cross-report preservation of benchmark/sentinel row meaning if normalized-index interpretation changes. |
| `scripts/check_bench_canonical_freshness.py` | Fail-closed policy enforcement for selected benchmark methodology. |

## Day 1 Outcome

Day 1 closes the intake setup. The sprint now has:

- a working-notes ledger for item status, evidence, risks, validation, and
  changed surfaces;
- an initial selected benchmark evidence inventory;
- a current selected benchmark boundary table;
- a threshold-readiness observation set that keeps the Day 5 decision honest;
- a validation plan for later methodology and documentation changes.

Day 2 should capture current benchmark baseline behavior and concrete metadata
gaps before any policy decision.

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

Day 1 validation command:

```sh
git diff --check
```
