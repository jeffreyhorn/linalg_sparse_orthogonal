# Sprint 163 Day 7 Report Implementation II

## Purpose

Day 7 completes the selected report enhancement patch set by adding sentinel
methodology fields, preserving S5/S2/S3 row meanings, and keeping the normalized
report index compatible with the Day 6 and Day 7 schema additions.

## Changed Files

- `scripts/performance_sentinels.sh`
- `scripts/normalize_report_index.py`
- `docs/planning/EPIC_14/SPRINT_163/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_163/artifacts/day7-report-implementation-2.md`

Day 6's canonical script changes remain part of the overall report enhancement
patch set:

- `scripts/bench_canonical_report.sh`

## Sentinel Schema Additions

`scripts/performance_sentinels.sh` now appends these fields to
`build/bench-reports/sentinels/sentinels.tsv`:

- `baseline_provenance`
- `repeat_semantics`
- `warmup`
- `variance`
- `methodology_notes`

The existing leading sentinel columns remain unchanged:

- `report_family`
- `sentinel_id`
- `status`
- `support_tier`
- `claim_boundary`
- `command`
- `build_mode`
- `omp_num_threads`
- `matrix_or_fixture`
- `metric`
- `value`
- `baseline`
- `threshold`
- `artifact`
- `backend_request`
- `backend_selected`
- `backend_fallback`
- `dense_kernel`
- `panel_solver`
- `notes`

## Row Meaning Preserved

| Row | Added Methodology | Meaning |
| --- | --- | --- |
| S5 wall-check rows | `baseline_provenance=docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`, `repeat_semantics=wall_check_configured_single_runs`, `warmup=not_recorded`, `variance=not_recorded`, `methodology_notes=thresholded_local_wall_gate;not_portable_performance_claim` | Existing hard local wall-check gate; may pass or fail and retains nonzero failure behavior. |
| S2 Cholesky CSC rows | `baseline_provenance=n/a`, `repeat_semantics=configured_repeat_1`, `warmup=not_recorded`, `variance=not_recorded`, `methodology_notes=threshold_free_local_backend_context;not_backend_superiority_claim` | Threshold-free local backend-context report, not pass/fail evidence. |
| S3 LDLT KKT rows | `baseline_provenance=n/a`, `repeat_semantics=configured_repeat_1`, `warmup=not_recorded`, `variance=not_recorded`, `methodology_notes=threshold_free_local_ldlt_backend_context;not_backend_superiority_claim` | Threshold-free local LDLT backend-context report, not pass/fail evidence. |

## Manifest Caveats

The sentinel manifest now records:

- S5 baseline provenance;
- S5/S2/S3 repeat semantics;
- warmup and variance states;
- S5 local wall-gate caveat;
- S2/S3 threshold-free backend-context caveat;
- non-superiority, non-portability, package, ABI, runtime-loader,
  external-library, OpenMP speedup, and backend-superiority non-claims.

## Normalizer Preservation

`scripts/normalize_report_index.py` now preserves the new generated
methodology fields in normalized `configuration` text:

- benchmark rows include row family, row status, support tier, claim boundary,
  fixture/workload, matrix size, repeat semantics, warmup, variance, baseline,
  threshold, backend context, and methodology notes;
- sentinel rows include baseline provenance, repeat semantics, warmup,
  variance, and methodology notes.

This does not change normalized field names or row-family meanings. Benchmark
rows remain advisory local measurements, S5 remains a sentinel hard gate, and
S2/S3 remain sentinel advisory measurements.

## Focused Validation

The following checks passed:

```sh
bash -n scripts/performance_sentinels.sh
make performance-sentinels
python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv
python3 tests/test_normalize_report_index.py
```

Observed generated sentinel behavior:

- `build/bench-reports/sentinels/sentinels.tsv` includes appended methodology
  fields.
- S5 rows include the wall-check baseline provenance path and thresholded local
  wall-gate methodology notes.
- S2 rows include `baseline_provenance=n/a`,
  `repeat_semantics=configured_repeat_1`, `warmup=not_recorded`,
  `variance=not_recorded`, and threshold-free backend-context methodology
  notes.
- `build/bench-reports/sentinels/manifest.txt` includes the strengthened
  non-superiority caveats.
- The combined normalized benchmark/sentinel index writes successfully with
  the new methodology fields visible in `configuration`.

## Remaining Follow-Through

- Day 8 should convert the selected report behavior into a gate-classification
  and publication policy.
- Day 9 and Day 10 should align maintainer and public documentation with the
  completed script behavior.
- Day 11 and Day 12 should re-run selected benchmark/sentinel validation after
  documentation and policy updates.

## Completion Check

- Selected report behavior is complete locally for canonical and sentinel
  bundles.
- Diagnostics are reviewable through explicit row fields and manifest caveats.
- Unsupported performance claims remain explicit non-claims.
