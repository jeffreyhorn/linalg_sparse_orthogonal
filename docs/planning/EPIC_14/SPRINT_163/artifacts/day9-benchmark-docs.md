# Sprint 163 Day 9 Benchmark Documentation Alignment

## Purpose

Day 9 aligns `benchmarks/README.md` with the selected Sprint 163
methodology-bound report behavior. The update documents the new canonical and
sentinel methodology fields, row-state interpretation, regeneration boundary,
and non-superiority caveats for benchmark/report users.

## Updated Documentation

- `benchmarks/README.md`

## Documentation Changes

### Canonical Report Fields

The `make bench-canonical-report` section now documents the appended
`index.tsv` methodology fields:

- `report_family`
- `status`
- `support_tier`
- `claim_boundary`
- `fixture_or_workload`
- `matrix_size`
- `repeat_semantics`
- `warmup`
- `variance`
- `baseline`
- `threshold`
- `backend_context`
- `methodology_notes`

The docs now state that:

- canonical rows use `status=measurement`;
- canonical rows use `support_tier=local_only`;
- canonical rows use `claim_boundary=local_threshold_free`;
- canonical `baseline=n/a` and `threshold=n/a` mean the rows are not hard
  timing gates;
- `warmup=not_recorded` and `variance=not_recorded` must be read literally.

### Sentinel Report Fields

The `make performance-sentinels` section now documents the appended
`sentinels.tsv` methodology fields:

- `baseline_provenance`
- `repeat_semantics`
- `warmup`
- `variance`
- `methodology_notes`

The docs now state that the sentinel manifest includes:

- S5 baseline provenance;
- S5/S2/S3 repeat semantics;
- warmup and variance state;
- non-superiority caveats.

### Gate Versus Report Wording

The benchmark docs now distinguish:

- S5 as the existing thresholded local `wall-check` gate whose `pass` or
  `fail` status is meaningful only with baseline, threshold, fixture, command,
  baseline provenance, and local machine context;
- S2 and S3 as threshold-free `status=report` backend-context rows that do not
  pass, fail, or prove backend superiority;
- canonical rows as threshold-free `status=measurement` rows rather than
  timing gates.

### Report-Index Handoff

The report-index handoff now says downstream summaries should preserve:

- canonical `support_tier`, `claim_boundary`, `repeat_semantics`, `warmup`,
  `variance`, `baseline`, `threshold`, and `methodology_notes`;
- sentinel `baseline_provenance`, `repeat_semantics`, `warmup`, `variance`,
  and `methodology_notes`;
- backend `n/a`, `unknown`, selected, and fallback fields.

## Regeneration Commands

The selected reports remain regenerated from maintained commands:

```sh
make bench-canonical-report
make performance-sentinels
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --output build/report-index/normalized-index.tsv
```

Generated benchmark, sentinel, and normalized-index outputs remain under
ignored `build/` paths and should not be hand-edited.

## Unsupported Claims Preserved

The documentation continues to block:

- portable performance guarantees;
- state-of-the-art claims;
- broad platform parity claims;
- OpenMP speedup claims;
- backend superiority claims;
- generated report rows as release proof;
- skipped rows as passing evidence.

Package, install, package-manager, shared-library, dynamic ABI, and
runtime-loader evidence remain outside Sprint 163 performance publication.

## Validation Notes

Documentation validation for Day 9 should include:

```sh
rg -n "baseline_provenance|repeat_semantics|methodology_notes|warmup=not_recorded|variance=not_recorded|status=measurement|status=report" benchmarks/README.md
rg -n "portable performance|state-of-the-art|backend superiority|OpenMP speedup" benchmarks/README.md
git diff --check
```

## Completion Check

- Users can reproduce selected reports from maintained commands.
- Benchmark docs explain methodology fields and caveats.
- Unsupported performance claims remain blocked.
