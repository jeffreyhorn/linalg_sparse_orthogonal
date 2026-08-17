# Sprint 163 Day 10 Public Documentation Alignment

## Purpose

Day 10 aligns top-level, maintainer, and report-index documentation with the
completed Sprint 163 methodology-bound report behavior. Day 9 handled the
benchmark README; Day 10 updates the public summary, authoritative maintainer
policy, and report-index schema notes.

## Updated Documentation

- `README.md`
- `docs/maintainer_guide.md`
- `tests/corpus/schemas/report_index_fields.md`

## README Alignment

The README now states that:

- `make bench-canonical-report` emits methodology context in generated
  `index.tsv` / `manifest.txt`;
- canonical rows are `status=measurement`, `support_tier=local_only`, and
  `claim_boundary=local_threshold_free`;
- `make performance-sentinels` keeps hard pass/fail behavior limited to S5
  wall-check rows;
- S5 rows carry baseline provenance;
- S2/S3 rows carry backend-context caveats rather than pass/fail meaning;
- generated benchmark, sentinel, and normalized report-index artifacts stay
  under ignored `build/` paths.

## Maintainer Guide Alignment

The maintainer guide now documents:

- appended canonical methodology fields;
- canonical row status, support tier, claim boundary, baseline, threshold,
  warmup, and variance interpretation;
- appended sentinel methodology fields;
- S5 baseline provenance and local-gate interpretation;
- S2/S3 `status=report` backend-context interpretation;
- normalized report-index preservation expectations for canonical and sentinel
  methodology fields;
- local-only generated-artifact policy;
- package/performance separation.

## Report-Index Schema Alignment

`tests/corpus/schemas/report_index_fields.md` now records that:

- Sprint 163 benchmark rows may expose methodology fields through generated
  report indexes or normalized `configuration` text;
- those fields preserve local context and do not create pass/fail benchmark
  proof;
- Sprint 163 sentinel rows may expose baseline provenance, repeat semantics,
  warmup, variance, and methodology notes;
- S5 remains the local wall-check hard gate;
- S2 and S3 remain threshold-free backend-context rows.

## Package And Performance Separation

The Day 10 docs keep these evidence families separate:

- package/install/static-first proof;
- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- performance report evidence.

No Sprint 162 package proof is reused as Sprint 163 performance proof.

## Unsupported-Claim Scan

Day 10 validation scans should verify that sensitive wording remains bounded:

```sh
rg -n "state-of-the-art|superiority|package-manager|shared-library|dynamic ABI|runtime-loader|broad platform|portable performance|OpenMP speedup|backend superiority" README.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/schemas/report_index_fields.md
```

Acceptable hits must be non-claims or boundaries, not positive claims.

## Completion Check

- Public docs match selected performance evidence.
- Package and performance evidence remain separate.
- Non-superiority boundaries are explicit.
