# Sprint 169 Day 2: Current Report Methodology Audit

## Purpose

Day 2 audits the current selected performance report methodology surface
before changing policy or schema. The audit maps generated fields to owners,
records local output shape, reviews hosted CI metadata, and identifies weak or
underdefined fields for Days 3 and 4.

## Audited Sources

| Source | Role |
| --- | --- |
| `scripts/bench_canonical_report.sh` | Generates the canonical benchmark CSV bundle, `index.tsv`, and `manifest.txt`. |
| `scripts/check_bench_canonical_freshness.py` | Validates selected-row freshness, selected-row methodology metadata, manifest agreement, and unselected-row claim boundaries. |
| `Makefile` | Owns `bench-canonical-report` and `bench-canonical-report-freshness` target wiring. |
| `.github/workflows/ci.yml` | Owns hosted selected-performance metadata, hosted checker invocation, summary output, and artifact upload. |
| `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md` | Own user-facing and maintainer-facing interpretation of selected performance evidence. |

## Current Report Schema Inventory

The canonical index currently emits 29 tab-separated fields:

| Field group | Fields | Owner |
| --- | --- | --- |
| report identity | `surface`, `category`, `report_label`, `generated_at_utc` | generator; checker validates selected row |
| source identity | `git_commit`, `git_branch` | generator; checker validates manifest agreement |
| platform identity | `platform`, `compiler`, `runner_context`, `build_flags`, `cpu_model` | generator and hosted CI env |
| build/thread context | `build_mode`, `omp_num_threads` | generator with optional env override |
| artifact identity | `artifact`, `relative_path`, `command`, `report_family`, `status` | generator; checker validates selected row |
| claim boundary | `support_tier`, `claim_boundary` | generator row logic; checker enforces selected and unselected rows |
| workload semantics | `fixture_or_workload`, `matrix_size`, `repeat_semantics` | generator; checker validates selected fixture/repeat |
| methodology state | `warmup`, `variance`, `baseline`, `threshold`, `backend_context`, `methodology_notes` | generator; checker validates non-empty selected metadata and threshold-free selected values |

## Local Output Shape

Day 2 regenerated the canonical report through:

```sh
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
```

The target passed and produced:

| Metric | Observed value |
| --- | --- |
| `index.tsv` columns | 29 |
| `index.tsv` data rows | 4 |
| selected artifact row | `bench_refactor_csc` |
| selected relative path | `bench_refactor_csc.csv` |
| selected local support tier | `local_only` |
| selected local claim boundary | `local_threshold_free` |
| selected repeat semantics | `configured_repeat_1` |
| unselected row support tier | `local_only` |
| unselected row claim boundary | `local_threshold_free` |

Observed local row boundaries:

| Artifact | Support tier | Claim boundary | Matrix size | Warmup | Variance | Repeat semantics |
| --- | --- | --- | --- | --- | --- | --- |
| `bench_refactor_csc` | `local_only` | `local_threshold_free` | `not_recorded` | `not_recorded` | `not_recorded` | `configured_repeat_1` |
| `bench_chol_csc` | `local_only` | `local_threshold_free` | `not_recorded` | `not_recorded` | `not_recorded` | `configured_repeat_1` |
| `bench_iterative_reuse` | `local_only` | `local_threshold_free` | `not_recorded` | `not_recorded` | `not_recorded` | `benchmark_default` |
| `bench_eigs_reuse` | `local_only` | `local_threshold_free` | `not_recorded` | `not_recorded` | `not_recorded` | `benchmark_default` |

Generated output remains under ignored `build/` paths and should not be
staged as source.

## Freshness Check Inventory

`scripts/check_bench_canonical_freshness.py` currently validates:

- required artifact presence for `bench_refactor_csc.csv`, `index.tsv`, and
  `manifest.txt`;
- non-empty selected-row metadata for required fields;
- `index.tsv` row width and required-column stability;
- exactly one selected `artifact=bench_refactor_csc` row;
- selected command, relative path, fixture, repeat semantics, report family,
  status, baseline, and threshold;
- UTC timestamp shape;
- `methodology_notes` includes `not_portable_performance_claim`;
- local mode selected-row support tier and claim boundary stay within
  threshold-free allowed values;
- hosted mode selected-row support tier is `hosted_selected`, claim boundary
  is `hosted_selected_threshold_free`, runner context is not `local`, build
  flags are recorded, and report label is not `unlabeled`;
- unselected rows remain `local_only` / `local_threshold_free`;
- selected-row values agree with `manifest.txt` for report label, commit,
  branch, platform, compiler, runner context, build flags, CPU model, build
  mode, thread state, support tier, claim boundary, baseline, threshold, and
  methodology notes.

## Hosted CI Metadata Inventory

The hosted lane is:

```text
Linux reviewed hosted selected performance freshness
```

It supplies:

| Metadata | Hosted value |
| --- | --- |
| `BENCH_CANONICAL_REPORT_LABEL` | `sprint-168-hosted-performance` |
| `SPARSE_CANONICAL_SUPPORT_TIER` | `hosted_selected` |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | `hosted_selected_threshold_free` |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | `github-actions-ubuntu-latest` |
| `SPARSE_CANONICAL_BUILD_FLAGS` | `default_make_flags` |
| `SPARSE_CANONICAL_BUILD_MODE` | `serial` |
| `SPARSE_CANONICAL_CPU_MODEL` | first `/proc/cpuinfo` model name, or `unknown` |

The job runs `make bench-canonical-report`, checks the report in hosted mode,
prints `sprint168-performance-summary` lines, and uploads the six-file
canonical report bundle under `sprint168-selected-performance-freshness`.

## Field Strength Assessment

| Field | Current strength | Gap or risk |
| --- | --- | --- |
| selected artifact identity | strong | Exact artifact, path, command, fixture, and repeat semantics are checked. |
| selected support/claim boundary | strong | Hosted-selected values are selected-row scoped; unselected rows are guarded. |
| generated timestamp | moderate | UTC shape is checked, but freshness age is not bounded. |
| compiler/platform strings | moderate | Values are captured but not normalized across hosts. |
| runner context | moderate | Hosted mode rejects `local`, but runner image version is not recorded. |
| build flags | moderate | Hosted mode rejects `not_recorded`, but `default_make_flags` is a label rather than expanded flags. |
| CPU model | moderate | Captured when available; `unknown` is intentionally accepted. |
| build mode | moderate | Can be overridden or inferred from OpenMP runtime linkage. |
| repeat semantics | weak | Selected row records `configured_repeat_1`; no sample-count or repeated-run policy exists. |
| warmup | weak | Always `not_recorded`; no policy says whether this is acceptable long term. |
| variance | weak | Always `not_recorded`; no variance or confidence policy exists. |
| matrix size | weak | Always `not_recorded`; selected fixture dimensions are not exposed in metadata. |
| baseline/threshold | intentional | `n/a` preserves threshold-free publication, but any future sentinel needs separate ownership. |
| manifest agreement | moderate | Selected row is checked, but unselected row manifest detail is minimal. |
| artifact discoverability | moderate | CI uploads artifacts and docs describe them, but no checked-in generated report index exists. |

## Missing Or Underdefined Methodology Fields

Day 2 identifies these as candidates for Days 3 and 4:

1. Repeat and sample policy: whether selected hosted performance remains
   `configured_repeat_1` or records a larger configured repeat/sample count.
2. Warmup policy: whether `not_recorded` remains an explicit non-measured
   state or becomes a concrete warmup policy.
3. Variance policy: whether variance remains unavailable or is computed from
   repeated samples.
4. Matrix-size policy: whether `nos4.mtx` dimensions should be derived and
   recorded for the selected row.
5. Build-flags policy: whether hosted `default_make_flags` is sufficient or
   should expand to captured `CC` / `CFLAGS` / `LDFLAGS` context.
6. Freshness-age policy: whether generated timestamps need a maximum age check
   in hosted mode.
7. Sentinel boundary: whether a regression sentinel should reuse existing
   wall-check/performance-sentinel infrastructure or remain deferred.
8. Report-index policy: whether selected performance evidence should stay as a
   focused checker or also integrate with normalized report-index output.

## Day 2 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected report fields are mapped to their owners. | Complete | Current schema inventory maps field groups to generator, checker, Makefile, CI, and docs. |
| Missing methodology semantics are explicit. | Complete | Repeat, warmup, variance, matrix-size, build-flags, freshness-age, sentinel, and report-index gaps are listed. |
| Unselected canonical rows remain local/advisory. | Complete | Local output audit and checker inventory confirm unselected rows are `local_only` / `local_threshold_free`. |

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation run:

```sh
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
```

Run after writing this artifact:

```sh
git diff --check
```
