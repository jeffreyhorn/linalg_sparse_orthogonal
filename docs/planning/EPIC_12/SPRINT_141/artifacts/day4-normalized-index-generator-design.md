# Day 4 Normalized Index Generator Design

## Purpose

Day 4 designs the normalized report index generator before implementation.
The generator will produce one deterministic TSV that indexes source-controlled
report metadata and any generated report bundles that already exist locally,
without requiring platform-specific benchmark, coverage, package, or CI report
generation.

This is a design artifact. It does not add the generator script, generated
index output, schemas, Make targets, tests, or public documentation.

## Ownership And Integration

| Decision | Design |
| --- | --- |
| Script owner | `scripts/normalize_report_index.py` |
| Primary output | `build/report-index/normalized-index.tsv` |
| Optional manifest | `build/report-index/manifest.txt` after implementation proves useful |
| Initial integration | Standalone Python CLI invoked directly by maintainers and tests |
| Later Make target | Candidate `make report-index` after Day 6 implementation and Day 13 validation |
| Input model | Read source-controlled metadata always; read generated report files only when present or requested |
| Claim posture | Index rows are navigation, freshness, and interpretation aids; they are not new proof by themselves |

The standalone-first choice keeps the first implementation testable without
adding build-system coupling before the data model is proven.

## CLI Shape

Proposed command:

```sh
python3 scripts/normalize_report_index.py \
  --corpus-root tests/corpus \
  --build-root build \
  --output build/report-index/normalized-index.tsv
```

Planned options:

| Option | Meaning |
| --- | --- |
| `--corpus-root <path>` | Override `tests/corpus` for tests and fixtures. |
| `--build-root <path>` | Override `build` so tests can use temporary generated-report fixtures. |
| `--output <path>` | Write normalized TSV to the requested path. |
| `--family <name>` | Restrict output to one report family; repeatable. |
| `--include-generated` | Include generated report rows when files exist. This should be on by default for local use. |
| `--no-generated` | Emit source-controlled and `not_generated` rows only. Useful for deterministic tests. |
| `--require-generated <family>` | Treat missing generated rows for a family as errors instead of advisory rows. Repeatable. |
| `--check` | Validate and print diagnostics without rewriting output. |
| `--format=tsv` | Initial output format. JSON remains deferred unless a downstream owner needs it. |

## Input Discovery Rules

| Family | Discovery | If present | If absent |
| --- | --- | --- | --- |
| Corpus fixtures | `tests/corpus/manifests/fixtures.tsv` | Emit `fixture_metadata` source-controlled rows. | Error; corpus root is malformed. |
| Corpus generators | `tests/corpus/manifests/generators.tsv` | Emit `generator_metadata` source-controlled rows. | Error; corpus root is malformed. |
| Optional data | `tests/corpus/manifests/optional_data.tsv` | Emit `optional_data_policy` source-controlled rows. | Error; corpus root is malformed. |
| Expected results | `tests/corpus/expected/*.tsv` | Emit `expected_result` source-controlled rows. | Warning if no expected files are found. |
| Corpus/oracle generated reports | `build/corpus/oracle/*.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | Emit `observed_oracle_comparison`, `solver_backed_fixture_proof`, and generated skip/defer rows. | Emit advisory `not_generated` rows unless `--require-generated oracle` is set. |
| Canonical benchmarks | `build/bench-reports/canonical/index.tsv`, `manifest.txt` | Emit `benchmark_measurement` rows. | Emit advisory `not_generated` row. |
| Performance sentinels | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt` | Emit `sentinel_hard_gate` and `sentinel_advisory_measurement` rows. | Emit advisory `not_generated` row. |
| Large-matrix guardrails | `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt` | Emit reviewed and supplemental guardrail lane rows. | Emit advisory `not_generated` row. |
| Dead-code reports | `build/deadcode/report.tsv`, `build/deadcode/report.md` | Emit `deadcode_classification` rows or a report-presence summary row. | Emit advisory `not_generated` row unless required by a selected gate. |
| Coverage reports | `coverage/coverage-src.info`, `coverage/html/index.html` | Emit `coverage_summary` advisory rows with backend/tool context when inferable. | Emit advisory `not_generated` row. |
| Package/install proof surfaces | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, package templates | Emit source-controlled proof-owner rows. | Error only if proof-owner script or template is missing. |
| CI workflows | `.github/workflows/*.yml` | Emit `ci_lane_definition` source-controlled rows for known jobs. | Warning if workflow directory is absent. |
| Documentation advisories | README, maintainer, cookbook, benchmark, and install docs | Emit source-controlled advisory rows only for known report interpretation anchors. | Warning if expected docs are missing. |

## Output Schema Draft

The generator should initially emit the Day 3 common fields in this order:

```text
row_id
report_family
subfamily
native_row_id
row_origin
row_meaning
status
status_reason
support_tier
claim_scope
non_claims
generator_command
source_commit
source_branch
generated_at_utc
platform
compiler
configuration
artifact_path
freshness_status
freshness_reason
skip_or_defer_reason
```

Family-specific detail should be encoded inside stable `configuration` text
or retained in the native artifact, not added as many sparse nullable columns
on Day 4. Candidate future detail columns are deferred until an implementation
test proves they are needed.

## Row ID Rules

Normalized row IDs should be deterministic and readable:

| Family | Row ID pattern |
| --- | --- |
| Corpus fixture | `corpus_fixture_<fixture_key>_v1` |
| Corpus generator | `corpus_generator_<generator_key>_v1` |
| Optional data | `corpus_optional_<optional_data_key>_v1` |
| Expected row | `corpus_expected_<oracle_row_id>_v1` |
| Oracle row | existing `report_row_id` when available; otherwise `oracle_<oracle_row_id>_v1` |
| Benchmark artifact | `benchmark_canonical_<artifact>_v1` |
| Sentinel metric | `sentinel_<sentinel_id>_<metric>_v1` |
| Guardrail lane | `guardrail_<lane_id>_v1` |
| Dead-code report | `deadcode_report_<native_id>_v1` |
| Coverage summary | `coverage_<backend>_<native_id>_v1` |
| Package/install proof | `package_<proof_name>_v1` or `install_<proof_name>_v1` |
| CI lane | `ci_<workflow_name>_<job_name>_v1` |
| Documentation advisory | `documentation_<doc_name>_<anchor>_v1` |

IDs must be normalized to lowercase snake case, with non-alphanumeric runs
collapsed to `_`.

## Deterministic Ordering

Rows should be sorted by:

1. `report_family`;
2. `subfamily`;
3. `row_origin`;
4. `row_meaning`;
5. `native_row_id`;
6. `artifact_path`;
7. `row_id`.

Within a native TSV input, duplicate normalized IDs are errors. Across
families, duplicate `native_row_id` values are allowed only because
`report_family` and `subfamily` disambiguate them.

## Generated, Missing, Skip, And Stale Behavior

| Situation | Default row behavior |
| --- | --- |
| Generated report exists and parses | Emit generated row with source, command, platform, compiler, configuration, and native status. |
| Generated report missing | Emit one family-level row with `status=unknown`, `freshness_status=not_generated`, and advisory support tier. |
| `--require-generated <family>` and report missing | Emit row plus nonzero check result in `--check` mode. |
| Optional data unavailable | Preserve `status=skip` with optional-data reason and non-claim boundary. |
| Deferred family | Emit `status=defer`, `freshness_status=deferred`, and owner/reason. |
| Unsupported platform or package lane | Emit `status=unsupported` when the source metadata makes the unsupported scope explicit. |
| Stale generated row | Day 4 design records the field shape; Day 10/11 own final stale comparison rules. Initial generator should preserve enough fields for the gate. |

The generator must not synthesize pass evidence for missing generated reports.

## Test Strategy

### Unit Tests

Use temporary directories and tiny TSV fixtures.

| Test | Expected behavior |
| --- | --- |
| Minimal corpus fixture/generator/expected rows | Emits stable source-controlled normalized rows. |
| Duplicate normalized row ID | Fails validation. |
| Missing generated oracle rows | Emits deterministic `not_generated` row by default. |
| Required generated family missing | Fails in `--check` mode when `--require-generated` names the family. |
| Optional-data unavailable row | Emits `skip` with non-claim boundary. |
| Deferred row | Emits `defer` with `freshness_status=deferred`. |
| Generated corpus report row | Preserves `oracle_row_id`, support tier, claim scope, non-claims, and freshness fields. |
| Benchmark/sentinel/guardrail fixture rows | Preserves local/advisory versus reviewed/thresholded distinctions. |

### Smoke Tests

Run the generator on the current repository with `--no-generated` so no
benchmark, coverage, package, or corpus reports need to be generated.

Expected smoke behavior:

- emits corpus fixture, generator, optional-data, and expected-result rows;
- emits package/install proof-owner rows;
- emits CI lane definition rows;
- emits documentation advisory rows;
- emits `not_generated` rows for generated corpus, benchmark, sentinel,
  guardrail, dead-code, and coverage report families;
- exits successfully because missing generated local reports are advisory by
  default.

### Golden Output

Use a small fixture directory under a future test-data path rather than the
full repository state. The golden output should include:

- one source-controlled fixture row;
- one expected-result row;
- one generated oracle row;
- one benchmark measurement row;
- one sentinel hard-gate row;
- one sentinel advisory row;
- one guardrail reviewed row;
- one optional-data skip row;
- one deferred runtime/backend row.

The golden should pin field order, deterministic sorting, status semantics,
and native ID preservation.

## Implementation Checklist

1. Add `scripts/normalize_report_index.py` with TSV reader/writer helpers.
2. Define constants for normalized fields, report families, row meanings,
   statuses, support tiers, and freshness statuses.
3. Implement path discovery using `Path` APIs, not ad hoc shell parsing.
4. Implement source-controlled corpus row emitters first.
5. Implement generated corpus/oracle report ingestion using existing
   `scripts/run_corpus_oracle.py` field names.
6. Implement benchmark, sentinel, and guardrail generated report ingestion with
   row-meaning preservation.
7. Implement advisory `not_generated` rows for generated families.
8. Implement package/install proof-owner and CI lane definition rows.
9. Add tests using temporary fixture roots and golden TSV output.
10. Run `python3 -m py_compile scripts/normalize_report_index.py` and focused
    generator tests.

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The generator design maps directly to the metadata contract. | Complete | Output schema uses the Day 3 common fields and row-meaning taxonomy. |
| Optional and generated-only report handling is deterministic. | Complete | Missing generated rows, optional-data skips, defer rows, and required-generated behavior are specified. |
| Tests can verify behavior without requiring platform-specific measurements. | Complete | Smoke and golden-output plans use source-controlled rows and small temp fixtures rather than requiring benchmark, coverage, package, or hosted CI generation. |
