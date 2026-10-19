# Sprint 215 Day 7: Schema Validation And Compatibility Design

## Decision And Test Fixture

Keep the Day 6 separate external-evidence schema. No external package should
be needed for schema validation. The companion
`day7-external-evidence-blueprint.example.json` is a complete, parseable
**planning-only** v1 record: it fixes the selected `symmetric_4.mtx` hash,
RHS, solver APIs and versioned baseline, while leaving unverified binary,
host, build and observed outputs null with reasons. Its `planned` status,
`record_kind=blueprint`, `evidence_origin=planning`, null `cell_id`, two
`not_run` correctness sides, `not_run` timing and empty artifacts must never
be projected as generated pass evidence.

This day designs the validator and fixtures; it does **not** create the
proposed `tests/corpus/schemas/external_baseline_evidence_v1.json` authority
or `scripts/validate_external_baseline_evidence.py`. Implement and accept
those only after the Day 8 schema decision. The example is deliberately not
an observed CHOLMOD result or a timing sample.

## Proposed Validator Interface And Order

The proposed stdlib-only command is:

```sh
python3 scripts/validate_external_baseline_evidence.py \
  --schema tests/corpus/schemas/external_baseline_evidence_v1.json \
  --input build/external-baseline/<selection_id>/<run_id>/evidence.json \
  --bundle-root build/external-baseline/<selection_id>/<run_id>
```

For a planning fixture, `--input` may point at the companion JSON and no
bundle root is needed because `artifacts=[]`. These commands describe the
future interface; **the validator does not exist yet**. The schema file
should hold v1 required fields, allowed enums, fixed selection controls and
claim tokens, while Python handles cross-field rules, byte hashes, derived
cell IDs and raw-sample checks. Do not make a third-party JSON Schema package
a dependency of `make test` or install checks.

Validate in this fail-closed sequence:

1. Parse UTF-8 JSON with duplicate-key rejection and nonfinite-number rejection (`NaN`, `Infinity`, `-Infinity` are invalid even when a permissive decoder accepts them). Reject unknown fields for v1 rather than silently ignoring a misspelled provenance field. Report JSON pointer, rule code and offending record ID.
2. Check required fields and exact types. Booleans must not pass integer checks. Reject empty strings, float-valued counts, negative dimensions/durations, unsupported enums and malformed UTC timestamps. SHA-256 values are 64 lowercase hex characters; Git commits are complete 40-character hex IDs for generated runs.
3. Match the `selection_id` against immutable v1 selection constraints: matrix path and bytes hash, `real symmetric`/4 x 4/seven stored/ten expanded, `spd_linear_solve`, exact RHS/oracle, float64, project no-reorder linked-list path, SuiteSparse `v7.12.2` commit `42151688813c45846a597edcb601435a0e38f3dd`, and CHOLMOD natural/simplicial `CHOLMOD_A`. A changed value requires a new selection decision/ID, not a permissive row under the old ID.
4. For a local generated record, verify actual fixture bytes, all methodology file hashes, project/baseline binaries and dependency provenance, environment and thread observations. For blueprint, permit null execution fields only with the named reasons in the example. A discovered Homebrew dylib hash is a *candidate*, not a verified runtime-loaded binary hash.
5. Reconstruct `cell_id` only when the full Day 3 tuple is present. Serialize an exact identity object (matrix SHA, problem/RHS, precision and both index ABIs, project commit/binary/build/solver/backend/ordering, baseline source commit/loaded binary/build/dependency hashes/solver/backend/ordering, requested and observed threads, host OS/architecture/CPU) as UTF-8 JSON with sorted keys, no insignificant spaces, ASCII escaping and integer/string/bool scalars only; use SHA-256 and `sha256:<64hex>`. Sort dependency entries by role and absolute loaded path before serialization; reject duplicate roles/paths. A different complete tuple must have a different ID. `cell_id=null` needs `cell_id_reason` and is never acceptable for `correctness_observed` or `measured`.
6. Apply the status matrix below, then verify every non-null metric against the Day 4 formulas and every timing aggregate against Day 5 raw samples. Follow artifact references only within the declared repo/bundle root; reject absolute paths, `..`, symlink escapes, missing files or SHA mismatch. Do not infer validity from a CSV/TSV row label, success process exit or a report summary.

The identity serialization is a *v1 proposal*; Day 8 must freeze the exact
JSON field list and canonical byte example before assigning production
`cell_id`s. Raw numerical values are not part of `cell_id` but must be
attached to the run and checked against its matrix/RHS. Grouping or comparing
runs additionally requires the same methodology hashes and Day 5 phase
definition, not merely an equal cell hash.

## Required Fields And Units

| Group | Blueprint requirement | Additional generated-run requirement |
| --- | --- | --- |
| Record identity | `schema_version=external_baseline_evidence_v1`; unique `record_id`; selected `selection_id`; `record_kind`, `evidence_origin`, `evidence_status`, nonempty reason and publication review. | Nonempty unique `run_id`; immutable attempt path; origin `generated_local` or `generated_ci`; full `cell_id` before an observed correctness or measured claim. |
| Matrix and operation | Exact path/hash, shape, storage/expansion, provenance review, class, `A*x=b`, float64, one RHS, exact RHS/oracle, both requested solver paths. | Rehash loaded matrix; verify expanded coefficients and RHS on both sides. Actual index ABIs, effective backend/ordering and any source/fixture ownership evidence must be recorded. |
| Package and build | SuiteSparse/CHOLMOD name, tag/version, 40-character source commit and acquisition route; project API/backend intent. | For observed results: project commit/dirty patch hash, compiler/flags/build mode/binary digest; actual CHOLMOD loaded library path/digest, package revision, module options, dependency and BLAS/OpenMP identity. A `not_run` or execution-blocked attempt may leave unavailable runtime fields null with reasons. No `unknown` disguised as a complete provenance value. |
| Methodology and platform | Day 3/4/5 paths, planned rules, `correctness_units` and timing unit. | For observed results: exact SHA-256 of all methodology files; OS/build, architecture/CPU, compiler/runtime, requested and observed threads, command/cwd/time, clock/resolution, host fingerprint; hosted records also need workflow/run/artifact IDs. A `not_run` or execution-blocked attempt may leave unobserved fields null with reasons. |
| Correctness | `project` and `baseline` state in `not_run|blocked|fail|pass`; null unobserved metrics with availability reason. | Per-phase statuses, factor diagnostics, finite raw `x`, residual/backward error and oracle error, Day 4 thresholds and hashed raw solution/metric artifacts when a side ran. `pass` requires all Day 4 gates, not just a success return. |
| Timing | State in `not_run|invalid|measured`, unit exactly `ns`, nullable observations with reason. | `measured` requires both sides' correctness `pass`, verified threads, valid monotonic clock, five warmup and 50 alternating-order timed pairs per reported window, hashed ordered raw samples and recomputed median/Q1/Q3/IQR/MAD. Resolution-limited or failed trials cannot be promoted. |
| Claims and artifacts | Narrow claim scope, required non-claim tokens, `publication_review`, artifact list (empty for a blueprint). | Every referenced raw input/log/sample/summary has path, role, media type and SHA-256. Publication needs separate `approved` review; local correctness alone does not grant it. |

Units are part of the schema: `residual_inf` is in RHS units,
`oracle_error_inf` in solution units, `backward_error` dimensionless, and
all timing durations/aggregates in integer nanoseconds. Do not accept `ms`
under a field ending in `_ns`, a stringified number, NaN/Inf, or zero as a
substitute for an unavailable metric. An unavailable optional memory or fill
value is JSON `null` with `availability_reason=not_collected|not_exposed|
not_applicable|failed` plus detail; timing `not_run` uses a named reason such
as `correctness_only_blueprint`. For correctness results, re-evaluate
the Day 4 limits from the preserved raw values, not from rounded display
strings. The selected tiny cell has no authorized speedup/winner verdict;
reject v1 keys such as `speedup`, `faster_solver`, `performance_verdict`,
`threshold_pass` or a `claim_scope` implying portable speed, hosted timing
threshold, platform parity or state-of-the-art status.

## Cross-Field Status Matrix

| `evidence_status` | Required relationship | Forbidden promotion |
| --- | --- | --- |
| `planned` | Blueprint/planning; `run_id=null`, `cell_id=null`, environment null, both correctness sides `not_run`, timing `not_run`, no generated artifacts. | No numerical or timing values. |
| `not_run` | Generated attempt ID with explicit reason, both sides `not_run`, timing `not_run`. | A prepared run directory does not count as a solver result. |
| `blocked` | Named `block_gate=execution|publication` and missing prerequisite. Execution block has no new solver output; a publication-only block may retain separately hashed local numerical observations. | No timing `measured` or performance verdict; absent optional package is never `pass`. |
| `non_comparable` | Name the mismatched Day 3 identity field or inapplicable matrix; do not reuse the selected complete `cell_id` for changed input/operation. | No paired correctness pass or ranked timing, even if one side emitted a vector. |
| `correctness_failed` | Complete cell ID and eligible input; at least one side `fail` with raw status/metric evidence. Other side may be `pass` or `not_run` if stopped. | Timing is `not_run` or `invalid`, never `measured`. |
| `correctness_observed` | Complete cell ID, both sides `pass`, raw artifacts and Day 4 metrics validated; timing `not_run`. | No third-party family-wide parity or runtime claim. |
| `timing_invalid` | Complete cell ID, initial two-sided correctness pass, named Day 5 failure and raw attempt evidence. | No valid summary/ranking; failed timed trial cannot be dropped. |
| `measured` | Complete cell ID, both sides `pass`, timing `measured`, complete raw paired samples and method/host provenance. | Only one-cell local descriptive latency; no winner, ratio, hosted threshold or cross-platform parity. |

Reject contradictory states before reading a rendered claim. In particular,
`measured` with a blocked baseline, `planned` with residuals, `blocked` with
timing samples presented as valid, or `non_comparable` with a selected-cell
speedup are errors. `publication_review=pending|denied` bars publication even
if locally observed correctness is otherwise valid; an approval must cite a
review artifact, not merely change the enum.

## Positive And Negative Fixture Matrix

The positive blueprint is the companion JSON; it must parse and pass the
future v1 validator in blueprint mode. Other positive rows below are
**synthetic test shapes to build**, not numerical results from this sprint.

| Fixture | Expected result / rule code |
| --- | --- |
| Complete companion blueprint | Accept as `planned`, no generated evidence. |
| Synthetic blocked run with absent CHOLMOD binary and `block_gate=execution` | Accept as `blocked`; no baseline pass or timing evidence. |
| Synthetic correctness-only run with two verified passes and hashed raw outputs | Accept as `correctness_observed` only with full cell ID and runtime provenance. |
| Synthetic fully measured run with 5 warmup/50 alternating pairs per phase | Accept as `measured` only with both passes, valid clock/threads, raw hashes and recomputed statistics; still no speed verdict. |
| Delete baseline source commit, loaded binary hash or dependency inventory from an observed record | Reject `missing_provenance`. |
| Change `selection_id` or fixture bytes/hash while retaining selected cell ID | Reject `selection_mismatch` or `cell_id_mismatch`. |
| Use unknown evidence status, solver state or support scope | Reject `unsupported_enum`. |
| Set timing unit to `ms`, `median_ns` to a string, or sample count to 1 | Reject `unit_or_sample_contract`. |
| Give `measured` a blocked baseline, missing raw-sample artifact or only one numerical pass | Reject `measured_requires_two_passes_and_raw_samples`. |
| Give `blocked` a performance verdict or make missing dependency `pass` | Reject `blocked_cannot_claim_performance`. |
| Insert NaN/Inf, `0` for unavailable memory, or nonfinite residual | Reject `nonfinite_or_missing_value_abuse`. |
| Point an artifact outside the bundle or change its bytes after hashing | Reject `artifact_path_or_hash`. |
| Add `speedup=2`, `faster_solver=cholmod`, platform parity or state-of-the-art claim | Reject `unearned_claim`. |
| Duplicate JSON key or misspell `loaded_library_sha256` | Reject `duplicate_or_unknown_field`, not silently accept a default. |

Tests should assert both exit status and a stable rule code, then restore the
untouched positive fixture. Mutation input should be generated in temporary
directories; no test may alter the source-controlled selected manifest or
canonical benchmark bundle. A validation failure is not a numerical solver
failure and must not be reported as one.

## Compatibility And Exact Checks

The separate schema is **not** added to `report_families.tsv`,
`selected_report_targets.tsv`, `scripts/normalize_report_index.py` or
`scripts/run_external_comparison.py` by this design. Before and after any
later implementation, compare the selected-target manifest bytes and
normalizer output for the existing families, including the source-controlled
helper `baseline_type` and the selected benchmark's `baseline=n/a`,
`threshold=n/a`, repeat-one, no-warmup/no-variance fields. A future optional
external index projection must be a new reviewed family and must preserve
current selected row IDs/counts and missing-generated semantics. It cannot
turn `planned`, `blocked` or `not_run` into `pass`.

Current executable compatibility baseline:

```sh
python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
make docs-check
```

After a validator exists, run its positive blueprint and synthetic blocked/
observed/measured fixtures, then the mutation matrix above. Keep the new
validator optional to the maintained build until Day 8 acceptance and a
reviewed pilot path. No selected workflow or report-index behavior was
changed on Day 7.
