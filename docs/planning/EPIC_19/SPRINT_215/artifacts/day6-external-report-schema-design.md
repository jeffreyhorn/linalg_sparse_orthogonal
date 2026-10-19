# Sprint 215 Day 6: External Report Schema Design

## Decision

Use a **separate, versioned external-baseline evidence schema** for the
Sprint 215 pilot. A future validator should read source-controlled schema
authority at `tests/corpus/schemas/external_baseline_evidence_v1.json` and
validate local generated records under
`build/external-baseline/<selection_id>/<run_id>/evidence.json`, with raw logs and
samples in the same bundle. These paths are proposed ownership boundaries,
not files created on Day 6. JSON is appropriate for one cell/run with nested
solver identities, per-side correctness, dependency provenance, and optional
timing; raw ordered samples may live in a separately hashed TSV/JSON file.

Do **not** extend the six selected `comparison` targets or the selected
`benchmark` target for this pilot. Do not add a new `report_families.tsv` or
`selected_report_targets.tsv` row yet. A standalone evidence record is not a
selected freshness row and cannot make an absent optional library count as
pass. Any later normalized-index projection needs its own reviewed family,
validator, normalizer and migration tests; it must not rename an existing
source-controlled-helper comparison as a third-party baseline.

## Inventory And Authority Boundaries

| Existing surface | Observed contract | Why it is not the external schema home |
| --- | --- | --- |
| `tests/corpus/schemas/report_index_fields.md` and `report_families.tsv` | Family contracts carry `row_meaning`, `row_origin`, default status, support tier, freshness policy, claims and non-claims. Contract rows cannot be `pass`; generated observations own results. | These are navigation/interpretation metadata, not a package/binary, matrix-hash, per-side residual or raw-sample record. |
| `selected_report_targets.tsv` | Six selected comparison targets have fixed expected rows/IDs and required generated files; one selected benchmark target is `SRT-BENCH-REFACTOR-CSC-NOS4`. | Adding a pilot row would widen selected freshness and hosted metadata without reviewed workflow/artifact evidence. The selected Cholesky helper fixture is `cholesky-spd-tridiag-5`, not the new `symmetric_4.mtx` CHOLMOD cell. |
| `scripts/run_external_comparison.py` | `study.tsv` has useful generic fields such as operation, baseline name/type/version, commands, platform, compiler, statuses and artifact path. It sets `baseline_type=external-process-source-controlled-helper`, with helper-file identity and local fixture outcomes. | Its baseline is a repository dense helper, not an independently packaged solver. Reusing its `pass` rows or `baseline_version` text would falsely imply external-library parity. |
| `scripts/normalize_report_index.py` | The normalized TSV has 22 navigation fields; it maps generated `pass/fail/skip/...` for comparisons, maps `measurement` to advisory, and has family-specific generated loaders and selected freshness checks. Missing selected output becomes `not_generated`. | Its current fields do not carry binary SHA-256, complete cell identity, raw numeric vectors or ordered timing samples. Adding a family without a loader would yield a contract row only; mapping `measured` to `pass` would be unsafe. |
| Canonical `build/bench-reports/canonical/index.tsv` and `manifest.txt` | `bench_refactor_csc` on `nos4.mtx --repeat 1` records local/hosted freshness metadata, `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`. | Its operation, matrix, baseline and sampling policy differ. It supplies no third-party solver comparison and no portable timing threshold. |

The current report-index metadata and selected manifests remain **unchanged**
by this Day 6 design. External evidence may link to a normalized-index row
for navigation later, but identity and validity must be verified in the
separate evidence document, never inferred from a target name.

## Proposed V1 Record Shape

One JSON document describes one selection and one attempt or blueprint state.
`selection_id` names the Day 3 fixed package/operation/matrix choice without
asserting a host or binary build. `cell_id` is a stable identifier computed
from the **complete** Day 3 identity tuple only after all runtime-dependent
components are known; it is null with a reason for an incomplete blueprint
or blocked attempt. `run_id` distinguishes attempts under a selection.
Define canonical serialization/hash rules on Day 7 before implementation.
Do not use `SRT-*` as either ID. Source-controlled planning text can describe
a blueprint, but cannot masquerade as a generated run.

| Field or group | Type and requirement | Meaning / constraint |
| --- | --- | --- |
| `schema_version` | string, required; fixed `external_baseline_evidence_v1` | Schema semantics must not change without a version bump. |
| `record_id`, `selection_id`, `cell_id`, `run_id` | strings; record/selection always required; cell ID nullable until complete provenance; run ID required for generated attempts | Unique record, selected workload, full frozen comparison identity, and distinct generated attempt. Do not overwrite prior attempts or fabricate a complete cell ID from a version string. |
| `record_kind`, `evidence_origin`, `evidence_status`, `status_reason` | enums/strings, required | `record_kind=blueprint|run`; origin `planning|generated_local|generated_ci`; status vocabulary below. `status_reason` names the first blocker or failure, not a favorable summary. |
| `claim_scope`, `non_claims`, `publication_review` | strings and enum, required | Local one-cell claim; explicit no portable speed/platform/SOTA claim. Publication review `pending|approved|denied`, independent of numerical status. |
| `cell.matrix` | object, required | Repo-relative fixture path, SHA-256 hex string, Matrix Market field/symmetry, dimensions, stored/expanded nnz, provenance/attribution. Pilot is `tests/data/symmetric_4.mtx`, hash `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. |
| `cell.problem`, `cell.rhs` | object, required | `spd_linear_solve`, `A*x=b`, float64, index ABI, one RHS, exact vector `[13,14,17,16]` and oracle `[1,1,1,1]`; a different operation is a new cell. |
| `cell.project` | object, required | Solver APIs, no-reorder and linked-list backend; exact Git SHA and dirty-tree/patch identity, source/build flags, compiler, binary hash at execution. Runtime fields may be null for blueprint/blocked attempts with a named reason. |
| `cell.baseline` | object, required | Independently released project, version, upstream commit/source URL, acquisition route/package revision, solver APIs, actual loaded library path/hash, ABI, natural ordering, simplicial mode, build/module options and dependency/BLAS hashes. A package name alone is insufficient. |
| `methodology` | object, required | Version/hash references to Day 3 identity, Day 4 correctness and Day 5 measurement contracts; units, phase definitions, tolerance formulas, warmup/repeat policy and clock rule. A correctness-only record marks timing policy `not_run`, not zero. |
| `environment` | object, nullable only before execution | OS/build, architecture, CPU, compiler/runtime, project and baseline flags, BLAS/OpenMP identity, requested and verified thread counts, host class, clock, command/cwd, UTC timestamp. Hosted origin additionally needs workflow/run/artifact IDs. |
| `correctness.project`, `correctness.baseline` | objects, required with per-side state | `not_run|blocked|fail|pass`; phase statuses, factor diagnostics, finite-value check, raw solution/residual/oracle-error values and Day 4 thresholds when observed. Null metrics require an explicit availability reason. |
| `timing` | object, required with state | `not_run|invalid|measured`; each measured window names phase, unit `ns`, ordered raw-sample reference, 5 warmup/50 paired trial contract, aggregation/dispersion and clock resolution. No number when not run or invalid. |
| `artifacts` | array of objects, required (possibly empty for blueprint) | Each generated input/log/solution/sample/report reference has repo-relative or bundle-relative path, media type, SHA-256 and role. No unhashed external URL can substitute for raw observations. |

Use JSON `null` plus an explicit availability/status reason for unknown build
options or unavailable memory/fill/timing metrics. Zero is a real numerical
value, never a missing-data sentinel. Numeric metrics and durations must be
finite and carry units. Preserve unrounded solution and residual values in
raw artifacts even if a rendered summary is shorter. A blueprint may contain
the Day 2 discovered CHOLMOD dylib SHA-256
`0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`,
but its actual runtime load path and full build provenance remain unverified
until execution.

## Evidence Status Vocabulary

`evidence_status` is not the current normalized-index `status`; do not map
these values to `pass` by string similarity.

| Status | Meaning | Required boundary |
| --- | --- | --- |
| `planned` | Blueprint only; no generated attempt. | `record_kind=blueprint`, `evidence_origin=planning`, `cell_id=null`; both correctness sides and timing `not_run`. |
| `not_run` | A run record was opened but no solver attempt occurred. | Named reason; no numeric observation or timing verdict. |
| `blocked` | Required package, provenance, license/publication gate, input or environment evidence is missing. | Name the missing prerequisite and gate; optional dependency absence cannot be `pass`. Publication-only block may coexist with locally observed numerical data but bars publication. |
| `non_comparable` | Matrix, RHS, operation, backend, build identity, precision or other Day 3 key differs, or the input is inapplicable. | State mismatch/eligibility reason; no paired correctness or speed verdict. |
| `correctness_failed` | Verified eligible cell ran, but at least one Day 4 side failed status, finiteness, residual or oracle gate. | Retain per-side raw diagnostics; timing is `not_run` or `invalid`, never evidence. |
| `correctness_observed` | Both sides passed Day 4 on the exact cell, with complete execution provenance. | Raw outputs and per-side metric artifacts required; timing remains `not_run`. This is one-cell numerical evidence, not external-library parity. |
| `timing_invalid` | A timing attempt was made but Day 5 sampling, clock, thread, provenance or per-trial correctness rule failed. | Preserve raw attempt/failure reason; no aggregate speed verdict. |
| `measured` | Both sides passed Day 4 and a separate Day 5 local timing run has complete, valid raw samples. | Complete non-null `cell_id` and provenance required; timing can be described only for that one host/build/cell, not ranked as portable speed or state of the art. |

Evidence status transitions are append-only run records, not in-place edits
that erase a previous block or failure. A status is an interpretation of
observed artifacts, not a proxy for their existence. A `measured` record
cannot have `correctness.baseline.state=blocked`, and a `planned` or
`not_run` record cannot carry non-null timing samples. Day 7 will make these
cross-field rules executable.

## Illustrative Rows (Not Evidence)

The following examples are **schema-design examples only**. No row below
claims an executed CHOLMOD solve or measured runtime. Every row uses the
same `selection_id`; `cell_id` is null until complete identity is available,
and cannot be reused for an identity mismatch. `example-*` IDs are not
production IDs.

| Example record | Kind/origin/status | Correctness / timing | Why it is distinguishable |
| --- | --- | --- | --- |
| `example-blueprint` | `blueprint`, `planning`, `planned` | Both sides `not_run`; timing `not_run`; raw artifacts empty | The versioned selection and methodology exist, but no execution is claimed. |
| `example-missing-package` | `run`, `generated_local`, `blocked` | Project `not_run`, baseline `blocked`; timing `not_run`; reason `cholmod_binary_unavailable` | Missing optional dependency produces a block, not a passing comparison or a zero duration. |
| `example-wrong-matrix` | `run`, `generated_local`, `non_comparable` | Both sides `not_run`; timing `not_run`; reason `matrix_sha256_mismatch` | A changed fixture cannot borrow the selected cell key. |
| `example-numerical-pass-shape` | `run`, `generated_local`, `correctness_observed` | Both sides `pass` with hashed raw solutions/metrics; timing `not_run` | This is a **hypothetical validator fixture shape**, not a result produced on Day 6. |
| `example-measured-shape` | `run`, `generated_local`, `measured` | Both sides `pass`; timing `measured` with hashed 50-pair raw samples in `ns` | This is a **hypothetical validator fixture shape**, not a speed claim or measured result. |

## Compatibility Risks And Day 7 Handoff

- `run_external_comparison.py` emits helper-based `pass` rows. Its generic `baseline_name/version` fields must not be copied into a third-party record without the actual package, binary and dependency identity.
- The normalized index has fixed fields and family-specific generated loaders. Adding a report-family contract row alone would not ingest external evidence; forcing this record into `configuration` would hide required nested provenance and status rules.
- `selected_report_targets.tsv` owns expected counts/IDs and hosted workflow metadata. Changing it for a local optional pilot would alter existing freshness gates and require a tested migration. No such migration is selected.
- Canonical benchmark `baseline=n/a`, `threshold=n/a`, repeat-one and no-variance fields must remain untouched. An external timing sample cannot be used to reinterpret that row or claim Linux/macOS timing parity.
- A later schema validator must reject missing hashes, mismatched cell keys, unsupported status/unit values, `measured` without two correctness passes or raw sample artifacts, `blocked` with a speed verdict, and nonfinite or zero-as-missing metrics. It should verify relative artifact paths stay within the bundle and hashes match bytes.

Day 7 should turn this dictionary into parseable positive/negative fixtures
and validation rules, then compare unchanged selected manifests and
normalizer output. Current compatibility checks to retain are
`python3 scripts/validate_corpus_schema.py`,
`python3 tests/test_normalize_report_index.py`,
`python3 tests/test_selected_report_targets_manifest.py`, and the selected
comparison and benchmark freshness tests if those surfaces ever change.
No schema, manifest, normalizer, workflow or generated report was modified on
Day 6.
