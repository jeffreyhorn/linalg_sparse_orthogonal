# Sprint 221 Day 3: Canonical Identity And Schema Authority

## Boundary And Sources

This freezes the v1 **selected-cell identity**, schema shape and validator
interface for the Sprint 215 CHOLMOD pilot. It does not validate an observed
run or execute CHOLMOD. The accepted source decisions are Sprint 215 Day 6
(`day6-external-report-schema-design.md`), Day 7
(`day7-schema-validation-design.md`) and Day 8
(`day8-pilot-feasibility-decision.md`). The Day 7 JSON example remains a
planning blueprint. The Sprint 215 Day 10 attempt remains blocked before
solver execution, not a numerical result.

The source-controlled authority is
`tests/corpus/schemas/external_baseline_evidence_v1.json`. It uses JSON
Schema Draft 2020-12 as a declarative shape, with `x-v1-policy` for rules
that require bytes, run context or cross-field interpretation. Runtime
validation will be stdlib-only; installing a JSON Schema package is not part
of the maintained build. Unknown fields, duplicate keys, nonfinite JSON
numbers and booleans used as integers are rejected. A schema shape pass alone
cannot establish correctness or timing.

## Selection Controls And Run Provenance

The fixed `selection_id` is
`s215-cholmod-v7_12_2-symmetric4-spdsolve-v1`. The selection fixes the
`symmetric_4.mtx` bytes/hash, Matrix Market real/symmetric 4 x 4 with seven
stored and ten expanded entries, `A*x=b`, float64, one RHS
`[13,14,17,16]`, oracle `[1,1,1,1]`, project int32/no-reorder/linked-list
Cholesky API, and SuiteSparse v7.12.2 commit
`42151688813c45846a597edcb601435a0e38f3dd` with CHOLMOD 5.3.4,
natural/simplicial/`CHOLMOD_A` and no postorder. The baseline ABI is unknown
in the blueprint and must be verified as int32 for this selected cell. A
different ABI, matrix, RHS, solver control or baseline version requires a new
selection decision, never reuse of this ID.

The historical blueprint abbreviates the matrix-origin commit as
`aa1c04ea`. Generated records must use the resolved full commit
`aa1c04eab3536031b9a92ee92587260520ce1913`; this is a provenance
normalization, not a changed matrix selection.

The following are **run provenance**, not selected controls or `cell_id`
substitutes: `record_id`, `run_id`, UTC execution time, command/cwd, Day 3/4/5
methodology paths and SHA-256 values, raw solutions and metrics, timing
samples, process logs, artifact hashes, review status and numerical status.
The Homebrew candidate dylib hash is discovery evidence only; the actual
loaded library path/hash belongs to the complete identity. Clean source
worktrees use SHA-256 of the empty patch bytes
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
For a dirty run, hash raw `git diff --binary HEAD` bytes and preserve the
status/patch artifact; untracked build inputs cannot silently enter an
observed record.

## Complete Cell ID

`identity_from_record()` selects only these exact fields. No value in a
complete identity may be null; partial blueprints and execution blocks use
`cell_id=null` and a nonempty `cell_id_reason` instead. A publication block
with otherwise complete local observations may retain its computed ID.

| Identity object | Exact fields |
| --- | --- |
| Root | `matrix_sha256`, `problem`, `project`, `baseline`, `threads`, `host` |
| `problem` | `class`, `operation`, `rhs`, `precision`, `project_index_abi`, `baseline_index_abi` |
| `project` | `git_commit`, `worktree_state`, `patch_sha256`, `binary_sha256`, `compiler`, `compile_flags`, `build_mode`, `solver_api`, `backend`, `ordering` |
| `baseline` | `source_commit`, `loaded_library_sha256`, `dependencies`, `build_options`, `module_composition`, `solver_api`, `backend`, `ordering`, `postorder` |
| `baseline.dependencies[]` | `role`, `loaded_path`, `sha256`; each role and loaded path is unique |
| `threads` | `requested`, `observed`, each containing positive integer `blas`, `openmp` counts |
| `host` | `os`, `os_build`, `architecture`, `cpu` |

Project and baseline commits must be complete 40-character lowercase hex;
all hashes are 64-character lowercase hex. Dependencies are sorted by
`(role, loaded_path)`; paths are normalized absolute loaded paths. The
serializer uses UTF-8 bytes of sorted-key, compact JSON
(`separators=(',', ':')`, `ensure_ascii=True`, `allow_nan=False`) containing
only strings, integers, booleans, arrays and objects. `cell_id` is
`sha256:` followed by the SHA-256 of those bytes. It excludes raw metrics
and methodology hashes, which still must match for a comparable run.

The synthetic canonical byte vector is the single line in
`tests/corpus/fixtures/external_baseline_identity_v1.canonical.json`, without
its final newline. Its expected ID is
`sha256:5ff34c9c3d5661942d98b94a4c5dabf62b10920bec03f364f69201126d54481d`.
The `a`/`b`/`c`/`d`/`e` digests, `clang 1` and `test-\u00b5CPU` in that
fixture are synthetic serialization inputs, **not** binary or host
observations. The test proves ASCII escaping, dependency ordering and exact
bytes/hash; it does not assign a production cell ID.

## Status, Units And Claim Rules

The schema's eight `evidence_status` values have the accepted Sprint 215
relationships. `planned` requires blueprint/planning, null run/cell IDs,
both sides `not_run`, timing `not_run` and no artifacts. `not_run` is a
generated attempt with reason but no observations. `blocked` names execution
or publication gate; an execution block cannot carry a new solver result,
and a publication block cannot publish. `non_comparable` names the changed
identity/eligibility field and cannot borrow the selected complete ID.
`correctness_failed` requires a complete eligible cell and raw failure
evidence. `correctness_observed` requires two independently validated passes
with raw artifacts and no timing result. `timing_invalid` retains failed
attempt data but no valid aggregate. `measured` requires the same two passes,
complete provenance, five warmup and 50 alternating timed pairs per phase,
valid clock and threads, hashed raw samples and recomputed statistics.

`residual_inf` uses RHS units; `oracle_error_inf` uses solution units;
`backward_error` is dimensionless. Raw durations are integer nanoseconds;
aggregates can be fractional nanoseconds. A missing metric is null with a
named reason, never zero, NaN or infinity. No selected-cell status grants a
speedup, winner, hosted threshold, portable timing, platform parity or
state-of-the-art claim. `publication_review=approved` also requires a
separate review artifact, not merely a changed enum. Artifact paths are
safe relative paths under the declared repository/bundle root and require
byte-level SHA-256 verification; symlink escapes are invalid.

## Interface And Current Gate

`scripts/validate_external_baseline_evidence.py` now accepts `--schema`,
`--input`, optional `--bundle-root`, or `--schema ... --self-test`. Errors
are intended to carry a stable rule code and JSON pointer. Day 3 implements
the canonical identity primitive and strict JSON parse, but **does not**
claim that shape/cross-field/byte validation is complete. Both `--input` and
`--self-test` currently return `implementation_pending /` with exit 1 after
basic parsing. Thus `run_cholmod_pilot.py` still stops before build/solver
execution. Day 4-8 must implement the remaining rules and negative self-test
before that gate may pass. The selected report manifests, normalizer,
canonical benchmark and workflows remain untouched.

## Rule-Code Handoff

The schema lists the stable proposed codes: `invalid_json`,
`duplicate_or_unknown_field`, `missing_required_field`, `exact_type`,
`unsupported_enum`, `selection_mismatch`, `missing_provenance`,
`cell_id_mismatch`, `status_contradiction`, `unit_or_sample_contract`,
`measured_requires_two_passes_and_raw_samples`,
`blocked_cannot_claim_performance`, `nonfinite_or_missing_value_abuse`,
`artifact_path_or_hash`, `unearned_claim` and `implementation_pending`.
Future tests must assert code and verdict; `implementation_pending` must
disappear from successful paths only after the negative fixture matrix runs.
