# Sprint 221 Day 9: Fixture Closure And Pilot Gate Design

## Scope And Evidence Boundary

The fixture matrix is synthetic validator input, not a CHOLMOD observation.
`tests/test_validate_external_baseline_evidence.py` now exercises all eight
schema statuses through `validate_record()`, including real byte reads for
the selected matrix and methodology and temporary hashed bundles where a
status has generated artifacts. The validator CLI still exits with
`implementation_pending`; a validated fixture shape does not release the
optional pilot or establish a solver result.

## Canonical Fixture Matrix

| V1 status | Positive source | Negative neighbor |
| --- | --- | --- |
| `planned` | Frozen `external_baseline_planned_v1.json` | Added observation or method hash contradicts planning state |
| `not_run` | `synthetic_not_run()` | Added solver/timing value or missing reason |
| `blocked` | `synthetic_execution_blocked()`; separately hashed publication-blocked case | Pass label without raw evidence or measured timing |
| `non_comparable` | `synthetic_non_comparable()` | Complete cell or paired result cannot be used to mask a mismatch |
| `correctness_failed` | `synthetic_correctness_failed()` with a failed oracle gate and hashed raw sides | Missing raw baseline or failure label with a passing solution |
| `correctness_observed` | `synthetic_observed()` with two passing raw sides | One side blocked, missing raw output, or raw/reported value mismatch |
| `timing_invalid` | `synthetic_timing_invalid()` with hashed attempt bytes and no summary | Missing timing-attempt artifact or attempted measured claim |
| `measured` | `synthetic_measured()` with three hashed raw windows | Bad counts/order/aggregate, clock or thread controls, or winner claim |

The integrated status test is
`ArtifactTests.test_all_evidence_statuses_have_integrated_synthetic_fixtures`.
The observed/measured cases and their negative neighbors are in
`ArtifactTests`; non-execution and publication-block cases are also in
`NonExecutionFixtureTests`. The temporary bundle is never a selected report.

## Required Field Groups

| Group | Acceptance and rejection coverage |
| --- | --- |
| Envelope, selection and claim scope | Structural selection mutations; semantic kind/status/claim mutations; synthetic status matrix |
| Cell, matrix, RHS and solver identity | Frozen identity vector, selected matrix byte hash, canonical cell-ID recomputation, ABI and backend mutations |
| Methodology | Frozen paths and actual file SHA-256 checks; missing/changed method hash mutations |
| Environment and provenance | Generated-run timestamp, compiler, loaded CHOLMOD/dependencies, thread and measured-clock mutations |
| Correctness sides | Hashed raw status/diagnostic/solution binding, Day 4 metrics, two-sided consistency and failed-gate mutations |
| Timing | Five warmups/50 timed AB/BA pairs per reported window, raw hash, recomputed summary and resolution gate |
| Artifacts and publication | Missing, changed, duplicate, unsafe and escaping path checks; independent publication-review evidence gate |

## Named Rule-Code Ledger

Every rule code in the schema's `x-v1-policy.rule_codes` has a negative test:

| Code | Exercised by |
| --- | --- |
| `invalid_json` | Structural malformed/UTF-8/surrogate cases |
| `duplicate_or_unknown_field` | Duplicate JSON key, unknown field, duplicate raw sample key |
| `missing_required_field` | Missing envelope or nested matrix field |
| `exact_type` | Boolean/floating integer fields and boolean solver status |
| `unsupported_enum` | Unknown schema/status/side state |
| `selection_mismatch` | Matrix, RHS, method path and frozen count mutations |
| `nonfinite_or_missing_value_abuse` | Nonfinite JSON and solver-value mutations |
| `missing_provenance` | Runtime, method, raw side, clock and publication omissions |
| `artifact_path_or_hash` | Missing/changed file, unsafe path, symlink, inaccessible root and digest mismatch |
| `cell_id_mismatch` | Changed complete identity or canonical ID |
| `status_contradiction` | Unrun values, false failure/pass, one-sided observed status and raw/reported mismatch |
| `unit_or_sample_contract` | Timing unit, order, summary, clock resolution and thread mutations |
| `measured_requires_two_passes_and_raw_samples` | Measured with one side or no verified raw pairs |
| `blocked_cannot_claim_performance` | Blocked record with measured timing |
| `unearned_claim` | Speed/winner token and unsupported claim scope |
| `implementation_pending` | CLI/pilot gate remains closed until the negative self-test exists |

Unsupported shapes are explicit: multiple cells, another matrix or solver
selection, a different baseline version, inferred performance from
correctness, a `measured` record without two verified sides and raw paired
samples, and publication approval without review evidence. A schema-valid
`generated_local` record is not itself evidence of a real execution; its
loaded binaries, dependencies and raw output must still be independently
verified by the pilot owner.

## Pilot Control Flow And Earliest Gate

Current `run_cholmod_pilot.py` order is selected fixture existence/hash,
`pkg-config` discovery, candidate dylib version/hash, Git/compiler/linkage
probes, schema/validator existence check, `schema_preflight --self-test`,
project archive build, helper compilation, project solve, baseline solve.
The existing gate is before all build/solver commands, and its failure test
asserts those command labels are absent. It currently fails because the
validator intentionally returns `implementation_pending`.

The earliest useful Day 10 stop point is immediately after the selected
fixture hash and before dependency discovery. Run the real v1 self-test from
there, capturing raw stdout/stderr and SHA-256 via `command()`. A missing or
unreadable schema or validator, failing self-test, or invalid selected
planning evidence must yield an immutable attempt with
`provisional_status=blocked`, `sides={}`, `timing=not_run`, no build/solver
commands, and a specific `schema_validation` failure reason. An invalid
generated result can only be checked after execution; it must never be
promoted to a validated correctness status. The pre-solve "invalid evidence"
case refers to the selected planning/input contract and self-test fixtures,
not an as-yet-unproduced solver result.

Day 10 tests should cover missing schema, missing validator, unreadable
schema, failing self-test and invalid planning input at this earliest gate;
assert raw preflight stream hashes, empty sides, unchanged timing and no
`project_library_build`, `compile_helper`, `project` or `baseline` command.
Day 11 should test a passing self-test followed by each downstream gate.
Passing the self-test authorizes only the next pilot step.

## Optional Commands

Local fixture and pilot tests need no CHOLMOD installation:

```sh
python3 tests/test_validate_external_baseline_evidence.py
python3 tests/test_run_cholmod_pilot.py
```

After Day 10 implements the real negative self-test, the optional Python
CI route can add:

```sh
python3 scripts/validate_external_baseline_evidence.py \
  --schema tests/corpus/schemas/external_baseline_evidence_v1.json --self-test
```

Until then, that command is expected to exit nonzero with
`implementation_pending`. It must not be registered as a passing CI check
or bypassed to run the solver. No selected report, normalizer output or
canonical benchmark row is changed by these commands.
