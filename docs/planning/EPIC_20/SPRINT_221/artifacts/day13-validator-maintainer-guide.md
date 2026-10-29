# External Baseline Evidence V1: Maintainer Guide

## Scope And Authority

V1 describes one selected, local, correctness-only comparison of project
Cholesky and SuiteSparse v7.12.2 / CHOLMOD 5.3.4 on
`tests/data/symmetric_4.mtx`. The authoritative shape and frozen selection
are `tests/corpus/schemas/external_baseline_evidence_v1.json` with
`$id=external_baseline_evidence_v1`; the stdlib implementation is
`scripts/validate_external_baseline_evidence.py`. The schema's `x-v1-policy`
lists cross-field rules that JSON Schema alone cannot decide. Changing the
matrix, RHS, solver, baseline version or index ABI requires a new selection
and schema version, not a silent edit to a v1 record. Unknown fields and
unknown statuses fail closed. Preserve old schemas when adding a version so
old attempt records remain interpretable.

A record is either a planning blueprint or one generated attempt. `record_id`
identifies that record; generated attempts also require `run_id`. The pilot's
`attempt.json` is an unvalidated attempt envelope, **not** a v1 evidence
record or an accepted external result. A passing preflight self-test only
allows the optional pilot to proceed to later gates.

## Status And Claim Contract

| Status | Required interpretation |
| --- | --- |
| `planned` | Planning blueprint; no run ID, cell ID, observations or generated artifacts. |
| `not_run` | Generated attempt with a reason; no solver or timing result. |
| `blocked` | Named execution or publication gate; an execution block has no solver result and neither gate can claim measured performance. |
| `non_comparable` | Named eligibility/identity mismatch; no reused complete cell ID or paired verdict. |
| `correctness_failed` | Complete provenance and at least one raw, independently evidenced failed side. |
| `correctness_observed` | Complete cell and two independently passing, hashed solver outputs; timing remains `not_run`. |
| `timing_invalid` | Two passing sides and retained failed timing-attempt evidence; no valid summary. |
| `measured` | Two passes plus complete verified raw paired samples, host/clock/thread controls and recomputed statistics. |

The selected pilot is correctness-only. The timing statuses define v1 shape
for future use; they do not authorize timing the selected pilot. Allowed
claims remain one local cell of correctness or, for a separately eligible
record, descriptive local latency. No v1 status supports a winner, speedup,
hosted threshold, portable speed, cross-platform parity or state-of-the-art
claim. `publication_review` is independent of numerical validation:
`approved` requires an identified, hashed review artifact and approved
matrix attribution. A validator pass does not perform or replace that human
review.

## Canonical Cell Identity

For complete records, `identity_from_record()` extracts exactly the matrix
SHA-256; problem class/operation/RHS/precision and both index ABIs; project
commit/worktree/patch/binary/compiler/build/solver controls; baseline
commit/loaded-library/dependency/build/solver controls; requested and
observed thread counts; and host OS/build/architecture/CPU. Dependency
entries have unique roles and normalized absolute loaded paths and are
sorted by `(role, loaded_path)` before serialization. No field in the
complete identity may be null.

The canonical bytes are `json.dumps(identity, sort_keys=True,
separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")`.
`cell_id` is `sha256:` plus the lowercase SHA-256 of those exact bytes. The
synthetic byte vector is
`tests/corpus/fixtures/external_baseline_identity_v1.canonical.json` **without
its final newline**. Its digest is a test vector, not a production cell ID.
Planning, incomplete and execution-blocked records use `cell_id=null` with
`cell_id_reason`; complete observed records require a matching ID and null
reason. Methodology hashes, raw results and publication review are run
provenance, not cell-ID inputs, but still require verification.

## CLI And Diagnostics

Run from the repository root:

```sh
python3 scripts/validate_external_baseline_evidence.py \
  --schema tests/corpus/schemas/external_baseline_evidence_v1.json --self-test
python3 scripts/validate_external_baseline_evidence.py \
  --schema tests/corpus/schemas/external_baseline_evidence_v1.json \
  --input tests/corpus/fixtures/external_baseline_planned_v1.json
python3 scripts/validate_external_baseline_evidence.py \
  --schema tests/corpus/schemas/external_baseline_evidence_v1.json \
  --input /path/to/record.json --bundle-root /path/to/attempt-bundle
```

Use exactly one of `--self-test` or `--input`; `--bundle-root` applies only
to `--input`. The validator always runs the negative self-test first. A
successful `--input` exit proves **format and internal consistency against
referenced bytes only**. A rejection exits nonzero and prints
`<rule_code> <JSON_pointer> record_id=<id>: <detail>`; do not scrape the
human detail text. The schema's `x-v1-policy.rule_codes` is the rule-code
namespace. `implementation_pending` is a fail-closed self-test failure,
never a successful evidence status.

| Code group | Meaning |
| --- | --- |
| `invalid_json`, `duplicate_or_unknown_field`, `missing_required_field`, `exact_type`, `unsupported_enum`, `nonfinite_or_missing_value_abuse` | Strict parse or declarative shape failed. |
| `selection_mismatch`, `cell_id_mismatch` | Frozen cell controls or canonical identity differ. |
| `missing_provenance`, `artifact_path_or_hash` | Required runtime/review source or referenced bytes are absent, unsafe or inconsistent. |
| `status_contradiction`, `blocked_cannot_claim_performance`, `measured_requires_two_passes_and_raw_samples` | A status asserts more than its independent side/timing evidence supports. |
| `unit_or_sample_contract`, `unearned_claim` | Timing units/counts/statistics or claim scope violate v1. |

Repository matrix and methodology references are checked against their
actual SHA-256 bytes. Generated artifacts require `--bundle-root`; paths
must be normalized relative POSIX paths under that root. The validator
rejects traversal, symlink traversal, missing/nonregular files and digest
mismatches. Observed raw solver JSON is parsed after hashing and bound to
reported statuses, diagnostics and solutions. Measured raw pairs are
rehashed and used to recompute counts and summaries. A matching hash alone
does not prove that the producer or solver was genuine.

## Retention And Reproducible Checks

Retain the original input record, schema version, methodology files, all
raw stdout/stderr bytes, solver outputs, sample files, commands, environment,
loaded binary/dependency identities and each SHA-256. Do not normalize raw
streams before hashing. Pilot output goes under
`build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/` in a
new attempt directory; the pilot refuses an existing directory. Treat each
attempt as append-only and never overwrite a previous result. Generated
local evidence stays local and out of selected canonical reports; publication
requires a separate review and artifact.

The following checks need no CHOLMOD installation:

```sh
python3 -m unittest tests.test_validate_external_baseline_evidence.NonExecutionFixtureTests
python3 -m unittest tests.test_validate_external_baseline_evidence.ArtifactTests
python3 -m unittest tests.test_validate_external_baseline_evidence.CliMutationTests
python3 tests/test_run_cholmod_pilot.py
```

The planning CLI command above checks the frozen blueprint. The
`NonExecutionFixtureTests` command checks synthetic `not_run`/blocked cases;
`ArtifactTests` checks synthetic observed/measured temporary bundles; and
`CliMutationTests` checks strict parser and semantic negatives with exact
diagnostic codes/pointers and restored positives. These synthetic tests are
not observations. Sprint 222 must supply the first real, independently
provenanced, two-sided CHOLMOD correctness cell before any external parity
claim can be considered.

Unsupported v1 shapes include multiple cells, a different selection or
baseline version, unverified baseline ABI, a one-sided correctness claim,
measured timing without all raw pairs, a performance conclusion inferred
from correctness, and publication approval without review evidence. Reject
or version these shapes; do not coerce them into a selected v1 cell.
