# Sprint 221 Day 7: Planning And Blocked Fixtures

## Evidence Boundary

This day adds positive fixtures for non-execution states and negative tests
for attempted status promotion. No CHOLMOD build, external solve or timing
trial was run. The new planning fixture is selection/methodology
metadata only; the generated-run records in tests are explicitly synthetic
in-memory attempts. Their acceptance by `validate_record()` is a schema and
artifact-contract test, not evidence of an external baseline result.

## Frozen Planning Fixture

`tests/corpus/fixtures/external_baseline_planned_v1.json` freezes the accepted
Sprint 215 Day 7 blueprint under a new fixture record ID. A regression
compares it field-for-field with the original blueprint after normalizing
only that record ID. It has `record_kind=blueprint`, `evidence_origin=planning`,
`evidence_status=planned`, null run and cell IDs, named missing-runtime and
method-hash reasons, two `not_run` correctness sides, `timing=not_run`, no
generated artifacts and mandatory no-result/non-performance tokens. The
integrated validator verifies its matrix bytes and repository method paths.

## Synthetic Non-Execution Matrix

`synthetic_not_run()` derives a generated-local run with a synthetic run ID,
`status=not_run` and named preflight reason, but no environment, loaded
CHOLMOD identity, solver values, timing samples or bundle artifacts.
`synthetic_execution_blocked()` derives an execution-gated record with the
baseline side blocked by `cholmod_runtime_unavailable`. Both retain null
cell IDs with reasons and `claim_scope=no_result`. Tests assert each absent
observation explicitly, then pass the integrated validator. Mutations that
remove reasons, add numerical values or timing medians, assert a correctness
claim, relabel the blocked baseline as a pass, or label the unrun record
`measured` are rejected.

The publication-blocked shape is included only as a synthetic, separately
hashed two-sided correctness record built in a temporary bundle. It keeps
`publication_review=pending`, matrix attribution pending, timing `not_run`,
and the narrow local-correctness claim. It cannot be relabeled as measured
performance. This case demonstrates that local correctness and publication
approval remain independent; it is not a claim that the selected pilot ran.

## Baseline Pass Exclusions

Starting from a synthetic two-sided observation with hashed temporary raw
files, focused mutations remove the loaded CHOLMOD path, remove the loaded
binary SHA-256, omit module composition/review, or mark it `unverified`.
Each fails at the corresponding baseline provenance pointer before it can
represent a valid pass. The no-run fixtures cannot manufacture a baseline
pass by changing a status label alone. A CLI invocation on the valid
execution-blocked fixture still exits with `implementation_pending`, so a
validated blocked shape does not release the optional pilot gate.

## Verification

`python3 tests/test_validate_external_baseline_evidence.py` passed 36 tests;
`python3 tests/test_run_cholmod_pilot.py` passed 13 tests;
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation of the
validator and test, and `make docs-check` passed. No `.c` or `.h` file
changed, so `make format && make lint && make test` was not required.
