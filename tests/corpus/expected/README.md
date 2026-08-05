# Expected Results

Expected-result files in this directory are small committed baselines for
maintained corpus fixtures. Each expected-result file must be tied to a row in
`../manifests/fixtures.tsv` and must preserve the fixture-local claim boundary.

Expected results are not observed oracle output. Observed rows, comparison
logs, generated indexes, and local run manifests belong under ignored
`build/corpus/` or `build/corpus-reports/`. The observed oracle row schema is
documented in `../schemas/oracle_fields.md`.

Raw QR basis vectors should not be the primary expected artifact for
rank-deficient QR fixtures. Prefer rank, nullity, residual, and
normalized null-vector residual comparisons with fixture-local tolerances.

Expected-result skeleton rows may use placeholder statuses such as
`placeholder_pending_generator` or `placeholder_pending_oracle_command`.
Placeholder rows are not pass evidence.

`ready_for_oracle` means the expected row is source-controlled and can be
consumed by `python3 scripts/run_corpus_oracle.py`. It is still not pass
evidence until an observed oracle row is generated and passes under its
fixture-local tolerance.
