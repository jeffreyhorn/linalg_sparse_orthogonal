# Sprint 221 Day 5: Semantic Status And Cell Validation

## Scope And Evidence Boundary

The v1 validator now checks relationships that the Day 3 schema cannot express
alone. The selected cell remains the Sprint 215 `symmetric_4.mtx`/CHOLMOD
blueprint. No external solve or timing run was performed. All observed and
measured records in the focused tests are explicitly synthetic in-memory
fixtures; their placeholder hashes do not attest to files or loaded binaries.

The CLI and pilot preflight still fail closed with `implementation_pending`.
Day 6 must verify referenced artifact bytes and hashes, and later days must
exercise a genuine negative self-test before a generated record can pass.

## Implemented Semantic Rules

- Match blueprint/planning and run/generated origin, run ID, status and claim
  scope. Planned and `not_run` records have two unrun sides, no timing samples,
  no cell ID and no generated artifacts. Execution-blocked records cannot
  carry solver observations; publication-blocked records cannot carry a
  measured-performance verdict.
- Require two independent `pass` sides for `correctness_observed`,
  `timing_invalid` and `measured`; require a demonstrable failed phase,
  effective control or numerical gate for `correctness_failed`. A changed
  status/reason alone cannot turn two successful solutions into a failure.
  A one-sided `non_comparable` record remains claim-free with a named mismatch.
- Require complete observed provenance for completed runs, including the full
  matrix-origin commit, runtime project and loaded-baseline identities,
  dependency inventory, verified index ABI, environment, UTC timestamp,
  methodology hashes and clean-tree empty-patch hash. Recompute the cell ID
  from canonical identity bytes. Reject duplicate dependency roles or paths
  and any complete-cell ID mismatch.
- Recompute residual infinity norm, backward error, oracle error and fixed
  Day 4 limits from each four-value solution. Reject nonfinite values,
  unreported or contradictory metrics, failed numerical gates on a passing
  side, missing success statuses/effective factor controls, and cross-solver
  inconsistency.
- For a measured record, require two correctness passes and in-memory raw
  paired samples for every reported window: five warmup pairs, 50 timed pairs,
  alternating AB/BA or BA/AB order, positive integer nanosecond durations,
  successful statuses, an eligible timing policy and a positive observed
  clock resolution. Recompute count, min, max, median, Q1, Q3, IQR, MAD and
  primary-window medians; reject resolution-limited medians. Invalid timing
  keeps a reason and a referenced raw attempt without a numerical summary.
- Keep publication review separate from local correctness and timing.
  Approval needs an attribution-approved fixture and review-artifact
  reference. Non-claim tokens remain mandatory; no portable-speed, hosted
  threshold, cross-platform parity or state-of-the-art claim is accepted.

## Focused Regressions

The Python suite now covers planning, execution-blocked, non-comparable,
correctness-failed, correctness-observed, timing-invalid and measured shapes;
complete cell-ID mismatch and dependency duplication; missing provenance;
wrong reported metrics and solver statuses; a fabricated failure label;
nonfinite in-memory values; unearned claims and unsupported approval; timing
summary, order, policy and artifact-reference mismatches. Both AB-first and
BA-first alternating pairs are accepted. The synthetic measured fixture is
passed directly to `validate_semantics()` with raw samples, not to the CLI.

## Verification And Next Boundary

`python3 tests/test_validate_external_baseline_evidence.py` passed 20 tests;
`python3 tests/test_run_cholmod_pilot.py` passed 13 tests;
`python3 scripts/validate_corpus_schema.py`, `python3 -m py_compile` for the
validator and test, and `make docs-check` passed. A direct `--self-test` call
exited 1 with `implementation_pending`. No `.c` or `.h` file changed, so the
C format/lint/test sequence was not required.

Day 6 must bind every artifact reference to safe in-root bytes and verify its
SHA-256, including raw solver and timing files. Until then, in-memory semantic
success is not an accepted run or a publication right.
