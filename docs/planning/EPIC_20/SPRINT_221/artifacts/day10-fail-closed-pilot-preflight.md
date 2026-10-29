# Sprint 221 Day 10: Fail-Closed Pilot Preflight

## Boundary

The optional pilot now invokes the real v1 validator `--self-test`
immediately after checking the selected matrix bytes. It cannot reach
`pkg-config`, the project archive build, helper compilation or either solver
without a passing self-test. The self-test validates the frozen planning
fixture against repository bytes, then requires rejection of duplicate JSON
keys, an unknown field, a changed selected RHS, a false solver pass and a
changed methodology hash. It performs no CHOLMOD build or solve. Its success
means that the format gate is working; it is not a correctness result.
Both CLI modes, `--self-test` and `--input`, run this negative self-test first;
the input mode cannot accept a record under a policy-relaxed schema.

## Failure Outcomes

Missing schema or validator files produce `schema_validation_unavailable`
with `v1_schema_or_validator_missing`. Unreadable files produce the same
blocked outcome with `v1_schema_or_validator_unreadable`. A malformed or
policy-relaxed schema, invalid selected planning fixture or failed negative
self-test produces `schema_validation_failed` with
`validator_self_test_failed`. All cases retain `sides={}`, `timing=not_run`
and no build/solver command. When the self-test subprocess was started, the
attempt records its command, return code, raw stdout/stderr files and their
SHA-256 digests. An absent file is rejected before a subprocess runs.

The pilot still creates an attempt directory with `exist_ok=False`; an
existing path is rejected with a clear CLI error and its prior `attempt.json`
is left unchanged. The matrix hash, CHOLMOD version/candidate-binary checks,
Git/compiler/linkage probes, solver-status parsing, loaded-library hash and
Day 4 numerical gates remain after the self-test. A passing self-test only
authorizes those next checks, not an `observed` or `measured` v1 record.

## Verification Scope

The pilot tests cover missing and unreadable inputs, a failing self-test with
byte-exact CRLF/non-UTF-8 preflight stream capture, malformed and relaxed
schemas through the actual validator subprocess, blocked dependency discovery
after a passing gate, and immutable attempt-path rejection. The validator
tests cover a successful self-test and planning/blocked input validation plus
an intentionally relaxed schema that must fail the negative self-test.
No external solver or timing command is used by these tests.
