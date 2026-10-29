# Sprint 221 Working Notes

## Day 3: Canonical Identity And Schema Authority (2026-10-28)

Source revision at start: `50dab57aa8cc6cbdeacd70357d0453bb5170993e`.
The branch initially had only the untracked Sprint 221 plan. To restore the
missing prerequisites before freezing Day 3, this work also recorded the
Day 1 field/owner inventory and Day 2 status/units/provenance matrix from
the accepted Sprint 215 Day 6-8 design. No Day 1 or Day 2 external run was
performed or inferred.

Changes: source-controlled v1 JSON schema, frozen identity layout and
synthetic canonical byte/hash vector, stdlib-only validator entry point and
focused identity/interface tests. Schema authority and the freeze are
documented in `artifacts/day3-canonical-identity-and-schema-authority.md`.
The catch-up contract records are
`artifacts/day1-evidence-contract-inventory.md` and
`artifacts/day2-status-units-provenance-freeze.md`.

Owner map: corpus owner verifies selected fixture bytes and attribution;
solver owner captures two independent statuses, vectors and metrics; report
owner implements semantic schema/identity/artifact validation; benchmark
owner owns any later Day 5 timing evidence; publication reviewer owns
license/attribution approval. No owner has asserted a new external result.

Open validation risks for Day 4: implement exact schema traversal without a
third-party package; enforce null-with-reason and status contradictions;
verify generated Git/build/dependency provenance and artifact byte hashes;
check UTC calendar validity, raw numerical formulas and timing samples;
add a negative self-test that actually exercises these rules. The pilot
preflight remains intentionally blocked by `implementation_pending` until
that work is complete. The observed-result boundary remains **none**.

Day 3 checks: `python3 tests/test_validate_external_baseline_evidence.py`
(six tests), `python3 tests/test_run_cholmod_pilot.py` (13 tests),
`python3 -m json.tool tests/corpus/schemas/external_baseline_evidence_v1.json`,
`python3 -m py_compile` on the new script and test,
`python3 scripts/validate_corpus_schema.py`, and `make docs-check` all passed.
No C/header files changed, so `make format && make lint && make test` was not
required for this day.

## Day 4: Strict JSON And Structural Validation (2026-10-28)

Implemented pointer-aware strict JSON decoding and recursive v1 schema shape
checking in the stdlib-only validator. Added focused positive planning-shape
and negative parser/selection/shape mutations with stable code/pointer
assertions; see `artifacts/day4-strict-json-and-structural-validation.md`.
No generated record is accepted: the CLI and pilot preflight remain closed
with `implementation_pending` until the semantic checks and genuine negative
self-test exist. No CHOLMOD solve or timing run was performed.

Day 4 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(10 tests), `python3 tests/test_run_cholmod_pilot.py` (13 tests),
`python3 scripts/validate_corpus_schema.py`, `python3 -m py_compile` for the
validator and test, and `make docs-check` all passed. A direct
`--schema ... --self-test` call exited 1 with `implementation_pending /`,
confirming the pilot gate remains closed. No `.c` or `.h` file changed, so
the C format/lint/test sequence was not required.

## Day 5: Semantic Status And Cell Validation (2026-10-28)

Implemented the v1 status/side/timing matrix, observed provenance and
canonical cell-ID recomputation, Day 4 metric and cross-solver checks,
in-memory paired timing aggregate validation, claim-scope limits and
separate publication review constraints. Synthetic positive and mutation
tests now cover all v1 status classes and reject unsupported failure labels,
contradictory identities, metrics, sample counts and claims. See
`artifacts/day5-semantic-status-and-cell-validation.md`. No actual CHOLMOD
run or timing result was produced; synthetic placeholder hashes are not
accepted evidence.

Day 5 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(20 tests), `python3 tests/test_run_cholmod_pilot.py` (13 tests),
`python3 scripts/validate_corpus_schema.py`, `python3 -m py_compile` for the
validator and test, and `make docs-check` passed. The CLI `--self-test`
remains intentionally closed with `implementation_pending` until Day 6
artifact-byte checks and later negative self-test coverage are complete.
No `.c` or `.h` files changed, so `make format && make lint && make test`
was not required.

## Day 6: Safe Artifacts And Fixture Harness (2026-10-28)

Implemented safe, descriptor-relative byte hashing for repository matrix and
frozen methodology inputs plus declared bundle artifacts; rejects unsafe
paths, symlink escapes, missing/non-regular/inaccessible files, duplicate
artifact paths and digest mismatches. Measured timing samples are decoded
only from verified raw bytes. Added temporary-bundle fixture helpers and an
integrated positive planning-blueprint assertion; see
`artifacts/day6-safe-artifacts-and-fixture-harness.md`. This closes the
first-pass item 221.2 validator, not the release gate. The pilot CLI remains
fail-closed pending the later negative self-test, and no actual CHOLMOD
result or timing was produced.

Day 6 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(29 tests), `python3 tests/test_run_cholmod_pilot.py` (13 tests),
`python3 scripts/validate_corpus_schema.py`, `python3 -m py_compile` for the
validator and test, and `make docs-check` passed. A direct CLI `--self-test`
exited 1 with `implementation_pending`. No `.c` or `.h` file changed, so
the C format/lint/test sequence was not required.

## Day 7: Planning And Blocked Fixtures (2026-10-28)

Froze the Sprint 215 blueprint as the source-controlled v1 planning fixture
and added synthetic generated `not_run`, execution-blocked and separately
hashed publication-blocked cases. Added non-promotion mutations for missing
reasons, stray numerical/timing values, unearned claims and baseline pass
labels without raw evidence. Missing CHOLMOD runtime, module review or
loaded-binary SHA-256 cannot pass baseline provenance. See
`artifacts/day7-planning-and-blocked-fixtures.md`. No external solve or
timing run was performed, and the pilot remains fail-closed.

Day 7 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(36 tests), `python3 tests/test_run_cholmod_pilot.py` (13 tests),
`python3 scripts/validate_corpus_schema.py`, `python3 -m py_compile` for the
validator and test, and `make docs-check` passed. No `.c` or `.h` file
changed, so the C format/lint/test sequence was not required.

## Day 8: Synthetic Observed And Measured Fixtures (2026-10-28)

Expanded the canonical synthetic two-sided correctness builder and temporary
bundle test to compare frozen methodology hashes, cell identity, both raw
solver-side outputs and Day 4 numerical gates. The synthetic measured builder
now covers all three planned timing windows with five warmups and 50
alternating-order timed pairs each, plus monotonic clock and observed
single-thread provenance. The validator binds hashed solver-side raw JSON to
reported statuses, controls and solutions, and enforces the measured clock
and thread controls. Paired tests reject one-sided or missing raw output,
altered values, invalid timing aggregates/counts and attempted winner claims.
See `artifacts/day8-synthetic-observed-and-measured-fixtures.md`. This is
test-only data, not a CHOLMOD observation; the pilot remains fail-closed.

Day 8 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(38 tests), `python3 tests/test_run_cholmod_pilot.py`,
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation and
`make docs-check` passed. No `.c` or `.h` file changed, so the C
format/lint/test sequence was not required.

## Day 9: Fixture Closure And Pilot Gate Design (2026-10-28)

Closed the integrated synthetic fixture matrix across all eight v1 statuses.
Added byte-backed `correctness_failed` and `timing_invalid` cases, a
`non_comparable` case, and negative neighbors for missing raw files and a
failure label without an observed failed gate. The coverage ledger maps all
named rule codes and required field groups to tests, identifies unsupported
shapes and records the current pilot command order and earliest Day 10
pre-solve stop point. See `artifacts/day9-fixture-closure-and-pilot-gate.md`.
The self-test CLI remains intentionally fail-closed; no CHOLMOD execution or
external result is inferred. Item 221.3's fixture allocation is closed.

Day 9 verification: `python3 tests/test_validate_external_baseline_evidence.py`
(40 tests), `python3 tests/test_run_cholmod_pilot.py`,
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation and
`make docs-check` passed. No `.c` or `.h` file changed, so the C
format/lint/test sequence was not required.

## Day 10: Fail-Closed Pilot Preflight (2026-10-28)

Replaced the validator CLI's unconditional `implementation_pending` result
with a real negative self-test in both CLI modes and
format/consistency-only success output.
Moved the pilot's schema/validator check and self-test to immediately after
the selected matrix hash, before dependency discovery, build or either solver.
Missing/unreadable inputs and failing self-tests now produce blocked attempts
with empty sides and no timing output. Preflight subprocess streams remain
byte-exact hashed artifacts, and existing attempt directories cannot be
overwritten. The later package, binary, status and numerical gates remain
fail-closed. See `artifacts/day10-fail-closed-pilot-preflight.md`. No CHOLMOD
solve or measured external evidence was produced.

Day 10 verification: validator `--self-test`,
`python3 tests/test_validate_external_baseline_evidence.py` (41 tests),
`python3 tests/test_run_cholmod_pilot.py` (15 tests),
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation and
`make docs-check` passed. No `.c` or `.h` file changed, so the C
format/lint/test sequence was not required.

## Day 11: Pilot Integration And Mutation Seeds (2026-10-28)

Added an integration test that runs the real passing validator self-test and
uses synthetic temporary files plus mocked downstream probes/build/solves to
prove gate order without CHOLMOD execution. Registered the stdlib-only
self-test, validator suite and pilot suite in the maintained Linux Python CI
lane. Added a temporary-copy CLI mutation harness with duplicate-key,
unknown-field, nonfinite-value and missing loaded-library-provenance seeds;
each asserts the exact rejection code/pointer and revalidates restored
input. See `artifacts/day11-pilot-integration-and-mutation-seeds.md`. Item
221.4 is closed; no actual external result or timing is claimed.

Day 11 verification: validator `--self-test`,
`python3 tests/test_validate_external_baseline_evidence.py` (43 tests),
`python3 tests/test_run_cholmod_pilot.py` (16 tests),
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation,
`python3 tests/test_selected_comparison_workflow.py`, Ruby YAML parsing of
`.github/workflows/ci.yml` and `make docs-check` passed. No `.c` or `.h` file
changed, so the C format/lint/test sequence was not required.

## Day 12: Semantic Mutation Proof (2026-10-28)

Added 16 syntax-valid CLI mutation cases for selected hashes, artifact bytes
and paths, canonical identity, solver control, dependency identity, status
relationships, timing units and paired samples, aggregate reporting, claims
and publication approval. Each case asserts its diagnostic code and pointer
and revalidates the restored positive fixture. All writes stay in temporary
bundles. See `artifacts/day12-semantic-mutation-proof.md`. No external solve,
timing result or publication approval was produced.

Day 12 verification: validator `--self-test` with the frozen schema,
`python3 tests/test_validate_external_baseline_evidence.py` (44 tests),
`python3 tests/test_run_cholmod_pilot.py` (16 tests),
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation and
`make docs-check` passed. No `.c` or `.h` file changed, so the C
format/lint/test sequence was not required.

## Day 13: Mutation Closure And Maintainer Guide (2026-10-28)

Added five CLI cross-field mutations for methodology bytes, raw-output
binding, blocked timing claims, duplicate dependencies and unlisted review
evidence. Together with Day 12's 16 syntax-valid cases and Day 11's parser
seeds, the mutation ledger now covers the selected v1 gates with exact rule
codes and pointers, positive fixture restoration and temp-only writes.
Published a [maintainer guide](artifacts/day13-validator-maintainer-guide.md)
for schema versioning, status/claim boundaries, canonical identity bytes,
CLI diagnostics, raw retention and independent publication review; linked it
from the README. See `artifacts/day13-mutation-closure-ledger.md`. Validator
success remains format/consistency-only; Sprint 222 still owns the first
real two-sided CHOLMOD correctness cell. Item 221.5 is closed.

Day 13 verification: validator `--self-test`, direct planning-fixture CLI
validation, `python3 tests/test_validate_external_baseline_evidence.py`
(45 tests), `python3 tests/test_run_cholmod_pilot.py` (16 tests), focused
non-execution (7), artifact (13) and CLI mutation (4 methods) suites,
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation and
`make docs-check` passed. No `.c` or `.h` file changed, so the C
format/lint/test sequence was not required.

## Day 14: Integrated Validation And Handoff (2026-10-28)

Re-ran the exact optional Python CI step and direct planning CLI, corpus,
selected-comparison/manifest/report-index/performance-doc regressions,
`make docs-check`, API-doc local-only routing with pinned PyYAML and Python
byte-compilation. All passed. Git status and ignore-rule checks show no
selected manifest, normalized report, canonical benchmark row, generated-API
publication route or external-baseline attempt output was added. The
[validation ledger and Sprint 222 handoff](artifacts/day14-integrated-validation-and-sprint222-handoff.md)
reconcile items 221.1-221.6 as complete while keeping real CHOLMOD execution,
loaded dependency provenance, repeat validation and independent rights review
open for Sprint 222. No `.c` or `.h` file changed this sprint, so the
conditional C format/lint/test gate did not apply. The optional pilot may
proceed only after the v1 self-test passes; neither that preflight nor a
synthetic validator success establishes solver parity or performance.
