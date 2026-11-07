# Sprint 221 Day 11: Pilot Integration And Mutation Seeds

## Pilot Gate Integration

`tests/test_run_cholmod_pilot.py` now exercises a passing **real** v1
`--self-test` followed by mocked package probes, build, helper compilation
and project/baseline solver outputs. The test uses synthetic files in a
temporary root. It proves the command order begins with `schema_preflight`,
then reaches the existing version, binary, build and solver gates; both raw
side outputs pass the pilot's status and Day 4 numerical checks. The result
remains `two_sided_correctness_candidate` with `timing=not_run` and
`publication_review=pending`, not a validated external observation. No
CHOLMOD library was installed or executed.

The Linux supplemental Python CI lane now runs the validator self-test,
validator tests and pilot tests before its C runtime work. These are
stdlib-only checks of local fixtures and mocked pilot behavior. They neither
require CHOLMOD nor create a generated evidence bundle. A failed negative
self-test exits nonzero in CI and blocks the pilot before build or solve.
This closes item 221.4's preflight allocation.

## Mutation Harness

`CliMutationTests` copies the valid planning fixture into a temporary
directory for each case, checks that the copy validates, mutates one field
or byte sequence, invokes the CLI, and asserts a nonzero exit with the
expected code and JSON pointer. Restoring the original bytes revalidates.
The observed-provenance case stages only synthetic hashed raw artifacts in
a separate temporary bundle and likewise restores the original record.

| Seed | Mutation | Required rejection |
| --- | --- | --- |
| Duplicate key | Add a second `record_id` JSON member | `duplicate_or_unknown_field /record_id` |
| Unknown field | Add root `synthetic_unknown` | `duplicate_or_unknown_field /synthetic_unknown` |
| Nonfinite value | Insert `NaN` in timing median | `nonfinite_or_missing_value_abuse /timing/median_ns` |
| Missing execution provenance | Clear loaded CHOLMOD library SHA-256 in a synthetic observed record | `missing_provenance /cell/baseline/loaded_library_sha256` |

These seeds establish the temporary-copy/CLI-verdict method for Day 12's
broader hash, path, cell-ID, status, units and claim mutations. No selected
manifest, source corpus or canonical benchmark output is modified.
