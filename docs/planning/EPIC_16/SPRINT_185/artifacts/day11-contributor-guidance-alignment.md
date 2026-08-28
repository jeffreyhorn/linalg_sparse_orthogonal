# Sprint 185 Day 11: Contributor Guidance Alignment

## Purpose

Promote the Day 10 LDLT CSC helper-layout maintenance draft into the
maintainer-facing documentation surface and record the focused validation.

## Documentation Updated

| File | Day 11 update |
| --- | --- |
| `docs/maintainer_guide.md` | Added Sprint 185 LDLT CSC helper-header ownership and validation guidance to the existing test fixture/helper ownership section. |
| `docs/planning/EPIC_16/SPRINT_185/WORKING_NOTES.md` | Recorded Day 11 documentation alignment, non-claims, validation, and Day 12 handoff. |

No test-local README was added. The existing maintainer guide is already the
repository-wide surface for proof ownership, quality-contract interpretation,
and helper ownership policy.

## Guidance Added

The maintainer guide now records that:

- `tests/test_ldlt_csc_fixtures.h` owns LDLT CSC family-local KKT,
  scaled-KKT, and analysis-backed two-pass fixture/setup helpers;
- `tests/test_ldlt_csc_oracle_helpers.h` owns LDLT CSC family-local dense
  oracles, symmetric-swap helpers, and native-wrapper comparison helpers;
- `tests/test_ldlt_csc_supernode_helpers.h` owns LDLT CSC family-local
  supernode fixtures, snapshots, dense-SPD setup, and factor-state comparison
  helpers;
- `tests/test_ldlt_csc.c` remains the registered LDLT CSC proof-owner binary;
- `main`, `RUN_TEST(...)` ordering, public test bodies, test names, fixture
  values, numerical tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` ownership stay in
  `tests/test_ldlt_csc.c`;
- external-process dense-reference state and platform skip behavior remain in
  `tests/test_ldlt_csc.c` unless a later boundary review approves extraction;
- `make ldlt-csc-helper-guard` is the maintained selected-cluster guard after
  helper-layout changes;
- the Day 10 artifact remains the detailed provenance record for helper
  placement rules.

## Stale Comment Review

Reviewed the current LDLT CSC proof-owner file and helper headers:

- `tests/test_ldlt_csc.c`
- `tests/test_ldlt_csc_fixtures.h`
- `tests/test_ldlt_csc_oracle_helpers.h`
- `tests/test_ldlt_csc_supernode_helpers.h`

No stale comments conflicting with the extracted layout were found. The
helper-header comments already describe their family-local ownership and the
existing proof-owner binary.

## Non-Claims

The Day 11 documentation alignment does not claim new solver behavior,
broader numerical coverage, performance improvement, package/platform support,
ABI stability, or external-library parity. It documents review-surface and
helper-ownership policy only.

## Validation

Validation completed:

```sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Results:

- `make ldlt-csc-helper-guard`: passed.
- `make source-list-check`: PASS, 49 library sources.
- `git diff --check`: passed.

Day 11 changed documentation/planning files only. No `.c` or `.h` files were
modified for this day, so the full C gate was not required.

## Day 12 Handoff

- Run focused selected-cluster validation.
- Include `make ldlt-csc-helper-guard` and `make source-list-check`.
- Review the accumulated Sprint 185 diff for accidental solver behavior,
  fixture, tolerance, or scope changes before the Day 13 full gate.
