# Sprint 185 Day 10: Maintenance Invariants

## Purpose

Draft maintainer guidance for extending the Sprint 185 LDLT CSC test layout
without re-growing the selected proof-owner surface or drifting registration.

## Current File Layout

| File | Ownership |
| --- | --- |
| `tests/test_ldlt_csc.c` | Registered proof-owner test binary, public test bodies, `main`, `RUN_TEST(...)` ordering, external dense-reference state, and helpers that remain too broad or stateful to extract. |
| `tests/test_ldlt_csc_fixtures.h` | Family-local KKT, scaled-KKT, and analysis-backed two-pass fixture/setup helpers. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Family-local dense-oracle, symmetric-swap, and native-wrapper comparison helpers. |
| `tests/test_ldlt_csc_supernode_helpers.h` | Family-local supernode fixture, snapshot, dense-SPD, and factor-state comparison helpers. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Selected-cluster drift guard for helper-header presence, include ownership, and registration boundaries. |
| `Makefile` `ldlt-csc-helper-guard` | Maintained command entry point for the selected-cluster guard. |

## Maintenance Invariants

- Keep `test_ldlt_csc` as the only Sprint 185 LDLT CSC proof-owner binary
  unless a later reviewed artifact explicitly selects a new proof owner.
- Keep `main`, `RUN_TEST(...)` ordering, test names, numerical tolerances,
  fixture values, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` ownership in `tests/test_ldlt_csc.c`.
- Keep extracted helper headers family-local to `test_ldlt_csc.c`; do not use
  them as public, shared, or production helper APIs.
- Keep helper definitions `static` or `static inline` so the helpers retain
  internal linkage inside the existing test translation unit.
- Keep helper headers self-contained with include guards and only the includes
  their helpers require.
- Do not add the helper headers to `Makefile` test registration,
  `CMakeLists.txt` test registration, or
  `build-metadata/library_sources.txt`.
- Preserve Make/CMake registration parity if a future reviewed change adds a
  new test binary.
- Preserve Makefile, CMake, and `build-metadata/library_sources.txt` parity if
  a future reviewed change adds a new library `.c` source.

## Where New Code Belongs

| New contribution type | Preferred location |
| --- | --- |
| New public LDLT CSC proof case | `tests/test_ldlt_csc.c`, with a `RUN_TEST(...)` entry in the existing proof-owner binary. |
| New KKT, scaled-KKT, or analysis-backed setup fixture | `tests/test_ldlt_csc_fixtures.h` if it is reused by multiple LDLT CSC tests; otherwise keep it local near the test. |
| New dense expected-value oracle or native-wrapper comparison helper | `tests/test_ldlt_csc_oracle_helpers.h` if it is a reusable family-local assertion/oracle; otherwise keep it local near the test. |
| New supernode fixture, snapshot, or factor-state comparison helper | `tests/test_ldlt_csc_supernode_helpers.h` if it serves the supernode proof surface; otherwise keep it local near the test. |
| New external-process dense-reference state or platform skip behavior | Keep in `tests/test_ldlt_csc.c` unless a later boundary review proves extraction is lower risk. |
| New broad random-matrix or solve residual helper | Keep in `tests/test_ldlt_csc.c` until a later artifact proves a tighter helper boundary. |
| New proof-owner binary | Defer unless review cost, registration impact, and CMake parity validation are explicitly documented. |

## Contribution Workflow

1. Decide whether the new code is a test body, reusable helper, fixture, or
   registration change.
2. Prefer keeping one-off helpers near their test body. Move only reused or
   review-surface-reducing helpers into the family-local headers.
3. If a helper moves, preserve behavior, names where possible, tolerances,
   fixture values, and process-global state restoration.
4. Run the selected-cluster guard after any helper layout change:

   ```sh
   make ldlt-csc-helper-guard
   ```

5. Run source-list parity when registration or source-list concerns are in
   scope:

   ```sh
   make source-list-check
   ```

6. If any `.c` or `.h` file changes, run the full C gate after focused
   validation:

   ```sh
   make format
   if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
   make build/test_ldlt_csc
   ./build/test_ldlt_csc
   make ldlt-csc-helper-guard
   make source-list-check
   make lint
   make test
   ```

7. If only planning or maintainer documentation changes, run the relevant
   guards and `git diff --check`.

## Relevant Existing Documentation

- `docs/maintainer_guide.md` owns repository-wide quality-contract
  interpretation and should remain the long-term home for maintainer-facing
  policy if Day 11 promotes this draft.
- `docs/planning/EPIC_16/SPRINT_185/WORKING_NOTES.md` owns Sprint 185
  provenance, decisions, validation results, and handoffs.
- `docs/planning/EPIC_16/SPRINT_185/artifacts/day9-drift-guard-update.md`
  records the selected-cluster guard coverage and limitations.
- The helper headers themselves own short file-local comments describing their
  family-local scope.

## Non-Claims

This maintenance note does not claim new solver behavior, broader numerical
coverage, performance improvement, package/platform support, ABI stability, or
external-library parity. Sprint 185 reduces review surface only.

## Day 11 Handoff

- Decide whether to promote this draft into `docs/maintainer_guide.md`, a
  test-local README, or both.
- Cross-link the final note from existing maintainer or testing documentation.
- Keep the final note aligned with `make ldlt-csc-helper-guard`.
- Re-run guard and docs/whitespace checks after any documentation integration.
