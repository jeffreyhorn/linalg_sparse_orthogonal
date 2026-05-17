# Sprint 30 Day 10 Test Warning Triage

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Turn the current `tests/` warning backlog into an explicit cleanup queue by separating dormant scaffolding, positional option-struct drift, and incidental numeric-literal warnings.

## Inventory Summary

Using the Day 8 serialized Apple Clang CMake baseline as input:

- total test warnings: `98`
- warning-bearing test files: `21`

By warning class:

- `-Wmissing-field-initializers`: `62`
- `-Wdouble-promotion`: `33`
- `-Wunused-function`: `3`

Derived support files:

- `day10-test-warning-counts-by-file.txt`
- `day10-test-warning-counts-by-file-and-class.txt`

Top warning-bearing test files:

1. `tests/test_ldlt.c`: `18`
2. `tests/test_sprint20_integration.c`: `9`
3. `tests/test_chol_csc.c`: `8`
4. `tests/test_colamd.c`: `8`
5. `tests/test_reorder_nd.c`: `7`
6. `tests/test_sprint18_integration.c`: `6`
7. `tests/test_sprint6_integration.c`: `6`
8. `tests/test_svd.c`: `6`
9. `tests/test_cholesky.c`: `5`
10. `tests/test_sprint12_integration.c`: `5`

## Triage Categories

### Category 1: positional option-struct drift

Dominant signal:

- `62` warnings
- `-Wmissing-field-initializers`

Representative files:

- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_colamd.c`
- `tests/test_cholesky.c`
- `tests/test_reorder.c`
- `tests/test_reorder_nd.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`

Representative examples:

- `backend` omitted from evolving options structs
- `progress_cb` omitted from evolving options structs
- `used_csc_path` omitted from telemetry-bearing options structs

Interpretation:

- This is maintainability debt, not algorithmic breakage.
- The warnings track public and semi-public option structs that gained fields over time while tests kept positional initializers.
- The test suite still passes, but the warning volume shows the tests are brittle against further struct evolution and obscure which defaults are intentional.

Cleanup style implied by the warnings:

- move touched tests to designated initializers
- make intentionally-defaulted fields explicit
- keep backend/progress/telemetry expectations visible at the call site

### Category 2: dormant or non-executed test scaffolding

Signal:

- `3` warnings
- all `-Wunused-function`
- all in `tests/test_reorder_nd.c`

Affected functions:

- `test_finest_fm_annealing_pres_poisson_close_to_target`
- `test_nd_root_spectral_pres_poisson_close_to_target`
- `test_non_pipeline_pres_poisson_close_to_target`

Interpretation:

- This is the only current test-warning cluster that directly reflects test-honesty debt rather than initializer syntax.
- The file comments explicitly describe these as failing-as-expected or commented-out future checks.
- Leaving them as ordinary unused static functions inside the main test translation unit is misleading: the suite compiles them, warns about them, but never executes them.

Why this is structurally important:

- it weakens the line between active regression coverage and historical planning notes
- it encourages “documented but non-running” assertions to accumulate in the normal suite
- it aligns directly with the Epic 3 review finding about dormant ND scaffolding

### Category 3: incidental numeric-literal precision warnings

Signal:

- `33` warnings
- all `-Wdouble-promotion`

Representative files:

- `tests/test_sprint6_integration.c`: `6`
- `tests/test_svd.c`: `6`
- `tests/test_sprint20_integration.c`: `4`
- `tests/test_bidiag.c`: `3`
- `tests/test_sprint18_integration.c`: `3`
- smaller one- and two-site clusters across `test_qr.c`, `test_sprint10_integration.c`, `test_bicgstab.c`, `test_block_solvers.c`, `test_ilu.c`, `test_lu_csr.c`, and `test_sprint5_integration.c`

Interpretation:

- This is mostly fixture-constant and assertion-literal debt rather than hidden numerical behavior problems.
- It is lower priority than the initializer cluster because it does not hide evolving struct defaults or dormant coverage.
- It is also lower priority than the Day 9 `src/` strict-pass finding because it remains outside the core library.

Likely cleanup style:

- replace float-typed macros or literals used in `double` contexts
- use explicitly double-typed constants where the test already expects double precision
- keep the cleanup mechanical and localized

## First-Fix Subset For Future Test Cleanup

### Priority A: Sprint 32 test-honesty close

Start with:

- `tests/test_reorder_nd.c`

Reasons:

- it is the only file carrying dormant non-executed test bodies
- it also carries initializer-drift warnings, so one focused pass can clean both the warning debt and the misleading inactive assertions
- it is the strongest Day 10 link between warnings and coverage honesty

Expected outcome:

- remove, relocate, or formalize the dormant test bodies
- eliminate the `-Wunused-function` cluster
- convert surviving option structs in the same file to designated initializers

### Priority B: high-volume initializer wave

Next cleanup group:

- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_colamd.c`
- `tests/test_cholesky.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_reorder.c`

Reasons:

- these files carry the largest initializer-warning volume
- they exercise multiple backends and dispatch paths where explicit defaults improve readability
- they are the clearest maintainability payoff for designated-initializer conversion

### Priority C: mechanical double-promotion sweep

After the two higher-signal categories:

- `tests/test_sprint6_integration.c`
- `tests/test_svd.c`
- `tests/test_bidiag.c`
- `tests/test_sprint20_integration.c`
- remaining one- and two-warning files

Reason:

- broad but low-ambiguity cleanup
- useful once the higher-signal test-honesty and initializer-drift debt is under control

## Broader Test-Structure Concerns

Day 10’s warning triage points to two non-feature problems in the existing test suite:

1. Test coverage honesty is weaker where “future intended checks” live as inactive functions inside normal test translation units.
2. Test maintainability is weaker where positional options-struct initialization hides whether omitted fields are intentional defaults or accidental drift after API evolution.

Neither issue means the tests are currently failing. The problem is that both make the suite harder to trust and harder to evolve cleanly.

## Day 10 Conclusion

Test warnings are no longer one undifferentiated backlog:

- the largest cluster is options-struct initializer drift
- the most structurally important cluster is dormant ND scaffolding
- the remaining broad cluster is mechanical double-promotion cleanup

That split gives later Epic 3 sprints a realistic queue instead of a generic “clean up test warnings” goal.

## Validation

End-of-day verification passed:

- `make format`
- `make lint`
- `make test`
