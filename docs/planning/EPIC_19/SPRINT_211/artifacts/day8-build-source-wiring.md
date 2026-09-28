# Sprint 211 Day 8: Build And Source Wiring

## Summary

Day 8 reconciles build wiring after the selected LDLT CSC native-parity helper
extraction. The new helper remains header-only, but the Makefile proof-owner
rule now explicitly depends on the LDLT CSC helper headers, and the helper guard
now enforces that wiring.

## Changed Files

| File | Change |
| --- | --- |
| `Makefile` | Added an explicit `build/test_ldlt_csc` rule with helper-header prerequisites. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added the native parity helper to the guarded helper set, enforced Makefile helper prerequisites, and added active/order-aware selected registration checks. |
| `tests/test_ldlt_csc_helper_guard.py` | Added focused guard regression fixtures. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added Day 8 implementation notes. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day8-build-source-wiring.md` | Added this artifact. |

## Build Wiring

`test_ldlt_csc` now has an explicit Makefile rule before the generic test rule:

```make
$(BUILDDIR)/test_ldlt_csc: $(TESTDIR)/test_ldlt_csc.c $(TESTDIR)/test_ldlt_csc_fixtures.h $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h $(TESTDIR)/test_ldlt_csc_oracle_helpers.h $(TESTDIR)/test_ldlt_csc_supernode_helpers.h $(LIB) | $(BUILDDIR)
```

No standalone helper executable was added. CMake still registers only
`add_sparse_test(test_ldlt_csc)`.

## Guard Coverage

The LDLT CSC helper guard now checks:

| Guard surface | Day 8 coverage |
| --- | --- |
| Helper presence | Includes `tests/test_ldlt_csc_native_parity_helpers.h` in `HELPERS`. |
| Include ownership | Requires each helper include exactly once in `tests/test_ldlt_csc.c`. |
| Makefile wiring | Requires the `build/test_ldlt_csc` prerequisite rule to list each helper header. |
| Header-only boundary | Rejects helper CMake test registration and library-source registration. |
| Selected registrations | Requires active selected native parity `RUN_TEST(...)` lines in increasing order. |
| Comment bypasses | Strips line and block comments before active registration matching. |

## Regression Fixture

`tests/test_ldlt_csc_helper_guard.py` covers:

- current tree and minimal fixture pass;
- missing native helper include;
- missing native helper Makefile prerequisite;
- missing active registration;
- line-commented registration;
- block-commented registration;
- reordered selected registrations;
- accidental standalone CMake registration;
- accidental library-source registration.

## Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed. |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions. |

Day 8 changed Makefile, shell, Python, and planning artifacts. No additional
`.c` or `.h` edits were made on Day 8; the latest full C gate remains the Day 7
`make format && make lint && make test` pass.

## Item Evidence

Item 211.5 is complete for build/source wiring. The extracted helper is
reachable through the intended proof-owner build/test path, source-list parity
risks are guarded, and no standalone helper target or library source was
introduced.
