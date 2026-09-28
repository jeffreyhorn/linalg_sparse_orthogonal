# Sprint 211 Day 4: Extraction Design

## Summary

Day 4 turns the Day 3 LDLT CSC native-parity boundary into an
implementation-ready extraction design. The design creates one new
family-local header for the selected native parity test bodies while keeping
`tests/test_ldlt_csc.c` as the only proof-owner executable and registration
owner.

No code is moved on Day 4. Day 5 must capture baseline behavior and metrics
before implementation begins.

## Chosen File Shape

| Field | Design |
| --- | --- |
| New owner file | `tests/test_ldlt_csc_native_parity_helpers.h`. |
| Include guard | `TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H`. |
| Linkage | Header-only `static` test functions included into `test_ldlt_csc.c`. |
| Registration owner | `tests/test_ldlt_csc.c` keeps all `RUN_TEST(...)` calls. |
| Build registration | No new Makefile, CMake, CTest, or library source registration. |
| Guard owner | `scripts/check_ldlt_csc_helper_guard.sh`. |

This matches the existing LDLT CSC helper-header pattern while giving the
selected native parity tests a distinct owner from the dense-oracle comparison
helpers.

## Planned Include Direction

`tests/test_ldlt_csc.c` should include the new helper between the fixture and
oracle helpers. The helper itself includes `test_ldlt_csc_oracle_helpers.h`
for native/wrapper comparison helpers:

```c
#include "test_ldlt_csc_fixtures.h"
#include "test_ldlt_csc_native_parity_helpers.h"
#include "test_ldlt_csc_oracle_helpers.h"
#include "test_ldlt_csc_supernode_helpers.h"
```

The new helper may depend on declarations already available to
`tests/test_ldlt_csc.c`, but it should include `<math.h>` itself if it owns the
inertia comparison that calls `fabs()`.

## Dependency Map

| Dependency | Owner after extraction | Rule |
| --- | --- | --- |
| Selected native 1x1 and 2x2 test bodies | `tests/test_ldlt_csc_native_parity_helpers.h` | Move without changing function names or assertion bodies. |
| `check_native_matches_wrapper()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `ldlt_factorizations_match()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `ldlt_column_nonzeros_match()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `rel_residual()` | `tests/test_ldlt_csc.c` unless Day 6 records a safer helper-local approach | Do not move blindly because many Day 9 solve tests also use it. |
| `RUN_TEST(...)` registrations | `tests/test_ldlt_csc.c` | Preserve names and order. |
| Sparse fixture data inside selected tests | New helper header | Preserve dimensions, values, symmetry inserts, and insertion order exactly. |
| Kernel override state | Selected tests and oracle helper | Preserve native/wrapper/default sequencing exactly. |

## Batch Plan

### Batch One: Native 1x1 Parity

Move the Sprint 18 Day 3 block:

- `test_native_1x1_diagonal_matches_wrapper`;
- `test_native_1x1_tridiagonal_matches_wrapper`;
- `test_native_1x1_mixed_indefinite_matches_wrapper`;
- `test_native_1x1_with_swap_matches_wrapper`;
- `test_native_1x1_tridiag_large_matches_wrapper`;
- `test_native_detects_near_zero_1x1_pivot`;
- `test_native_1x1_identity_matches_wrapper`.

### Batch Two: Native 2x2 And Solve Parity

Move the Sprint 18 Day 4 block:

- `test_native_2x2_forced_matches_wrapper`;
- `test_native_2x2_nonadjacent_partner_matches_wrapper`;
- `test_native_mixed_pivots_matches_wrapper`;
- `test_native_mixed_pivots_larger_matches_wrapper`;
- `test_native_2x2_solve_matches_linked_list`;
- `test_native_2x2_inertia_matches_wrapper`.

## Source-List Plan

Do not add the new helper to:

- `TEST_SRCS`;
- `TEST_BINS`;
- `CMakeLists.txt`;
- CTest registration;
- `build-metadata/library_sources.txt`;
- public headers or install metadata.

If Makefile prerequisite updates become necessary, they must attach only to the
existing `test_ldlt_csc` proof-owner target and must be covered by the LDLT CSC
helper guard.

The final implementation does require that explicit Makefile prerequisite so
helper-only edits rebuild `build/test_ldlt_csc`; the helper must still not be
listed in `TEST_SRCS`, `TEST_BINS`, CMake/CTest registration, or library
source metadata.

## Guard Strategy

When the new helper is created, update `scripts/check_ldlt_csc_helper_guard.sh`
to enforce:

| Guard | Requirement |
| --- | --- |
| Helper presence | `tests/test_ldlt_csc_native_parity_helpers.h` exists with the expected include guard. |
| Include ownership | `tests/test_ldlt_csc.c` includes the new helper exactly once. |
| Header-only status | The helper is absent from Makefile, CMake, CTest registration, and `build-metadata/library_sources.txt`. |
| Proof-owner registration | `test_ldlt_csc` remains registered in Makefile and CMake. |
| Selected active registrations | The selected native parity `RUN_TEST(...)` lines remain active in `tests/test_ldlt_csc.c`. |
| Registration order | Sprint 18 Day 3 1x1 registrations remain before Sprint 18 Day 4 2x2 registrations, and both remain before Day 9 solve registrations. |
| Comment safety | Registration checks ignore line comments and block comments instead of counting raw substrings. |

## Behavior Preservation Rules

Implementation must not change:

- test names;
- registration order;
- fixture dimensions, values, symmetric insertion pairs, or insertion order;
- numerical tolerances;
- expected status codes;
- residual comparisons;
- kernel override reset behavior;
- stdout/stderr behavior;
- CTest count;
- public API, ABI, install metadata, package metadata, workflows, benchmarks,
  or generated artifact formats.

## Validation Plan

Day 5 baseline should capture:

```sh
make ldlt-csc-helper-guard
```

and the smallest available focused `test_ldlt_csc` execution path. After any
`.c` or `.h` implementation edits, the required closeout gate is:

```sh
make format && make lint && make test
```

## Item 211.3 Evidence

Item 211.3 is complete. The extraction design is ready for implementation:
owner file, include direction, dependency ownership, source-list behavior,
guard updates, and validation expectations are documented before code movement.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

`git diff --check` is the Day 4 validation command.
