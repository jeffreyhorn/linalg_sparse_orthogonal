# Sprint 211 Day 3: Cluster Boundary

## Summary

Day 3 defines the selected Sprint 211 extraction boundary before code edits.
The selected cluster is the Sprint 18 Day 3/Day 4 native-kernel parity block in
`tests/test_ldlt_csc.c`. The proof-owner binary remains `test_ldlt_csc`; the
planned reduction is a behavior-preserving helper extraction, not a production
LDLT CSC refactor.

## Selected Boundary

| Boundary field | Decision |
| --- | --- |
| Proof owner | `tests/test_ldlt_csc.c`. |
| Build owner | Existing `Makefile` and `CMakeLists.txt` `test_ldlt_csc` registration. |
| Selected cluster | Sprint 18 Day 3/Day 4 native 1x1 and 2x2 Bunch-Kaufman parity tests. |
| Primary helper destination | New selected family-local header, tentatively `tests/test_ldlt_csc_native_parity_helpers.h`, pending Day 4 design. |
| Existing helper dependency | `tests/test_ldlt_csc_oracle_helpers.h` for `check_native_matches_wrapper()`, factorization comparison, and dense-oracle utilities. |
| Guard owner | `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard`. |
| Non-goal | Moving production LDLT CSC code, changing solver behavior, changing public API/ABI, or broadening support/performance/platform claims. |

## Owned Tests

The selected cluster owns these test bodies and their current behavior:

| Test | Current purpose |
| --- | --- |
| `test_native_1x1_diagonal_matches_wrapper` | Pure diagonal indefinite 1x1 native/wrapper parity. |
| `test_native_1x1_tridiagonal_matches_wrapper` | Tridiagonal SPD 1x1 column-loop parity. |
| `test_native_1x1_mixed_indefinite_matches_wrapper` | Mixed indefinite 1x1 parity with weak off-diagonals. |
| `test_native_1x1_with_swap_matches_wrapper` | Criterion-3 1x1 pivot with symmetric row swap. |
| `test_native_1x1_tridiag_large_matches_wrapper` | Larger deterministic tridiagonal 1x1 stress fixture. |
| `test_native_detects_near_zero_1x1_pivot` | Native singular detection for near-zero 1x1 pivot. |
| `test_native_1x1_identity_matches_wrapper` | Identity native/wrapper trivial path parity. |
| `test_native_2x2_forced_matches_wrapper` | Forced 2x2 Bunch-Kaufman block parity. |
| `test_native_2x2_nonadjacent_partner_matches_wrapper` | Non-adjacent 2x2 partner swap parity. |
| `test_native_mixed_pivots_matches_wrapper` | Mixed 1x1/2x2 cmod cross-term parity. |
| `test_native_mixed_pivots_larger_matches_wrapper` | Larger mixed-pivot parity fixture. |
| `test_native_2x2_solve_matches_linked_list` | Native factor plus solve residual check against original matrix. |
| `test_native_2x2_inertia_matches_wrapper` | Native/wrapper D and D_offdiag parity for inertia evidence. |

Their `RUN_TEST(...)` registrations remain in `tests/test_ldlt_csc.c` and must
stay in the current relative order.

## Dependency Trace

| Dependency | Current owner | Day 3 boundary |
| --- | --- | --- |
| `check_native_matches_wrapper()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `ldlt_factorizations_match()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `ldlt_column_nonzeros_match()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse unchanged. |
| `rel_residual()` | `tests/test_ldlt_csc.c` Day 9 solve block | Keep in proof-owner unless Day 4 explicitly designs a solve-helper owner. |
| `ldlt_csc_set_kernel_override()` calls | selected native tests and oracle helper | Preserve native/wrapper/default sequencing exactly. |
| Sparse fixture construction | selected native tests | May move with test bodies; matrix entries and insertion order must not change. |
| Test registration | `tests/test_ldlt_csc.c` main | Preserve exact registration names and order. |

## No-Behavior-Change Invariants

Later implementation must preserve:

- selected test names;
- selected `RUN_TEST(...)` registrations and relative order;
- all fixture matrix sizes, values, symmetry insertions, and insertion order;
- all tolerances, especially `1e-12` native/wrapper and residual comparisons;
- expected statuses, including `SPARSE_ERR_SINGULAR` for the near-zero 1x1
  pivot test;
- kernel override reset to `LDLT_CSC_KERNEL_DEFAULT` after every selected
  native or wrapper override;
- `ldlt_csc_validate()` checks inside native/wrapper comparison flow;
- existing stdout/stderr behavior and absence of new selected-test diagnostics;
- CTest test count unless a later artifact records and guards a deliberate
  registration change;
- helper headers remaining header-only and absent from Makefile, CMake test
  registration, and `build-metadata/library_sources.txt`.

## Owner Files

Allowed owner files for this selected reduction:

- `tests/test_ldlt_csc.c` as proof-owner, include owner, and registration
  owner;
- `tests/test_ldlt_csc_oracle_helpers.h` as existing dense-oracle comparison
  owner;
- a possible new `tests/test_ldlt_csc_native_parity_helpers.h` for selected
  native parity test bodies;
- `scripts/check_ldlt_csc_helper_guard.sh` for helper ownership and
  registration guards;
- planning artifacts under `docs/planning/EPIC_19/SPRINT_211/`.

## Non-Owner Files

These files and surfaces must not take ownership of Day 3's selected cluster:

- production sources such as `src/sparse_ldlt_csc.c`;
- public headers under `include/`;
- installed/package metadata;
- `tests/test_ldlt.c` and the Sprint 210 allocation-failure gate;
- `tests/test_svd.c` and the Sprint 201 SVD helper guard;
- `tests/test_chol_csc.c`, unless the Day 2 fallback is explicitly activated;
- unrelated Python report tooling, generated artifacts, benchmark manifests,
  or platform workflow evidence.

## Guard Implications

The existing LDLT CSC helper guard already proves:

- `tests/test_ldlt_csc.c` remains registered in Make and CMake;
- known helper headers are included exactly once by `tests/test_ldlt_csc.c`;
- helper headers remain header-only and absent from library source metadata;
- helper headers do not become independent CMake tests without a new
  proof-owner decision.

Day 4 should decide whether to extend that guard with:

- the new selected native-parity helper header if created;
- active, uncommented registration checks for the selected native tests;
- registration-order checks for the Day 3/Day 4 native parity block;
- a prohibition against registering the selected helper as a standalone test
  or library source.

## Validation Expectations

Day 3 does not modify code. Later implementation days should plan:

| Validation | Trigger |
| --- | --- |
| `make ldlt-csc-helper-guard` | Any helper header or guard change. |
| Focused `test_ldlt_csc` execution | Any selected test-body/helper move. |
| `ctest -N` or CMake parity checks | Any CMake/test registration touch. |
| `make format && make lint && make test` | Any `.c` or `.h` modification. |

## Item 211.2 Evidence

Item 211.2 is complete: the selected extraction boundary, owner model,
non-owner surfaces, observable behavior invariants, and guard expectations are
defined before implementation. The selected cluster is narrow enough for a
complete Sprint 211 closure and intentionally avoids production behavior,
public API, ABI, package, platform, performance, release, external-library
parity, and state-of-the-art claims.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

`git diff --check` is the Day 3 validation command.
