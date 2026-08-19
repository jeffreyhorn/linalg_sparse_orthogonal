# Sprint 170 Day 11: Guard Implementation

## Purpose

Day 11 implements the guard updates scoped by Day 10. The implementation keeps
the Sprint 170 product decision unchanged: supported package behavior remains
static-first-only, while shared-library packaging and dynamic ABI compatibility
remain unsupported and deferred.

## Implemented Changes

| File | Change |
| --- | --- |
| `scripts/static_package_deferral_check.sh` | Added a Sprint 170 decision-record guard that requires the Day 9 product decision artifact and selected static-first/non-claim wording. |
| `scripts/static_package_deferral_check.sh` | Added a Makefile static archive contract guard requiring `libsparse_lu_ortho.a`, install of `$(LIB)` to `$(INSTALL_LIB)`, uninstall removal of the installed archive, and no shared-library install behavior. |
| `scripts/static_package_deferral_check.sh` | Tightened `sparse.pc.in` source metadata checks for exact static archive description, installed include flags, and static archive link flags. |

## Guard Behavior

The static package deferral guard now fails if:

- the Sprint 170 Day 9 product decision record is missing;
- the decision record no longer selects the static-first-only package posture;
- the decision record drops the explicit shared-library and dynamic ABI
  deferral;
- the decision record stops naming `BUILD_SHARED_LIBS=ON`,
  `Sparse::sparse_lu_ortho`, or Windows `pkg-config` execution parity;
- the Makefile stops defining the maintained library as
  `libsparse_lu_ortho.a`;
- the Makefile stops installing or uninstalling the static archive through the
  maintained install paths;
- the Makefile gains shared-library install behavior before a shared ABI
  product decision exists;
- `sparse.pc.in` drifts away from the exact static archive metadata selected by
  the package contract.

## Preserved Behavior

No package behavior was changed:

- CMake still rejects `BUILD_SHARED_LIBS=ON`.
- CMake still builds and exports `sparse_lu_ortho` as an explicit static
  target.
- Make install still installs the static archive, headers, generated
  `sparse_version.h`, and `sparse.pc`.
- CMake install still exports archive-only package metadata.
- Windows package validation remains CMake-first with metadata-only
  `sparse.pc` inspection.

## Validation

Focused validation was run after the guard implementation:

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
git diff --check
```

No `.c` or `.h` files were modified, so the full C quality gate
`make format && make lint && make test` was not required for Day 11.

## Day 11 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Implemented guard updates | Complete | Decision-record, Makefile static archive, and `sparse.pc.in` metadata guards were added. |
| Package metadata check updates | Complete | Exact `Description`, `Cflags`, and `Libs` expectations are enforced in source metadata. |
| Negative-check coverage | Complete | Makefile shared install behavior is now rejected before a shared ABI decision exists. |
| Focused validation log | Complete | Commands are listed in the validation section. |
| Day 11 guard-implementation artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Unsupported shared-library or ABI claims fail mechanically where feasible. | Complete | The deferral guard now protects the canonical decision record and Makefile static package path. |
| Package metadata matches the selected decision. | Complete | `sparse.pc.in` exact static archive metadata is checked directly. |
| Focused guard validation passes. | Complete | Focused commands passed locally. |
