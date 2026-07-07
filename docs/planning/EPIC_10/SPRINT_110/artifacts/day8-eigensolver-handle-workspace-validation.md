# Sprint 110 Day 8 Eigensolver Handle/Workspace Validation

## Purpose

Day 8 validates the Day 7 selected eigensolver behavior owner: the public
handle/workspace bridge. The result is a no-move contract for Sprint 110 plus
focused Make and CMake validation evidence.

## Selected Owner

The selected owner is the public handle/workspace bridge around:

- `sparse_eigs_handle_init`;
- `sparse_eigs_handle_free`;
- `sparse_eigs_handle_prepare`;
- `sparse_eigs_sym_with_handle`;
- `s49_eigs_handle_ensure`;
- `s49_eigs_handle_prepare_backend`;
- grow-m workspace preparation;
- thick-restart workspace preparation;
- LOBPCG workspace preparation.

## No-Move Contract

No eigensolver source moved on Day 8.

The handle/workspace bridge remains behavior-sensitive because it touches:

- public handle lifetime;
- caller-visible prepare semantics;
- solve-without-prepare on-demand allocation;
- repeated solve reuse;
- later-call workspace growth;
- backend-specific workspace sizing;
- invalid option and shape error behavior;
- cleanup idempotence and public handle zeroing.

Moving this owner would require direct proof that each behavior above remains
unchanged across grow-m, thick-restart, and LOBPCG paths. Day 8 validates the
current owner rather than creating another source boundary.

## Direct Test Coverage

`tests/test_eigs.c` directly covers the selected owner through:

- `test_public_handle_growm_prepare_reuse_and_growth`;
- `test_public_handle_prepare_and_reuse`;
- `test_public_handle_validation_and_on_demand`;
- `test_public_handle_thick_restart_prepare_reuse_and_growth`;
- `test_public_handle_lobpcg_prepare_reuse_and_growth`.

Cross-backend and integration coverage is provided by:

- `tests/test_eigs_thick_restart.c`;
- `tests/test_eigs_lobpcg.c`;
- `tests/test_sprint29_integration.c`.

## Make Validation

Focused Make build passed:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
```

Focused Make execution passed:

```sh
build/test_eigs
build/test_eigs_thick_restart
build/test_eigs_lobpcg
build/test_sprint29_integration
```

Observed results:

- `test_eigs`: 31 tests passed.
- `test_eigs_thick_restart`: 21 tests passed.
- `test_eigs_lobpcg`: 27 tests passed.
- `test_sprint29_integration`: 3 tests passed.

## CMake And CTest Validation

CTest registration no-drift check:

```sh
ctest -N --test-dir build/quality-review-cmake
```

Result:

- total registered CTest tests: 54.

Focused CTest execution:

```sh
ctest --test-dir build/quality-review-cmake --output-on-failure -R '^(test_eigs|test_eigs_thick_restart|test_eigs_lobpcg|test_sprint29_integration)$'
```

Result:

- 4 of 4 tests passed.

## Source-List And Public-Surface Review

`make source-list-check` passed with 48 library sources.

Day 8 introduced no changes to:

- `include/sparse_eigs.h`;
- other public headers;
- install/export rules;
- Makefile test registration;
- CMake test registration;
- helper targets;
- eigensolver source files;
- eigensolver private headers.

The only existing source/build-system diffs on the branch remain the earlier
Sprint 110 Matrix Market split.

## Deferred Movement

The following eigensolver movement remains deferred:

- defaults and option validation;
- backend dispatch;
- grow-m sizing and retry behavior;
- refinement defaults and budgets;
- shift-invert setup;
- shared Lanczos kernels;
- public handle/workspace source movement.

Any future movement must add direct owner-specific tests before source-list or
public-contract changes are made.

## Completion Status

- Selected eigensolver behavior remains externally unchanged.
- Focused eigensolver gates passed through Make and CTest.
- No public API or install-header drift occurred.
- Unsafe behavior-sensitive movement is explicitly deferred.
