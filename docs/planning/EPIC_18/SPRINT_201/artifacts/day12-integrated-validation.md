# Sprint 201 Day 12 Integrated Validation

## Scope

Day 12 performed the integrated local validation pass for the selected SVD
helper review-surface reduction. The validation covers focused selected-cluster
behavior, ownership guards, source/CMake/doc parity, and the required full C
quality gate for the branch's `.c`/`.h` changes.

## Focused Regression Checks

```sh
make build/test_svd && ./build/test_svd
```

Result: passed.

- `make`: `build/test_svd` was already up to date.
- `./build/test_svd`: 114 tests run, 0 failed, 0 skipped, 2067 assertions.
- Final summary: `ALL TESTS PASSED`.

```sh
make svd-helper-guard
```

Result: passed.

- `svd-helper-guard: required files ok`
- `svd-helper-guard: proof-owner registration ok`
- `svd-helper-guard: helper boundary ok`
- `svd-helper-guard: selected cluster ownership ok`
- `svd-helper-guard: header-only registration ok`
- `svd-helper-guard: passed`

```sh
python3 tests/test_svd_helper_guard.py
```

Result: passed.

## Ownership, Source, CMake, and Docs Checks

```sh
make source-list-check
```

Result: passed.

- `source-list-check: PASS (49 library sources)`

```sh
cmake -S . -B build/sprint201-day12-validation-check
```

Result: passed.

- C compiler detected: AppleClang 17.0.0.17000604.
- Configure and generate completed.
- Build files written under ignored `build/sprint201-day12-validation-check`.

```sh
make docs-check
```

Result: passed.

- Doxygen generated the API docs under ignored `docs/api/html`.
- `api-docs-coverage: PASS`.
- Checked-in public headers: 18.
- Generated reference pages: 18.
- Generated source pages: 18.

```sh
python3 tests/test_qr_external_ref_helper_guard.py
```

Result: passed.

The QR helper guard was run as an adjacent review-surface guard to confirm the
new SVD helper guard did not disturb the existing helper-boundary pattern.

## Required Full C Quality Gate

Because Sprint 201 moved selected SVD test bodies into
`tests/test_svd_selected_helpers.h`, retained shared fixtures in
`tests/test_svd_helpers.h`, and changed `tests/test_svd.c`, the required quality
gate was run.

```sh
make format
```

Result: passed.

```sh
make lint
```

Result: passed.

- Built 16 benchmark binaries without execution.
- Built 14 example binaries without execution.
- Strict warning compile completed.
- `clang-tidy` completed for 49 production source files.
- `cppcheck` completed for 109 files.

```sh
make test
```

Result: passed.

- Full test runner completed through `test_reorder_amd_qg`.
- Final summary: `All tests passed.`

## Generated Output Hygiene

- `build/sprint201-day12-validation-check` is generated CMake output under the
  ignored `build/` tree.
- `docs/api/html` is generated Doxygen output under the ignored `docs/api/`
  tree.
- No generated validation output is intended for commit.

## Risk Register

| Risk | Day 12 Status | Residual |
| --- | --- | --- |
| Selected SVD helper extraction changes behavior. | Focused `test_svd` and full `make test` passed. | No known selected-cluster behavior residual. |
| Selected SVD helper ownership drifts from `tests/test_svd.c`. | `make svd-helper-guard` and `python3 tests/test_svd_helper_guard.py` passed. | No known selected-cluster ownership residual. |
| Registration/source/CMake parity regresses. | `make source-list-check` and CMake configure passed. | No known parity residual. |
| Docs or maintainer guidance overclaim. | `make docs-check` passed and Day 11 documentation recorded selected-scope non-claims. | Broader review-surface cleanup remains future selected-cluster work. |
| Public API, ABI, or library implementation is unintentionally changed. | No Sprint 201 implementation/header API changes are part of the selected SVD helper extraction. | Public API/ABI non-claim remains unchanged. |
| Platform, package-manager, performance, or state-of-the-art claims are implied. | Day 12 validation is local selected-cluster evidence only. | These remain explicit non-claims for Sprint 201. |

## Completion

Day 12 completes item 201.6 validation evidence for Sprint 201's selected SVD
rank, pseudoinverse, and dense low-rank helper review-surface reduction.
