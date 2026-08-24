# Sprint 178 Day 12: Integrated Validation

## Scope

Day 12 validates all changed Sprint 178 surfaces:

- focused Make gate for the selected `sparse_matmul()` allocation-failure
  proof;
- registration guard for Makefile, CMake, and `test_matmul` test registration;
- CMake/CTest label discoverability for `matmul` and `allocation_failure`;
- documentation hygiene for allocation-failure terminology;
- full C quality gate because Sprint 178 modified `tests/test_matmul.c`.

## Focused Gate Result

Passed:

```sh
make matmul-allocation-failure-gate
```

The gate ran the registration guard and `test_matmul`. `test_matmul` reported
18 tests, 0 failures, and 185 assertions.

## Registration Guard Result

Passed:

```sh
python3 tests/test_matmul_allocation_failure_gate_registration.py
```

The guard confirmed:

- `Makefile` exposes `matmul-allocation-failure-gate`;
- the Make target depends on `$(BUILDDIR)/test_matmul`;
- the Make target runs the registration guard;
- `CMakeLists.txt` registers `test_matmul`;
- `test_matmul` is labeled `matmul;allocation_failure`;
- the selected `test_matmul` allocation-failure regressions remain registered.

## CMake And CTest Result

Passed:

```sh
cmake -S . -B build-sprint178-day12
cmake --build build-sprint178-day12 --target test_matmul test_iterative --parallel 1
ctest --test-dir build-sprint178-day12 -N -L allocation_failure
ctest --test-dir build-sprint178-day12 --output-on-failure -L matmul
ctest --test-dir build-sprint178-day12 --output-on-failure -L allocation_failure
```

`ctest -N -L allocation_failure` discovered two focused lanes:

- `test_matmul`;
- `test_iterative`.

`ctest -L matmul` passed 1 of 1 test. `ctest -L allocation_failure` passed 2
of 2 tests.

## Documentation Hygiene Result

Passed:

```sh
rg -n "allocator-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178 || true
git diff --check
```

The only `allocator-failure` spelling found is intentional evidence text that
warns against that terminology drift.

## Full C Quality Gate Result

Passed:

```sh
make format && make lint && make test
```

This covered formatting, strict warnings, `clang-tidy`, `cppcheck`, and the
full Make test suite. The run ended with `All tests passed.`

## Validation Interpretation

The Day 12 pass supports only the selected Sprint 178 claim:

- `sparse_matmul()` accumulator, nonzero-flag, and touched-column workspace
  allocation failure cleanup;
- no stale output publication on those failures;
- successful retry after resetting the private allocation-failure hook.

It does not establish broad allocation-failure coverage for matrix shell
construction, insertion/product flush, conversions, solver families,
package/install flows, generated tooling, or unrelated allocation paths.
