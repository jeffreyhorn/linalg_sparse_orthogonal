# Sprint 178 Day 10: Focused Gate Registration

## Scope

Day 10 adds maintained focused validation for the selected Sprint 178
allocation-failure proof:

- subsystem: `sparse_matmul()`;
- selected allocation sites: `acc`, `nz_flag`, and `touched`;
- proof owner: `tests/test_matmul.c`;
- gate owner: `Makefile`;
- CTest registration owner: `CMakeLists.txt`.

## Make Gate

Day 10 adds:

```sh
make matmul-allocation-failure-gate
```

The target builds `build/test_matmul`, runs the registration guard, runs the
matrix multiply test executable, and reports a scoped pass message.

This gate intentionally runs the full `test_matmul` executable because the
selected allocation-failure regressions share local fixture helpers and retry
assertions with the existing matrix multiply correctness tests.

## CTest Registration

Day 10 labels `test_matmul` with:

```cmake
matmul;allocation_failure
```

This keeps CTest discoverability aligned with the existing Sprint 176
`allocation_failure` label while adding a subsystem-specific `matmul` selector.

## Drift Guard

Day 10 adds:

```sh
python3 tests/test_matmul_allocation_failure_gate_registration.py
```

The guard asserts:

- `Makefile` contains the `matmul-allocation-failure-gate` phony target;
- the gate depends on `$(BUILDDIR)/test_matmul`;
- the gate runs the registration guard;
- `CMakeLists.txt` registers `test_matmul`;
- `CMakeLists.txt` labels `test_matmul` with `matmul;allocation_failure`;
- `tests/test_matmul.c` still registers the selected allocation-failure,
  stale-output, and error-precedence tests.

## Boundary

The focused gate does not claim broad allocation-failure coverage. It covers
only the selected `sparse_matmul()` workspace allocation proof and the support
tests needed by that executable.

Out of scope remains:

- `sparse_create()` shell allocation;
- `sparse_insert()` product-flush allocation;
- matrix copy, transpose, CSR/CSC conversion, and build-helper allocation;
- direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines,
  reorder routines, package/install flows, and generated-report tooling.

## Validation

- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `make matmul-allocation-failure-gate`
- CMake configure/build for `test_matmul`
- `ctest --test-dir build-sprint178-day10 -N -L allocation_failure`
- `ctest --test-dir build-sprint178-day10 --output-on-failure -L matmul`
- `make format`
- `make lint`
- `make test`

## Handoff

Day 11 should update README and maintainer guidance with the exact scoped
`make matmul-allocation-failure-gate` command and preserve broad
allocation-failure non-claims.
