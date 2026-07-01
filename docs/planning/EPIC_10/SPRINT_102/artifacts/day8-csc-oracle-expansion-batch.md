# Sprint 102 Day 8 CSC Oracle Expansion Batch

## Purpose

Day 8 implements the CSC-family oracle expansion selected on Day 7. The batch
adds one bounded LDLT CSC external dense-reference fixture, keeps the proof
family-local, and reuses the Day 5 external-reference vector parser.

## Implemented Fixture

Added fixture key:

```text
ldlt_kkt_scaled_10
```

The fixture is a deterministic 10x10 symmetric indefinite KKT matrix:

```text
A = [ H  C^T ]
    [ C   0  ]
```

where `H` is a 6x6 SPD tridiagonal block:

```text
diag(H) = [8, 10, 12, 14, 16, 18]
offdiag(H) = [-1, -1.25, -1.5, -1.75, -2]
```

and `C` has mixed-scale coupling entries:

| row in `C` | nonzero columns and values |
|---:|---|
| 0 | `C[0,0] = 1`, `C[0,4] = 0.125` |
| 1 | `C[1,1] = -2`, `C[1,5] = 0.25` |
| 2 | `C[2,2] = 0.5`, `C[2,4] = -0.375` |
| 3 | `C[3,3] = 3`, `C[3,5] = 0.5` |

The same construction now exists in:

- `tests/ldlt_external_dense_reference.py`, as
  `build_kkt_scaled_10()` and fixture dispatch for `ldlt_kkt_scaled_10`;
- `tests/test_ldlt_csc.c`, as `build_kkt_scaled_10x10()`.

## Implemented Test

Added:

```c
test_s102_external_dense_reference_scaled_kkt_10x10
```

The test reuses the existing LDLT external dense-reference harness:

1. builds the C-side sparse fixture;
2. sets `x_true[i] = i + 1`;
3. computes `b = A*x_true`;
4. factors through the existing two-pass LDLT CSC indefinite path;
5. solves in permuted ordering and maps the result back;
6. reads the Python dense reference through
   `tf_read_external_reference_vector(...)`;
7. checks `x` against both `x_ref` and `x_true`;
8. checks the residual against the original sparse matrix.

The new test is registered next to the existing Sprint 98 LDLT CSC external
fixtures `kkt5` and `kkt10`.

## Focused Validation Results

| command | result |
|---|---|
| `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10` | passed; emitted `OK 10` and a dense solution equal to `1..10` to roundoff |
| `make build/test_ldlt_csc` | passed; target rebuilt before formatting and was up to date after formatting |
| `./build/test_ldlt_csc` | passed; 99 tests, 0 failures, 0 skips, 2318 assertions |

Focused LDLT external-reference metrics:

| fixture | max error | residual |
|---|---:|---:|
| `kkt5` | `0.000e+00` | `0.000e+00` |
| `kkt10` | `3.553e-15` | `2.292e-16` |
| `ldlt_kkt_scaled_10` | `8.882e-15` | `1.692e-17` |

The Day 7 tolerance of `1e-10` remained valid; no relaxation was needed.

## Full Validation Results

Because Day 8 changed `tests/test_ldlt_csc.c`, the required quality gate was
run:

| command | result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed; `All tests passed.` |

## Failure Behavior

Day 8 did not add a new expected-failure fixture because Day 7 selected only a
positive correctness expansion.

Preserved behavior:

- unknown helper fixture keys still emit `ERROR unknown fixture ...`;
- malformed helper output or dimension mismatch remains a test failure through
  `tf_read_external_reference_vector(...)`;
- unavailable helper pipes remain skip/unsupported, not correctness proof;
- Windows keeps the existing external-helper skip behavior;
- Cholesky does not claim support for the indefinite KKT fixture.

## Touched Files

| file | change |
|---|---|
| `tests/ldlt_external_dense_reference.py` | added `ldlt_kkt_scaled_10` dense fixture |
| `tests/test_ldlt_csc.c` | added matching sparse builder, positive external-reference test, and registration |
| `docs/planning/EPIC_10/SPRINT_102/artifacts/day8-csc-oracle-expansion-batch.md` | implementation evidence |
| `docs/planning/EPIC_10/SPRINT_102/WORKING_NOTES.md` | Day 8 working notes |

No public headers, library source files, build files, public documentation, or
Cholesky CSC tests were changed for Day 8.

## Non-Claims

Day 8 earns only this bounded evidence claim:

> LDLT CSC solves for the named scaled KKT fixture
> `ldlt_kkt_scaled_10` agree with an external-process dense reference under
> the recorded tolerance and validation commands.

Day 8 does not claim:

- LDLT CSC handles all indefinite matrices;
- external factorization, pivot, or storage-layout parity exists;
- Cholesky supports indefinite KKT inputs;
- direct CSR/CSC solver APIs exist;
- portable helper availability on every platform;
- runtime, fill, or benchmark improvement.

## Day 8 Conclusion

The selected CSC-family oracle expansion is implemented and validated. Sprint
102 now has one additional bounded LDLT CSC external dense-reference fixture
with moderate scaling, while the proof remains family-local and claim wording
stays limited to named fixtures and recorded validation.
