# Sprint 102 Day 7 CSC Oracle Boundary

## Purpose

Day 7 freezes the CSC direct-family oracle expansion before implementation.
It selects one LDLT/Cholesky lane, defines the exact fixture contract,
tolerances, failure handling, validation plan, and non-claims that Day 8 must
honor.

## Inputs Reviewed

| input | role in Day 7 decision |
|---|---|
| Sprint 102 Day 2 gap audit | identified LU as the broadest direct-solver gap and LDLT CSC as a high-value CSC-family expansion candidate |
| Sprint 102 Day 3 fixture taxonomy | defined `indef-kkt-scaled` as the LDLT CSC expansion class |
| Sprint 102 Day 5 helper extraction | made the external dense-reference vector parser reusable by future direct-solver lanes |
| Sprint 102 Day 6 closeout and rerank | selected LDLT CSC scaled/reordered KKT as the Day 7 boundary target |
| Sprint 98 LDLT CSC external lane | established deterministic `kkt5` and `kkt10` dense-reference fixtures |
| Sprint 100 Cholesky CSC pilot | confirmed Cholesky CSC already owns external SPD proof on `nos4` and `bcsstk04` |

## Selected Lane

Day 7 selects an LDLT CSC external dense-reference expansion, not a Cholesky
CSC expansion.

| field | selected boundary |
|---|---|
| fixture key | `ldlt_kkt_scaled_10` |
| taxonomy class | `indef-kkt-scaled` |
| solver family | LDLT CSC |
| implementation owner | `tests/test_ldlt_csc.c` |
| external helper owner | `tests/ldlt_external_dense_reference.py` |
| helper parser | `tf_read_external_reference_vector(...)` from `tests/test_solver_helpers.h` |
| expected outcome | success |
| proof type | user-visible solve agreement against an external-process dense reference |
| primary assertion | `max|x - x_ref| <= 1e-10` and `rel_residual < 1e-10` |

Cholesky CSC is the backup lane only if the LDLT scaled KKT fixture cannot be
kept deterministic and bounded during Day 8 implementation.

## Fixture Definition

`ldlt_kkt_scaled_10` is a deterministic 10x10 symmetric indefinite KKT
fixture. The matrix is:

```text
A = [ H  C^T ]
    [ C   0  ]
```

where `H` is a 6x6 SPD tridiagonal block:

```text
diag(H) = [8, 10, 12, 14, 16, 18]
offdiag(H) = [-1, -1.25, -1.5, -1.75, -2]
```

and the 4x6 coupling block `C` has these nonzero entries:

| row in `C` | nonzero columns and values |
|---:|---|
| 0 | `C[0,0] = 1`, `C[0,4] = 0.125` |
| 1 | `C[1,1] = -2`, `C[1,5] = 0.25` |
| 2 | `C[2,2] = 0.5`, `C[2,4] = -0.375` |
| 3 | `C[3,3] = 3`, `C[3,5] = 0.5` |

Day 8 must mirror this construction exactly in:

- `tests/ldlt_external_dense_reference.py`, as a new fixture key accepted by
  `fixture_matrix(...)`;
- `tests/test_ldlt_csc.c`, as a C-side sparse fixture builder passed to the
  existing LDLT external dense-reference harness.

The fixture intentionally introduces moderate scale variation across the SPD
block and coupling block without becoming a near-singular tolerance stress.
It remains a correctness fixture, not a conditioning benchmark.

## Reference Vector And RHS

Day 8 must preserve the existing LDLT external-lane target:

```text
x_true[i] = i + 1
b = A * x_true
```

The Python helper must compute a dense reference solution from the same matrix
and RHS. The C harness must:

1. build the same sparse matrix;
2. build `x_true` and `b`;
3. factor through the existing LDLT CSC two-pass indefinite path;
4. solve the permuted system;
5. unpermute the result to original ordering;
6. read the external dense-reference vector through
   `tf_read_external_reference_vector(...)`;
7. compare both `x_ref` and `x_true`;
8. assert the residual against the original sparse matrix.

## Acceptance Criteria

| check | required threshold |
|---|---:|
| external helper returns `OK 10` | required |
| `max|x - x_ref|` | `<= 1e-10` |
| `max|x - x_true|` | `<= 1e-10` |
| `rel_residual(A, x, b)` | `< 1e-10` |
| helper command exit status | clean exit required |
| malformed or unknown fixture key | comparison failure, not a pass |
| unavailable helper pipe | skip/unsupported, not correctness evidence |

A one-off dense Gaussian-elimination sanity check of the proposed fixture
recovered `x_true = 1..10` with `max|x - x_true| = 8.882e-15`. That check is
only boundary evidence; Day 8 implementation must earn the solver proof.

If Day 8 observes stable roundoff slightly above `1e-10`, it may relax the
threshold no further than `1e-9`, but only if the implementation artifact
records the observed metrics and explains why the relaxation is numerical
rather than masking a solver regression.

## Failure And Unsupported Behavior

| condition | required interpretation |
|---|---|
| `python3` or pipe unavailable | `SKIP_TEST(reason)` is allowed and must not count as correctness proof |
| helper emits `ERROR reason` | fail the test |
| helper emits malformed `OK` header or truncated vector | fail the test |
| helper returns wrong vector length | fail the test |
| LDLT CSC factor or solve returns an error | fail the test |
| Windows helper path remains disabled | skip/unsupported with existing wording |
| Cholesky on this indefinite fixture | unsupported; do not add a Cholesky correctness pass for this fixture |

No new expected-failure fixture is selected for Day 8. The existing unknown
fixture behavior in `tests/ldlt_external_dense_reference.py` remains the
helper-level failure model for this batch.

## Day 8 Implementation Plan

Day 8 should touch only the focused implementation surface:

| file | intended Day 8 change |
|---|---|
| `tests/ldlt_external_dense_reference.py` | add `build_kkt_scaled_10()` and fixture dispatch for `ldlt_kkt_scaled_10` |
| `tests/test_ldlt_csc.c` | add matching sparse builder, add one external dense-reference test, register it next to the existing Sprint 98 external tests |
| Sprint 102 Day 8 artifact | record implementation evidence and validation results |
| `WORKING_NOTES.md` | record Day 8 actions, findings, and validation |

Day 8 should not change public headers, library sources, build registration,
public documentation, or Cholesky CSC tests unless the LDLT boundary proves
unworkable.

## Validation Plan

Because Day 8 is expected to change `.c` and helper code, the required quality
gate is:

```sh
make format
python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10
make build/test_ldlt_csc
./build/test_ldlt_csc
make lint
make test
git diff --check
rg -n "[ \t]+$" tests/ldlt_external_dense_reference.py tests/test_ldlt_csc.c docs/planning/EPIC_10/SPRINT_102
```

The focused LDLT run must show the existing `kkt5` and `kkt10` lanes still
passing, plus a new `ldlt_kkt_scaled_10` external-reference line with recorded
`max|x - x_ref|` and residual metrics.

## Non-Claims

This boundary does not claim:

- new solver evidence has landed yet;
- LDLT CSC handles all indefinite matrices;
- LDLT CSC external factorization or pivot-layout parity exists;
- Cholesky CSC supports indefinite KKT inputs;
- direct CSR/CSC solver APIs exist;
- Python helper availability is portable correctness evidence;
- runtime or fill behavior improved.

## Day 7 Conclusion

Sprint 102 should proceed to Day 8 with `ldlt_kkt_scaled_10` as the selected
CSC-family oracle expansion. The fixture is deterministic, moderately scaled,
nonsingular by dense sanity check, and bounded to user-visible LDLT CSC solve
agreement against an external dense reference.
