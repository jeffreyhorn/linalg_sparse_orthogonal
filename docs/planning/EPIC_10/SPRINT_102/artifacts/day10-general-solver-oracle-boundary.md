# Sprint 102 Day 10 General Solver Oracle Boundary

## Purpose

Day 10 freezes the selected LU/QR/SVD oracle and failure-mode expansion before
implementation. Day 9 selected LU as the highest-value general direct-solver
gap; this artifact defines the exact LU fixture keys, helper contract,
tolerances, failure behavior, validation plan, and public trust boundary for
Day 11.

## Inputs Reviewed

| input | Day 10 use |
|---|---|
| Day 2 direct-solver gap audit | confirmed LU has the highest external-oracle gap among LU/QR/SVD |
| Day 3 fixture taxonomy | supplied `nonsym-square-small` and `square-rank-def` classes |
| Day 9 CSC closeout and rerank | selected LU and proposed `lu_nonsym_square_5` plus `lu_singular_square_4` |
| `tests/test_sparse_lu.c` | chosen Day 11 implementation owner for linked-list LU one-shot solve proof |
| `tests/test_lu_csr.c` | reviewed as a later CSR LU follow-up, not Day 11 owner |
| `include/sparse_lu.h` | confirms one-shot LU contract on a fresh matrix or copy |
| QR/SVD test surfaces | retained as backup/follow-up because they already have broad internal invariant coverage |

## Selected Lane

Day 10 selects a linked-list LU external dense-reference lane.

| field | selected boundary |
|---|---|
| solver family | linked-list LU |
| test owner | `tests/test_sparse_lu.c` |
| implementation owner | no library implementation change planned |
| external helper owner | new `tests/lu_external_dense_reference.py` |
| shared parser | `tf_read_external_reference_vector(...)` from `tests/test_solver_helpers.h` |
| success fixture key | `lu_nonsym_square_5` |
| expected-failure fixture key | `lu_singular_square_4` |
| pivot policy | `SPARSE_PIVOT_COMPLETE` for the external positive lane |
| proof type | user-visible solve agreement against an external-process dense reference |

QR remains the backup lane if Day 11 finds the LU helper boundary cannot stay
bounded. SVD remains deferred because its external oracle shape is heavier and
its internal invariant coverage is already broad.

## Positive Fixture Definition

`lu_nonsym_square_5` is a deterministic 5x5 nonsymmetric full-rank matrix:

```text
[ 4.0  -1.0   0.0   2.0   0.5 ]
[ 1.5   5.0  -2.0   0.0   1.0 ]
[ 0.0   2.0   6.0  -1.0   0.0 ]
[ 3.0   0.0   1.0   7.0  -2.0 ]
[-1.0   0.5   0.0   2.0   8.0 ]
```

Fixture class:

```text
nonsym-square-small
```

Expected outcome:

- external helper returns `OK 5`;
- LU factorization succeeds on a fresh copy of the matrix;
- `sparse_lu_solve(...)` returns `SPARSE_OK`;
- solver result matches dense reference and known target within tolerance;
- residual against the original matrix is below tolerance.

The fixture should use:

```text
x_true[i] = i + 1
b = A*x_true
```

Day 11 must mirror this matrix exactly in the Python helper and the C harness.

## Expected-Failure Fixture Definition

`lu_singular_square_4` is a deterministic 4x4 singular matrix with dependent
rows:

```text
[ 1.0   2.0  -1.0   0.0 ]
[ 2.0   4.0  -2.0   0.0 ]
[ 0.0   1.0   3.0   1.0 ]
[ 1.0   0.0   0.5  -1.0 ]
```

Fixture class:

```text
square-rank-def
```

Expected outcome:

- dense helper emits `ERROR ... singular ...` for this fixture;
- C LU factorization returns `SPARSE_ERR_SINGULAR`;
- the test asserts the expected failure rather than skipping or treating the
  fixture as correctness evidence.

Day 11 may implement this expected-failure check as C-only if a dense helper
failure assertion would duplicate the existing helper unknown-fixture behavior
too much. If implemented as C-only, the Day 11 artifact must state that the
external oracle lane contains one positive dense-reference fixture and one
local expected singular fixture.

## Helper Contract

Recommended Day 11 helper:

```text
tests/lu_external_dense_reference.py
```

Required CLI behavior:

```sh
python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
python3 tests/lu_external_dense_reference.py lu_singular_square_4
```

Output contract:

| condition | output |
|---|---|
| successful dense solve | `OK n` followed by `n` solution values, one per line |
| singular dense reference | `ERROR matrix is singular to dense reference tolerance` or equivalent clear error |
| unknown fixture | `ERROR unknown fixture <key>` |
| wrong argument count | `ERROR expected one fixture key` |

The C harness should use the existing Day 5 parser:

```c
tf_read_external_reference_vector(cmd, "external LU reference", x_ref, n,
                                  reason, sizeof(reason));
```

Command construction, C-side matrix construction, pivot policy, factor/solve
calls, residual computation, tolerance choice, and assertion interpretation
must remain LU-local in `tests/test_sparse_lu.c`.

## Acceptance Criteria

| check | required threshold or status |
|---|---|
| helper success fixture | `OK 5` |
| `max|x - x_ref|` | `<= 1e-10` |
| `max|x - x_true|` | `<= 1e-10` |
| `||A*x - b||_inf` | `< 1e-10` |
| LU factorization pivot policy | `SPARSE_PIVOT_COMPLETE` |
| singular C fixture | `SPARSE_ERR_SINGULAR` |
| helper singular fixture, if checked | `ERROR`, not `OK` or `SKIP` |

If the positive fixture produces stable roundoff slightly above `1e-10`, Day
11 may relax to no more than `1e-9`, but only with recorded metrics and an
explicit numerical justification.

## Failure And Unsupported Behavior

| condition | required interpretation |
|---|---|
| helper unavailable or pipe open fails | skip/unsupported; not correctness proof |
| helper emits `ERROR` for success fixture | fail the positive external-reference test |
| helper emits malformed `OK` header | fail the test |
| helper returns wrong vector length | fail the test |
| helper output is truncated or unparsable | fail the test |
| LU factorization fails on `lu_nonsym_square_5` | fail the test |
| LU solve fails on `lu_nonsym_square_5` | fail the test |
| LU factorization succeeds on `lu_singular_square_4` | fail the expected-failure test |
| Windows helper path | may keep existing external-helper skip convention if needed |

Expected skips must not be counted as correctness passes.

## Day 11 Implementation Plan

Day 11 should touch only:

| file | intended change |
|---|---|
| `tests/lu_external_dense_reference.py` | add bounded dense LU reference helper for two fixture keys |
| `tests/test_sparse_lu.c` | enable external parser helper, add fixture builders, helper wrapper, positive external-reference test, and expected singular fixture test |
| Day 11 artifact | implementation evidence and validation results |
| `WORKING_NOTES.md` | Day 11 notes |

Day 11 should not change public headers, `src/`, build registration, QR, SVD,
LU CSR, Cholesky, LDLT, or public documentation unless implementation reveals
an unavoidable blocker.

## Validation Plan

Because Day 11 is expected to change `.c` code, the required quality gate is:

```sh
make format
python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
python3 tests/lu_external_dense_reference.py lu_singular_square_4
make build/test_sparse_lu
./build/test_sparse_lu
make lint
make test
git diff --check
rg -n "[ \t]+$" tests/lu_external_dense_reference.py tests/test_sparse_lu.c docs/planning/EPIC_10/SPRINT_102
```

The focused `test_sparse_lu` run should show one new positive
external-reference line with recorded max error and residual, plus one
deterministic expected singular failure assertion.

## Public Trust Boundary

This boundary can earn only a named-fixture LU evidence claim:

> Linked-list LU solves for `lu_nonsym_square_5` agree with an
> external-process dense reference under the recorded tolerance and validation
> commands; `lu_singular_square_4` is detected as singular.

It must not claim:

- LU CSR external dense-reference coverage;
- QR or SVD external oracle coverage;
- every nonsymmetric matrix is externally validated;
- all LU pivot patterns are covered;
- direct CSR/CSC solver APIs exist;
- portable performance, fill, or runtime superiority.

## Day 10 Conclusion

Sprint 102 should proceed to Day 11 with a bounded linked-list LU external
dense-reference lane: one positive nonsymmetric 5x5 solve fixture and one
deterministic singular 4x4 expected-failure fixture. QR remains the backup
lane, and SVD remains deferred.
