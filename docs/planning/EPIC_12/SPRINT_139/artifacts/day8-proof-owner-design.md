# Sprint 139 Day 8: Proof Owner Design

## Purpose

Day 8 defines the focused QR proof owner for the selected Sprint 139 residual:
`qr_rank_deficient_6x4_nullspace_v1`. The design keeps the proof narrow,
discoverable, and tied to the maintained corpus lane without weakening the
existing broad QR test coverage in `tests/test_qr.c`.

## Decision

Add a dedicated test executable:

```text
tests/test_qr_corpus.c
```

Do not append the Sprint 139 corpus proof to `tests/test_qr.c`.

Rationale:

- `tests/test_qr.c` already owns many QR factorization, Q, rank, nullspace,
  economy, sparse-mode, reorder, threshold, and external-reference checks.
- The selected Sprint 139 closure needs one clear owner for one maintained
  corpus fixture, not another case hidden inside the largest QR proof file.
- A dedicated executable makes failures local: a failing
  `test_qr_corpus` result should point directly to the corpus-backed QR
  residual closure.
- Existing QR tests should remain registered and unchanged. The new owner adds
  evidence; it does not transfer or remove existing coverage.

## Ownership Map

| Surface | Day 8 ownership decision |
| --- | --- |
| `tests/test_qr_corpus.c` | New focused owner for the maintained corpus-backed QR residual proof. |
| `tests/test_qr_helpers.h` | Add only reusable fixture/residual helpers needed by the focused owner. |
| `tests/test_qr.c` | Retain existing broad QR factorization, rank, nullspace, projector, Q, economy, sparse-mode, reorder, and threshold coverage. |
| `tests/test_qr_solve.c` | Retain solve, least-squares, rank-deficient solve residual, and minimum-norm ownership; do not absorb this nullspace fixture. |
| `scripts/run_corpus_oracle.py` | Continue owning generated-reference and opt-in solver-backed oracle/report rows. |
| `tests/corpus/*` | Continue owning source-controlled fixture, generator, and expected-result metadata. |

## Focused Test Plan

`tests/test_qr_corpus.c` should contain a small suite with these tests:

| Test name | Behavior | Assertions |
| --- | --- | --- |
| `test_qr_corpus_rankdef_6x4_fixture_shape` | Fixture builder emits the maintained 6x4 fixture. | rows `6`, cols `4`, nonzero count or inserted-entry success, no allocation/insert failure. |
| `test_qr_corpus_rankdef_6x4_rank_and_nullity` | QR factorization reports the maintained rank/nullity. | `sparse_qr_factor()` returns `SPARSE_OK`, `sparse_qr_rank(&qr, 0.0) == 3`, `sparse_qr_nullspace(&qr, 0.0, NULL, &ndim) == SPARSE_OK`, `ndim == 1`. |
| `test_qr_corpus_rankdef_6x4_nullspace_residual` | Solver-produced nullspace basis satisfies the maintained residual threshold. | basis extraction succeeds, basis norm is nonzero, `||A*v||_2 / ||v||_2 <= 1e-10`. |
| `test_qr_corpus_rankdef_6x4_reference_direction` | The deterministic reference direction remains a true structural null vector. | `[-1, -1, 0, 1]` gives normalized residual `0` within exact arithmetic tolerance, independent of QR basis orientation. |

The reference-direction check is a fixture integrity check, not a raw QR basis
parity claim. The QR solver proof should rely on residual/subspace-safe
assertions against the solver-produced vector.

## Helper Plan

Add these helpers to `tests/test_qr_helpers.h` on Day 9:

| Helper | Purpose |
| --- | --- |
| `tf_qr_make_rankdef_6x4_nullspace_v1()` | Build the exact maintained corpus fixture in C with dependency `c3 = c0 + c1`. |
| `tf_qr_normalized_matvec_residual(const SparseMatrix *A, const double *x, idx_t x_len)` | Compute `||A*x||_2 / ||x||_2` with explicit zero-norm failure handling. |

The fixture builder should use `tf_qr_insert_or_free()` so allocation or insert
failures clean up consistently with existing QR helpers.

The residual helper should stay metric-only. It should not encode QR semantics,
rank policy, SuiteSparse behavior, or external-library parity.

## Fixture Entries

Day 9 should mirror the generated corpus fixture exactly:

```text
(0,0)=1, (0,3)=1
(1,1)=1, (1,3)=1
(2,2)=1
(3,0)=1, (3,1)=1, (3,3)=2
(4,1)=1, (4,2)=1, (4,3)=1
(5,0)=1, (5,2)=1, (5,3)=1
```

Because the sparse matrix does not need explicit zero entries, the C fixture
builder should insert only these 14 nonzero entries:

```text
(0,0)=1, (0,3)=1
(1,1)=1, (1,3)=1
(2,2)=1
(3,0)=1, (3,1)=1, (3,3)=2
(4,1)=1, (4,2)=1, (4,3)=1
(5,0)=1, (5,2)=1, (5,3)=1
```

This preserves the corpus dependency `c3 = c0 + c1` and the reference null
vector direction `[-1, -1, 0, 1]`.

## Assertion Semantics

The focused owner should assert:

- factorization succeeds for the selected fixture;
- rank is exactly `3` at tolerance `0.0`;
- nullity is exactly `1` at tolerance `0.0`;
- extracted nullspace basis has nonzero norm;
- normalized residual for the extracted basis is at most `1e-10`;
- normalized residual for `[-1, -1, 0, 1]` is at most `1e-12`;
- failure messages mention the fixture key or test name clearly.

The focused owner should not assert:

- exact QR basis components;
- basis sign or orientation;
- broad rank-threshold policy;
- least-squares or minimum-norm solve behavior;
- SuiteSparse, LAPACK, NumPy, SciPy, platform, performance, or
  state-of-the-art parity.

## Build-System Touch Plan

If Day 9 adds `tests/test_qr_corpus.c`, update:

| Surface | Required edit |
| --- | --- |
| `Makefile` | Add `$(TESTDIR)/test_qr_corpus.c` near `test_qr.c` and `test_qr_solve.c` in `TEST_SRCS`. |
| `CMakeLists.txt` | Add `add_sparse_test(test_qr_corpus)` near `add_sparse_test(test_qr)` and `add_sparse_test(test_qr_solve)`. |
| CI/test manifests | No separate workflow edit appears required because the repository builds registered Make/CMake tests from those lists. Recheck `.github` before Day 9 finalization. |

The focused Make validation command should be:

```sh
make build/test_qr_corpus && ./build/test_qr_corpus
```

The focused CMake validation command should be:

```sh
cmake -S . -B build/qr-corpus-proof && cmake --build build/qr-corpus-proof --target test_qr_corpus && ./build/qr-corpus-proof/test_qr_corpus
```

Because Day 9 will modify `.c`, `.h`, `Makefile`, and `CMakeLists.txt`, it must
also run the required full gate after focused checks:

```sh
make format && make lint && make test
```

## Retained-Coverage Checklist

Day 9 must confirm:

- `test_qr` remains in both Makefile and CMake registration.
- `test_qr_solve` remains in both Makefile and CMake registration.
- No existing `RUN_TEST(...)` line is removed from `tests/test_qr.c`.
- No solve/minimum-norm case moves into `test_qr_corpus`.
- The new focused owner is additive and does not replace Day 7 oracle rows.
- The new focused owner uses the same fixture semantics as the corpus generator
  and expected rows.

## Day 9 Implementation Sequence

1. Add `tf_qr_make_rankdef_6x4_nullspace_v1()` to
   `tests/test_qr_helpers.h`.
2. Add a metric-only normalized residual helper if no existing helper is
   precise enough.
3. Create `tests/test_qr_corpus.c` with the four focused tests above.
4. Register `test_qr_corpus` in Makefile and CMake.
5. Run focused Make and CMake test commands.
6. Run the Day 7 oracle command with `--include-solver-qr` to keep corpus and
   focused proof semantics aligned.
7. Run `make format && make lint && make test`.
8. Write the Day 9 implementation artifact with focused output and retained
   coverage evidence.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Proof-owner scope is narrower than broad QR correctness. | Complete | Dedicated `test_qr_corpus` design covers one fixture, one rank/nullity contract, and one normalized residual threshold. |
| Existing QR tests are not weakened or silently bypassed. | Complete | Existing `test_qr` and `test_qr_solve` remain retained owners; Day 9 has an explicit no-removal checklist. |
| Build and source-list implications are explicit before code changes. | Complete | Makefile, CMake, focused test commands, and full quality gate requirements are listed above. |
