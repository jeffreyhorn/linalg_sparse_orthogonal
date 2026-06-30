# Sprint 98 Day 4: External Correctness Boundary Freeze

## Purpose

Day 4 freezes the first Sprint 98 external correctness expansion before any
implementation changes. The selected lane is LDLT CSC external correctness.
This artifact defines the exact algorithms, fixtures, reference output,
tolerances, skip behavior, validation commands, landing sequence, and rollback
notes for Days 5-6.

No tests, helpers, source files, headers, benchmarks, or workflows are changed
on Day 4.

## Frozen Correctness Lane

Selected lane:

- LDLT CSC external correctness expansion

Touched implementation surfaces for Days 5-6:

- `tests/test_ldlt_csc.c`
- new helper: `tests/ldlt_external_dense_reference.py`

Surfaces explicitly not touched by this lane:

- `src/*.c`
- `include/*.h`
- `CMakeLists.txt`
- `Makefile`
- `.github/workflows/*.yml`
- benchmark sources and scripts
- public README or install/package docs

## Frozen Fixture Boundary

The first LDLT external correctness lane uses deterministic in-memory KKT
fixtures, not SuiteSparse Matrix Market files.

Frozen fixtures:

| Fixture key | Matrix class | Size | Purpose |
|---|---|---:|---|
| `kkt5` | symmetric indefinite KKT | 5x5 | smallest smoke fixture with expected factor/solve stability |
| `kkt10` | symmetric indefinite KKT | 10x10 | non-trivial off-block coupling that exercises the existing analysis-aware LDLT CSC path |

The C harness should reuse the existing deterministic builders in
`tests/test_ldlt_csc.c`:

- `build_kkt_5x5`
- `build_kkt_10x10`

The external helper should construct the same dense matrices from fixture keys
instead of reading implementation-owned CSC state or internal pivot data.

Deferred fixture classes:

- SPD-as-LDLT fallback:
  - deferred because it is weaker LDLT-specific evidence
- random indefinite fixtures:
  - deferred because skip variability is inappropriate for a maintained
    external lane
- broader Matrix Market fixtures:
  - deferred until a later design decides how to select deterministic
    indefinite corpus entries and reference tolerances

## Frozen Reference Helper Shape

Add a new helper:

- `tests/ldlt_external_dense_reference.py`

Helper contract:

```text
python3 tests/ldlt_external_dense_reference.py <fixture-key>
```

Accepted fixture keys:

- `kkt5`
- `kkt10`

Output contract:

```text
OK <n>
<x0>
<x1>
...
<xN-1>
```

Skip and failure output:

```text
SKIP <reason>
ERROR <reason>
```

Reference computation:

- construct the dense KKT matrix for the fixture key
- build the known true solution vector:
  - `x_true[i] = i + 1`
- compute:
  - `b = A * x_true`
- solve:
  - `A * x_ref = b`
- emit `x_ref`

Allowed dense algorithm:

- partial-pivoting Gaussian elimination or another small deterministic dense
  solve that does not mirror LDLT CSC internals

Disallowed helper behavior:

- no call into project C code
- no replication of Bunch-Kaufman pivot logic
- no CSC row/column storage assumptions
- no dependency on SuiteSparse, NumPy, SciPy, LAPACK, or platform package
  installation
- no timing or fill reporting

## Frozen C Harness Shape

Add a harness near the existing Sprint 20 KKT/analysis-aware LDLT tests in
`tests/test_ldlt_csc.c`.

Preferred helper functions:

- `read_ldlt_external_dense_reference_solution`
- `assert_ldlt_external_dense_reference`

Preferred tests:

- `test_s98_external_dense_reference_kkt_5x5`
- `test_s98_external_dense_reference_kkt_10x10`

Register the tests near the existing KKT and analysis-aware LDLT block:

- after `test_s20_supernodal_heuristic_vs_with_analysis_residuals`
- before later argument-contract or unrelated LDLT CSC tests

Harness behavior:

1. Build the deterministic KKT fixture in C.
2. Build `x_true[i] = i + 1`.
3. Compute `b = A * x_true`.
4. Factor through the selected LDLT CSC analysis-aware path.
5. Solve for `x`.
6. Invoke the external helper by fixture key.
7. Compare `x` against `x_ref`.
8. Compare `x` against `x_true`.
9. Check final residual against the original fixture.

The C harness should compare user-visible solve behavior. It should not assert
external factor entries, pivot-size arrays, permutation arrays, or CSC
structure as part of this new external lane.

## Frozen Algorithm Path

The lane should exercise the existing LDLT CSC analysis-aware path:

- scalar pre-pass through `ldlt_csc_from_sparse`
- symmetric pre-permutation using the scalar pass permutation
- `sparse_analyze` on the pre-permuted matrix
- `ldlt_csc_from_sparse_with_analysis`
- seed `pivot_size` from the scalar pass
- `ldlt_csc_eliminate_supernodal`
- `ldlt_csc_solve`

Implementation may reuse `s20_two_pass_indefinite_factor` and
`s20_solve_residual` where appropriate, but the final external comparison must
read the helper solution vector and compare against it.

## Frozen Tolerances

Initial tolerances:

| Assertion | Tolerance |
|---|---:|
| `max|x - x_ref|` for `kkt5` | `1e-10` |
| `max|x - x_ref|` for `kkt10` | `1e-10` |
| `max|x - x_true|` for both fixtures | `1e-10` |
| relative residual for both fixtures | `< 1e-10` |

If Day 5 implementation discovers that the dense helper and LDLT CSC path
agree at round-off but one fixture needs a slightly wider tolerance, stop and
record evidence before widening. Do not silently relax above `1e-9` without a
new artifact note.

## Optional Dependency and Skip Behavior

Required external command:

- `python3`

Skip behavior:

- on Windows:
  - skip the external helper tests, matching the current Cholesky external
    helper behavior
- if the pipe cannot be opened:
  - skip with a clear reason

Failure behavior:

- unknown fixture key from the helper:
  - fail
- malformed helper output:
  - fail
- helper exits non-zero:
  - fail
- helper prints `ERROR`:
  - fail
- helper dimension mismatch:
  - fail
- LDLT CSC factor/solve failure:
  - fail
- tolerance or residual miss:
  - fail

No SuiteSparse file absence skip is needed for this lane because fixtures are
constructed in memory.

## Validation Commands

Focused Day 5 development commands:

```sh
python3 tests/ldlt_external_dense_reference.py kkt5
python3 tests/ldlt_external_dense_reference.py kkt10
make build/test_ldlt_csc
./build/test_ldlt_csc
```

Required validation after C harness changes:

```sh
make format && make lint && make test
```

Additional hygiene after helper/docs changes:

```sh
git diff --check
rg -n "[ \t]+$" tests/ldlt_external_dense_reference.py tests/test_ldlt_csc.c docs/planning/EPIC_9/SPRINT_98
```

CI-equivalent reading:

- Linux is the strongest reviewed source of truth for the full local suite.
- CMake registration already includes `test_ldlt_csc`; no CMake registration
  change is expected.
- Windows should retain helper skip behavior and existing staged exclusions.

## Landing Sequence

1. Add `tests/ldlt_external_dense_reference.py`.
2. Manually run the helper for `kkt5` and `kkt10`.
3. Add reader and assertion helpers to `tests/test_ldlt_csc.c`.
4. Add `kkt5` external reference test.
5. Build and run `test_ldlt_csc`.
6. Add `kkt10` external reference test.
7. Re-run focused helper and `test_ldlt_csc` commands.
8. Run required full validation for C changes:
   - `make format && make lint && make test`
9. Update Sprint 98 working notes and Day 5 artifact with observed behavior.

## Rollback Notes

If the helper is the blocker:

- remove `tests/ldlt_external_dense_reference.py`
- remove the helper invocation from `tests/test_ldlt_csc.c`
- preserve Day 4 artifact and record the blocker in Day 5 notes

If `kkt10` is the blocker but `kkt5` is stable:

- stop and record evidence
- do not silently ship only `kkt5` unless the Day 5 artifact explains why the
  smaller lane still carries useful maintained evidence

If both KKT fixtures are too coupled to implementation internals:

- fall back only after a new note to SPD-as-LDLT external solve comparison
- keep the claim fence explicit that the lane is direct-family solve evidence,
  not broad indefinite LDLT proof

If full source validation fails:

- stop and fix the validation failure before proceeding
- do not commit or continue to Day 6 with a failing validation chain

## Day 4 Result

The Sprint 98 external correctness boundary is frozen. Days 5-6 should
implement a bounded LDLT CSC external dense-reference solve comparison on
deterministic KKT fixtures, using a small Python helper and a C harness that
asserts solution agreement and residual strength without coupling the external
reference to LDLT CSC internal pivot or storage details.
