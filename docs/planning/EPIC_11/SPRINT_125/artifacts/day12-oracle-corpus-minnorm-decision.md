# Sprint 125 Day 12 QR-vs-SVD and SuiteSparse Minimum-Norm Decision

## Decision

Accepted two bounded Day 12 minimum-norm lanes in `tests/test_colamd.c`:

- `qr_minnorm_vs_svd_pinv_crosscheck`
- `qr_minnorm_suitesparse_submatrix`

Both lanes strengthen existing owner-local tests. Day 12 does not add generic
minimum-norm helpers, change SVD semantics, add a new external-reference
protocol, or update public solver claims.

## Accepted Lanes

| Owner key | Test | Added evidence | Claim boundary |
| --- | --- | --- | --- |
| `qr_minnorm_vs_svd_pinv_crosscheck` | `test_minnorm_vs_pinv` | Asserts QR and SVD-pseudoinverse solutions both equal `[0.5, 0.5, 0.5, 0.5]`, both have norm `1.0`, both satisfy `A*x=b`, and both match each other. | Bounded cross-check for one 2x4 fixture; SVD pseudoinverse is not a global QR oracle. |
| `qr_minnorm_suitesparse_submatrix` | `test_minnorm_ss_submatrix` | Asserts the checked-in `west0067` submatrix dimensions, residual below `1e-8`, positive solution norm, and `||x_min|| <= ||ones|| + 1e-8`. | One default checked-in corpus submatrix smoke; no broad SuiteSparse minimum-norm or platform parity claim. |

## Oracle Role Decision

The SVD-pseudoinverse lane is accepted as a bounded cross-check, not as an
oracle role for all QR minimum-norm behavior.

The matching SVD owner remains `tests/test_svd.c`, especially
`test_pinv_underdetermined_minnorm_solution`, which independently checks the
SVD-side underdetermined pseudoinverse solution and Moore-Penrose residual.
Day 12 uses that owner for validation, while keeping QR-vs-SVD comparison
assertions in `tests/test_colamd.c`.

## SuiteSparse Support-Tier Decision

The SuiteSparse lane remains a default checked-in corpus smoke because it uses
`tests/data/suitesparse/west0067.mtx`, which is already required by the default
suite. Missing `west0067.mtx` is therefore a default-data failure, not an
optional skip.

The accepted corpus evidence is limited to:

- matrix path: `tests/data/suitesparse/west0067.mtx`
- extraction: first 30 rows into a 30 x 67 underdetermined system
- RHS: `b = A * ones`
- residual metric: max absolute row residual below `1e-8`
- norm metric: `||x_min|| <= ||ones|| + 1e-8`

The lane does not prove SuiteSparse-wide minimum-norm behavior, large-matrix
behavior, optional-corpus behavior, platform parity, or performance.

## Deferred Work

| Deferred lane | Reason | Promotion gate |
| --- | --- | --- |
| Additional QR-vs-SVD fixtures | One bounded cross-check is enough for Sprint 125; broader SVD oracle use would blur ownership. | Define fixture keys, SVD tolerance, QR residual/norm metric, and cross-check wording per fixture. |
| Optional-large SuiteSparse minimum-norm | Day 12 accepts only a default checked-in `west0067` submatrix smoke. | Apply Day 8-9 optional-corpus gates, skip diagnostics, support tier, and focused validation. |
| SuiteSparse rank-deficient minimum-norm corpus | Requires expected rank, nullity, residual, norm, and corpus metadata not available today. | Pin rank/threshold metadata and support tier before registration. |
| Generic QR/SVD minimum-norm helper movement | Would hide whether the owner is QR solve, SVD pseudoinverse, corpus, or cross-check behavior. | Future helper owner must use behavior-specific names and keep tolerances at call sites. |

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/test_colamd.c` | Strengthened `test_minnorm_vs_pinv` and `test_minnorm_ss_submatrix` with explicit solution, residual, norm, dimension, and corpus-bound assertions. |
| `tests/test_svd.c` | No code change; used as companion validation for the SVD pseudoinverse owner. |
| `tests/qr_external_dense_reference.py` | No change; Day 12 does not add a helper protocol. |
| `docs/maintainer_guide.md` | No Day 12 update; Day 13 owns final evidence-table and claim-gate refresh. |

## Non-Claims Preserved

- No SVD pseudoinverse as a global QR oracle.
- No broad QR minimum-norm parity.
- No global minimum-norm optimality beyond named fixtures.
- No broad SuiteSparse corpus, optional-data, platform, or performance claim.
- No COLAMD, reorder, fallback, refinement, rank-deficient, zero-row,
  pseudoinverse, or SuiteSparse superiority claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, dense-library,
  backend, package, ABI, public API, CMake, CTest, performance, scalability,
  memory, or state-of-the-art claim.

## Validation

Focused validation passed:

```text
make build/test_colamd && ./build/test_colamd
make build/test_svd && ./build/test_svd
```

Focused results:

| Command | Result |
| --- | --- |
| `./build/test_colamd` | 70 tests, 0 failures, 0 skips, 299 assertions |
| `./build/test_svd` | 109 tests, 0 failures, 0 skips, 1802 assertions |

Because Day 12 changed a C file, full required validation passed:

```text
make format && make lint && make test
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 6 is complete or explicitly deferred. | Complete | Day 11 accepted core lanes; Day 12 accepted bounded oracle/corpus lanes and deferred broader work. |
| QR and SVD oracle roles are not conflated. | Complete | SVD is documented and validated as a bounded cross-check only. |
| SuiteSparse behavior follows optional-corpus support rules. | Complete | Accepted path uses default checked-in `west0067`; optional-large work remains deferred. |
| Full C-file quality gate passed. | Complete | `make format && make lint && make test`, `git diff --check`, trailing-whitespace scan, and Python-cache scan passed. |
