# Sprint 215 Day 3: Baseline Selection Decision

## Decision

Select [SuiteSparse `v7.12.2`](https://github.com/DrTimothyAldenDavis/SuiteSparse/releases/tag/v7.12.2), tag commit
`42151688813c45846a597edcb601435a0e38f3dd`, CHOLMOD 5.3.4, as the
external baseline for the **blueprint**. The one pilot cell is a local,
correctness-only real-float64 SPD factor-and-one-solve comparison against the
project Cholesky solver on `tests/data/symmetric_4.mtx`. No external solver
has been run or numerical result accepted by this decision.

The fixture's exact checked-in SHA-256 is
`ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`.
It is a 4 x 4 Matrix Market `real symmetric` matrix with seven stored entries
and ten entries after one symmetric expansion. Set `x_ref = [1, 1, 1, 1]` and
`b = A*x_ref = [13, 14, 17, 16]`, both in float64. The comparison operation is
`A*x=b`, with successful factorization and one solve on each side. It is not a
factor-layout comparison, repeated-solve test, refactor benchmark, or timing
ranking. The exact input and cell contract are in
`day3-methodology-inputs.md`.

## Selected Solver Paths

| Side | Pinned identity and control | Verification still required |
| --- | --- | --- |
| Project | Current pilot Git commit must be recorded at execution; `sparse_load_mm`, a fresh matrix copy, `sparse_cholesky_factor_opts` with `SPARSE_REORDER_NONE` and `SPARSE_CHOL_BACKEND_LINKED_LIST`, then `sparse_cholesky_solve`. Use repository default float64 and `SPARSE_IDX_BITS=32`; capture compiler/flags and `used_csc_path=0`. | Build and run the harness; confirm solver return statuses, finite output, backend selection, and unmodified reference input. |
| External | SuiteSparse tag above; use the discovered macOS x86-64 Homebrew `suite-sparse 7.12.2` / CHOLMOD 5.3.4 C API. Analyze, factorize, and solve a lower-triangular CSC view. Pin `nmethods=1`, `method[0].ordering=CHOLMOD_NATURAL`, `postorder=0`, `supernodal=CHOLMOD_SIMPLICIAL`; use real double. | Record effective `cholmod_common`, factor ordering/mode and status; confirm the loaded binary and its dependency graph and thread settings. Version strings alone do not satisfy the pilot gate. |

Local acquisition evidence from Day 2: `pkg-config --modversion cholmod`
returned 5.3.4; `pkg-config --variable=prefix cholmod` returned
`/usr/local/Cellar/suite-sparse/7.12.2`; and
`libcholmod.5.3.4.dylib` SHA-256 was
`0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`.
These identify the **discovered** local library, not a validated run.
The [tagged CHOLMOD header](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/CHOLMOD/Include/cholmod.h)
defines the API and method controls. At execution, recheck the package
version, absolute library path, digest, ABI/index width, compiler flags,
loaded BLAS/LAPACK and SuiteSparse dependencies; if any differ, create a new
cell record rather than treating the result as this one.

## Why This Cell

The two sides expose the same narrow SPD solve operation. The small
source-controlled matrix requires no optional download and has a fixed
byte-level identity and hand-checkable RHS. Both ordering and project backend
can be set explicitly. Local CHOLMOD acquisition is already visible. These
facts make a one-cell *correctness* pilot plausible, not a representative
external performance benchmark.

## Deferred And Excluded Choices

| Choice | Disposition and missing evidence |
| --- | --- |
| SciPy `v1.17.0` / SuperLU | Deferred. The pinned wheel route is documented in the Day 2 survey, but SciPy and NumPy are absent locally. The genuinely nonsymmetric inline `lu_nonsym_square_5` fixture lacks canonical standalone bytes/hash; NumPy/BLAS provenance and pivot/ordering equivalence remain unresolved. It cannot replace CHOLMOD without a new decision and cell identity. |
| `tests/data/tridiagonal_20.mtx` | Future SPD expansion only. Different matrix hash and size require a separate cell; authorship/file terms still need review for published evidence. |
| `tests/data/suitesparse/nos4.mtx` | Future SPD/CSC expansion only. Require source attribution, upstream/checked-in byte verification, and CC-BY 4.0 metadata handling. Its existing internal canonical benchmark row is not an external baseline. |
| `tests/data/unsymm_5.mtx` | Excluded as a nonsymmetric LU input: its `general` header does not change its symmetric numerical values. |
| QR, SVD, LU and CSC refactor families | Excluded from the selected pilot because operation, rank/pivot rules or phase boundaries differ. A solver-family change needs an independent version, fixture and methodology decision. |

## Gates And Claim Boundary

Before executing the selected cell, verify exact fixture bytes, installed
binary and transitive-library fingerprints, build/ABI configuration, natural
ordering, simplicial mode, single-thread controls, conversion invariants and
both solver statuses. Record any missing item as **blocked**, not a passing
result. The Homebrew binary's module composition and build flags are not yet
certified; the [SuiteSparse tag](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/README.md)
lists required dependencies, while CHOLMOD's optional modules have distinct
licenses. Review module and fixture authorship/redistribution terms before
publishing or redistributing an external-evidence package. Local pilot
execution does not itself confer publication approval.

Correctness acceptance thresholds are Day 4 work. Timing phases, warmups,
repeats, dispersion, thread/BLAS checks and host comparability are Day 5
work; **no timing is authorized by this decision**. A one-cell correctness
outcome cannot establish solver parity, platform parity, portable speed, or
state-of-the-art status. Any change to package version, library digest,
matrix bytes, RHS, solver family, backend, ordering or precision requires a
new cell and an explicit decision-log entry.

## Day 3 Evidence

The decision uses the Day 2 version/fixture survey, repository loader and
Cholesky API inspection, and the recorded local package discovery. No solver,
benchmark or external comparison was executed. The local package's full
build provenance and the publishability of the small fixture remain named
gates; they are not inferred from availability or the repository license.
