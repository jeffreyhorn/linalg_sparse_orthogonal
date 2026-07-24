# Sprint 130 Day 11: SuiteSparse Corpus Gate

## Purpose

Day 11 decides whether SuiteSparse partial-SVD evidence is ready to become an
accepted Sprint 130 evidence lane. The answer is no for independent corpus
expected values: the checked-in SuiteSparse files are available for product
smoke and regression coverage, but the repository does not currently carry
independent singular-value, singular-vector, projector, conditioning, or
optimality metadata for them.

Day 12 may proceed with local analytic low-rank evidence. Corpus evidence may
proceed only as explicitly bounded smoke unless external metadata is added.

## Existing Corpus Inventory

| Matrix | Path | Shape | NNZ | Market type | Current support tier | Day 11 classification |
|---|---|---:|---:|---|---|---|
| `west0067` | `tests/data/suitesparse/west0067.mtx` | 67x67 | 294 | real general | checked-in unit fixture | Small nonsymmetric smoke candidate; no independent spectrum oracle. |
| `nos4` | `tests/data/suitesparse/nos4.mtx` | 100x100 | 347 | real symmetric | checked-in unit fixture | Small symmetric smoke candidate; already used by partial-SVD value/vector smoke. |
| `bcsstk04` | `tests/data/suitesparse/bcsstk04.mtx` | 132x132 | 1890 | real symmetric | checked-in unit fixture | Small symmetric low-rank/corpus safety candidate; no partial-SVD oracle. |
| `steam1` | `tests/data/suitesparse/steam1.mtx` | 240x240 | 3762 | real general | checked-in iterative-solver fixture | Nonsymmetric corpus candidate; not currently a partial-SVD evidence fixture. |
| `fs_541_1` | `tests/data/suitesparse/fs_541_1.mtx` | 541x541 | 4285 | real general | checked-in solver fixture | Medium nonsymmetric candidate; needs runtime and oracle metadata. |
| `orsirr_1` | `tests/data/suitesparse/orsirr_1.mtx` | 1030x1030 | 6858 | real general | checked-in solver fixture | Medium nonsymmetric candidate; needs runtime and oracle metadata. |
| `bcsstk14` | `tests/data/suitesparse/bcsstk14.mtx` | 1806x1806 | 32630 | real symmetric | checked-in reordered/partition fixture | Expensive SVD candidate; existing low-rank comments exclude it from unit low-rank SVD. |
| `s3rmt3m3` | `tests/data/suitesparse/s3rmt3m3.mtx` | 5357x5357 | 106526 | real symmetric | checked-in reorder fixture | Large corpus candidate; opt-in/benchmark only for SVD evidence. |
| `Kuu` | `tests/data/suitesparse/Kuu.mtx` | 7102x7102 | 173651 | real symmetric | checked-in reorder fixture | Large corpus candidate; opt-in/benchmark only for SVD evidence. |
| `bloweybq` | `tests/data/suitesparse/bloweybq.mtx` | 10001x10001 | 39996 | real symmetric | checked-in solver/reorder fixture | Large corpus candidate; opt-in/benchmark only for SVD evidence. |
| `Pres_Poisson` | `tests/data/suitesparse/Pres_Poisson.mtx` | 14822x14822 | 365313 | real symmetric | checked-in graph/reorder fixture | Large corpus candidate; opt-in/benchmark only for SVD evidence. |
| `tuma1` | `tests/data/suitesparse/tuma1.mtx` | 22967x22967 | 50560 | real symmetric | checked-in solver/reorder fixture | Large corpus candidate; opt-in/benchmark only for SVD evidence. |

## Existing Partial-SVD Corpus Coverage

| Test | Corpus data | Metric | Evidence status |
|---|---|---|---|
| `test_partial_svd_nos4` | `nos4`, `k=5` | partial singular values versus product full SVD within `0.1 * sigma` | Internal smoke; not independent expected-value evidence. |
| `test_partial_svd_west0067` | `west0067`, `k=3` | partial singular values versus product full SVD within `0.1 * sigma` | Internal smoke; not independent expected-value evidence. |
| `test_partial_svd_vectors_nos4` | `nos4`, `k=5` | product full-SVD values plus `A v - sigma u` residual | Internal vector smoke; no `A^T u` residual and no external oracle. |
| `test_partial_svd_vectors_west0067` | `west0067`, `k=3` | `A v - sigma u` residual only | Internal vector smoke; no external value, projector, or `A^T u` oracle. |
| `test_sparse_svd_lowrank_outer_product_corpus_safety` | `nos4`, `bcsstk04`, `k in {10, 50}` | env-off versus env-on sparse low-rank output agreement | Implementation parity smoke for accumulator path, not partial-SVD residual or optimality evidence. |

## Support-Tier Policy

| Tier | Data requirement | Allowed in default unit gate | Claim strength |
|---|---|---|---|
| Tier 0: local analytic | Matrix generated in the test with analytic singular values/projectors/residual target | Yes | Fixture-specific evidence when metrics and tolerances are declared. |
| Tier 1: checked-in smoke | Matrix file exists in `tests/data/suitesparse` and runtime is bounded on default CI | Yes, only with explicit skip/diagnostic messages if load fails | Product-regression smoke only unless independent metadata is present. |
| Tier 2: checked-in expensive | Matrix file exists but full/partial SVD runtime is too high or platform-sensitive | No, unless guarded by slow/experimental opt-in | Benchmark or opt-in diagnostics only. |
| Tier 3: optional external corpus | Matrix may be absent on developer or CI machines | No default requirement | Requires explicit skip behavior, source/version metadata, and independent oracle files before evidence promotion. |

Missing Tier 1 files may skip only when the test is documented as optional
smoke. Accepted evidence lanes must fail closed when their required local
fixture or oracle cannot be loaded.

## Oracle Policy

Product-observed values are not independent expected values. The following are
not sufficient to promote a corpus lane beyond smoke:

- `sparse_svd_partial` compared to `sparse_svd_compute` on the same matrix;
- a residual computed only from vectors returned by the product;
- env-off versus env-on product output agreement;
- historical console output copied from a prior product run.

A corpus lane may become accepted evidence only if it carries one of these
oracles:

| Oracle type | Required metadata |
|---|---|
| External singular values | Source/tool version, matrix checksum or fixture version, selected `k`, expected values, tolerance rationale, and failure class. |
| External projector/subspace | Source/tool version, basis or projector format, rank/subspace dimension, sign/rotation policy, tolerance rationale, and failure class. |
| Published corpus facts | Citation or checked-in metadata file, matrix version, fact type, tolerance window, and evidence owner. |
| Analytic corpus slice | Exact construction rule, expected values/projectors/residuals, and proof that the slice is not product-observed. |

## Diagnostics Policy

Every future corpus partial-SVD evidence lane must print or record:

- matrix key, path, shape, nnz, market symmetry, and support tier;
- selected `k`, options, tolerance, and max-iteration/budget settings;
- availability outcome: loaded, skipped as optional, or failed as required;
- singular-value error only against an independent oracle;
- `A v - sigma u` and `A^T u - sigma v` residuals when vectors are claimed;
- U/V orthogonality diagnostics when vectors are published;
- projector or principal-angle diagnostics when subspace ambiguity exists;
- runtime class: default unit, slow opt-in, benchmark, or deferred.

## Runtime Policy

Default Sprint 130 evidence should stay on Tier 0 analytic fixtures or small
Tier 1 smoke fixtures. Large checked-in matrices are not automatically safe for
partial-SVD evidence just because other solver tests use them. Full or partial
SVD work on `bcsstk14` and larger matrices must be slow/experimental or
benchmark-only unless a focused runtime measurement and skip policy is added.

## Day 12 Decision

Day 12 should not promote SuiteSparse corpus residual parity. It may do one of
these:

| Candidate | Decision | Reason |
|---|---|---|
| Local analytic low-rank optimality fixture | Accept for Day 12 | Can provide independent retained singular values and exact Frobenius tail error without corpus metadata. |
| Existing rectangular low-rank reconstruction fixture upgrade | Accept only if it stays fixture-specific | Already local and analytic; must state dense reconstruction semantics and avoid sparse-output/drop-tolerance claims. |
| `nos4` or `west0067` corpus residual evidence | Defer or keep smoke-only | Existing values are product-observed; no independent corpus oracle. |
| `bcsstk04` low-rank corpus safety | Defer or keep implementation parity smoke | Current check compares env-off/env-on product paths, not optimality. |
| `bcsstk14` or larger SVD corpus lane | Defer | Runtime and oracle metadata are missing for default evidence. |

Recommended Day 12 lane:
`partial_svd_lowrank_diag6x4_k2_frobenius_optimality`, a local analytic
rectangular diagonal fixture that checks `||A - U_k Sigma_k V_k^T||_F` against
the exact discarded singular-value tail. The claim must remain
fixture-specific and dense-reconstruction-only.

## Non-Claims

Day 11 does not claim:

- SuiteSparse singular-value parity;
- SuiteSparse vector, residual, or projector parity;
- broad corpus support across platforms;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  MATLAB parity;
- low-rank global optimality;
- sparse-output or drop-tolerance optimality;
- convergence-budget behavior;
- public solver-selection guidance.

## Completion Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Corpus evidence cannot proceed without skip and diagnostic metadata. | Complete | Support-tier, diagnostics, runtime, and optional-data policies are defined above. |
| Expected values are independent or explicitly bounded as smoke diagnostics. | Complete | Product-observed full-SVD comparisons and env-off/env-on agreement are classified as smoke only. |
| Runtime and platform expectations are explicit. | Complete | Tier policy separates default unit fixtures from slow/experimental and benchmark-only corpus lanes. |
