# Sprint 131 Day 2 - Numerical Fixture Inventory

## Purpose

Day 2 inventories checked-in Matrix Market fixtures and generated numerical
families without assigning new support claims. The goal is visibility:
dimensions, Matrix Market structure, current owner, default interpretation,
and missing metadata are recorded before Sprint 131 defines corpus taxonomy or
report indexes.

This artifact separates checked-in corpus files from local analytic and
generated fixtures. Generated families remain product regression inputs unless
later artifacts define independent corpus status, oracle provenance, support
tier, validation, and non-claim boundaries.

## Checked-In Matrix Market Fixtures

| Fixture | Path | Shape | Stored entries | Market type | Structure and hints | Current owner | Default interpretation |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `bad_header` | `tests/data/bad_header.mtx` | malformed | n/a | invalid header | Parser negative case. | `tests/test_sparse_io.c` | Expected parse failure; not numerical evidence. |
| `identity_5` | `tests/data/identity_5.mtx` | 5x5 | 5 | real general | Diagonal identity; SPD and full rank by construction, but stored as general. | `tests/test_sparse_io.c`, `tests/test_known_matrices.c` | Local checked-in analytic fixture for IO and LU residual regression. |
| `diagonal_10` | `tests/data/diagonal_10.mtx` | 10x10 | 10 | real general | Positive diagonal with entries 1-10; SPD and full rank by construction, but stored as general. | `tests/test_sparse_io.c`, `tests/test_known_matrices.c` | Local checked-in analytic fixture for IO, LU residual, and conditioning-style behavior. |
| `pattern_3` | `tests/data/pattern_3.mtx` | 3x3 | 5 | pattern general | Pattern-only sparse structure; no numeric definiteness or conditioning metadata. | `tests/test_sparse_io.c` | Parser and structural IO fixture only. |
| `symmetric_4` | `tests/data/symmetric_4.mtx` | 4x4 | 7 | real symmetric | Small symmetric matrix; definiteness/rank are implied by tests only, not recorded as independent metadata. | `tests/test_sparse_io.c`, `tests/test_known_matrices.c` | Local checked-in fixture for symmetric Matrix Market expansion and LU residual regression. |
| `tridiagonal_20` | `tests/data/tridiagonal_20.mtx` | 20x20 | 58 | real general | 1D Poisson-style tridiagonal with explicit upper/lower entries; SPD by construction, stored as general. | `tests/test_sparse_io.c`, `tests/test_known_matrices.c` | Local checked-in analytic fixture for IO and LU/refinement residual regression. |
| `unsymm_5` | `tests/data/unsymm_5.mtx` | 5x5 | 13 | real general | Small nonsymmetric matrix; rank/conditioning metadata not recorded. | `tests/test_sparse_io.c` | Parser and unsymmetric load fixture. |
| `bcsstk01` | `tests/data/bcsstk01.mtx` | 6x6 | 12 | real symmetric | Small structural-engineering stiffness-inspired matrix; comments call it SPD. | `tests/test_known_matrices.c` | Local checked-in named-structure fixture for LU residual regression, not SuiteSparse corpus parity. |

## Checked-In SuiteSparse-Derived Fixtures

The stored-entry counts below are Matrix Market stored entries before symmetric
expansion. The current default support tier is checked-in corpus data, but
not independent SuiteSparse parity unless an owner-specific artifact adds
oracle metadata.

| Fixture | Path | Shape | Stored entries | Market type | Structure and hints | Current owner | Default interpretation |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `west0067` | `tests/data/suitesparse/west0067.mtx` | 67x67 | 294 | real general | Chemical engineering nonsymmetric matrix; structurally zero diagonal noted in algorithm docs for ILU behavior. | QR, SVD, COLAMD, BiCGSTAB, benchmarks | Small nonsymmetric corpus smoke; no independent spectrum or solve oracle by default. |
| `nos4` | `tests/data/suitesparse/nos4.mtx` | 100x100 | 347 | real symmetric | Structural symmetric beam matrix; used as small SPD-like corpus surface in Cholesky, IC, eigensolver, SVD, benchmark, and report lanes. | Cholesky CSC, IC, SVD, QR sparse-mode, eigensolver, benchmarks, report scripts | Tier-1 checked-in smoke and benchmark/report fixture; only owner-specific bounded claims apply. |
| `bcsstk04` | `tests/data/suitesparse/bcsstk04.mtx` | 132x132 | 1890 | real symmetric | Structural stiffness matrix; commonly treated as small SPD corpus fixture. | Cholesky CSC, IC, eigensolver, reorder, benchmarks | Tier-1 checked-in smoke; no broad corpus or eigensolver parity. |
| `steam1` | `tests/data/suitesparse/steam1.mtx` | 240x240 | 3762 | real general | Thermal/oil-reservoir nonsymmetric matrix. | BiCGSTAB, SVD benchmark | Iterative and SVD benchmark/corpus candidate; no independent expected values. |
| `fs_541_1` | `tests/data/suitesparse/fs_541_1.mtx` | 541x541 | 4285 | real general | Nonsymmetric facsimile convergence matrix. | SuiteSparse download/inventory and solver candidates | Medium nonsymmetric corpus candidate; owner metadata incomplete. |
| `orsirr_1` | `tests/data/suitesparse/orsirr_1.mtx` | 1030x1030 | 6858 | real general | Oil reservoir nonsymmetric matrix. | BiCGSTAB benchmark, SVD benchmark | Medium nonsymmetric benchmark/corpus candidate; no independent oracle metadata. |
| `bcsstk14` | `tests/data/suitesparse/bcsstk14.mtx` | 1806x1806 | 32630 | real symmetric | Structural stiffness matrix; large enough to serve as reorder/eigs/guardrail fixture. | Reorder, graph, eigensolver, Cholesky/LDLT benchmarks, wall-check, guardrails | Checked-in expensive fixture for structural and benchmark/report lanes; not default SVD corpus evidence. |
| `s3rmt3m3` | `tests/data/suitesparse/s3rmt3m3.mtx` | 5357x5357 | 106526 | real symmetric | Cylindrical shell FEM matrix. | Reorder and factorization benchmarks | Large checked-in benchmark/report candidate; not default reviewed numerical oracle. |
| `Kuu` | `tests/data/suitesparse/Kuu.mtx` | 7102x7102 | 173651 | real symmetric | Finite-element stiffness matrix. | Reorder and factorization benchmarks, algorithm documentation | Large checked-in benchmark/report candidate; no independent runtime or solver claim. |
| `bloweybq` | `tests/data/suitesparse/bloweybq.mtx` | 10001x10001 | 39996 | real symmetric | Symmetric indefinite matrix from GHS_indef. | LDLT/reorder candidate surfaces | Large indefinite corpus candidate; factorization disposition is owner-specific. |
| `Pres_Poisson` | `tests/data/suitesparse/Pres_Poisson.mtx` | 14822x14822 | 365313 | real symmetric | Pressure Poisson CFD matrix. | Reorder, graph, wall-check, large-matrix guardrails | Large structural guardrail and benchmark/report fixture; not broad scalability proof. |
| `tuma1` | `tests/data/suitesparse/tuma1.mtx` | 22967x22967 | 50560 | real symmetric | Symmetric indefinite mine-model matrix. | LDLT/reorder candidate surfaces | Large indefinite corpus candidate; runtime and oracle metadata incomplete. |

## Generated-Family Inventory

| Family | Representative owners | Structure and numerical hints | Current evidence role | Missing metadata before corpus promotion |
| --- | --- | --- | --- | --- |
| Identity and diagonal systems | `tests/test_bicgstab.c`, `tests/test_chol_csc.c`, `tests/test_ic.c`, `tests/test_known_matrices.c`, `tests/test_svd.c`, `tests/test_dense.c` | Analytic diagonal, identity, scaled identity, and positive diagonal matrices; rank and conditioning often known by construction. | Local analytic regression and exact-value checks. | Stable fixture key, taxonomy tags, tolerance policy, and owner-specific claim boundary. |
| Tridiagonal SPD / 1D Poisson systems | `tests/test_ic.c`, `tests/test_chol_csc.c`, `tests/test_cross_solver_oracle.c`, `tests/test_eigs.c`, `tests/test_known_matrices.c`, `tests/test_integration_fixtures.h` | Symmetric positive-definite banded matrices with known residual/eigenvalue behavior in selected tests. | Solver, preconditioner, cross-solver, eigensolver, and integration regression. | Explicit construction metadata, dimensions, spectral facts when claimed, support tier, and validation owner. |
| Banded SPD and random SPD systems | `tests/test_chol_csc.c`, `tests/test_ic.c`, `tests/test_direct_csc_dispatch.c` | Banded or generated lower-triangular products, sometimes seeded. | Fill, allocation, direct dispatch, factorization, and preconditioner regression. | Seed policy, determinism statement, conditioning/rank metadata, and reviewed versus stress classification. |
| Rank-deficient, duplicate-column, dependent-row, threshold-rank families | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Rectangular or square fixtures with exact or near rank deficiencies, threshold perturbations, and range/nullspace behavior. | Bounded QR/SVD rank, residual, projector, and partial-SVD evidence where previous sprint artifacts accepted it; otherwise local regression. | Taxonomy tags for rank/nullity, perturbation scale, oracle source, tolerance, and support tier. |
| Rectangular full-rank and low-rank SVD families | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Tall, wide, rectangular, low-rank, and diagonal fixtures with local analytic or external-reference value checks. | Bounded SVD/partial-SVD evidence only for named accepted lanes; otherwise implementation regression. | Clear split between analytic, external-reference, and product-observed expectations. |
| Nonsymmetric known-solution systems | `tests/test_bicgstab.c`, `tests/test_integration_fixtures.h`, `tests/test_qr_solve.c`, `tests/test_svd_partial_helpers.h` | Small dense-ish or sparse nonsymmetric matrices with generated RHS or reference comparisons. | Iterative, QR, SVD, and integration regression. | Solver family tags, oracle provenance, expected-failure semantics, and tolerance policy. |
| Graph fixtures | `tests/test_graph_fixtures.h`, `tests/test_graph.c`, `tests/test_reorder*.c` | 1D paths, 2D grids, 3D meshes, complete graphs, bipartite graphs, cliques with bridges, asymmetric boundary graphs. | Structural graph/reorder/partition regression and large-matrix guardrail support. | Matrix-versus-graph taxonomy tags, scale class, ordering ownership, and reviewed/supplemental report status. |
| KKT and indefinite generated systems | `tests/test_integration_fixtures.h`, `tests/test_eigs_thick_restart.c`, direct-solver tests | Saddle-point or block indefinite matrices, often symmetric but not positive definite. | Direct lifecycle, LDLT, eigensolver, and integration regression. | Definiteness tags, nonsingularity proof, conditioning hints, and failure interpretation. |
| Dense helper matrices | `tests/test_dense.c`, `tests/test_svd.c` | Dense identities, known products, small eigenvalue examples, bidiagonal/arrowhead reductions. | Dense algebra and decomposition regression. | Sparse/dense scope tag, exact expected values, and relation to sparse corpus taxonomy. |

## Support-Tier Boundary Notes

| Tier | Sources currently in scope | Default claim strength |
| --- | --- | --- |
| Local analytic checked-in fixture | `identity_5`, `diagonal_10`, `tridiagonal_20`, `symmetric_4`, `bcsstk01`, and selected generated exact families | Fixture-specific regression or bounded evidence only when owner/tolerance/oracle are explicit. |
| Parser/structural checked-in fixture | `bad_header`, `pattern_3`, Matrix Market shape and symmetry fixtures | Parser or structural IO evidence, not numerical solver evidence. |
| Checked-in corpus smoke | Small SuiteSparse-derived files such as `west0067`, `nos4`, `bcsstk04`, `steam1` where runtime is bounded for current owner tests | Product regression smoke unless independent metadata and oracle provenance exist. |
| Checked-in expensive corpus/report fixture | `bcsstk14`, `s3rmt3m3`, `Kuu`, `Pres_Poisson`, `bloweybq`, `tuma1`, and medium/large generated stress families | Benchmark, guardrail, slow, or supplemental report context unless promoted by a later gate. |
| Optional external corpus | Downloaded or absent-by-default data outside the checked-in files | Not required for default validation; must have skip, checksum/version, source, and independent oracle policy before reviewed use. |

## Missing-Metadata Queue

| Gap | Affected sources | Required before reviewed corpus evidence |
| --- | --- | --- |
| Fixture owner is implicit or shared across unrelated tests. | Most checked-in Matrix Market files and generated families. | A single corpus owner or explicit multi-owner split. |
| Matrix metadata is incomplete. | `symmetric_4`, `unsymm_5`, many generated families. | Symmetry, definiteness, rank/nullity, conditioning, scale, sparsity pattern, and ordering tags. |
| Stored versus expanded nonzero count is not normalized. | Symmetric Matrix Market files and SuiteSparse-derived fixtures. | Policy that records stored entries and expanded logical nnz separately when needed. |
| Independent oracle provenance is missing. | SuiteSparse-derived fixtures and product-observed generated families. | External values/projectors/residuals, published facts, or analytic proof with source/version and tolerance. |
| Runtime/support tier is not recorded per fixture. | Medium and large SuiteSparse files, graph stress families, benchmark-only inputs. | Default-unit, slow, benchmark, supplemental, or optional classification. |
| Generated fixtures lack stable taxonomy keys. | Most in-code builders. | Stable names, construction parameters, and owner files. |
| Expected failures and skips are not tied to fixture metadata. | Parser failures, optional file loads, SuiteSparse smoke tests, unsupported solver modes. | Failure class, skip condition, required versus optional status, and diagnostics. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every checked-in numerical corpus source has an owner or explicit unknown owner. | Complete | Checked-in Matrix Market and SuiteSparse-derived fixture tables list current owners; owner gaps are recorded in the missing-metadata queue. |
| Generated fixtures are not confused with independent external corpus evidence. | Complete | Generated-family inventory classifies local analytic/generated inputs separately from checked-in corpus files. |
| Missing metadata is recorded as a blocker, not left implicit. | Complete | Missing-metadata queue records owner, metadata, oracle, runtime, taxonomy-key, and skip/failure blockers. |

## Day 3 Handoff

Day 3 should inventory external-reference helper scripts, fixture keys,
expected failures, skips, and optional-data gates. It should connect helper
protocols and skip/failure behavior to this Day 2 fixture inventory without
promoting product-observed SuiteSparse or generated-family outputs into
independent oracle evidence.

