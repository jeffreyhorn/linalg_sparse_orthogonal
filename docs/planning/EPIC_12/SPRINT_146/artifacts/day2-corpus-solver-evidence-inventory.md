# Day 2 Corpus And Solver Evidence Inventory

## Scope

Day 2 inventories Epic 12 numerical evidence for corpus, QR, partial-SVD, and
solver-correctness surfaces. It records the exact fixture-local claims that are
supported, identifies the source-controlled evidence owners, separates
generated local rows from source-controlled pass proof, and lists unresolved
numerical gaps that remain after Sprints 137-140.

## Evidence Owner Table

| Evidence Area | Supported Claim | Source-Controlled Owner | Generated-Local Evidence | Validation Command | Freshness Status |
| --- | --- | --- | --- | --- | --- |
| Corpus fixture manifest | The maintained corpus currently has source-controlled generated fixture metadata for `qr_rank_deficient_6x4_nullspace_v1` and `partial_svd_clustered_repeated_diag8x6_k3_v1`. | `tests/corpus/manifests/fixtures.tsv`; `tests/corpus/manifests/generators.tsv`; `tests/corpus/README.md` | none required for manifest existence | `python3 scripts/validate_corpus_schema.py` | Source-controlled metadata is current until fixture/generator rows change. |
| QR expected rows | The QR fixture has source-controlled expected rank `3`, nullity `1`, and normalized null-vector residual `<= 1e-10` rows. | `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv`; `tests/corpus/schemas/oracle_fields.md` | `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` when regenerated | `python3 scripts/run_corpus_oracle.py --include-solver-qr` | Expected rows are source-controlled; observed rows are local generated evidence only. |
| QR compiled proof owner | For `qr_rank_deficient_6x4_nullspace_v1`, the project QR implementation reports rank `3`, reports nullity `1`, and produces a solver-backed nullspace vector residual at or below `1e-10`. | `tests/test_qr_corpus.c`; `tests/test_qr_helpers.h`; `Makefile`; `CMakeLists.txt` | optional oracle/report rows from `--include-solver-qr` | `make build/test_qr_corpus && ./build/test_qr_corpus`; CMake target `test_qr_corpus`; `make test` | Compiled proof is source-controlled; run output must be regenerated per validation baseline. |
| Partial-SVD expected rows | The partial-SVD fixture has source-controlled expected rows for top-3 singular values, left/right top-k subspace projectors, triplet residuals, orthogonality, default-budget success, tight-budget non-convergence, and no partial arrays on tight-budget failure. | `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`; `tests/corpus/schemas/oracle_fields.md` | `build/corpus/oracle/partial_svd_clustered_repeated_diag8x6_k3_v1.oracle.tsv` when regenerated | `python3 scripts/run_corpus_oracle.py --include-partial-svd` | Expected rows are source-controlled; observed rows are local generated evidence only. |
| Partial-SVD compiled proof owner | For `partial_svd_clustered_repeated_diag8x6_k3_v1`, the implementation verifies top-3 values, left/right subspace projectors, triplet residuals, orthogonality, default success, tight-budget fail-closed status, and no visible partial arrays on failure. | `tests/test_svd_partial_corpus.c`; `tests/test_svd_partial_shared_helpers.h`; `Makefile`; `CMakeLists.txt`; `include/sparse_svd.h` | optional oracle/report rows from `--include-partial-svd` | `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus`; CMake target `test_svd_partial_corpus`; `make test` | Compiled proof is source-controlled; run output must be regenerated per validation baseline. |
| Corpus schema and false-pass guardrails | Corpus TSV shape, required references, enum values, generator hashes, expected rows, known-generator parameters, and selected false-pass guardrails are validated. | `scripts/validate_corpus_schema.py`; `tests/corpus/schemas/*.md`; `tests/corpus/manifests/*.tsv`; `tests/corpus/expected/*.tsv` | validator output only | `python3 scripts/validate_corpus_schema.py` | Validation must be rerun after corpus metadata or expected rows change. |
| Oracle/report generation | Local oracle/report commands can emit observed rows, report indexes, skip/defer rows, and run manifests under ignored `build/` paths. | `scripts/run_corpus_oracle.py`; `tests/corpus/README.md`; `docs/maintainer_guide.md` | `build/corpus/oracle/*.tsv`; `build/corpus/reports/*`; run manifests | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Generated rows are reproducibility evidence only and must not be treated as checked-in pass proof. |
| Solver-selection public boundary | Public solver guidance states QR and partial-SVD evidence as fixture-local and rejects broad parity claims. | `docs/solver_selection.md`; `README.md`; `docs/maintainer_guide.md` | none | Markdown claim audit plus referenced validation commands | Current wording remains bounded by named fixtures and proof owners. |

## QR Evidence Boundary

The earned QR claim is deliberately narrow:

- fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`
- shape: 6 by 4
- nonzeros: 14
- expected rank: 3
- expected nullity: 1
- source-controlled expected rows:
  - `qr_rank_deficient_6x4_nullspace_v1_rank`
  - `qr_rank_deficient_6x4_nullspace_v1_nullity`
  - `qr_rank_deficient_6x4_nullspace_v1_projector_residual`
- compiled proof owner: `tests/test_qr_corpus.c`
- helper owner: `tests/test_qr_helpers.h`
- fixture-local tolerance: normalized null-vector residual `<= 1e-10`

Supported QR wording may say that the project QR implementation proves rank,
nullity, and normalized nullspace residual behavior for this maintained
generated fixture. It may not claim raw QR basis parity, global rank-threshold
policy, broad rank-deficient solve behavior, SuiteSparse parity, hosted
platform parity, performance, or state-of-the-art status.

## Partial-SVD Evidence Boundary

The earned partial-SVD claim is also fixture-local:

- fixture key: `partial_svd_clustered_repeated_diag8x6_k3_v1`
- generator key: `partial_svd_clustered_repeated_diag8x6_generator_v1`
- shape: 8 by 6
- nonzeros: 5
- requested rank: `k = 3`
- expected rank: 5
- expected nullity: 1
- diagonal values: `10,10,9.999999,4,1,0`
- source-controlled expected rows cover:
  - top-3 singular values
  - left top-k subspace projector distance
  - right top-k subspace projector distance
  - maximum triplet residual
  - maximum orthogonality residual
  - default-budget success
  - tight-budget `SPARSE_ERR_NOT_CONVERGED`
  - no partial `sigma`, `U`, or `Vt` arrays on tight-budget failure
- compiled proof owner: `tests/test_svd_partial_corpus.c`
- helper owner: `tests/test_svd_partial_shared_helpers.h`
- fixture-local tolerance: `1e-8` for value, projector, residual, and
  orthogonality rows where applicable

Supported partial-SVD wording may say that the implementation proves the named
clustered/repeated fixture behavior. It may not claim broad partial-SVD
correctness, raw singular-vector identity, broad repeated-spectrum coverage,
external-library parity, convergence-rate guarantees, partial-result
guarantees after non-convergence, performance, hosted platform parity, or
state-of-the-art status.

## Source-Controlled Vs Generated-Local Classification

| Artifact Type | Examples | Classification | Closeout Rule |
| --- | --- | --- | --- |
| Fixture and generator metadata | `tests/corpus/manifests/fixtures.tsv`; `tests/corpus/manifests/generators.tsv` | Source-controlled evidence contract | May support fixture existence, parameters, ownership, non-claims, and validation command references. |
| Expected oracle rows | `tests/corpus/expected/*.tsv` | Source-controlled expected-result contract | May support expected comparison semantics, not observed solver pass status by itself. |
| Focused compiled tests | `tests/test_qr_corpus.c`; `tests/test_svd_partial_corpus.c` | Source-controlled proof owners | May support fixture-local solver claims when the relevant validation command passes. |
| Helper headers | `tests/test_qr_helpers.h`; `tests/test_svd_partial_shared_helpers.h` | Source-controlled proof implementation support | May support maintainability and shared comparison semantics, not an independent claim. |
| Oracle/report rows | `build/corpus/oracle/*.tsv`; `build/corpus/reports/*` | Generated local evidence | May support reproducibility for the command, commit, platform, compiler, configuration, and support tier that generated it. |
| Run manifests | ignored `build/` corpus manifests | Generated local freshness evidence | Must be regenerated or reconciled before final validation claims. |
| Public docs | `README.md`; `docs/solver_selection.md`; `docs/maintainer_guide.md`; `tests/corpus/README.md` | Source-controlled claim boundary | May describe only claims supported by the evidence owners above. |

## Validation Commands For Day 2 Evidence

| Command | Evidence It Refreshes | Notes |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Corpus manifest, generator, expected-row, schema, and false-pass guardrails | Required after corpus metadata or expected rows change. |
| `python3 scripts/run_corpus_oracle.py --include-solver-qr` | QR solver-backed local oracle/report rows | Requires the static library or an explicit solver library path. Generated rows stay under ignored `build/` paths. |
| `python3 scripts/run_corpus_oracle.py --include-partial-svd` | Partial-SVD local oracle/report rows | Generated rows stay under ignored `build/` paths. |
| `make build/test_qr_corpus && ./build/test_qr_corpus` | Focused QR compiled proof | Required when QR corpus proof owner changes; also useful for Sprint 146 final validation. |
| `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus` | Focused partial-SVD compiled proof | Required when partial-SVD corpus proof owner changes; also useful for Sprint 146 final validation. |
| `make test` | Full Make test suite | Required when `.c` or `.h` files change and planned for the Sprint 146 full quality baseline. |

## Unresolved Numerical Gaps

| Gap | Current Owner Surface | Current Status | Promotion Gate |
| --- | --- | --- | --- |
| Broad QR correctness | QR tests and solver docs | Not claimed | More fixtures across shape, rank, conditioning, reorder, solve, minimum-norm, and optional external corpora, with reviewed solver-backed rows. |
| Raw QR basis parity | QR docs and maintainer guidance | Explicit non-claim | A mathematically appropriate basis-invariant comparison policy, not raw basis equality. |
| Global QR rank-threshold policy | QR implementation/tests | Residual | Scale/perturbation fixture families and documented threshold semantics. |
| Broad rank-deficient QR solve and minimum-norm behavior | QR solve tests and solver-selection docs | Residual/non-claim | Focused solve/minimum-norm corpus rows and proof owners. |
| Broad partial-SVD correctness | SVD tests and public SVD docs | Not claimed | Multiple maintained fixtures across spectra, shapes, rank deficiency, nonsymmetry, and sparse structures. |
| Broad repeated-spectrum behavior | Partial-SVD corpus/docs | Fixture-local only | Additional repeated/clustered fixture families with subspace-safe comparisons. |
| Partial-SVD convergence-rate guarantees | Partial-SVD docs/tests | Explicit non-claim | Budgeted convergence study with reproducible thresholds and platform/implementation constraints. |
| Partial-result guarantees after non-convergence | Partial-SVD docs/tests | Explicit non-claim except no visible arrays for the named tight-budget failure | Contracted API semantics and tests for partial-output policy. |
| External library parity | Public docs and maintainer guidance | Explicit non-claim | Direct comparative evidence against named libraries and fixture families. |
| State-of-the-art numerical claim | Epic 12 closeout | Explicit non-claim unless final comparative evidence exists | Direct comparative benchmarks and feature/correctness evidence against appropriate sparse linear algebra baselines. |

## Day 3 Handoff

Day 3 should inventory the non-numerical support families: report indexes,
runtime/backend governance, package and ABI support, platform lanes, adoption
surfaces, and validation evidence. It should reuse this Day 2 distinction
between source-controlled contracts, compiled proof owners, generated local
rows, and public claim wording.
