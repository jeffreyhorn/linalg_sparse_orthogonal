# Sprint 150 Day 1: QR Intake

## Purpose

Establish the Sprint 150 baseline before selecting QR fixture families. Day 1
ties the sprint scope to current repository files, prior sprint handoffs,
existing QR proof owners, corpus/report rows, and explicit stop conditions.

## Prior Sprint Context

| Source | Relevant Sprint 150 Input |
| --- | --- |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Sprint 150 must close two or three bounded QR fixture families with fixture metadata, oracle semantics, focused tests, report rows, docs, and validation. |
| `docs/planning/EPIC_13/SPRINT_147/artifacts/day8-corpus-family-evidence-gate.md` | Corpus-family promotion requires fixture rows, generator rows, expected rows, proof-owner tests, oracle/report rows, validation commands, support tier, claim scope, and non-claims. |
| `docs/planning/EPIC_12/SPRINT_139/RETROSPECTIVE.md` | Sprint 139 closed only `qr_rank_deficient_6x4_nullspace_v1` with rank `3`, nullity `1`, and normalized nullspace residual `<= 1e-10`. |
| `docs/planning/EPIC_13/SPRINT_149/artifacts/day14-closeout-handoff.md` | Sprint 150 QR work must not infer Windows Makefile, Windows `pkg-config`, package-manager, shared-library, dynamic ABI, runtime-loader, broad Windows parity, or QR platform proof from package-lane evidence. |

## Current QR Implementation Surface

| Surface | Files | Day 1 Interpretation |
| --- | --- | --- |
| Public API | `include/sparse_qr.h` | Owns QR factorization, apply/form Q, solve, refinement, rank, nullspace, rank diagnostics, condition estimate, minimum-norm solve, and minimum-norm refinement declarations. |
| Implementation | `src/sparse_qr.c` | Owns factorization, solve, rank/nullspace, minimum-norm, refinement, COLAMD option handling, and QR lifecycle behavior. |
| Householder internals | `src/sparse_qr_householder.c`, `src/sparse_qr_internal.h` | Owns private QR Householder helper kernels; not a public corpus proof owner. |

## Current QR Test And Proof-Owner Surface

| Surface | Files | Coverage Already Present |
| --- | --- | --- |
| Broad QR behavior | `tests/test_qr.c` | Internal invariants, reconstruction, rank, nullspace, Q application, economy mode, sparse mode, rank thresholds, and QR refinement. |
| QR solve behavior | `tests/test_qr_solve.c` | Square, overdetermined, compatible/incompatible, rank-deficient, residual-only, external dense-reference, and selected SuiteSparse local solve behavior. |
| Focused QR corpus proof | `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h` | Current maintained corpus proof for `qr_rank_deficient_6x4_nullspace_v1`: shape, rank, nullity, solver nullspace residual, and reference direction residual. |
| QR/COLAMD and minimum-norm owner-local behavior | `tests/test_colamd.c` | COLAMD ordering, QR+COLAMD solves, COLAMD-vs-AMD residual comparison, sparse mode, minimum-norm cases, rank diagnostics, and condition estimate checks. |
| External dense reference helper | `tests/qr_external_dense_reference.py` | Existing helper for bounded external dense-reference rows in QR tests; not a broad external-library parity claim. |

## Current Source-Controlled QR Corpus Rows

| Row Surface | Current QR Row(s) | Owner / Interpretation |
| --- | --- | --- |
| Fixture metadata | `qr_rank_deficient_6x4_nullspace_v1` in `tests/corpus/manifests/fixtures.tsv` | One maintained generated QR fixture, family `qr_rank_deficient`, rank `3`, nullity `1`, local-only support tier. |
| Generator metadata | `qr_rank_deficient_6x4_nullspace_generator_v1` in `tests/corpus/manifests/generators.tsv` | Deterministic fixed-columns generator with structure/value hashes and regeneration command. |
| Expected results | `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Three ready-for-oracle rows: rank, nullity, and normalized null-vector residual. |
| Optional data | `suitesparse_rank_deficient_qr_subset_v1` in `tests/corpus/manifests/optional_data.tsv` | Disabled optional SuiteSparse QR subset; skip/defer policy only, not QR pass evidence. |
| Report family | `oracle/generated_reference` and `oracle/solver_backed` in `tests/corpus/manifests/report_families.tsv` | Generated-local oracle rows may support only named fixtures, commands, commit/platform/compiler/configuration, support tier, claim scope, and non-claims. |

## Current QR Oracle And Report Surface

| Surface | File / Command | Current Meaning |
| --- | --- | --- |
| Schema validation | `python3 scripts/validate_corpus_schema.py` | Validates source-controlled corpus rows and hashes. |
| QR oracle generation | `python3 scripts/run_corpus_oracle.py --include-solver-qr` | Emits generated-reference rows plus three solver-backed QR rows for the Sprint 139 fixture when the static library is available. |
| Report index normalization | `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` and freshness variants | Normalizes source-controlled and generated-local report rows; absence of generated rows is not pass evidence. |
| Generated artifacts | `build/corpus/`, `build/corpus-reports/`, `build/report-index/` | Ignored local output paths; not committed unless a later sprint explicitly changes policy. |

## Existing QR Coverage Not Yet Maintained Corpus Families

The repository already contains useful owner-local QR evidence that can seed
Sprint 150 family selection, but these are not yet complete maintained corpus
families:

| Candidate Area | Existing Evidence | Missing For Sprint 150 Corpus Promotion |
| --- | --- | --- |
| Rank-deficient rectangular | `tests/test_qr.c`, `tests/test_qr_solve.c`, and helper fixtures for duplicate columns, dependent rows, rank thresholds, nullspace projector/subspace checks, and rank-only/residual-only dense-reference checks. | Source-controlled fixture rows, generator rows, expected rows, family claim scopes, tolerances, oracle rows, report rows, and focused corpus proof ownership for selected fixtures. |
| Underdetermined minimum-norm | `tests/test_qr_solve.c` and `tests/test_colamd.c` cover known 2x4, 3x6, 5x10, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-pseudoinverse, and `west0067` submatrix behavior. | Source-controlled fixture rows, expected norm/residual semantics, minimum-norm oracle rows, tolerances, report rows, and non-claims that prevent global minimum-norm guarantees. |
| Reorder/COLAMD QR | `tests/test_qr.c` and `tests/test_colamd.c` cover AMD/COLAMD QR solves, fill/residual comparisons, sparse mode, backward compatibility, and selected SuiteSparse local matrices. | Ordering-specific claim scope, permutation/fill/status semantics, report rows, and non-claims for reorder optimality, COLAMD parity, performance, and SuiteSparse corpus completeness. |

## Current Claim Boundaries

- Current maintained QR corpus proof is fixture-local to
  `qr_rank_deficient_6x4_nullspace_v1`.
- It proves rank `3`, nullity `1`, and normalized solver-produced nullspace
  residual `<= 1e-10`.
- Expected rows are source-controlled metadata and are not observed solver pass
  evidence by themselves.
- Solver-backed QR rows are generated-local and local-only until a later sprint
  promotes hosted evidence.
- Optional external QR data is disabled and skip/defer evidence only.
- Current docs explicitly reject broad QR correctness, raw QR basis parity,
  global rank-threshold policy, broad rank-deficient solve behavior, broad
  nullspace/minimum-norm/economy/reorder claims, external-library parity,
  SuiteSparse corpus completeness, platform parity, package/ABI claims,
  performance claims, and state-of-the-art claims.

## Stop Conditions

| Stop Condition | Why It Stops Work |
| --- | --- |
| A proposed expected row requires raw Q/R basis equality, sign, orientation, column order, or raw basis-vector identity. | Valid QR bases are not unique; Sprint 150 must use residual, projector, rank, nullity, norm, status, or subspace-safe metrics. |
| A proposed family cannot name fixture keys, generator rows, expected rows, tolerances, support tier, proof owner, validation command, and non-claims. | The family is not ready for maintained corpus promotion. |
| A source-controlled expected row is cited as observed pass evidence. | Expected rows define targets; they do not prove solver behavior. |
| Generated oracle/report rows omit command, commit, platform, compiler, configuration, support tier, claim scope, or non-claims. | The report cannot support a bounded generated-local claim. |
| Optional-data skip/defer rows are counted as solver pass evidence. | Disabled optional data is policy evidence only. |
| Documentation widens selected fixtures into broad QR, external-library, platform, package, performance, or state-of-the-art claims. | The sprint would overstate its evidence. |
| Sprint 149 Windows package evidence is reused as QR platform proof. | Package-lane validation is not QR numerical or platform proof. |
| Required focused tests, corpus schema, oracle/report checks, or full C gates fail after implementation changes. | The sprint cannot close with failing required validation. |

## Day 2 Handoff

Day 2 should audit candidate QR families with concrete repository evidence:

1. Rank-deficient rectangular QR candidates.
2. Underdetermined minimum-norm QR candidates.
3. Reorder/COLAMD QR candidates.
4. Fixture/generator/expected-row needs for each candidate.
5. Oracle semantics and tolerance readiness.
6. Closure value versus implementation and documentation risk.
