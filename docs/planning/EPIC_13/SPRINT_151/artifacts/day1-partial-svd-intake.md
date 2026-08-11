# Sprint 151 Day 1: Partial-SVD Intake

## Purpose

Establish the Sprint 151 scope, artifact structure, current partial-SVD
corpus/test/report baseline, proof owners, claim boundaries, and stop
conditions before selecting new maintained corpus families.

## Source Context

| Source | Day 1 Finding |
| --- | --- |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Sprint 151 targets a broader but still bounded maintained partial-SVD corpus family beyond the single Sprint 140 fixture. |
| `docs/planning/EPIC_12/SPRINT_140/RETROSPECTIVE.md` | Sprint 140 closed one fixture-local partial-SVD lane with source-controlled metadata, expected rows, generated-reference oracle rows, focused proof-owner tests, docs, and full validation. |
| `docs/planning/EPIC_13/SPRINT_150/artifacts/day14-closeout-handoff.md` | Sprint 150 provides the expansion pattern: select a small closable family, define comparison semantics first, add focused proof owners, reset generated-local reports, and keep broad/platform/package claims out of scope. |
| `docs/planning/EPIC_13/SPRINT_151/PLAN.md` | Day 1 owns intake, artifact setup, inventory, current claim-boundary snapshot, and stop conditions. |

## Artifact Setup

Created the Sprint 151 planning surface:

- `docs/planning/EPIC_13/SPRINT_151/PLAN.md`
- `docs/planning/EPIC_13/SPRINT_151/WORKING_NOTES.md`
- `docs/planning/EPIC_13/SPRINT_151/artifacts/day1-partial-svd-intake.md`

## Implementation Inventory

Current partial-SVD implementation and public API surfaces:

| Surface | Files | Day 1 Notes |
| --- | --- | --- |
| Public SVD API | `include/sparse_svd.h` | Public full-SVD, partial-SVD, condition, pseudoinverse, dense low-rank, and sparse low-rank declarations and comments. |
| SVD implementation | `src/sparse_svd.c` | Full SVD, condition, pseudoinverse, low-rank, and shared SVD behavior. |
| Partial-SVD implementation | `src/sparse_svd_partial.c` | Lanczos-based partial-SVD behavior and convergence-budget handling. |
| Internal SVD declarations | `src/sparse_svd_internal.h` | Internal helpers and shared implementation contracts. |
| Examples and benchmarks | `examples/example_svd_lowrank.c`, `benchmarks/bench_svd.c` | Adoption and performance-adjacent surfaces; not proof owners for corpus claims. |

## Test And Helper Inventory

Current partial-SVD proof surfaces:

| Surface | Owner Status | Evidence |
| --- | --- | --- |
| `tests/test_svd_partial_corpus.c` | Maintained corpus proof owner | Owns the Sprint 140 fixture-local partial-SVD corpus closure with six focused tests. |
| `tests/test_svd_partial_shared_helpers.h` | Shared focused helper owner | Owns reusable triplet-residual and coordinate-range projector helpers used by the corpus proof owner. |
| `tests/test_svd_partial_helpers.h` | Owner-local candidate lane | Contains many partial-SVD behaviors in the broader `test_svd` binary: external singular values, vector residuals, rank-deficient projectors, dense low-rank, sparse low-rank, tight-budget failure, and nonsymmetric/rectangular cases. |
| `tests/test_svd.c` | Broad SVD owner | Runs full-SVD, partial-SVD, rank, condition, pseudoinverse, and low-rank tests; should not become the primary maintained corpus registry. |
| `tests/test_svd_helpers.h` | Shared broad SVD helpers | Fixture and numerical helper surface used by broad SVD tests. |
| `tests/svd_external_dense_reference.py` | Optional external reference helper | Useful for named external dense-reference fixtures, but not broad LAPACK/NumPy/SciPy parity proof. |

Focused current corpus proof count from Sprint 140:

- `test_partial_svd_corpus_clustered_repeated_default_success`
- `test_partial_svd_corpus_clustered_repeated_projectors`
- `test_partial_svd_corpus_clustered_repeated_residuals`
- `test_partial_svd_corpus_clustered_repeated_tight_budget_fail_closed`
- `test_partial_svd_corpus_clustered_repeated_recovery_after_failure`
- `test_partial_svd_corpus_full_rank_truncate_path`

## Corpus And Report Inventory

Current maintained partial-SVD corpus files:

| Surface | Current State |
| --- | --- |
| `tests/corpus/manifests/fixtures.tsv` | Contains one maintained partial-SVD fixture row: `partial_svd_clustered_repeated_diag8x6_k3_v1`. |
| `tests/corpus/manifests/generators.tsv` | Contains generator row `partial_svd_clustered_repeated_diag8x6_generator_v1`. |
| `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` | Contains eight expected rows for singular values, left/right subspace, vector residual, orthogonality, default status, tight-budget status, and no partial arrays. |
| `tests/corpus/manifests/report_families.tsv` | Allows `oracle` generated-local rows for `qr` and `partial_svd` through `scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`. |
| `scripts/validate_corpus_schema.py` | Owns deterministic generated fixture validation for the Sprint 140 partial-SVD fixture. |
| `scripts/run_corpus_oracle.py` | Owns `--include-partial-svd` generated-reference oracle rows and report manifest row counts. |
| `scripts/normalize_report_index.py` | Owns normalized report-index and freshness interpretation. |

Current local partial-SVD oracle command:

```sh
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

Expected generated-local partial-SVD report surface today:

- `partial_svd_row_count=8`
- fixture key `partial_svd_clustered_repeated_diag8x6_k3_v1`
- generated-reference rows under ignored `build/` outputs
- `support_tier=local_only`
- no source-controlled generated oracle/report output

## Current Claim Boundary

Currently closed from Sprint 140:

The generated 8 by 6 clustered/repeated diagonal partial-SVD corpus fixture
`partial_svd_clustered_repeated_diag8x6_k3_v1` verifies top-3 singular values,
left and right top-k subspace projectors, triplet residuals, orthogonality,
default-budget success, tight-budget fail-closed behavior, and no partial
`sigma`, `U`, or `Vt` arrays on tight-budget failure.

Current non-claims:

- no broad SVD or partial-SVD correctness;
- no raw singular-vector identity;
- no sign, orientation, phase, or arbitrary basis-order parity;
- no broad repeated-spectrum behavior;
- no broad rectangular, nonsymmetric, rank-deficient, null-space,
  pseudoinverse, or minimum-norm behavior;
- no sparse-output/drop-tolerance optimality;
- no convergence-rate or portable iteration-count claim;
- no partial-result guarantee after non-convergence;
- no LAPACK, NumPy, SciPy, SuiteSparse, ARPACK, or other external-library
  parity;
- no hosted-platform, package, ABI, shared-library, performance, or
  state-of-the-art claim.

## Candidate Families For Day 2 Audit

Day 1 does not select Sprint 151 families. Candidate families for Day 2 audit:

| Candidate Family | Existing Owner-Local Evidence | Main Audit Question |
| --- | --- | --- |
| Repeated spectra beyond Sprint 140 | `test_svd_repeated`, Sprint 140 corpus fixture, and full SVD repeated-spectrum tests | Can a second repeated-spectrum partial-SVD fixture close new evidence without raw-vector identity? |
| Rank-deficient rectangular projectors | `test_partial_svd_rankdef_diag6x4_k2_range_projector` | Can source-controlled rows promote projector/range evidence without overclaiming null-space behavior? |
| Sparse low-rank output | `test_lowrank_sparse_*`, `test_sparse_svd_lowrank_outer_product_*`, `example_svd_lowrank.c` | Can sparse output be compared with bounded structural/value semantics without performance or optimality overreach? |
| Convergence/fail-closed behavior | `test_partial_svd_max_iter_fail_closed_diag6_k2`, Sprint 140 tight-budget tests | Can additional fail-closed fixtures add meaningful behavior without portable iteration-count claims? |
| External dense-reference singular values | `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, `partial_svd_nonsym_rect10x8_k3` | Can these remain named fixture-local references without broad LAPACK/NumPy/SciPy parity claims? |

## Stop Conditions

- Do not select a fixture family that requires raw singular-vector equality,
  sign parity, orientation parity, or arbitrary basis ordering.
- Do not cite generated-reference oracle rows as solver-backed hosted-platform
  pass evidence.
- Do not cite optional-data skip/defer rows as partial-SVD pass evidence.
- Do not add source-controlled expected rows without a matching proof owner,
  generator, tolerance, validation command, support tier, claim scope, and
  non-claim text.
- Do not infer partial-SVD correctness from Sprint 150 QR evidence.
- Do not widen fixture-local evidence into broad partial-SVD, external-library,
  platform, package, ABI, performance, or state-of-the-art claims.
- Stop and fix before closeout if corpus schema validation, focused
  partial-SVD tests, oracle/report checks, or required full C quality gates
  fail.

## Day 2 Handoff

Day 2 should audit the candidate families against closure value,
implementation risk, metadata needs, comparison semantics, oracle/report
readiness, documentation impact, and claim-boundary clarity. The likely
high-value candidates are rank-deficient rectangular range projectors,
sparse low-rank output, and one additional convergence/fail-closed lane, but
selection must wait for the Day 2 evidence table.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Sprint 151 scope is tied to current repository files and prior sprint handoffs. | Complete | Day 1 reviewed Sprint 151 project plan, Sprint 140 retrospective, and Sprint 150 handoff. |
| Every current partial-SVD proof surface has an owner or is marked unowned. | Complete | Inventory separates maintained corpus owner, broad SVD owner, helper owners, optional external helper, and candidate owner-local lanes. |
| Stop conditions are explicit before fixture-family selection begins. | Complete | Stop conditions recorded in this artifact and `WORKING_NOTES.md`. |
