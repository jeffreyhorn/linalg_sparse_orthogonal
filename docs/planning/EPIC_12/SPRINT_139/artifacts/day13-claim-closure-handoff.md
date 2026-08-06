# Day 13 Claim Closure and Sprint 140 Handoff

## Closed Claim

Sprint 139 closes one fixture-local QR residual:

`qr_rank_deficient_6x4_nullspace_v1` solver-backed rank, nullity, and
normalized nullspace residual.

The earned claim is:

For the maintained generated 6 by 4 rank-deficient QR corpus fixture
`qr_rank_deficient_6x4_nullspace_v1`, the project QR implementation reports
rank `3`, reports nullity `1`, and produces a nullspace vector whose normalized
matrix-vector residual is at or below `1e-10`.

This is a local corpus/proof-owner claim. It is not a broad QR correctness,
least-squares, minimum-norm, external-library parity, SuiteSparse parity,
platform, performance, corpus completeness, or state-of-the-art claim.

## Evidence Added

| Evidence layer | Sprint 139 deliverable | Claim support |
| --- | --- | --- |
| Residual selection | `docs/planning/EPIC_12/SPRINT_139/artifacts/day2-qr-residual-reaudit.md` | Selects `qr_rank_deficient_6x4_nullspace_v1` as the only priority closure target and defers adjacent QR residuals. |
| Closure design | `docs/planning/EPIC_12/SPRINT_139/artifacts/day3-closure-design.md` | Defines rank, nullity, normalized residual, proof-owner, oracle, and non-claim boundaries. |
| Fixture specification | `docs/planning/EPIC_12/SPRINT_139/artifacts/day4-fixture-batch-design.md` | Records canonical matrix facts, expected rank/nullity, null-vector direction, tolerance, and support tier. |
| Fixture implementation | `docs/planning/EPIC_12/SPRINT_139/artifacts/day5-fixture-batch-implementation.md` | Confirms source-controlled corpus fixture, generator, expected rows, and schema validation. |
| Oracle design | `docs/planning/EPIC_12/SPRINT_139/artifacts/day6-oracle-comparison-design.md` | Defines generated-reference versus solver-backed QR row semantics. |
| Oracle implementation | `scripts/run_corpus_oracle.py` and `docs/planning/EPIC_12/SPRINT_139/artifacts/day7-oracle-comparison-implementation.md` | Adds `--include-solver-qr` and emits three solver-backed QR rows with `solver_family=qr`. |
| Proof-owner design | `docs/planning/EPIC_12/SPRINT_139/artifacts/day8-proof-owner-design.md` | Defines `test_qr_corpus` as the focused owner while preserving broad QR tests. |
| Proof-owner implementation | `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h`, `Makefile`, `CMakeLists.txt`, and `docs/planning/EPIC_12/SPRINT_139/artifacts/day9-proof-owner-implementation.md` | Adds focused C proof for fixture shape, rank/nullity, solver-produced residual, and reference direction. |
| Public and maintainer wording | `README.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `docs/cookbook.md`, `examples/README.md`, `tests/corpus/README.md`, `docs/maintainer_guide.md`, and Day 10-11 artifacts | Publishes earned fixture-local wording and keeps broad QR non-claims visible. |
| Validation | `docs/planning/EPIC_12/SPRINT_139/artifacts/day12-focused-validation.md` | Records schema, QR proof, oracle/report, CMake, docs hygiene, generated-artifact, and full quality-gate evidence. |

## Validation-to-Claim Traceability

| Claim component | Validation evidence | Result |
| --- | --- | --- |
| Fixture identity and shape | `test_qr_corpus_rankdef_6x4_fixture_shape` | 6 rows, 4 columns, and 14 nonzeros proved by focused C test. |
| Rank | `test_qr_corpus_rankdef_6x4_rank_and_nullity` and oracle row `qr_rank_deficient_6x4_nullspace_v1_qr_rank` | Observed rank `3`; comparison status `pass`. |
| Nullity | `test_qr_corpus_rankdef_6x4_rank_and_nullity` and oracle row `qr_rank_deficient_6x4_nullspace_v1_qr_nullity` | Observed nullity `1`; comparison status `pass`. |
| Solver-produced nullspace residual | `test_qr_corpus_rankdef_6x4_nullspace_residual` and oracle row `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual` | Observed residual approximately `2.220e-16`, below tolerance `1e-10`; comparison status `pass`. |
| Deterministic reference direction | `test_qr_corpus_rankdef_6x4_reference_direction` | Reference direction `[-1, -1, 0, 1]` residual `0.000e+00`. |
| Oracle/report reproducibility | `python3 scripts/run_corpus_oracle.py --include-solver-qr` | 6 oracle rows, `solver_families=qr,unknown`, `solver_qr_row_count=3`. |
| Build-system ownership | Make and CMake focused target runs | `test_qr_corpus` is registered and runnable through both build surfaces. |
| Full touched-surface safety | `make format && make lint && make test` | Passed with final output `All tests passed.` |
| Generated artifact hygiene | `git status --short --ignored build/corpus build/corpus-reports` and `git ls-files build/corpus build/corpus-reports` | Generated report files are ignored and untracked. |

## Remaining QR Non-Claims

Sprint 139 does not close:

- broad QR correctness;
- raw Q-basis, sign, scale, orientation, or exact basis-vector parity;
- global QR rank-threshold policy across scales and perturbations;
- broad rank-deficient QR solve behavior;
- broad rectangular least-squares residual behavior;
- broad minimum-norm behavior;
- COLAMD/reordered QR behavior;
- SuiteSparse optional-data QR pass evidence;
- LAPACK, NumPy, SciPy, SuiteSparse, or other external-library parity;
- hosted platform, cross-platform, performance, package, ABI, or
  state-of-the-art claims;
- broad numerical corpus completeness.

These non-claims are consistent with the public and maintainer documentation
updated during Days 10 and 11.

## Remaining QR Residual Queue

| Residual | Current status | Future gate |
| --- | --- | --- |
| Global rank-threshold policy | Deferred | Needs tolerance-family design across scaling and perturbation fixtures. |
| Rank-deficient solve and least-squares residual behavior | Deferred | Needs solve-side fixture rows, oracle semantics, and proof ownership. |
| Minimum-norm behavior | Deferred | Needs a separate minimum-norm claim gate and validation matrix family. |
| COLAMD/reordered QR | Deferred | Needs ordering-specific fixture, fill/permutation semantics, and proof owner. |
| Optional SuiteSparse QR subset | Deferred | Needs optional-data provenance, reviewed pass evidence, and support-tier promotion. |
| Broad external-library parity | Non-goal for this sprint | Needs a separate product decision and external parity strategy. |
| Partial-SVD rank-deficient subspace overlap | Sprint 140 dependency | Needs partial-SVD specific fixture and oracle design. |

## Sprint 140 Handoff

Sprint 140 can start from these assumptions:

- The QR corpus lane for `qr_rank_deficient_6x4_nullspace_v1` is closed only
  for QR rank, nullity, and normalized nullspace residual.
- Partial-SVD should not reuse the QR lane as SVD correctness evidence. It can
  reuse the corpus/oracle/report pattern, support-tier language, stale-report
  signals, and residual/subspace-safe comparison style.
- Partial-SVD work should define its own fixture keys, expected rows, oracle
  row IDs, proof owner, tolerances, and non-claims.
- Clustered or repeated singular-value fixtures need explicit sign/basis
  ambiguity rules like the QR lane uses for raw QR bases.
- Rank-deficient partial-SVD range-projector follow-through should compare
  projector or residual metrics, not raw singular-vector identity.
- Optional external data remains skip/defer evidence unless Sprint 140
  explicitly promotes reviewed support-tier proof.

## Closeout Readiness

Sprint 139 is ready for Day 14 closeout when the final closeout artifact:

- restates the closed QR fixture-local claim without broadening it;
- references the Day 12 full validation result;
- keeps remaining non-claims visible;
- confirms generated oracle/report outputs were not committed;
- confirms Sprint 140 has enough handoff detail to proceed without
  rediscovering QR boundaries.
