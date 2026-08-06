# Sprint 139 Day 10: Solver Documentation Update

## Purpose

Day 10 updates QR-facing documentation with the evidence earned by Sprint 139
Days 7 and 9. The wording is intentionally narrow: it names the selected
fixture, the proof owner, the opt-in oracle path, and the remaining non-claims.

## Updated Surfaces

| Surface | Update |
| --- | --- |
| `README.md` | Added the fixture-local QR corpus proof to workflow and QR API wording. |
| `docs/solver_selection.md` | Added a QR evidence-boundary section and narrowed the QR direct-solver table wording. |
| `docs/algorithm.md` | Added rank/nullspace evidence wording under Sparse QR factorization. |
| `docs/cookbook.md` | Added a short solver-choice note that the QR corpus proof is fixture-local. |
| `examples/README.md` | Clarified that teaching QR examples are separate from the maintained QR corpus proof. |
| `tests/corpus/README.md` | Updated the Sprint 139 QR handoff into an implemented QR lane with the proof owner and `--include-solver-qr` command. |
| `docs/maintainer_guide.md` | Added `tests/test_qr_corpus.c` and the opt-in oracle command to the QR maintained evidence row. |

## Earned Claim

The documentation now supports this narrow statement:

`qr_rank_deficient_6x4_nullspace_v1` is a maintained fixture-local QR corpus
lane with:

- shape 6x4;
- 14 stored nonzeros;
- rank `3`;
- nullity `1`;
- deterministic null-vector direction `[-1, -1, 0, 1]`;
- solver-backed QR rank/nullity/nullspace residual evidence from
  `tests/test_qr_corpus.c`;
- opt-in oracle/report rows from
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`;
- normalized solver-produced nullspace residual `<= 1e-10`.

## Preserved Non-Claims

The updated wording keeps these boundaries explicit:

- no broad QR correctness;
- no raw QR basis, basis sign, orientation, or normalization parity;
- no global rank-threshold policy;
- no broad rank-deficient solve;
- no broad least-squares or minimum-norm claim;
- no SuiteSparse, LAPACK, NumPy, SciPy, or external-library parity;
- no broad corpus completeness;
- no platform, performance, or state-of-the-art claim.

## Artifact Link Map

| Evidence | Path |
| --- | --- |
| Focused proof owner | `tests/test_qr_corpus.c` |
| Reusable fixture/residual helpers | `tests/test_qr_helpers.h` |
| Corpus fixture metadata | `tests/corpus/manifests/fixtures.tsv` |
| Corpus expected rows | `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` |
| Corpus/oracle runner | `scripts/run_corpus_oracle.py` |
| Oracle implementation artifact | `docs/planning/EPIC_12/SPRINT_139/artifacts/day7-oracle-comparison-implementation.md` |
| Proof implementation artifact | `docs/planning/EPIC_12/SPRINT_139/artifacts/day9-proof-owner-implementation.md` |

Generated local outputs remain under ignored `build/corpus/` and
`build/corpus-reports/`.

## Validation

Documentation validation:

```sh
git diff --check
rg -n '[[:blank:]]$' README.md docs/solver_selection.md docs/algorithm.md docs/cookbook.md examples/README.md tests/corpus/README.md docs/maintainer_guide.md docs/planning/EPIC_12/SPRINT_139
ruby -e '... markdown link validation under docs/planning/EPIC_12 ...'
```

Day 10 changed documentation only. The Day 9 full C gate already passed after
the C/helper/build-system changes, and no additional `.c` or `.h` files were
modified during Day 10.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public wording matches earned evidence exactly. | Complete | Public docs name only `qr_rank_deficient_6x4_nullspace_v1`, `tests/test_qr_corpus.c`, and `--include-solver-qr`. |
| Unsupported QR behavior remains fenced by explicit non-claims. | Complete | README, solver-selection, algorithm, corpus, and maintainer wording preserve broad QR and parity non-claims. |
| Documentation points to reproducible fixture/oracle evidence. | Complete | Docs link to source-controlled proof owner and corpus/oracle command instead of ignored generated outputs. |
