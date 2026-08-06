# Day 14 Closeout Validation Summary

## Sprint 139 Outcome

Sprint 139 completed the selected QR residual closure for
`qr_rank_deficient_6x4_nullspace_v1`.

Closed claim:

For the maintained generated 6 by 4 rank-deficient QR corpus fixture
`qr_rank_deficient_6x4_nullspace_v1`, the project QR implementation reports
rank `3`, reports nullity `1`, and produces a nullspace vector whose normalized
matrix-vector residual is at or below `1e-10`.

The sprint deliberately keeps this claim fixture-local. It does not claim broad
QR correctness, global rank-threshold policy, raw basis parity, broad
least-squares or minimum-norm behavior, external-library parity, SuiteSparse
parity, platform support, performance, package/ABI behavior, corpus
completeness, or state-of-the-art status.

## Sprint Item Completion

| Sprint requirement | Status | Evidence |
| --- | --- | --- |
| Re-audit QR residuals and select one bounded priority residual to close | Complete | Day 1 intake and Day 2 reaudit artifacts select `qr_rank_deficient_6x4_nullspace_v1` and defer adjacent QR residuals. |
| Add deterministic corpus fixtures for the selected QR residual family | Complete | Day 4 fixture design and Day 5 fixture implementation artifacts confirm canonical fixture facts, generator metadata, expected rows, hashes, and schema validation. |
| Add oracle comparison rows with explicit tolerance, skip, and failure semantics | Complete | Day 6 design and Day 7 implementation add opt-in solver-backed QR rows behind `scripts/run_corpus_oracle.py --include-solver-qr`. |
| Create or extract a focused QR proof owner without weakening existing QR coverage | Complete | Day 8 design and Day 9 implementation add `tests/test_qr_corpus.c`, QR helper support, and Make/CMake registration while preserving existing broad QR tests. |
| Update solver, algorithm, cookbook, and maintainer documentation with earned QR wording and preserved non-claims | Complete | Day 10 documentation updates and Day 11 maintainer guidance update public, corpus, example, and maintainer docs. |
| Run focused QR/corpus validation and broader quality gates when code changes require them | Complete | Day 12 validation passes schema, Make/CMake QR proof, oracle/report generation, docs hygiene, generated-artifact hygiene, and `make format && make lint && make test`. |
| Publish closed QR claims, remaining non-claims, and Sprint 140 partial-SVD handoff requirements | Complete | Day 13 claim closure and this Day 14 closeout summarize the closed claim, remaining QR residuals, and Sprint 140 dependencies. |

## Artifact Inventory

| Artifact | Purpose |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_139/PLAN.md` | Day-by-day Sprint 139 execution plan. |
| `docs/planning/EPIC_12/SPRINT_139/WORKING_NOTES.md` | Running implementation, validation, and decision log. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day1-qr-residual-intake.md` | Initial inherited QR/corpus evidence inventory. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day2-qr-residual-reaudit.md` | Residual ranking and selected closure target. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day3-closure-design.md` | Claim, fixture, oracle, proof-owner, and non-claim design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day4-fixture-batch-design.md` | Deterministic QR fixture specification. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day5-fixture-batch-implementation.md` | Corpus fixture and expected-row implementation evidence. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day6-oracle-comparison-design.md` | Oracle comparison semantics and row design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day7-oracle-comparison-implementation.md` | Solver-backed oracle runner implementation evidence. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day8-proof-owner-design.md` | Focused QR proof-owner design. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day9-proof-owner-implementation.md` | Focused QR proof-owner implementation evidence. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day10-solver-documentation-update.md` | Public and maintainer wording update evidence. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day11-maintainer-guidance-residual-queue.md` | Regeneration guidance, stale-report signals, and residual queue. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day12-focused-validation.md` | Full validation record for touched surfaces. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day13-claim-closure-handoff.md` | Closed claim, validation traceability, non-claims, and Sprint 140 handoff. |
| `docs/planning/EPIC_12/SPRINT_139/artifacts/day14-closeout-validation-summary.md` | Final Sprint 139 closeout summary. |

## Code and Documentation Inventory

| Surface | Change summary |
| --- | --- |
| `scripts/run_corpus_oracle.py` | Adds opt-in solver-backed QR oracle/report rows through `--include-solver-qr`. |
| `tests/test_qr_corpus.c` | Adds the focused QR corpus proof owner for fixture shape, rank/nullity, solver-produced residual, and reference direction. |
| `tests/test_qr_helpers.h` | Adds the reusable fixture builder and normalized residual helper. |
| `Makefile` | Registers `test_qr_corpus` for Make builds/tests. |
| `CMakeLists.txt` | Registers `test_qr_corpus` for CMake builds/tests. |
| `README.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `docs/cookbook.md`, `examples/README.md` | Publish earned fixture-local QR wording without broadening unsupported claims. |
| `tests/corpus/README.md` | Documents QR lane regeneration, generated output expectations, stale signals, support tier, and residual queue. |
| `docs/maintainer_guide.md` | Adds maintainer-facing QR corpus maintenance and claim-boundary guidance. |

## Validation Summary

Latest full validation is Day 12:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
cmake -S . -B build/qr-corpus-proof && cmake --build build/qr-corpus-proof --target test_qr_corpus && ./build/qr-corpus-proof/test_qr_corpus
python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py
rg -n "test_qr_corpus" Makefile CMakeLists.txt
git diff --check
make format && make lint && make test
```

Results:

- Corpus schema validation passed.
- Focused Make and CMake `test_qr_corpus` runs passed with 4 tests, 0
  failures, 0 skips, and 83 assertions.
- Solver QR oracle/report generation passed with 6 oracle rows,
  `solver_families=qr,unknown`, and `solver_qr_row_count=3`.
- Generated corpus/report files remained ignored and untracked.
- Script compile, source-list, whitespace, trailing-whitespace, and Markdown
  link checks passed.
- Required full quality gate passed with final output `All tests passed.`

Day 13 and Day 14 changed planning documentation only. The Day 12 full gate
remains current for the code and build-system changes.

## Deferred Work

Deferred QR work:

- global rank-threshold policy across scales and perturbations;
- broad rank-deficient QR solve and rectangular least-squares behavior;
- broad QR minimum-norm behavior;
- COLAMD/reordered QR behavior;
- optional SuiteSparse QR pass evidence and reviewed support-tier promotion;
- broad external-library parity;
- hosted platform, package/ABI, performance, corpus completeness, and
  state-of-the-art claims.

Sprint 140 dependency:

- partial-SVD clustered/repeated singular-value and rank-deficient
  range-projector follow-through should reuse the corpus/oracle/report pattern
  but define partial-SVD-specific fixtures, expected rows, proof owner,
  tolerances, basis/subspace ambiguity rules, and non-claims.

## Retrospective Input

What worked:

- Keeping the QR closure fixture-local avoided broad claim drift.
- Separating generated-reference rows from solver-backed QR rows preserved the
  Sprint 138 corpus semantics.
- A focused `test_qr_corpus` owner made the claim easier to validate than
  folding the fixture into broad QR tests.
- Day 12 validation gave a clean single source for final claim evidence.

What to watch:

- The generated oracle/report files are useful evidence but remain ignored
  local artifacts; future hosted promotion needs an explicit support-tier gate.
- Raw basis comparisons remain unsafe for QR and partial-SVD; use residual,
  projector, or subspace-safe metrics.
- The static-library oracle probe depends on the built library path unless a
  caller passes `--solver-library`.

## Closeout Readiness

Sprint 139 is ready for retrospective creation and pull-request packaging.
Items 1-7 are complete, deferred work is explicit, validation is tied to the
touched surfaces, and Sprint 140 can proceed from the documented partial-SVD
handoff without rediscovering QR boundaries.
