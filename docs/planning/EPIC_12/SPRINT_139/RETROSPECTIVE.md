# Sprint 139 Retrospective

**Sprint:** 139 - QR Priority Residual Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-139`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 139 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Re-read Sprint 137 evidence contracts, Sprint 138 corpus/oracle
      handoff, QR tests, QR examples, solver documentation, and residual
      queues.
- [x] Re-audited QR residual candidates and selected one bounded priority
      residual: `qr_rank_deficient_6x4_nullspace_v1`.
- [x] Preserved the Sprint 138 deterministic corpus fixture lane and confirmed
      its generator metadata, expected rank, expected nullity, null-vector
      direction, hashes, and schema validation.
- [x] Designed and implemented opt-in solver-backed QR oracle rows behind
      `scripts/run_corpus_oracle.py --include-solver-qr`.
- [x] Kept generated-reference rows separate from solver-backed QR rows:
      generated metadata remains `solver_family=unknown`, while QR proof rows
      use `solver_family=qr`.
- [x] Added the focused C proof owner `tests/test_qr_corpus.c`.
- [x] Added reusable QR corpus helpers in `tests/test_qr_helpers.h`.
- [x] Registered `test_qr_corpus` in both `Makefile` and `CMakeLists.txt`.
- [x] Updated public, corpus, example, solver-selection, algorithm, cookbook,
      and maintainer documentation with earned fixture-local QR wording.
- [x] Published maintainer guidance for QR evidence regeneration, expected
      outputs, stale-report signals, support-tier interpretation, optional-data
      boundaries, and remaining QR residuals.
- [x] Published claim closure, remaining QR non-claims, and Sprint 140
      partial-SVD handoff requirements.
- [x] Ran Sprint 139 focused and full validation:
  - `python3 scripts/validate_corpus_schema.py`;
  - `make build/test_qr_corpus && ./build/test_qr_corpus`;
  - `python3 scripts/run_corpus_oracle.py --include-solver-qr`;
  - `cmake -S . -B build/qr-corpus-proof && cmake --build
    build/qr-corpus-proof --target test_qr_corpus &&
    ./build/qr-corpus-proof/test_qr_corpus`;
  - `python3 -m py_compile scripts/run_corpus_oracle.py
    scripts/validate_corpus_schema.py`;
  - source-list parity check for `test_qr_corpus`;
  - generated oracle/report metadata checks;
  - generated-artifact ignored/untracked checks;
  - `git diff --check`;
  - trailing-whitespace scans;
  - focused Markdown relative-link validation;
  - `make format && make lint && make test`.
- [x] The required full C quality gate passed because Sprint 139 modified
      `.c` and `.h` files.

## What Went Well

1. **The sprint closed one QR residual completely instead of widening claims.**
   The selected fixture `qr_rank_deficient_6x4_nullspace_v1` now has
   fixture-local solver-backed evidence for rank `3`, nullity `1`, and a
   normalized nullspace residual at or below `1e-10`.

2. **Generated-reference rows and solver-backed rows stayed distinct.** The
   oracle command preserves Sprint 138 corpus semantics while adding opt-in QR
   proof rows with stable `_qr_*` row IDs, `solver_family=qr`, row counts,
   support tier, and non-claim fences.

3. **The focused proof owner made the claim easy to validate.**
   `tests/test_qr_corpus.c` owns exactly the maintained corpus fixture closure:
   shape/nnz, rank/nullity, solver-produced residual, and deterministic
   reference direction. Existing broad QR tests remained intact.

4. **Documentation moved from handoff wording to earned wording.** Public and
   maintainer docs now point to the fixture, proof owner, oracle command,
   regeneration steps, stale-report signals, and explicit non-claims.

5. **Day 12 produced a clean validation anchor.** The sprint has a single
   validation artifact tying schema, focused Make/CMake proof, oracle/report
   metadata, docs hygiene, generated-artifact hygiene, and the full Make
   quality gate together.

## What Didn't Go Well

1. **The closure required a mirrored C fixture builder.** Day 9 caught an
   early mismatch where the helper copy had 15 nonzeros instead of the
   canonical 14. The helper and Day 8 artifact were corrected, but this shows
   future corpus-backed tests need extra care to avoid metadata/code drift.

2. **The solver-backed oracle path depends on a built static library.** The
   `--include-solver-qr` path is reproducible, but it requires
   `build/libsparse_lu_ortho.a` or an explicit `--solver-library` path.

3. **Report freshness remains a documented interpretation rule.** Sprint 139
   records stale-report signals and manifest expectations, but automatic
   freshness normalization is still Sprint 141 work.

4. **The sprint intentionally left adjacent QR behavior open.** Rank-threshold
   policy, solve residuals, minimum-norm behavior, COLAMD/reordered QR,
   optional SuiteSparse rows, broad external parity, platform, performance,
   package/ABI, and state-of-the-art claims remain deferred.

## Final Metrics

### Validation

| Metric | Sprint 139 close state |
|---|---|
| tracked `.c`/`.h` changes | yes: focused QR corpus proof/helper changes |
| `python3 scripts/validate_corpus_schema.py` | passed |
| focused Make `test_qr_corpus` | passed: 4 tests, 0 failures, 0 skips, 83 assertions |
| focused CMake `test_qr_corpus` | passed: 4 tests, 0 failures, 0 skips, 83 assertions |
| solver-produced normalized residual | approximately `2.220e-16` |
| reference-direction normalized residual | `0.000e+00` |
| `python3 scripts/run_corpus_oracle.py --include-solver-qr` | passed |
| generated oracle rows | 6 |
| generated solver-backed QR rows | 3 |
| generated optional-data skip rows | 1 |
| generated optional-data pass rows | 0 |
| `solver_families` | `qr,unknown` |
| `solver_qr_row_count` | 3 |
| source-list parity check | passed for Make and CMake |
| generated corpus/report files tracked | no |
| generated corpus/report files ignored | yes |
| script compile check | passed |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |
| focused Markdown relative-link validation | passed |
| full C quality gate | passed: `make format && make lint && make test` |

### Artifact Package

| Metric | Sprint 139 close state |
|---|---:|
| daily artifacts under `SPRINT_139/artifacts/` | 14 |
| final retrospective files | 1 |
| focused QR corpus test files added | 1 |
| QR helper headers changed | 1 |
| build-system surfaces changed | 2 |
| public/maintainer documentation surfaces changed | 7 |
| oracle/report script surfaces changed | 1 |
| source-controlled generated oracle/report files | 0 |

## Closed Claim

Sprint 139 closes this claim:

For the maintained generated 6 by 4 rank-deficient QR corpus fixture
`qr_rank_deficient_6x4_nullspace_v1`, the project QR implementation reports
rank `3`, reports nullity `1`, and produces a nullspace vector whose normalized
matrix-vector residual is at or below `1e-10`.

This claim is supported by:

- `tests/test_qr_corpus.c`;
- `tests/test_qr_helpers.h`;
- `scripts/run_corpus_oracle.py --include-solver-qr`;
- `tests/corpus/README.md`;
- `docs/maintainer_guide.md`;
- Day 12 validation evidence.

## Sprint 140 Readiness

Sprint 140 should reuse the Sprint 139 pattern but define partial-SVD-specific
evidence:

| Handoff field | Sprint 140 requirement |
| --- | --- |
| fixture ownership | Define partial-SVD fixture keys and generator/expected rows rather than reusing QR rows as SVD correctness evidence. |
| comparison semantics | Prefer residual, projector, or subspace-safe metrics over raw singular-vector identity. |
| ambiguity handling | Explicitly document sign, scale, basis, clustered-spectrum, and repeated-singular-value ambiguity. |
| proof ownership | Add or identify a focused partial-SVD proof owner for the selected residual. |
| oracle/report rows | Keep generated-reference rows distinct from solver-backed rows with stable row IDs and support-tier fields. |
| support tier | Keep optional external data as skip/defer evidence until reviewed support-tier promotion exists. |
| public wording | Publish only earned fixture-local partial-SVD wording and preserve broad SVD/partial-SVD non-claims. |

## Residual Deferred Debt

Most important carry-forward work:

- global QR rank-threshold policy across scales and perturbations;
- broad rank-deficient QR solve behavior;
- broad rectangular least-squares residual behavior;
- broad QR minimum-norm behavior;
- COLAMD/reordered QR behavior;
- optional SuiteSparse QR pass evidence and reviewed support-tier promotion;
- Sprint 140 partial-SVD clustered/repeated singular-value and
  rank-deficient range-projector follow-through;
- Sprint 141 report freshness normalization and stale-report diagnostics.

Still consciously constrained rather than silently solved:

- no broad QR correctness claim;
- no raw QR basis/sign/orientation parity claim;
- no LAPACK, NumPy, SciPy, SuiteSparse, or broad external-library parity claim;
- no broad corpus completeness claim;
- no hosted platform parity claim;
- no package, ABI, shared-library, loader, or package-manager support claim;
- no portable performance claim;
- no state-of-the-art claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-qr-residual-intake.md](./artifacts/day1-qr-residual-intake.md)
- [day2-qr-residual-reaudit.md](./artifacts/day2-qr-residual-reaudit.md)
- [day3-closure-design.md](./artifacts/day3-closure-design.md)
- [day4-fixture-batch-design.md](./artifacts/day4-fixture-batch-design.md)
- [day5-fixture-batch-implementation.md](./artifacts/day5-fixture-batch-implementation.md)
- [day6-oracle-comparison-design.md](./artifacts/day6-oracle-comparison-design.md)
- [day7-oracle-comparison-implementation.md](./artifacts/day7-oracle-comparison-implementation.md)
- [day8-proof-owner-design.md](./artifacts/day8-proof-owner-design.md)
- [day9-proof-owner-implementation.md](./artifacts/day9-proof-owner-implementation.md)
- [day10-solver-documentation-update.md](./artifacts/day10-solver-documentation-update.md)
- [day11-maintainer-guidance-residual-queue.md](./artifacts/day11-maintainer-guidance-residual-queue.md)
- [day12-focused-validation.md](./artifacts/day12-focused-validation.md)
- [day13-claim-closure-handoff.md](./artifacts/day13-claim-closure-handoff.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 139 is complete. It closes the QR priority residual sprint with a
fixture-local corpus-backed QR claim, opt-in solver-backed oracle/report rows,
a focused QR corpus proof owner, updated public and maintainer claim wording,
final validation evidence, explicit remaining QR non-claims, and a Sprint 140
partial-SVD handoff. It does not promote generated reports into source control
or widen public claims beyond the named fixture-local QR behavior.
