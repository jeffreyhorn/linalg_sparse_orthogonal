# Sprint 139 Day 3: Closure Design

## Purpose

Day 3 turns the Day 2 selected residual into an implementation-ready closure
design. It defines the exact QR behavior to close, fixture class, oracle row
semantics, proof-owner boundary, touched surfaces, and validation requirements.

This is a design artifact. It does not change QR source, tests, corpus rows,
oracle commands, public documentation, or support-tier claims.

## Selected Residual

Sprint 139 will close solver-backed QR rank/nullity/nullspace residual behavior
for:

`qr_rank_deficient_6x4_nullspace_v1`

The selected lane remains fixture-local. It closes only this maintained
generated corpus fixture unless later days add reviewed evidence for a broader
support tier.

## Fixture Class Definition

| Field | Value |
| --- | --- |
| fixture family | `qr_rank_deficient` |
| fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| storage kind | `generated` |
| dimensions | `6 x 4` |
| nonzeros | `14` |
| rank | `3` |
| nullity | `1` |
| sparsity class | `structured_sparse` |
| scale class | `unit` |
| conditioning class | `moderate` |
| right-hand-side policy | `generated_rhs` in the existing fixture row, but Day 3 closure uses the matrix/nullspace behavior only |
| dependency | `c3 = c0 + c1` |
| reference null vector | `[-1, -1, 0, 1]` spans the right nullspace |
| support tier before promotion | `local_only` |

Expected matrix entries, zero-based:

| Row | Col | Value |
| ---: | ---: | ---: |
| 0 | 0 | `1.0` |
| 0 | 3 | `1.0` |
| 1 | 1 | `1.0` |
| 1 | 3 | `1.0` |
| 2 | 2 | `1.0` |
| 3 | 0 | `1.0` |
| 3 | 1 | `1.0` |
| 3 | 3 | `2.0` |
| 4 | 1 | `1.0` |
| 4 | 2 | `1.0` |
| 4 | 3 | `1.0` |
| 5 | 0 | `1.0` |
| 5 | 2 | `1.0` |
| 5 | 3 | `1.0` |

Day 5 implementation should add a C helper that mirrors these entries exactly
rather than inventing an adjacent QR fixture.

## Behavior Cases

| Case | Expected behavior | Closure interpretation |
| --- | --- | --- |
| Success | `sparse_qr_factor()` returns `SPARSE_OK`, rank is `3`, nullity is `1`, and one solver-produced nullspace vector has normalized residual `<= 1e-10`. | Closes fixture-local solver-backed QR rank/nullity/nullspace residual behavior. |
| Diagnostic failure | Factorization, nullity query, or basis extraction returns a non-`SPARSE_OK` status. | Report as `fail_oracle_mismatch` or a focused test failure; do not treat as unsupported pass evidence. |
| Tolerance boundary | Rank/nullity are exact comparisons; residual uses absolute tolerance `1e-10` on `||A*v||_2 / ||v||_2`. | Fail closed if residual is above tolerance or if the nullspace vector has zero norm. |
| Raw-basis mismatch | Solver basis differs from `[-1, -1, 0, 1]` by sign, scale, or orientation but satisfies residual/nullity checks. | Pass; raw basis equality is not a claim. |
| Optional external-data unavailable | Optional SuiteSparse rows remain disabled/skipped. | Not part of this closure and not pass evidence. |

## Expected Result and Oracle Semantics

| Oracle row ID | Operation | Comparison kind | Tolerance kind | Expected value | Failure class on mismatch |
| --- | --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank_info` | `rank` | `exact` | `3` | `fail_oracle_mismatch` |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `rank_info` | `nullity` | `exact` | `1` | `fail_oracle_mismatch` |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `nullspace` | `residual_norm` | `absolute` | `normalized_null_vector_residual<=1e-10` | `fail_oracle_mismatch` |

Day 6 should design the observed solver-backed oracle row source so it records
solver output, not just generated reference metadata. Day 7 should keep the
existing generated-reference row meaning distinct from solver-backed QR pass
evidence.

## Residual Comparison Decision

Primary Sprint 139 comparison:

```text
normalized_nullspace_residual = ||A * v_solver||_2 / ||v_solver||_2
```

where `v_solver` is the single vector returned by `sparse_qr_nullspace()` for
the selected fixture.

Decision rationale:

- The Sprint 138 expected row already defines `residual_norm` with absolute
  tolerance `1e-10`.
- The lane has nullity `1`, so residual plus rank/nullity is sufficient to
  prove the selected fixture-local behavior without raw basis equality.
- Projector or two-way projection distance can be added later if Sprint 139
  decides to prove subspace orientation against an external reference, but it
  is not required to close this first lane.

## Proof-Owner Decision

Chosen proof owner:

`tests/test_qr_corpus.c`

Rationale:

- `tests/test_qr.c` already owns many QR behaviors and is large enough that a
  new maintained-corpus lane would be harder to find there.
- A dedicated focused test can make the Sprint 139 evidence discoverable by
  fixture key and oracle row IDs.
- Existing QR tests remain intact and do not need to be weakened or moved.
- A focused test can still reuse `tests/test_qr_helpers.h` and the public QR
  API.

Day 8 should finalize the exact test shape. Day 9 should add the test only if
Day 4-7 fixture/oracle work confirms no better proof surface exists.

## Proposed C Proof Shape

Candidate test file:

`tests/test_qr_corpus.c`

Candidate helper addition:

`tf_qr_make_rankdef_6x4_nullspace_v1()` in `tests/test_qr_helpers.h`

Candidate focused checks:

1. Build the 6x4 fixture entries exactly as defined by the corpus generator.
2. Factor with `sparse_qr_factor()`.
3. Assert `sparse_qr_rank(&qr, 0.0) == 3`.
4. Query `sparse_qr_nullspace(&qr, 0.0, NULL, &ndim)` and assert `ndim == 1`.
5. Extract one nullspace vector.
6. Compute `||A*v||_2 / ||v||_2`.
7. Assert the normalized residual is `<= 1e-10`.
8. Print fixture key, rank, nullity, and normalized residual for validation
   traceability.

The test should not assert exact equality to `[-1, -1, 0, 1]`.

## Touched Surface Map

| Surface | Likely Day | Reason | Validation requirement |
| --- | ---: | --- | --- |
| `docs/planning/EPIC_12/SPRINT_139/*` | 3-14 | Planning, design, validation, and closeout artifacts. | `git diff --check`, sprint trailing-whitespace scan, Markdown link validation. |
| `tests/corpus/manifests/fixtures.tsv` | 5 if needed | Add or update QR fixture batch rows only if Day 4 expands beyond the first lane. | `python3 scripts/validate_corpus_schema.py`, TSV width check. |
| `tests/corpus/expected/*.tsv` | 5 or 7 if needed | Add solver-backed expected rows only if Day 6 chooses distinct observed-row IDs. | Schema validation and row non-claim review. |
| `scripts/run_corpus_oracle.py` | 7 if needed | Emit solver-backed QR observed rows and report rows. | `python3 -m py_compile`, schema/oracle command validation, report metadata checks. |
| `tests/test_qr_helpers.h` | 9 | Add exact C fixture builder. | Focused QR test plus full C quality gate because `.h` changed. |
| `tests/test_qr_corpus.c` | 9 | Add focused proof owner. | Add to Make/CMake, run focused test, then full C quality gate. |
| `Makefile` | 9 | Register new test source if dedicated proof owner lands. | Source-list parity and focused build/test. |
| `CMakeLists.txt` | 9 | Register new test target if dedicated proof owner lands. | CMake configure/build parity where feasible. |
| QR docs | 10-11 | Publish earned fixture-local claim and remaining non-claims. | Markdown link validation and claim-boundary scan. |

## Build and Test-Runner Implications

If Day 9 adds `tests/test_qr_corpus.c`:

- add the file to the Makefile test source list;
- add `add_sparse_test(test_qr_corpus)` to CMake near `test_qr` and
  `test_qr_solve`;
- keep `test_qr` and `test_qr_solve` registered and unchanged unless a later
  day intentionally transfers a helper;
- run focused build/test for `test_qr_corpus`;
- run `make format && make lint && make test` because `.c`/`.h` files changed.

If Day 9 instead extends `tests/test_qr.c`:

- no new source-list entries should be needed;
- the test must print the maintained fixture key;
- the existing QR tests must remain registered;
- full C quality gates still apply because `.c` changes.

## Claim Boundary

The selected closure may eventually earn this wording:

> The QR implementation has fixture-local evidence for rank, nullity, and
> normalized nullspace residual on the maintained
> `qr_rank_deficient_6x4_nullspace_v1` corpus fixture.

It must not claim:

- broad QR correctness;
- raw QR basis parity;
- broad rank-deficient solve behavior;
- global rank-threshold policy;
- global least-squares or minimum-norm correctness;
- SuiteSparse or external-corpus parity;
- LAPACK, NumPy, SciPy, or ecosystem parity;
- platform parity, portable performance, or state-of-the-art status.

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Fixtures and oracle rows have unambiguous expected semantics. | Complete | Fixture class, behavior cases, oracle row table, and residual comparison decision above. |
| Proof ownership is scoped before implementation begins. | Complete | Dedicated `tests/test_qr_corpus.c` proof-owner decision and fallback extension path are documented. |
| Validation requirements are known for docs-only, script, and C/H changes. | Complete | Touched surface map and build/test-runner implications define required checks. |
