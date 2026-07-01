# Sprint 102 Day 9 CSC Closeout and General Solver Rerank

## Purpose

Day 9 closes the CSC-family oracle expansion before Sprint 102 moves to LU,
QR, or SVD. It confirms the Day 8 LDLT CSC evidence against the Day 7
boundary, records proof-owner and residual notes, defers remaining CSC work,
and selects the Day 10 general direct-solver oracle boundary target.

## Focused Validation Rerun

Day 9 reran the helper and LDLT CSC focused checks touched by Day 8:

| command | result |
|---|---|
| `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10` | passed; emitted `OK 10` and dense solution values equal to `1..10` to roundoff |
| `make build/test_ldlt_csc` | passed; target was up to date |
| `./build/test_ldlt_csc` | passed; 99 tests, 0 failures, 0 skips, 2318 assertions |

Focused external-reference metrics from the Day 9 rerun:

| fixture | max error | residual |
|---|---:|---:|
| `kkt5` | `0.000e+00` | `0.000e+00` |
| `kkt10` | `3.553e-15` | `2.292e-16` |
| `ldlt_kkt_scaled_10` | `8.882e-15` | `1.692e-17` |

Day 8 already ran the full required quality chain after the `.c` change:

| command | Day 8 result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed; `All tests passed.` |

## Day 7 Boundary Check

| Day 7 criterion | Day 9 closeout result |
|---|---|
| fixture key is `ldlt_kkt_scaled_10` | met |
| taxonomy class is `indef-kkt-scaled` | met |
| helper returns `OK 10` | met |
| `max|x - x_ref| <= 1e-10` | met; `8.882e-15` |
| `max|x - x_true| <= 1e-10` | met through the existing assertion path |
| `rel_residual(A, x, b) < 1e-10` | met; `1.692e-17` |
| existing `kkt5` and `kkt10` lanes still pass | met |
| unknown or malformed helper output is not counted as proof | preserved through `tf_read_external_reference_vector(...)` |
| unavailable helper is skip/unsupported, not correctness proof | preserved |
| no Cholesky correctness pass on indefinite KKT | preserved |

No tolerance relaxation was required.

## CSC Proof-Owner Notes

| owner | closeout note |
|---|---|
| `tests/ldlt_external_dense_reference.py` | owns deterministic dense reference solves for `kkt5`, `kkt10`, and `ldlt_kkt_scaled_10` |
| `tests/test_ldlt_csc.c` | owns LDLT CSC sparse fixture construction, two-pass factor/solve behavior, permutation mapping, tolerance assertions, and residual checks |
| `tests/test_solver_helpers.h` | owns only shared external-reference vector parsing when explicitly enabled |
| `tests/test_chol_csc.c` | unchanged by Day 8 and remains the Cholesky CSC external SPD proof owner |
| `tests/chol_external_dense_reference.py` | unchanged and remains limited to Cholesky CSC SPD Matrix Market fixtures |

The CSC proof remains family-local: parser reuse is shared, but matrix
construction, solver path, permutation handling, tolerances, residual checks,
and claim wording remain in the LDLT CSC test owner.

## Residual CSC Queue

| residual | disposition |
|---|---|
| broader LDLT CSC Matrix Market indefinite corpus | defer; needs fixture taxonomy and runtime cap before adding corpus proof |
| reordered variant of `kkt10` | defer; current `ldlt_kkt_scaled_10` already covers the selected Day 7 scale/coupling gap |
| Cholesky CSC scaled SPD control fixture | defer; Cholesky CSC already owns external SPD proof on `nos4` and `bcsstk04` |
| direct CSC dispatch external oracle lane | defer; dispatch should consume family proof instead of becoming an independent oracle owner |
| helper malformed-output unit test | defer until a dedicated helper-focused test binary is justified |
| public docs claim update | defer until Sprint 102 Day 12 support-surface reconciliation |

Day 9 deliberately stops CSC work here so the Sprint 102 general direct-solver
window can start from a clean boundary.

## General Solver Rerank

The Day 2 gap audit ranked LU, QR, and SVD by user value, external-oracle gap,
implementation risk, and validation cost. Day 9 rechecks that ranking after
the CSC closeout:

| rank | lane | Day 9 disposition | reason |
|---:|---|---|---|
| 1 | LU external dense-reference solve | select for Day 10 boundary | LU is central, has residual and singular coverage, and still lacks an external dense-reference oracle |
| 2 | QR dense least-squares or rank reference | backup/follow-up | QR has broad internal invariant coverage but no external least-squares/rank oracle |
| 3 | SVD dense singular-value/vector reference | defer | valuable but heavier oracle shape and broad internal coverage already exists |

Day 10 should select LU unless implementation inspection finds the helper
boundary cannot be kept bounded.

## Selected Day 10 Target

Recommended LU lane:

| field | recommendation |
|---|---|
| success fixture key | `lu_nonsym_square_5` |
| expected-failure fixture key | `lu_singular_square_4` |
| taxonomy classes | `nonsym-square-small`; `square-rank-def` |
| likely owner | `tests/test_sparse_lu.c` for linked-list LU first |
| external helper owner | new LU dense-reference helper only if Day 10 confirms it stays small |
| reference contract | dense solve of `A*x = b` with `x_true[i] = i + 1` |
| success tolerance starting point | `1e-10` for `max|x - x_ref|` and residual, adjusted only with recorded numerical reason |
| failure behavior | singular fixture must produce expected solver/reference failure, not a skipped correctness pass |

The LU lane should reuse `tf_read_external_reference_vector(...)` only if the
helper can emit the same `OK n` vector contract as Cholesky and LDLT. Command
construction, matrix construction, LU factor/solve calls, pivot policy, and
tolerances should remain LU-local.

## Claim Boundaries

Day 9 closes only a bounded CSC claim:

> LDLT CSC solves for `kkt5`, `kkt10`, and `ldlt_kkt_scaled_10` agree with
> external-process dense references under recorded fixture, tolerance, and
> validation conditions.

Day 9 does not claim:

- external LU, QR, or SVD coverage exists yet;
- LDLT CSC handles all indefinite matrices;
- Cholesky supports indefinite KKT fixtures;
- direct CSR/CSC solver APIs exist;
- dispatch evidence proves family-level numerical correctness independently;
- portable performance, fill, or runtime superiority.

## Day 9 Conclusion

The CSC-family oracle expansion is validated and closed. Sprint 102 should
move to LU on Day 10, with `lu_nonsym_square_5` and `lu_singular_square_4` as
the proposed success/failure fixture pair and QR retained as the backup
general-solver oracle lane.
