# Sprint 102 Day 6 Helper Closeout and Rerank

## Purpose

Day 6 closes the Day 5 helper extraction before any new solver evidence
expands. It confirms the focused validation state, re-checks direct-solver
evidence gaps with the shared parser in place, and reranks the remaining
Sprint 102 implementation lanes.

## Helper Extraction Closeout

Day 5 introduced one opt-in test-support helper:

```c
tf_external_reference_status_t tf_read_external_reference_vector(
    const char *cmd,
    const char *label,
    double *x_out,
    idx_t n,
    char *reason,
    size_t reason_cap);
```

The helper is in `tests/test_solver_helpers.h` and is compiled only when a
test file defines:

```c
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
```

This keeps the external-reference subprocess reader out of unrelated solver
tests that include `tests/test_solver_helpers.h` only for residual helpers.

## Focused Validation Results

Day 6 reran focused checks for both test files touched by the helper
extraction:

| command | result |
|---|---|
| `make build/test_chol_csc` | passed; target was up to date |
| `./build/test_chol_csc` | passed; 92 tests, 0 failures, 0 skips, 20844 assertions |
| `make build/test_ldlt_csc` | passed; target was up to date |
| `./build/test_ldlt_csc` | passed; 98 tests, 0 failures, 0 skips, 2288 assertions |

Day 5 already ran the required full quality chain after the `.c` and `.h`
changes:

| command | recorded Day 5 result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

## Evidence Gap Recheck

| lane | Day 2 state | Day 6 state after helper extraction |
|---|---|---|
| Cholesky CSC external dense reference | strongest existing external lane; parser duplicated in family test | parser duplication removed; no new fixture coverage |
| LDLT CSC external dense reference | useful KKT external lane; parser duplicated in family test | parser duplication removed; still best CSC-family expansion candidate |
| LU external dense reference | highest-value missing external lane | still missing; helper now lowers parser cost for a future LU helper |
| QR dense least-squares/rank reference | second-highest missing external lane | still missing; helper may be reusable if QR uses the same vector output contract |
| SVD dense reference | valuable but heavier and less urgent | still deferred; helper may not fit singular-value/vector matrix output needs |
| direct CSC dispatch | route/regression proof only | still should consume family evidence rather than lead oracle expansion |

The helper extraction improves maintainability and reduces future parser cost,
but it does not by itself change any solver-family correctness coverage.

## Updated Expansion Ranking

The Sprint 102 project plan fixes Day 7-9 as the CSC direct-family expansion
window and Day 10-11 as the LU/QR/SVD expansion window. Within that sequencing,
the updated ranking is:

| rank | lane | disposition | reason |
|---:|---|---|---|
| 1 | LDLT CSC scaled/reordered KKT external fixture expansion | select for Day 7 boundary | strongest CSC-family gap after Cholesky already has `nos4` and `bcsstk04` external proof |
| 2 | Cholesky CSC scaled SPD control fixture | backup for Day 7 | useful only if LDLT boundary finds fixture/tolerance risk too high |
| 3 | LU external dense-reference solve lane | select for Day 10 boundary unless Day 7-9 changes priorities | highest general direct-solver external-oracle gap |
| 4 | QR dense least-squares/rank lane | backup or follow-up for Day 10 | broad internal coverage but no external dense least-squares reference |
| 5 | SVD dense singular-value/vector lane | defer unless scope remains | heavier oracle shape and already broad internal invariant coverage |
| 6 | direct CSC dispatch oracle lane | do not select as independent oracle lane | should reuse family evidence and route assertions |

## Day 7 Recommendation

Day 7 should freeze an LDLT CSC external fixture expansion, not a Cholesky CSC
expansion.

Recommended fixture lane:

| field | recommendation |
|---|---|
| fixture key | `ldlt_kkt_scaled_10` |
| taxonomy class | `indef-kkt-scaled` |
| source | extend the existing LDLT external helper with a deterministic scaled KKT builder |
| expected status | success |
| target behavior | compare LDLT CSC solve against dense reference for a scaled indefinite KKT fixture |
| suggested tolerance | define on Day 7 after inspecting scale choice; start from the existing `1e-10` lane and relax only if justified |
| non-claim | does not prove all indefinite matrices, all pivot patterns, or external factorization parity |

Backup fixture lane:

| field | backup |
|---|---|
| fixture key | selected scaled SPD control |
| taxonomy class | `scaled-near-singular` or `spd-mm-small` |
| family | Cholesky CSC |
| reason to use | only if LDLT scaled KKT cannot be made deterministic and bounded |

## Day 10 Recommendation

Day 10 should prefer LU over QR/SVD for the general direct-solver expansion
window.

Recommended LU lane:

| field | recommendation |
|---|---|
| success fixture | `lu_nonsym_square_5` |
| expected-failure fixture | `lu_singular_square_4` |
| taxonomy classes | `nonsym-square-small`; `square-rank-def` |
| oracle style | dense solve helper producing `OK`, `ERROR`, and possibly `SKIP` status |
| reason | LU is central, currently residual-heavy, and lacks external dense-reference coverage |

## Helper Residual Queue

| residual | disposition |
|---|---|
| shared command construction | defer; command quoting and helper arguments remain lane-specific |
| shared RHS construction | defer; direct-solver families may need different targets |
| shared residual helpers for infinity norm | defer; existing local helpers are family-tuned |
| explicit malformed-output self-test for `tf_read_external_reference_vector(...)` | defer until a future helper-focused test binary is justified |
| LU helper reuse | candidate for Day 10 if the LU helper emits the same `OK n` vector format |
| QR/SVD helper reuse | candidate only if their oracle output can stay vector-shaped and bounded |

## Claim Boundaries

Day 6 preserves the Day 5 claim boundary:

- shared parser maintainability is earned;
- no new solver correctness fixture is earned;
- no LU, QR, or SVD external oracle coverage exists yet;
- no direct CSR/CSC solver APIs exist;
- no broad direct-solver parity or state-of-the-art claim is introduced.

## Day 6 Conclusion

The helper extraction is validated and closed. Sprint 102 should proceed to
Day 7 with an LDLT CSC scaled-KKT external fixture boundary, then reserve the
general direct-solver expansion window for a bounded LU external dense-reference
lane unless Day 7-9 evidence changes the ranking.
