# Sprint 102 Day 2 Direct Solver Gap Audit

## Purpose

Day 2 inventories direct-solver correctness evidence before Sprint 102 adds
new fixture, helper, or oracle coverage. The audit ranks Cholesky, LDLT, LU,
QR, SVD, and direct-dispatch paths by oracle depth, fixture diversity,
failure-mode clarity, proof-owner concentration, implementation risk, and
validation cost.

## Scope

Audited surfaces:

| family or path | primary test owners | primary implementation owners |
|---|---|---|
| linked-list Cholesky | `tests/test_cholesky.c` | `src/sparse_cholesky.c` |
| CSC Cholesky | `tests/test_chol_csc.c`; `tests/test_chol_csc_supernodal.c`; `tests/chol_external_dense_reference.py` | `src/sparse_chol_csc.c`; `src/sparse_chol_csc_supernodal.c` |
| linked-list LDLT | `tests/test_ldlt.c` | `src/sparse_ldlt.c`; `src/sparse_ldlt_dense.c` |
| CSC LDLT | `tests/test_ldlt_csc.c`; `tests/ldlt_external_dense_reference.py` | `src/sparse_ldlt_csc.c`; `src/sparse_ldlt_csc_supernodal.c` |
| LU | `tests/test_sparse_lu.c`; `tests/test_lu_csr.c` | `src/sparse_lu.c`; `src/sparse_lu_csr.c` |
| QR | `tests/test_qr.c` | `src/sparse_qr.c` |
| SVD | `tests/test_svd.c`; `tests/test_svd_partial_helpers.h` | `src/sparse_svd.c`; `src/sparse_svd_partial.c` |
| direct CSC dispatch | `tests/test_direct_csc_dispatch.c`; `tests/test_direct_csc_regression.c`; `tests/test_ldlt_backend_dispatch.c` | direct solver wrappers and backend dispatch paths |

## Test and Source Concentration

| owner | line count | `RUN_TEST` count | concentration note |
|---|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3878 | 98 | largest direct-solver proof owner; includes external dense reference lane |
| `tests/test_qr.c` | 3234 | 73 | large single owner for QR factorization, solve, rank, sparse mode, and refinement |
| `tests/test_ldlt.c` | 2977 | 88 | large linked-list LDLT owner |
| `tests/test_svd.c` | 2766 | 97 | large single owner for SVD, partial SVD, rank, pseudoinverse, low-rank, and condition estimates |
| `tests/test_chol_csc.c` | 2617 | 92 | large CSC Cholesky owner; includes external dense reference lane |
| `tests/test_chol_csc_supernodal.c` | 2482 | 61 | large supernodal Cholesky owner |
| `tests/test_lu_csr.c` | 1899 | 53 | large LU CSR and block-solver owner |
| `tests/test_sparse_lu.c` | 908 | 37 | linked-list LU owner with singular/failure tests |
| `tests/test_ldlt_backend_dispatch.c` | 935 | 20 | mixed backend dispatch plus adjacent eigensolver evidence |
| `tests/test_direct_csc_regression.c` | 569 | 8 | focused direct CSC regression owner |
| `tests/test_direct_csc_dispatch.c` | 492 | 10 | focused dispatch owner |
| `tests/test_cholesky.c` | 535 | 21 | smaller linked-list Cholesky owner |

Implementation concentration:

| owner | line count | concentration note |
|---|---:|---|
| `src/sparse_ldlt_csc.c` | 2174 | largest direct-family implementation owner in this audit |
| `src/sparse_lu_csr.c` | 1665 | large LU CSR owner |
| `src/sparse_qr.c` | 1563 | large QR owner |
| `src/sparse_ldlt.c` | 1535 | large linked-list LDLT owner |
| `src/sparse_svd.c` | 1319 | large SVD owner |
| `src/sparse_chol_csc.c` | 1279 | large CSC Cholesky owner |
| `src/sparse_lu.c` | 1042 | medium linked-list LU owner |
| `src/sparse_cholesky.c` | 615 | smaller linked-list Cholesky owner |

## Existing Oracle Evidence Classification

| family | current strongest evidence | fixture diversity | failure-mode clarity | oracle-depth classification |
|---|---|---|---|---|
| Cholesky CSC | external dense-reference tests for `nos4` and `bcsstk04`; many residual and SuiteSparse SPD checks | moderate SPD coverage; limited conditioning/scale taxonomy | good SPD/indefinite distinction; Windows helper skip explicit | external-helper plus fixture-corpus residual proof |
| LDLT CSC | external dense-reference tests for `kkt5` and `kkt10`; many indefinite/SPD residual checks | moderate synthetic indefinite coverage; limited external fixture variety | good pivot/failure and backend notes; large-owner complexity remains | external-helper plus internal fixture proof |
| linked-list Cholesky | residual tests on small synthetic and SuiteSparse fixtures | moderate SPD fixture coverage | limited external oracle depth | fixture-corpus residual proof |
| linked-list LDLT | broad internal and dense-style fixture checks | broad synthetic coverage but concentrated | moderate; needs taxonomy separation | internal/dense-reference style proof |
| LU linked-list | residual and singular/failure tests; pivot strategy checks | small synthetic plus residual matrix cases | good singular/null/nonsquare failure checks | internal residual and failure-mode proof |
| LU CSR | residual tests on synthetic, `orsirr_1`, `steam1`, block paths, dense-block fallback | good sparse and block diversity; no external oracle helper | good singular/block fallback coverage | fixture-corpus residual proof |
| QR | reconstruction, solve, rank, refinement, sparse-mode, SuiteSparse checks | broad rectangular/rank/sparse-mode diversity | good null/rank-deficient coverage | internal invariant and cross-check proof |
| SVD | singular values, rank, pseudoinverse, low-rank, partial/vector reconstruction, condition checks | broad rank/rectangular/SuiteSparse coverage | good null/bad-k/rank-deficient coverage | internal invariant and cross-check proof |
| direct CSC dispatch | path routing, residuals, backend flags, batched/refactor behavior | moderate SPD and indefinite dispatch fixtures | good path/failure notes | dispatch smoke and regression proof |

## External Helper Inventory

| helper | current owner | current fixture keys or inputs | current behavior |
|---|---|---|---|
| `tests/chol_external_dense_reference.py` | `tests/test_chol_csc.c` | Matrix Market paths such as `nos4` and `bcsstk04` | emits dense Cholesky reference solution or `SKIP`/`ERROR` state |
| `tests/ldlt_external_dense_reference.py` | `tests/test_ldlt_csc.c` | `kkt5`, `kkt10` | emits dense LDLT/KKT reference solution or `ERROR` state |

Current gap: both helpers encode useful command/status conventions, but the C
harness glue is family-local and duplicated. Sprint 102 should decide whether
to extract a small shared test-support reader/status helper or preserve
family-local code to keep tolerance and fixture semantics explicit.

## Gap Scores

Scoring scale:

- `5` = highest priority or largest gap
- `1` = lowest priority or smallest gap

| candidate lane | user value | current evidence gap | implementation risk | validation cost | total priority | notes |
|---|---:|---:|---:|---:|---:|---|
| LU external dense-reference solve lane | 5 | 4 | 3 | 3 | 15 | LU is central and has strong residual/failure tests but no external oracle helper |
| QR dense-reference least-squares/rank lane | 4 | 4 | 3 | 3 | 14 | QR has broad internal coverage but no external dense least-squares reference lane |
| LDLT CSC external fixture expansion | 4 | 3 | 3 | 3 | 13 | existing KKT lane is useful; fixture taxonomy can add scale/order/indefinite variation |
| Cholesky CSC external fixture expansion | 4 | 2 | 2 | 3 | 11 | existing external SPD lane is already strongest; expansion should be selective |
| SVD dense-reference singular-value/vector lane | 4 | 3 | 4 | 4 | 11 | broad internal invariant coverage exists; external dense SVD helper may be heavier |
| direct CSC dispatch oracle/reporting lane | 3 | 3 | 3 | 3 | 9 | dispatch proof is valuable but should rely on family evidence rather than become a separate oracle owner |
| linked-list Cholesky oracle expansion | 3 | 2 | 2 | 2 | 9 | less urgent than CSC/direct-family expansion |
| linked-list LDLT oracle expansion | 3 | 2 | 3 | 3 | 8 | large owner; likely better served through shared taxonomy and CSC LDLT lane first |

## Ranked Expansion Queue

1. **LU external dense-reference solve lane.**
   Add a bounded dense-reference helper or reusable oracle pattern for a small
   nonsymmetric square fixture plus one singular/expected-failure fixture. LU
   is high-value, currently residual-heavy, and lacks external oracle depth.

2. **QR dense-reference least-squares or rank lane.**
   Add a bounded dense least-squares/rank comparison for a tall or
   rank-deficient fixture. QR has broad internal invariants but no external
   oracle process.

3. **LDLT CSC fixture expansion.**
   Extend the existing `kkt5`/`kkt10` external lane only after Day 3 taxonomy
   names fixture classes and expected failure behavior. This is the best CSC
   direct-family expansion candidate if Sprint 102 chooses family continuity
   over LU/QR breadth.

4. **Cholesky CSC fixture expansion.**
   Keep as a secondary option because external `nos4`/`bcsstk04` proof already
   exists. Any expansion should target a taxonomy gap such as scale,
   conditioning, or reorder variation rather than simply adding another SPD
   fixture.

5. **SVD dense-reference lane.**
   Valuable but heavier. Defer until fixture taxonomy clarifies whether Sprint
   102 needs singular-value-only, vector reconstruction, rank, or
   pseudoinverse oracle evidence.

6. **Direct CSC dispatch lane.**
   Treat as proof that routing preserves family behavior, not as an independent
   external oracle lane. Dispatch should consume Cholesky/LDLT/LU evidence
   rather than lead Sprint 102.

## Day 3 Fixture Taxonomy Inputs

The fixture taxonomy should classify at least:

| class | reason |
|---|---|
| SPD Matrix Market fixtures | already used by Cholesky CSC and QR/SVD tests |
| synthetic indefinite KKT fixtures | already used by LDLT CSC external helper |
| nonsymmetric square solve fixtures | needed for LU external oracle depth |
| tall full-rank least-squares fixtures | needed for QR external oracle depth |
| rank-deficient rectangular fixtures | needed for QR/SVD expected-failure or rank semantics |
| singular square fixtures | needed to separate expected failure from solver regression |
| scaled or nearly singular fixtures | needed for tolerance policy |
| reorder-sensitive fixtures | needed for Cholesky/LDLT trust-boundary wording |

## Non-Claims Preserved

This audit does not claim:

- any new solver correctness evidence has landed;
- every direct solver has external oracle coverage;
- LU, QR, or SVD are externally validated;
- direct CSR/CSC solver APIs exist;
- dispatch evidence proves family-level numerical correctness by itself;
- local timing output is portable performance evidence;
- the library is broadly state-of-the-art.

## Day 2 Conclusion

Sprint 102 should use Day 3 to define fixture taxonomy before implementation.
The highest-value new external-oracle gap is LU, followed by QR. The best CSC
continuity lane is an LDLT fixture expansion, while Cholesky CSC already has
the strongest external dense-reference baseline and should expand only if the
taxonomy identifies a meaningful fixture-class gap.
