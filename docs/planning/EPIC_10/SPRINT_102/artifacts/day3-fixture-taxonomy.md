# Sprint 102 Day 3 Fixture Taxonomy

## Purpose

Day 3 defines fixture classes and expected outcomes before Sprint 102 adds
new direct-solver oracle coverage. The taxonomy keeps future tests from
mixing correctness proof, expected failure behavior, helper availability, and
unsupported cases.

## Taxonomy Rules

Every new Sprint 102 solver comparison or oracle artifact must identify:

- fixture key;
- source or construction path;
- solver family and exact API path;
- matrix shape;
- symmetry;
- definiteness;
- rank expectation;
- scaling or conditioning note;
- sparsity and ordering note;
- expected success, expected failure, or unsupported status;
- oracle/reference behavior;
- acceptance criteria;
- non-claims.

Fixture classes should be solver-neutral where possible. Family-local fixture
keys are allowed only when a solver needs a specific mathematical structure,
such as SPD Cholesky inputs, indefinite KKT LDLT inputs, or rectangular QR/SVD
inputs.

## Fixture Class Catalog

| class id | fixture class | shape | symmetry | definiteness | rank expectation | primary use |
|---|---|---|---|---|---|---|
| `spd-mm-small` | small SPD Matrix Market fixture | square | symmetric | positive definite | full rank | Cholesky, LDLT, QR/SVD cross-checks |
| `spd-mm-medium` | medium SPD Matrix Market fixture | square | symmetric | positive definite | full rank | Cholesky CSC, dispatch, reorder-sensitive residuals |
| `spd-mm-large` | larger SPD Matrix Market fixture | square | symmetric | positive definite | full rank | bounded stress or dispatch residuals, not default oracle expansion |
| `indef-kkt-small` | synthetic indefinite KKT fixture | square | symmetric | indefinite | nonsingular by construction | LDLT CSC external and backend checks |
| `indef-kkt-scaled` | scaled or reordered KKT fixture | square | symmetric | indefinite | nonsingular by construction | future LDLT tolerance/order sensitivity |
| `nonsym-square-small` | synthetic nonsymmetric square solve fixture | square | unsymmetric | n/a | full rank | LU external dense-reference lane |
| `nonsym-mm-medium` | Matrix Market nonsymmetric solve fixture | square | unsymmetric | n/a | expected nonsingular | LU residual or external-reference candidate |
| `tall-full-rank` | overdetermined full-rank least-squares fixture | rectangular tall | unsymmetric | n/a | full column rank | QR dense-reference least-squares lane |
| `wide-full-rank` | underdetermined or wide rectangular fixture | rectangular wide | unsymmetric | n/a | full row or bounded rank | QR/SVD shape behavior |
| `rect-rank-def` | rectangular rank-deficient fixture | rectangular | unsymmetric | n/a | deficient by construction | QR rank and SVD rank semantics |
| `square-rank-def` | singular or rank-deficient square fixture | square | any | singular or semidefinite | deficient by construction | expected failure for LU/Cholesky/LDLT; rank proof for QR/SVD |
| `scaled-near-singular` | scaled or nearly singular fixture | square or rectangular | any | family-specific | full rank or borderline | tolerance policy and failure boundary |
| `malformed-input` | malformed Matrix Market or invalid compressed input | n/a | n/a | n/a | n/a | parser/constructor expected failure, not solver correctness |

## Existing Fixture Sources

| fixture or source | taxonomy class | current evidence role | notes |
|---|---|---|---|
| `tests/data/suitesparse/nos4.mtx` | `spd-mm-small` | Cholesky CSC external reference, QR/SVD tests, dispatch | standard small SPD fixture |
| `tests/data/suitesparse/bcsstk04.mtx` | `spd-mm-medium` | Cholesky CSC external reference and factor/residual checks | used with AMD in external Cholesky lane |
| `tests/data/suitesparse/bcsstk14.mtx` | `spd-mm-large` | dispatch and larger SPD residual/fill checks | useful but higher cost; avoid default external expansion |
| `tests/data/suitesparse/Kuu.mtx` | `spd-mm-large` | direct CSC regression/stress | not a first Sprint 102 oracle candidate |
| `tests/data/suitesparse/orsirr_1.mtx` | `nonsym-mm-medium` | LU CSR residual and block checks | possible LU external-reference candidate if helper cost is bounded |
| `tests/data/suitesparse/steam1.mtx` | `nonsym-mm-medium` | LU CSR residual and block checks | possible LU residual candidate; external dense solve may be costlier |
| `tests/data/suitesparse/west0067.mtx` | `nonsym-mm-medium` or SVD corpus | QR/SVD tests | possible QR/SVD fixture but needs rank/conditioning notes |
| `tests/data/unsymm_5.mtx` | `nonsym-square-small` | small unsymmetric Matrix Market input | good low-cost LU external-reference candidate |
| `tests/data/tridiagonal_20.mtx` | `spd-mm-small` | small structured SPD fixture | possible low-cost Cholesky/QR/SVD fixture |
| `tests/data/diagonal_10.mtx` | `spd-mm-small` or SVD diagonal | exact diagonal/reference behavior | good tolerance-control fixture |
| `tests/data/bad_header.mtx` | `malformed-input` | parser failure | not solver correctness proof |
| `kkt5` | `indef-kkt-small` | LDLT CSC external dense-reference fixture | synthetic helper key in `ldlt_external_dense_reference.py` |
| `kkt10` | `indef-kkt-small` | LDLT CSC external dense-reference fixture | synthetic helper key in `ldlt_external_dense_reference.py` |

## Expected Outcome Matrix

| class id | Cholesky | LDLT | LU | QR | SVD | dispatch |
|---|---|---|---|---|---|---|
| `spd-mm-small` | expected success | expected success | expected success if nonsingular | expected success | expected success | expected success for SPD route |
| `spd-mm-medium` | expected success | expected success | expected success if nonsingular | expected success | expected success | expected success for SPD route |
| `spd-mm-large` | expected success if cost bounded | expected success if cost bounded | not default | not default | not default | expected success for selected stress lanes |
| `indef-kkt-small` | expected failure or unsupported | expected success | expected success if nonsingular | expected success if rank permits | expected success | expected LDLT route success |
| `indef-kkt-scaled` | expected failure or unsupported | expected success or tolerance-bound failure | not default | not default | not default | expected LDLT route behavior |
| `nonsym-square-small` | unsupported | unsupported unless symmetrized | expected success if nonsingular | expected success | expected success | expected LU route only if dispatch supports it |
| `nonsym-mm-medium` | unsupported | unsupported unless symmetrized | expected success if nonsingular | expected success | expected success | route-specific |
| `tall-full-rank` | unsupported | unsupported | unsupported for square LU solve | expected least-squares success | expected success | unsupported |
| `wide-full-rank` | unsupported | unsupported | unsupported for square LU solve | expected shape-specific success or documented limitation | expected success | unsupported |
| `rect-rank-def` | unsupported | unsupported | unsupported for square LU solve | expected rank-deficient behavior | expected rank-deficient behavior | unsupported |
| `square-rank-def` | expected failure | expected singular failure | expected singular failure | expected rank-deficient behavior | expected rank-deficient behavior | route-specific expected failure |
| `scaled-near-singular` | tolerance-bound success or failure | tolerance-bound success or failure | tolerance-bound success or singular failure | tolerance-bound rank/refinement behavior | tolerance-bound rank/condition behavior | route-specific |
| `malformed-input` | not solver proof | not solver proof | not solver proof | not solver proof | not solver proof | not solver proof |

## Expected-Failure Classes

| expected-failure class | examples | required test interpretation |
|---|---|---|
| singular square input | zero row, rank-deficient square matrix, zero matrix | expected solver failure for LU/Cholesky/LDLT; not a correctness regression |
| indefinite input to Cholesky | KKT or matrix with negative eigenvalue | expected Cholesky factor failure or unsupported status |
| rectangular input to square solver | tall/wide matrix passed to LU/Cholesky/LDLT | expected `BADARG` or unsupported status |
| rank-deficient QR/SVD input | dependent columns or zero singular values | expected rank/least-squares semantics, not necessarily solver failure |
| near-singular or scaled input | tiny pivots, large scale gaps, borderline condition | tolerance-bound outcome must be declared before implementation |
| unavailable external helper | Windows helper skip or missing fixture | skip/unsupported; must not be counted as oracle pass |
| malformed data | bad Matrix Market header or invalid compressed arrays | parser/constructor failure; not solver numerical evidence |

## Solver-Family Mapping

| family | preferred Sprint 102 fixture classes | candidate proof type | trust-boundary wording |
|---|---|---|---|
| Cholesky CSC | `spd-mm-small`, `spd-mm-medium`, selected `scaled-near-singular` | external dense Cholesky or residual plus dense reference | proves bounded SPD fixtures only; does not prove indefinite or universal SPD corpus |
| LDLT CSC | `indef-kkt-small`, selected `indef-kkt-scaled`, `square-rank-def` expected failures | external dense solve for KKT, residual checks, pivot failure checks | proves named indefinite/symmetric fixtures only; does not prove every indefinite matrix |
| LU linked-list/CSR | `nonsym-square-small`, `nonsym-mm-medium`, `square-rank-def`, selected `scaled-near-singular` | dense solve helper plus residual/failure checks | proves named nonsymmetric solves and singular behavior; not every pivoting case |
| QR | `tall-full-rank`, `rect-rank-def`, `wide-full-rank`, `nonsym-mm-medium` | dense least-squares/rank reference plus reconstruction | proves selected shape/rank behavior; not full numerical rank taxonomy |
| SVD | `rect-rank-def`, `wide-full-rank`, `scaled-near-singular`, diagonal controls | dense singular-value or reconstruction reference | proves selected singular-value/rank behavior; not full LAPACK parity |
| direct CSC dispatch | `spd-mm-small`, `spd-mm-medium`, `indef-kkt-small` | route assertion plus family residual/oracle reuse | proves routing and no residual regression; family correctness comes from family lanes |

## Fixture Naming Rules

New synthetic fixture keys should use:

```text
<family-or-neutral>_<class>_<size>[_<variant>]
```

Examples:

- `lu_nonsym_square_5`
- `lu_singular_square_4`
- `qr_tall_fullrank_6x3`
- `qr_rect_rankdef_6x4`
- `ldlt_kkt_scaled_10`
- `svd_diag_scaled_8`

Rules:

1. Include the mathematical class in the key.
2. Include dimensions or a size hint.
3. Add a variant suffix only for meaningful distinctions such as `scaled`,
   `amd`, `near_singular`, or `rankdef`.
4. Do not use sprint numbers in fixture keys unless the fixture exists only as
   transitional planning evidence.
5. A fixture key may be shared across solver families only when the expected
   outcome is documented for each family.

## Storage and Generation Rules

| fixture type | preferred storage | rule |
|---|---|---|
| small synthetic dense fixture | helper-builder function in test or helper script | keep deterministic and cheap; document exact entries or construction |
| reusable Matrix Market fixture | `tests/data` or `tests/data/suitesparse` | use existing fixtures first; add new files only when generated fixtures are insufficient |
| external helper fixture | Python helper key plus C harness builder or loader | helper and C harness must build/load the same matrix class |
| expected-failure fixture | local builder in focused test | keep small and obvious; expected error must be asserted |
| large stress fixture | existing `tests/data/suitesparse` file | use only after cost and platform behavior are explicit |

Generation rules:

- use deterministic construction only;
- avoid random fixtures unless a fixed seed and expected skip/failure policy
  already exist;
- choose `x_true[i] = i + 1` and `b = A*x_true` for direct solve oracle lanes
  unless the solver family requires a different target;
- record tolerances before implementation;
- keep local timing fields out of correctness claims.

## Sprint 102 Lane Recommendations

| ranked lane | recommended taxonomy entry | Day 4/5 implication |
|---|---|---|
| LU external dense-reference solve | `lu_nonsym_square_5` plus `lu_singular_square_4` | helper boundary should support one success and one expected failure without becoming a broad LU helper framework |
| QR dense-reference least-squares/rank | `qr_tall_fullrank_6x3` or `qr_rect_rankdef_6x4` | helper boundary should decide whether dense least-squares reference is in scope after LU |
| LDLT CSC external fixture expansion | `ldlt_kkt_scaled_10` or a reordered `kkt10` variant | keep external helper family-local unless shared status parsing is extracted |
| Cholesky CSC expansion | selected scaled SPD or existing `tridiagonal_20`/`diagonal_10` control | lower priority because existing external SPD lane already covers `nos4` and `bcsstk04` |
| SVD dense-reference lane | `svd_diag_scaled_8` or `svd_rect_rankdef_6x4` | defer unless LU/QR scope leaves capacity |

## Non-Claims Preserved

This taxonomy does not claim:

- any new direct-solver oracle evidence has landed;
- fixture classes are complete for all future sprints;
- external oracle coverage exists for LU, QR, or SVD;
- direct CSR/CSC solver APIs exist;
- expected skips are correctness passes;
- local timing output is portable performance evidence;
- the library has broad state-of-the-art solver parity.

## Day 3 Conclusion

Sprint 102 now has fixture classes and expected-outcome rules that later days
can cite before implementation. The taxonomy points Day 4 toward a narrow
helper boundary, with LU external dense-reference coverage as the highest-value
new lane and QR/LDLT as the next ranked candidates.
