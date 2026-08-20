# Day 2: Candidate Family Inventory

## Purpose

Inventory candidate solver families for the next bounded external comparison
report family and rank them before Sprint 174 selects exactly one family on
Day 3.

## Existing Generated Comparison Baseline

The current generated comparison report/freshness family covers:

| Subfamily | Target | Fixture | Rows | Current status |
| --- | --- | --- | ---: | --- |
| `qr_minnorm` | `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | 6 | selected generated comparison family |
| `qr_compatible_ls` | `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | 6 | selected generated comparison family |
| `partial_svd_diag6_k2` | `partial-svd-diag6-k2` | `partial_svd_diag6_k2` | 10 | selected generated comparison family |

The current selected comparison freshness target is:

```sh
make report-index-comparison-freshness
```

It expects three source-controlled comparison contract rows and 22 generated
selected comparison rows under `build/comparison/*/study.tsv`.

## Candidate Matrix

| Candidate | Existing evidence | Comparator availability | Harness fit | User value | Implementation risk | Claim containment |
| --- | --- | --- | --- | --- | --- | --- |
| Cholesky CSC SPD generated comparison | `tests/test_chol_csc.c` checks `nos4.mtx` and `bcsstk04.mtx` against `tests/chol_external_dense_reference.py`. | Existing Python helper; external process dense reference; Matrix Market inputs. | Medium: runner would need a direct-solver target shape, but metrics are solution/residual oriented and simple. | High: SPD direct solve is a core workflow and currently lacks generated comparison report rows. | Medium: Matrix Market fixture path and reorder choice need stable report schema. | Strong if bounded to one named SPD fixture and residual/solution delta; must avoid broad Cholesky ecosystem parity. |
| LDLT CSC KKT generated comparison | `tests/test_ldlt_csc.c` checks `kkt5`, `kkt10`, and `ldlt_kkt_scaled_10` against `tests/ldlt_external_dense_reference.py`. | Existing Python helper; deterministic KKT fixtures. | Medium: deterministic fixtures are easier than Matrix Market paths, but LDLT status/pivot interpretation needs careful wording. | High: indefinite direct solves are important and not yet represented in generated comparison reports. | Medium: claim wording must avoid factorization-layout, pivot-internal, and broad indefinite parity claims. | Strong if bounded to one KKT solve and residual/solution delta. |
| linked-list LU generated comparison | `tests/test_sparse_lu.c` checks `lu_nonsym_square_5` and singular `lu_singular_square_4` against `tests/lu_external_dense_reference.py`. | Existing Python helper; deterministic nonsymmetric fixture. | High: one square solve maps cleanly to project/baseline status, residual, solution norm, solution values, and max delta. | High: general nonsymmetric LU is a front-door direct solve workflow. | Low/Medium: simpler fixture, but docs must avoid LU CSR, CSR/CSC public solve API, and broad nonsymmetric parity. | Strong if scoped to linked-list LU on `lu_nonsym_square_5`; singular behavior can remain deferred. |
| Additional QR generated comparison | Many QR helper-backed fixtures already exist; generated comparison already covers minimum-norm and compatible least-squares. | Existing QR helper and runner. | High: existing runner shape fits QR. | Medium: adds depth but not a new solver family. | Low/Medium: easiest extension, but less closure value than adding a new family. | Medium/Strong if bounded to one fixture; risk of overreading as broader QR parity because QR already has many lanes. |
| Additional partial-SVD/SVD generated comparison | SVD helper-backed fixtures exist; generated comparison already covers `partial_svd_diag6_k2`. | Existing SVD helper and runner. | High: existing runner shape fits SVD/partial-SVD. | Medium: useful depth, but Sprint 161 already added first partial-SVD comparison family. | Medium: repeated spectrum/vector orientation and subspace wording must stay tight. | Medium if bounded; raw vector/sign and repeated-spectrum claims are easy to overstate. |
| LU CSR generated comparison | Sprint 167 listed LU CSR dense solve as a candidate, but Day 2 found no existing external helper-backed generated comparison shape. | No obvious dedicated LU CSR external dense helper in the current comparison runner. | Low/Medium: likely requires new fixture and solver API boundary design. | High if implemented, but higher setup cost. | High: public API and direct CSR/CSC solve claims are sensitive. | Weak until a Day 3 decision freezes exact public API and fixture scope. |
| Iterative solver external comparison | Existing tests include internal comparisons such as BiCGSTAB vs GMRES/ILU. | No source-controlled external helper architecture identified for PETSc/SciPy/Trilinos style comparison. | Low for Sprint 174. | Medium/High long-term. | High: convergence semantics, optional dependencies, and iteration/performance interpretation are hard to bound. | Weak for this sprint; should be deferred unless separately architected. |
| Eigensolver external comparison | LOBPCG and thick-restart tests have closed-form/internal evidence. | No external helper architecture identified for ARPACK/SciPy comparison. | Low for Sprint 174. | Medium/High long-term. | High: eigenpair ordering, clusters, tolerances, residuals, and optional dependencies need a separate design. | Weak for this sprint; should be deferred unless separately architected. |

## Ranking

| Rank | Candidate | Reason |
| ---: | --- | --- |
| 1 | linked-list LU generated comparison | Best balance of user value, existing helper coverage, simple deterministic fixture, clean report metrics, and claim containment. |
| 2 | LDLT CSC KKT generated comparison | High value and deterministic fixtures, but pivot/factorization wording requires more care than LU. |
| 3 | Cholesky CSC SPD generated comparison | High value and existing helper coverage, but Matrix Market path/reorder choices make report schema slightly heavier. |
| 4 | Additional QR generated comparison | Low implementation risk, but lower closure value because QR already has two selected generated comparison families. |
| 5 | Additional partial-SVD/SVD generated comparison | Low/medium implementation risk, but lower closure value because partial-SVD already has one selected generated comparison family. |
| 6 | LU CSR generated comparison | Valuable but likely needs more API and fixture architecture than a 14-day bounded report-family sprint should absorb. |
| 7 | Iterative solver external comparison | Too broad for Sprint 174 without a new external-helper and convergence-claim architecture. |
| 8 | Eigensolver external comparison | Too broad for Sprint 174 without a new external-helper and eigenpair-claim architecture. |

## Recommended Day 3 Selection Criteria

Day 3 should select a candidate only if it can satisfy all of these criteria:

- exactly one solver family and one fixture are selected;
- the external comparator is source-controlled and already runnable locally;
- generated report rows can use the existing `scripts/run_external_comparison.py`
  output shape or a narrowly extended variant;
- report-family proof-owner rows can be added without changing unrelated
  families;
- missing helper dependencies fail as skip/defer, not pass evidence;
- the comparison can be described without broad parity, platform, performance,
  package, ABI, runtime-loader, or state-of-the-art claims;
- focused validation can run without requiring optional external packages
  beyond what the helper already handles.

## Recommended Day 3 Shortlist

The Day 3 decision should start from this shortlist:

1. `linked-list LU` on `lu_nonsym_square_5`
2. `LDLT CSC` on one KKT fixture, preferably `kkt5` or `kkt10`
3. `Cholesky CSC` on one named SPD Matrix Market fixture

The strongest candidate is `linked-list LU` on `lu_nonsym_square_5` because it
has an existing source-controlled dense-reference helper, a deterministic
fixture, simple solve-quality metrics, high user value, and a clear non-claim
boundary excluding LU CSR and broad nonsymmetric ecosystem parity.

## Deferred Or Rejected For Sprint 174

| Candidate | Sprint 174 disposition |
| --- | --- |
| Additional QR comparison | Defer unless Day 3 rejects direct-solver candidates; QR already has selected generated comparison coverage. |
| Additional partial-SVD/SVD comparison | Defer unless Day 3 rejects direct-solver candidates; partial-SVD already has selected generated comparison coverage. |
| LU CSR comparison | Defer to a sprint that can explicitly design public API, fixture, and direct CSR/CSC solve claim boundaries. |
| Iterative solver comparison | Defer pending external-helper architecture and convergence/tolerance policy. |
| Eigensolver comparison | Defer pending external-helper architecture and eigenpair ordering/tolerance policy. |

## Day 2 Validation

Day 2 is planning-only inventory. No `.c` or `.h` files changed, so the full C
quality gate is not required. `git diff --check` is the required day-level
hygiene check.

## Completion Check

Day 2 completion criteria are met:

- candidate families are comparable before selection;
- no harness implementation started before selecting one family;
- unsupported broad parity claims remain excluded.
