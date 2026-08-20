# Day 1: Sprint Intake And Comparison Boundary

## Purpose

Establish Sprint 174 scope, inherited evidence rules, existing comparison
families, and external-comparison non-claims before selecting or implementing
an additional bounded comparison family.

## Active Planning Source

The user prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but the
active merged Sprint 174 project-plan section is:

```text
docs/planning/EPIC_15/PROJECT_PLAN.md
Sprint 174: Additional Bounded External Comparison Family
```

The active Epic 15 section defines the sprint goal as adding one more complete
external comparison family with generated report, freshness checks, and
bounded claims.

## Sprint 174 Project-Plan Items

| Item | Intake interpretation |
| --- | --- |
| 174.1 Family Selection | Select exactly one solver family, fixture set, comparator, and tolerance policy before implementation. |
| 174.2 Fixture Design | Add bounded matrices only after the family and claim scope are frozen. |
| 174.3 Harness Extension | Extend the existing external comparison runner rather than inventing a parallel report format unless blocked. |
| 174.4 Report Integration | Add generated report/index/freshness behavior that fits the existing comparison family architecture. |
| 174.5 Claim Documentation | Document exact fixture, comparator, metric, tolerance, and non-claim boundaries. |
| 174.6 Validation | Run generation, freshness, focused tests, claim scans, and relevant deferral guards. |

## Existing Selected Comparison Gate

The maintained selected comparison freshness target is:

```sh
make report-index-comparison-freshness
```

It regenerates and checks selected local comparison output for:

| Target | Fixture | Artifact | Rows |
| --- | --- | --- | ---: |
| `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | `build/comparison/qr_minnorm/study.tsv` | 6 |
| `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | `build/comparison/qr_compatible_ls/study.tsv` | 6 |
| `partial-svd-diag6-k2` | `partial_svd_diag6_k2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | 10 |

The selected comparison freshness surface currently expects 22 generated rows
plus three source-controlled comparison contract rows.

## Existing Proof-Owner Rows

`tests/corpus/manifests/report_families.tsv` currently owns selected generated
comparison report rows for:

- `comparison/qr_minnorm`;
- `comparison/qr_compatible_ls`;
- `comparison/partial_svd_diag6_k2`.

Each row is `generated_local`, `local_only`, and bounded to a named fixture and
selected source-controlled dense reference helper.

## Existing External Dense-Reference Helpers

The current helper inventory includes:

| Helper | Current role |
| --- | --- |
| `tests/chol_external_dense_reference.py` | Cholesky CSC external dense-reference tests. |
| `tests/ldlt_external_dense_reference.py` | LDLT CSC KKT external dense-reference tests. |
| `tests/lu_external_dense_reference.py` | Linked-list LU external dense-reference tests. |
| `tests/qr_external_dense_reference.py` | QR selected comparison and oracle helper. |
| `tests/svd_external_dense_reference.py` | SVD and partial-SVD selected comparison and oracle helper. |

Only QR and partial-SVD currently participate in the generated comparison
report/freshness family. Direct-solver helper-backed tests are evidence, but
they are not yet generated comparison report families under the selected
comparison freshness gate.

## Sprint 173 Handoff

Sprint 173 closed generated API HTML as guarded local-only output. Sprint 174
must keep that boundary separate:

- generated API HTML uses `make api-docs-freshness`;
- generated comparison reports use `make report-index-comparison-freshness`;
- generated API HTML under `docs/api/` must not become comparison evidence;
- generated comparison outputs under `build/` must remain ignored local output
  unless a later decision explicitly promotes them.

## Initial Candidate Surface For Day 2

Day 2 should rank candidate families using existing helpers and residual
comparison gaps. Initial candidates visible from Day 1 intake are:

| Candidate | Current evidence | Day 2 question |
| --- | --- | --- |
| Cholesky CSC generated comparison report | external dense-reference helper-backed tests exist | Is it high-value to promote one named SPD fixture into generated comparison reports? |
| LDLT CSC generated comparison report | KKT external dense-reference helper-backed tests exist | Is a KKT fixture the best next generated comparison family? |
| linked-list LU generated comparison report | nonsymmetric/singular helper-backed tests exist | Can one LU fixture become a report family without implying LU CSR or broad nonsymmetric parity? |
| additional QR comparison family | QR helper and runner already exist | Would another QR fixture add more value than a new solver family? |
| additional partial-SVD/SVD comparison family | SVD helper and runner already exist | Would another SVD/partial-SVD fixture close a more important residual? |
| iterative/eigensolver comparison | existing tests include internal comparisons | Does the lack of external helper architecture make this too broad for Sprint 174? |

Day 2 should select candidates for detailed ranking, not implement any of
them.

## Retained Non-Claims

Sprint 174 must not imply:

- broad solver-family parity;
- LAPACK, NumPy, SciPy, SuiteSparse, Eigen, PETSc, Trilinos, ARPACK, or broad
  external-library ecosystem parity;
- raw QR basis or raw singular-vector identity;
- vector sign/orientation identity;
- global rank-threshold behavior;
- broad rank-deficient solve behavior;
- hosted CI proof beyond already reviewed selected lanes;
- release evidence;
- broad platform support;
- package-manager support;
- shared-library ABI support;
- runtime-loader behavior;
- performance superiority;
- state-of-the-art sparse linear algebra status.

## Day 1 Validation

Day 1 is planning-only intake. No `.c` or `.h` files changed, so the full C
quality gate is not required. `git diff --check` is the required day-level
hygiene check.

## Completion Check

Day 1 completion criteria are met:

- Sprint 174 scope is tied to the active Epic 15 project plan;
- existing comparison families and report owners are visible;
- unsupported platform, performance, package, ABI, and state-of-the-art claims
  remain protected before family selection.
