# Sprint 155 Day 6 Header Cleanup Selection

## Purpose

Day 6 selects the Sprint 155 public-header cleanup batch before any header
edits. The selection uses the Day 4-5 tutorial alignment results and preserves
the Sprint 155 constraint: selected cleanup must not change declarations,
signatures, typedefs, enum values, macros, struct fields, installed header
names, include guards, or ABI/package/platform claims.

## Public Header Inventory

| Header | Lines | Tutorial/Adoption Role | Current Disposition |
| --- | ---: | --- | --- |
| `include/sparse_analysis.h` | 499 | Explicit repeated-run direct lifecycle and advanced analysis/reordering controls. | Candidate. Tutorial now routes repeated direct lifecycle and advanced controls here. |
| `include/sparse_bidiag.h` | 72 | Lower-level SVD helper surface. | Defer; compact and less tutorial-facing. |
| `include/sparse_cholesky.h` | 227 | SPD one-shot direct solve and backend controls. | Defer; lower Day 5 mismatch than LDLT/eigs/IC/analysis. |
| `include/sparse_csr.h` | 161 | CSR/CSC conversion and compressed-first construction. | Defer; already concise and aligned with Day 4 data-input route. |
| `include/sparse_dense.h` | 197 | Dense helper utilities. | Defer; not a first tutorial cleanup target. |
| `include/sparse_eigs.h` | 651 | Symmetric eigensolver, AUTO backend, diagnostics, and advanced handle surface. | Candidate. Tutorial now adds eigensolver section and delegates exact options/results here. |
| `include/sparse_ic.h` | 116 | IC(0) preconditioner, CG/MINRES assumptions, callback semantics. | Candidate. Tutorial now routes IC/preconditioning through this surface. |
| `include/sparse_ilu.h` | 200 | ILU/ILUT preconditioners for general/indefinite workflows. | Defer; related to Day 5, but `sparse_ic.h` is smaller and currently missing from prior cleanup. |
| `include/sparse_iterative.h` | 731 | Iterative solver result and handle contracts. | Defer; cleaned in Sprint 145 and should be a reference, not default rework. |
| `include/sparse_ldlt.h` | 332 | Symmetric indefinite direct solve, LDLT backend dispatch, solve/refine/condest contracts. | Candidate. Tutorial now includes LDLT explicitly and routes symmetric-indefinite users here. |
| `include/sparse_lu.h` | 360 | General one-shot LU and refinement. | Defer; no Day 5 gap targeted it. |
| `include/sparse_lu_csr.h` | 322 | CSR LU dispatch and structural details. | Defer; specialized backend/API surface. |
| `include/sparse_matrix.h` | 585 | Core matrix shell and Matrix Market I/O. | Defer; cleaned in Sprint 145. Matrix Market comments are already concise enough for this sprint. |
| `include/sparse_qr.h` | 373 | QR rank, least-squares, and minimum-norm surface. | Defer; cleaned in Sprint 145 and current Day 5 work did not require revisiting it. |
| `include/sparse_reorder.h` | 186 | Reorder algorithms and ordering modes. | Defer; important, but secondary to tutorial Day 5 gaps. |
| `include/sparse_svd.h` | 243 | SVD, partial-SVD, condition, pseudoinverse, low-rank APIs. | Defer; cleaned in Sprint 145. Tutorial now delegates evidence boundary here indirectly through solver-selection. |
| `include/sparse_types.h` | 316 | Public scalar/index/error/progress types. | Defer; broad foundational surface best handled only if Day 12 reconciliation requires it. |
| `include/sparse_vector.h` | 70 | Sparse vector helpers. | Defer; compact and lower adoption impact. |
| `include/sparse_version.h.in` | 25 | Generated version header template. | Defer; package/version metadata is already owned by INSTALL and package checks. |

## Declaration-Preservation Tooling Inventory

No dedicated single-purpose declaration-preservation script was found during
the Day 6 inventory. Sprint 145 used focused git-diff and text scans after
header edits. Day 7 should formalize that into a repeatable checklist.

Required preservation approach for Days 8-9:

```sh
git diff -- include/*.h include/*.h.in
git diff --word-diff=porcelain -- include/*.h include/*.h.in
git diff --name-only -- '*.c' '*.h' '*.h.in'
rg -n "^[A-Za-z_].*\\);$|^typedef |^typedef enum|^typedef struct|^#define " include
git diff --check
```

Because Days 8-9 will modify public `.h` files, the full gate remains:

```sh
make format && make lint && make test
```

Day 7 should define the exact before/after capture files or command log before
any public header is edited.

## Candidate Scorecard

Scores use a 1-5 scale where 5 is strongest. Cleanup risk is inverted in the
total: lower risk produces a higher score.

| Header | User Impact | Cross-Doc Value | Cleanup Need | Low Risk | Total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `include/sparse_ldlt.h` | 5 | 5 | 4 | 3 | 17 | Select |
| `include/sparse_eigs.h` | 5 | 5 | 5 | 2 | 17 | Select |
| `include/sparse_ic.h` | 4 | 4 | 4 | 5 | 17 | Select |
| `include/sparse_analysis.h` | 5 | 5 | 4 | 2 | 16 | Select |
| `include/sparse_csr.h` | 4 | 4 | 2 | 5 | 15 | Defer |
| `include/sparse_ilu.h` | 4 | 4 | 3 | 4 | 15 | Defer |
| `include/sparse_cholesky.h` | 3 | 3 | 3 | 4 | 13 | Defer |
| `include/sparse_lu.h` | 3 | 3 | 3 | 4 | 13 | Defer |
| `include/sparse_reorder.h` | 3 | 3 | 3 | 4 | 13 | Defer |
| `include/sparse_types.h` | 4 | 4 | 3 | 1 | 12 | Defer |
| Sprint 145 cleaned headers | 4 | 4 | 2 | 3 | 13 | Defer by policy |

## Selected Cleanup Batch

Sprint 155 should clean these public headers:

1. `include/sparse_ldlt.h`
2. `include/sparse_ic.h`
3. `include/sparse_eigs.h`
4. `include/sparse_analysis.h`

### Batch Rationale

- `include/sparse_ldlt.h` now sits directly on the tutorial's symmetric
  indefinite route. It has high user value and several dense comment blocks
  around backend selection, tolerance semantics, and version/ABI wording that
  need clearer call-site boundaries without changing contracts.
- `include/sparse_ic.h` is compact but now more visible after the tutorial
  preconditioning rewrite. It should clarify IC(0), SPD/MINRES/CG assumptions,
  callback use, and local preconditioner caveats while preserving the existing
  API.
- `include/sparse_eigs.h` is the largest uncleaned tutorial-visible header.
  Day 5 added eigensolver tutorial coverage and delegates exact options,
  diagnostics, backend behavior, and handle details here. Cleanup should make
  that reference easier to scan without diluting backend or residual contracts.
- `include/sparse_analysis.h` owns the repeated-run direct lifecycle and
  advanced analysis/reordering controls referenced by the tutorial. Cleanup
  should clarify the public repeated-run contract and keep lower-level ND
  routing comments from becoming a tutorial burden.

## Batch Split For Days 8-9

Recommended implementation split:

| Day | Headers | Reason |
| --- | --- | --- |
| Day 8 | `include/sparse_ldlt.h`, `include/sparse_ic.h` | Direct/preconditioner headers tie most closely to the Day 4-5 tutorial updates and form a smaller first tranche. |
| Day 9 | `include/sparse_eigs.h`, `include/sparse_analysis.h` | Larger advanced-control headers need more careful declaration-preservation review. |

Day 7 should write the cleanup contract before either tranche is edited.

## Deferred Header Register

| Header | Deferred Reason |
| --- | --- |
| `include/sparse_matrix.h` | Cleaned in Sprint 145; Matrix Market comments are already concise and aligned enough for Sprint 155. |
| `include/sparse_iterative.h` | Cleaned in Sprint 145; Day 5 tutorial diagnostics can rely on it without rework. |
| `include/sparse_qr.h` | Cleaned in Sprint 145; no Day 5 QR wording gap requires revisiting it. |
| `include/sparse_svd.h` | Cleaned in Sprint 145; tutorial delegates current partial-SVD evidence to solver-selection. |
| `include/sparse_csr.h` | Aligned with Day 4 compressed-first route; no high-risk stale wording found. |
| `include/sparse_ilu.h` | Related to preconditioning, but lower priority than IC because the tutorial's missing handoff centered on IC(0)/MINRES. |
| `include/sparse_cholesky.h` | Important but not the strongest residual tutorial/API mismatch this sprint. |
| `include/sparse_lu.h` | First-solve path is already covered by examples and Sprint 145 matrix/iterative cleanup. |
| `include/sparse_lu_csr.h` | Specialized backend surface; defer until backend/API reference work needs it. |
| `include/sparse_reorder.h` | Reordering is important, but Day 5 tutorial keeps it as advanced solver-selection context. |
| `include/sparse_types.h` | Foundational and broad; defer unless Day 12 API/reference reconciliation identifies a narrow issue. |
| `include/sparse_bidiag.h`, `include/sparse_dense.h`, `include/sparse_vector.h`, `include/sparse_version.h.in` | Compact or lower tutorial-facing impact. |

## Preservation Constraints For Selected Batch

The selected cleanup must preserve:

- all function declarations and signatures;
- typedef names and layouts;
- enum names, values, and order;
- struct field names, order, and comments that carry ownership/default
  semantics;
- public macros and numeric values;
- include guards and includes unless Day 7 explicitly proves no surface
  change;
- documented ownership/freeing rules;
- documented default values and zero-init behavior;
- documented error returns;
- input mutation and non-mutation guarantees;
- identity-permutation, shape, symmetry, SPD, and same-pattern preconditions;
- non-claims for package, ABI, platform, performance, external parity,
  generated reports, and state-of-the-art status.

## Day 7 Cleanup Contract Handoff

Day 7 should produce:

1. allowed edit rules for comments, examples, ownership notes, error contracts,
   and cross-doc links;
2. disallowed edit rules for declarations, signatures, exported names, enum
   values, macros, struct fields, include guards, and unsupported claims;
3. exact declaration-preservation command log;
4. a maintainer checklist for public-header cleanup;
5. a Day 8 implementation checklist for `include/sparse_ldlt.h` and
   `include/sparse_ic.h`.

## Day 6 Completion Check

- Public header inventory exists.
- Candidate scorecard exists.
- Selected cleanup batch is bounded to four headers.
- Deferred headers have rationale.
- Signature, declaration, and ABI preservation constraints are explicit.
- Day 7 has a concrete cleanup-contract handoff.
