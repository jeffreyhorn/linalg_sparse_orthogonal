# Sprint 102 Day 12 - Direct Solver Guidance Update

## Scope

Day 12 updates documentation wording only where Sprint 102 evidence supports
the claim. The touched documentation surfaces are:

- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

No public headers, examples, tests, library sources, build files, or generated
API HTML were changed for Day 12.

## Public Guidance Updates

### README

The README workflow section now names the intended direct-solver selection
boundary:

- LU for general square systems;
- Cholesky for symmetric positive-definite systems;
- LDL^T for symmetric indefinite systems;
- QR for rectangular or rank-deficient least-squares workflows.

It also states that maintained external dense-reference evidence is
family-local and bounded to named Cholesky CSC, LDLT CSC, and linked-list LU
lanes.

### Tutorial

The tutorial direct-solver section now adds bounded trust notes:

- LU has a maintained external dense-reference lane for one nonsymmetric
  square fixture and one singular expected-failure fixture.
- Cholesky external dense-reference evidence remains bounded to CSC-backed SPD
  fixture lanes.
- LDL^T guidance now explicitly points symmetric indefinite users to LDL^T and
  binds its external dense-reference evidence to deterministic KKT fixtures.

The tutorial does not claim broad solver-family external-oracle coverage.

## Maintainer Trust Boundary

`docs/maintainer_guide.md` now records a Sprint 102 direct-solver trust
boundary table:

| Family / lane | Maintained evidence | Trust boundary |
|---|---|---|
| Cholesky CSC SPD | `tests/test_chol_csc.c` plus `tests/chol_external_dense_reference.py` | named SPD Matrix Market fixtures |
| LDLT CSC indefinite | `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py` | `kkt5`, `kkt10`, and `ldlt_kkt_scaled_10` |
| Linked-list LU | `tests/test_sparse_lu.c` plus `tests/lu_external_dense_reference.py` | `lu_nonsym_square_5` solve and `lu_singular_square_4` expected failure |
| QR | `tests/test_qr.c` | internal rank, residual, and scalar seam coverage |
| SVD | `tests/test_svd.c` | internal reconstruction, rank, condition, and partial-SVD coverage |

## Non-Claims Preserved

Day 12 does not claim:

- complete external oracle coverage for every direct solver family;
- LU CSR external dense-reference coverage;
- direct public CSR/CSC solve APIs for LU, Cholesky, LDLT, QR, or SVD;
- broad solver superiority or ecosystem parity;
- external dense SVD or QR oracle coverage;
- performance superiority from correctness fixtures.

## Validation

Day 12 is documentation-only. Required validation:

- `git diff --check`
- trailing-whitespace scan on touched documentation and Sprint 102 planning
  files

## Closeout

Sprint 102 now has public and maintainer-facing wording inputs for Sprint 103
comparison work. The wording remains tied to implemented evidence and named
test owners rather than broad product claims.
