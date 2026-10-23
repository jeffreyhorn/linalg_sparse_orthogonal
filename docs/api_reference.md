# API Reference

Use this page when you need exact public declarations, option/result structs,
ownership rules, output-buffer expectations, or return-code contracts after
you have chosen a workflow from the README, tutorial, cookbook, or
solver-selection guide.

## Source Of Truth

The checked-in public headers under [`include/`](../include/) are the source of
truth for API declarations and call-site contracts:

| Header | Primary surface |
| --- | --- |
| `sparse_matrix.h` | Matrix construction, mutation, copy/free, norms, and dense conversion |
| `sparse_types.h` | Shared scalar, error, and compressed-format types |
| `sparse_vector.h` | Vector helpers, including scaled 2-norm and NaN-aware infinity norm |
| `sparse_lu.h` | One-shot LU and LU option/result contracts |
| `sparse_lu_csr.h` | CSR LU support surface |
| `sparse_cholesky.h` | Cholesky factorization and solve contracts |
| `sparse_ldlt.h` | LDL^T factorization, solve, backend, and telemetry contracts |
| `sparse_analysis.h` | Analyze-once/factor-many direct-solver lifecycle |
| `sparse_qr.h` | QR factorization/lifecycle, least-squares, minimum-norm, rank/nullspace, R-diagonal diagnostics, and cancellation contracts |
| `sparse_svd.h` | Full SVD, partial SVD, pseudoinverse, and low-rank contracts |
| `sparse_eigs.h` | Symmetric eigensolver options, backends, handles, and result contracts |
| `sparse_iterative.h` | CG, GMRES, MINRES, BiCGSTAB, matrix-free, and handle contracts |
| `sparse_ilu.h` | ILU(0), ILUT, and preconditioner callback contracts |
| `sparse_ic.h` | IC(0) and IC preconditioner callback contracts |
| `sparse_csr.h` | CSR storage helpers |
| `sparse_dense.h` | Dense matrix helpers |
| `sparse_bidiag.h` | Bidiagonalization helpers |
| `sparse_reorder.h` | Reordering APIs and option contracts |

Installed packages also include a generated `sparse_version.h` derived from
`VERSION` and `include/sparse_version.h.in`. Use the installed header,
`VERSION`, and the install-validation tests for version macro behavior. For
the current support/readiness status of installed packages, generated API HTML,
platforms, package managers, shared libraries, and ABI boundaries, use
[`INSTALL.md#support-readiness-matrix`](../INSTALL.md#support-readiness-matrix).

## Selected Numerical Diagnostics

[`sparse_vector.h`](../include/sparse_vector.h) specifies scaled `vec_norm2`
accumulation: finite inputs with a representable nonzero 2-norm do not become
zero or infinity merely because direct component squaring underflows or
overflows; tiny contributions can still round away. A final norm too large
for `double` is infinity. Both norm helpers propagate any NaN, even in the
presence of infinity, and retain the historical zero result for NULL or
nonpositive length. These
arithmetic properties do not imply accuracy for every ill-conditioned solve.

[`sparse_iterative.h`](../include/sparse_iterative.h) distinguishes non-finite
caller vectors (`SPARSE_ERR_BADARG`) from selected solver-detected invalid
norms or residual ratios (`SPARSE_ERR_NUMERIC`, no convergence, NaN result
diagnostic). The selection is CG, GMRES, MINRES, BiCGSTAB, their documented
matrix-free variants, and block CG; it is not all-solver numerical parity.
[`sparse_ilu.h`](../include/sparse_ilu.h) specifies that an ILUT factor solve
can report `SPARSE_ERR_NUMERIC`, with partial output on failure. A successful
factor-system solve does not prove `A*z=r` or useful preconditioning. These
bounded contracts do not establish state-of-the-art numerical performance.

## Generated HTML

`make docs-check` runs Doxygen with [`Doxyfile`](../Doxyfile), writes generated
HTML under `docs/api/html/`, and checks generated page coverage and freshness
for the checked-in public headers.

`make api-docs-freshness` runs the selected local current-output proof:
Doxygen generation, generated page coverage/freshness, local-only staging
enforcement, and API routing validation for the generated API tree.

The generated HTML tree is local-only generated output. It remains ignored by
the repository and is not a hosted or source-controlled publication surface.
Treat it as current only for the branch and checkout where
`make api-docs-freshness` has just passed.

For durable links, use this page and checked-in public headers rather than
`docs/api/html/`. If you need the generated view, regenerate it locally with
`make api-docs-freshness`; do not use CI artifacts, release assets, or hosted
pages as substitutes for the source-controlled API route.

The routing guard keeps user-facing links on the source-controlled API path:
this page, checked-in public headers, the Doxygen configuration, workflow
guides, and the INSTALL support/readiness matrix. It rejects links that would
send readers to generated HTML under `docs/api/` or to an unsupported hosted
API publication.

Sprint 179 made the original local-only product decision, Sprint 186 retained
it during closeout, and Sprint 204 strengthened the local checks. Sprint 213
is the latest publication-policy decision: this generated tree stays local-only
rather than hosted, artifact-published, or committed. Use this page and the
public headers above as the source-controlled API reference path.

Passing generated API checks is evidence for the configured local Doxygen
input/output contract and staging guard only; it is not evidence for hosted
HTML, retained CI artifacts, package-manager distribution, dynamic ABI
compatibility, broad Windows parity, or completeness beyond checked-in public
headers selected by `Doxyfile`.

The current Doxygen configuration reads checked-in headers under `include/`.
Generated install headers such as `sparse_version.h` are owned by install
artifacts, `VERSION`, and install-validation tests rather than by a generated
Doxygen page. If local generated HTML is missing, stale, or incomplete, prefer
the public headers above for exact declarations until `make api-docs-freshness`
passes.

## Workflow Guides

Use the higher-level guides before dropping into declarations:

- [README.md](../README.md) for the short project front door;
- [tutorial.md](tutorial.md) for the fuller learning path;
- [cookbook.md](cookbook.md) for CSR, CSC, and Matrix Market first-use routes;
- [solver_selection.md](solver_selection.md) for choosing a solver family;
- [INSTALL.md](../INSTALL.md) for installed static-first downstream consumers;
- [maintainer_guide.md](maintainer_guide.md) for generated-reference
  current-output diagnostics, evidence, package, ABI, and support-tier
  interpretation.

## Claim Boundaries

This API reference index does not imply dynamic ABI compatibility,
shared-library support, package-manager distribution, broad platform parity,
external-library parity, portable performance, or state-of-the-art coverage.
Those boundaries remain owned by the install, benchmark, solver-selection, and
maintainer documentation.
