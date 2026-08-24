# Sprint 179 Day 2: Doxygen Surface Audit

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Inventory the configured generated API surface before warning triage,
publication decision, or guard changes. Day 2 accounts for Doxygen inputs,
generated outputs, ignored paths, staging policy, and source-header authority.

## Doxygen Configuration Inventory

| Area | Setting | Value | Day 2 interpretation |
| --- | --- | --- | --- |
| Project | `PROJECT_NAME` | `linalg_sparse_orthogonal` | Generated pages are branded for this project. |
| Project | `PROJECT_BRIEF` | `Sparse Linear Algebra Library with Orthogonal Linked-List Representation` | Brief text appears in generated output and may need claim review if publication is selected. |
| Input | `INPUT` | `include/` | Public headers are the only configured generated API inputs. |
| Input | `FILE_PATTERNS` | `*.h` | Only header files participate. |
| Input | `RECURSIVE` | `NO` | Nested include directories would not be scanned under the current policy. |
| Output | `OUTPUT_DIRECTORY` | `docs/api` | Generated output lands under the ignored `docs/api/` tree. |
| Output | `GENERATE_HTML` | `YES` | HTML generation is enabled locally. |
| Output | `GENERATE_LATEX` | `NO` | No LaTeX output is generated. |
| Output | `HTML_OUTPUT` | `html` | HTML files are written under `docs/api/html/`. |
| Extraction | `EXTRACT_ALL` | `NO` | Undocumented entities are not force-extracted as a broad completeness claim. |
| Extraction | `EXTRACT_STATIC` | `NO` | Static/internal implementation members are not part of the public generated surface. |
| Extraction | `EXTRACT_PRIVATE` | `NO` | Private details are not generated. |
| Warnings | `WARN_IF_UNDOCUMENTED` | `YES` | Missing documentation is visible during generation. |
| Warnings | `WARN_IF_DOC_ERROR` | `YES` | Documentation errors are visible during generation. |
| Warnings | `WARN_NO_PARAMDOC` | `YES` | Missing parameter docs are visible during generation. |
| Warnings | `WARN_AS_ERROR` | `NO` | Doxygen warnings do not currently fail generation by themselves. |
| Source browsing | `SOURCE_BROWSER` | `NO` | Source-browsing pages are not enabled beyond generated header source pages. |

## Configured Input Inventory

Doxygen currently scans only top-level checked-in public headers under
`include/`. Day 2 found 18 configured header inputs:

| Header | Primary public surface |
| --- | --- |
| `include/sparse_analysis.h` | Analyze-once/factor-many direct-solver lifecycle |
| `include/sparse_bidiag.h` | Bidiagonalization helpers |
| `include/sparse_cholesky.h` | Cholesky factorization and solve contracts |
| `include/sparse_csr.h` | CSR storage helpers |
| `include/sparse_dense.h` | Dense matrix helpers |
| `include/sparse_eigs.h` | Symmetric eigensolver APIs |
| `include/sparse_ic.h` | IC preconditioner APIs |
| `include/sparse_ilu.h` | ILU preconditioner APIs |
| `include/sparse_iterative.h` | Iterative solvers, matrix-free, and handle APIs |
| `include/sparse_ldlt.h` | LDL^T solver APIs and telemetry |
| `include/sparse_lu.h` | One-shot LU APIs |
| `include/sparse_lu_csr.h` | CSR LU APIs |
| `include/sparse_matrix.h` | Matrix construction, mutation, copy/free, norms, and dense conversion |
| `include/sparse_qr.h` | QR, least-squares, rank, nullspace, and minimum-norm APIs |
| `include/sparse_reorder.h` | Reordering APIs |
| `include/sparse_svd.h` | Full SVD, partial SVD, pseudoinverse, and low-rank APIs |
| `include/sparse_types.h` | Shared scalar, error, and compressed-format types |
| `include/sparse_vector.h` | Vector helpers |

Day 2 found no nested `include/**/*.h` files. If nested public headers are
introduced later, the current `RECURSIVE = NO` setting will exclude them until
the input policy changes.

## Non-Input Surfaces

The following repository surfaces do not currently contribute directly to
Doxygen HTML generation:

| Surface | Current Doxygen status | Implication |
| --- | --- | --- |
| `examples/` | Not configured as input | Examples can guide users elsewhere, but generated API HTML does not include example pages from source files. |
| `docs/tutorial.md` | Not configured as input | Tutorial navigation remains separate from generated API output. |
| `docs/cookbook.md` | Not configured as input | Cookbook workflows are not generated Doxygen pages. |
| `docs/solver_selection.md` | Not configured as input | Solver-selection guidance is not included in generated HTML. |
| `docs/maintainer_guide.md` | Not configured as input | Maintainer policy is source-controlled Markdown, not Doxygen output. |
| `docs/planning/**` | Not configured as input | Planning artifacts are outside the generated API product surface. |
| `include/sparse_version.h.in` | Not matched as checked-in `*.h` input | Generated `sparse_version.h` is governed by install/version policy, not Doxygen page coverage. |

## Generated Output Inventory

The local generated output tree currently exists under `docs/api/html/` and
contains 214 files in this checkout. The generated page inventory includes:

| Output category | Current files |
| --- | --- |
| Entry point | `docs/api/html/index.html` |
| Navigation and indexes | `annotated.html`, `classes.html`, `files.html`, `globals*.html`, `functions*.html`, `navtree*.js`, `search/**` |
| Styling and runtime assets | `doxygen.css`, `doxygen.svg`, `jquery.js`, `menu.js`, `dynsections.js`, `clipboard.js`, `cookie.js` |
| Header reference pages | 18 `*_8h.html` files |
| Header source pages | 18 `*_8h_source.html` files |

Current header page coverage:

| Header | Reference page | Source page |
| --- | --- | --- |
| `include/sparse_analysis.h` | `sparse__analysis_8h.html` | `sparse__analysis_8h_source.html` |
| `include/sparse_bidiag.h` | `sparse__bidiag_8h.html` | `sparse__bidiag_8h_source.html` |
| `include/sparse_cholesky.h` | `sparse__cholesky_8h.html` | `sparse__cholesky_8h_source.html` |
| `include/sparse_csr.h` | `sparse__csr_8h.html` | `sparse__csr_8h_source.html` |
| `include/sparse_dense.h` | `sparse__dense_8h.html` | `sparse__dense_8h_source.html` |
| `include/sparse_eigs.h` | `sparse__eigs_8h.html` | `sparse__eigs_8h_source.html` |
| `include/sparse_ic.h` | `sparse__ic_8h.html` | `sparse__ic_8h_source.html` |
| `include/sparse_ilu.h` | `sparse__ilu_8h.html` | `sparse__ilu_8h_source.html` |
| `include/sparse_iterative.h` | `sparse__iterative_8h.html` | `sparse__iterative_8h_source.html` |
| `include/sparse_ldlt.h` | `sparse__ldlt_8h.html` | `sparse__ldlt_8h_source.html` |
| `include/sparse_lu.h` | `sparse__lu_8h.html` | `sparse__lu_8h_source.html` |
| `include/sparse_lu_csr.h` | `sparse__lu__csr_8h.html` | `sparse__lu__csr_8h_source.html` |
| `include/sparse_matrix.h` | `sparse__matrix_8h.html` | `sparse__matrix_8h_source.html` |
| `include/sparse_qr.h` | `sparse__qr_8h.html` | `sparse__qr_8h_source.html` |
| `include/sparse_reorder.h` | `sparse__reorder_8h.html` | `sparse__reorder_8h_source.html` |
| `include/sparse_svd.h` | `sparse__svd_8h.html` | `sparse__svd_8h_source.html` |
| `include/sparse_types.h` | `sparse__types_8h.html` | `sparse__types_8h_source.html` |
| `include/sparse_vector.h` | `sparse__vector_8h.html` | `sparse__vector_8h_source.html` |

## Repository Policy For Generated Output

| Path | Current policy | Evidence |
| --- | --- | --- |
| `docs/api/` | Ignored local generated output | `.gitignore:40:docs/api/` |
| `docs/api/html/` | Ignored local generated output | `.gitignore:40:docs/api/` |
| `docs/api/html/index.html` | Ignored local generated output | `.gitignore:40:docs/api/` |

Day 2 command evidence:

| Check | Result |
| --- | --- |
| `git ls-files docs/api` | Empty output; no tracked generated API files. |
| `git check-ignore -v docs/api docs/api/html docs/api/html/index.html` | All paths ignored by `.gitignore:40:docs/api/`. |
| `bash scripts/check_api_docs_local_only.sh` | Passed; generated API tree is ignored, untracked, unstaged, and not visible as non-ignored untracked output. |
| `python3 scripts/check_api_docs_coverage.py` | Passed; 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |

## Source-Header Authority Rules

Day 2 records these source-authority rules for later publication decision work:

1. Exact public declarations and call-site contracts are owned by checked-in
   public headers under `include/`.
2. `docs/api_reference.md` is the compact user-facing API reference entry
   point and routes exact declaration questions back to `include/`.
3. Generated HTML under `docs/api/html/` is derived output, not an editing
   surface.
4. Generated HTML freshness is only valid for the checkout where
   `make api-docs-freshness` has just passed.
5. Generated install headers such as `sparse_version.h` are owned by install
   artifacts, `VERSION`, and install-validation checks rather than Doxygen page
   coverage.

## Drift Risks Found On Day 2

| Risk | Why it matters for Sprint 179 |
| --- | --- |
| Doxygen warnings are not fatal by configuration. | Day 3 must determine whether warnings are clean enough or need a fail-closed policy. |
| Generated HTML is ignored and local-only. | Publication or committed-output decisions must explicitly change guard behavior rather than relying on the existing tree. |
| Generated output exists locally but is not tracked. | Stale local output can be inspected accidentally unless commands are rerun before citing it. |
| Examples and Markdown guides are not Doxygen inputs. | Hosted generated HTML alone would not publish tutorial/cookbook/solver-selection guidance. |
| Input scope is non-recursive. | Future nested public headers would be missed unless input policy and coverage checks change. |

## Day 2 Deliverables

- Doxygen input inventory
- generated output inventory
- ignored-path and staging-policy notes
- source-header authority notes
- Day 2 Doxygen surface audit artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every configured Doxygen input path is accounted for. | Complete | `include/` is the only configured input path; all 18 top-level public headers are listed. |
| Generated output paths are tied to current repository policy. | Complete | `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` are ignored local generated output. |
| Source authority for API text is explicit. | Complete | Public headers own exact declarations; generated HTML is derived output. |
