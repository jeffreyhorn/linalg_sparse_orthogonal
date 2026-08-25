# Sprint 179 Day 3: Warning And Coverage Audit

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Determine whether generated API HTML has concrete warning or page-coverage
blockers before the sprint selects hosted publication, retained CI artifact,
committed output, or stronger local-only status. Day 3 uses command evidence
from the current generated-doc path and separates publish blockers from
policy/polish risks.

## Commands Run

```bash
make docs-check
make api-docs-freshness
```

`make docs-check` runs `make docs` and then
`scripts/check_api_docs_coverage.py`. `make api-docs-freshness` repeats the
docs-check path and adds `scripts/check_api_docs_local_only.sh`.

## Warning Inventory

The captured `make docs-check` output was:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

Day 3 found no emitted Doxygen warning lines in the captured docs-check output.

| Warning class | Day 3 result | Interpretation |
| --- | --- | --- |
| Undocumented public header/member warning | None observed in captured output | No immediate warning blocker found for configured inputs. |
| Documentation syntax warning | None observed in captured output | No immediate Doxygen syntax blocker found. |
| Parameter documentation warning | None observed in captured output | No immediate parameter-doc warning blocker found. |
| Fatal warning policy | Not enabled by `Doxyfile` | `WARN_AS_ERROR = NO`; warnings are visible but not fail-closed by Doxygen itself. |

## Page Coverage Matrix

`scripts/check_api_docs_coverage.py` passed with:

- checked-in public headers: 18;
- generated reference pages: 18;
- generated source pages: 18;
- generated `sparse_version.h`: separate installed-header policy row, not an
  expected Doxygen page.

| Header | Major API family | Reference page | Source page | Day 3 status |
| --- | --- | --- | --- | --- |
| `include/sparse_analysis.h` | Analyze-once/factor-many direct solver lifecycle | `sparse__analysis_8h.html` | `sparse__analysis_8h_source.html` | Covered |
| `include/sparse_bidiag.h` | Bidiagonalization | `sparse__bidiag_8h.html` | `sparse__bidiag_8h_source.html` | Covered |
| `include/sparse_cholesky.h` | Cholesky | `sparse__cholesky_8h.html` | `sparse__cholesky_8h_source.html` | Covered |
| `include/sparse_csr.h` | CSR helpers | `sparse__csr_8h.html` | `sparse__csr_8h_source.html` | Covered |
| `include/sparse_dense.h` | Dense helpers | `sparse__dense_8h.html` | `sparse__dense_8h_source.html` | Covered |
| `include/sparse_eigs.h` | Symmetric eigensolvers | `sparse__eigs_8h.html` | `sparse__eigs_8h_source.html` | Covered |
| `include/sparse_ic.h` | IC preconditioner | `sparse__ic_8h.html` | `sparse__ic_8h_source.html` | Covered |
| `include/sparse_ilu.h` | ILU preconditioner | `sparse__ilu_8h.html` | `sparse__ilu_8h_source.html` | Covered |
| `include/sparse_iterative.h` | Iterative solvers | `sparse__iterative_8h.html` | `sparse__iterative_8h_source.html` | Covered |
| `include/sparse_ldlt.h` | LDL^T | `sparse__ldlt_8h.html` | `sparse__ldlt_8h_source.html` | Covered |
| `include/sparse_lu.h` | Linked-list LU | `sparse__lu_8h.html` | `sparse__lu_8h_source.html` | Covered |
| `include/sparse_lu_csr.h` | CSR LU | `sparse__lu__csr_8h.html` | `sparse__lu__csr_8h_source.html` | Covered |
| `include/sparse_matrix.h` | Matrix construction and mutation | `sparse__matrix_8h.html` | `sparse__matrix_8h_source.html` | Covered |
| `include/sparse_qr.h` | QR and least-squares | `sparse__qr_8h.html` | `sparse__qr_8h_source.html` | Covered |
| `include/sparse_reorder.h` | Reordering | `sparse__reorder_8h.html` | `sparse__reorder_8h_source.html` | Covered |
| `include/sparse_svd.h` | SVD and partial SVD | `sparse__svd_8h.html` | `sparse__svd_8h_source.html` | Covered |
| `include/sparse_types.h` | Shared public types | `sparse__types_8h.html` | `sparse__types_8h_source.html` | Covered |
| `include/sparse_vector.h` | Vector helpers | `sparse__vector_8h.html` | `sparse__vector_8h_source.html` | Covered |

## Adoption And Supplemental Page Coverage

These user-facing surfaces are important for adoption, but they are not
configured Doxygen inputs:

| Surface | Generated HTML coverage | Day 3 interpretation |
| --- | --- | --- |
| `README.md` | Not included | README remains the project front door outside generated API HTML. |
| `INSTALL.md` | Not included | Install and package guidance is not part of generated API HTML. |
| `docs/tutorial.md` | Not included | Tutorial flow is not published by Doxygen. |
| `docs/cookbook.md` | Not included | Cookbook workflows are not published by Doxygen. |
| `docs/solver_selection.md` | Not included | Solver-selection guidance is not published by Doxygen. |
| `docs/maintainer_guide.md` | Not included | Maintainer policy remains source-controlled Markdown. |
| `examples/*.c` | Not included | Example source files do not produce Doxygen example pages under current configuration. |

This is not a configured-input coverage failure, but it matters for a hosted
publication decision: generated API HTML alone would publish declaration-level
reference pages, not the project adoption path.

## Missing, Stale, And Orphan Findings

| Finding type | Day 3 result | Notes |
| --- | --- | --- |
| Missing configured header reference pages | None found | Coverage check passed for 18 of 18 checked-in public headers. |
| Missing configured header source pages | None found | Coverage check passed for 18 of 18 checked-in public headers. |
| Missing generated index | None found | Coverage check requires `docs/api/html/index.html`. |
| Stale local output | Not found after `make api-docs-freshness` | Freshness is checkout-local; generated files remain ignored. |
| Unexpected tracked generated API files | None found | Local-only guard passed. |
| Unexpected staged generated API files | None found | Local-only guard passed. |
| Non-ignored generated API files | None found | Local-only guard passed. |
| Orphan generated pages | Not proven absent | Day 3 did not add a graph-level orphan detector; Doxygen navigation/index assets are generated and ignored. |

## Publish Blockers Versus Polish Risks

| Category | Item | Day 3 classification |
| --- | --- | --- |
| Blocker | Missing configured header pages | Not present. |
| Blocker | Doxygen warnings in captured output | Not present. |
| Blocker | Tracked or staged generated API HTML under current local-only policy | Not present. |
| Blocker if hosting is selected | No hosted publication metadata exists yet | Must be designed before any hosted claim. |
| Blocker if committing output is selected | `docs/api/` is currently ignored | Must be explicitly changed if committed generated output is selected. |
| Polish/policy risk | `WARN_AS_ERROR = NO` | Warnings are not fail-closed by Doxygen configuration. |
| Polish/policy risk | Examples and Markdown guides are not Doxygen inputs | Generated HTML is declaration reference only, not a complete adoption site. |
| Polish/policy risk | No orphan-page detector | Current checks prove required pages exist, not that every generated page is navigationally meaningful. |

## Claim-Risk Notes

Current generated API evidence supports these claims:

- local Doxygen HTML can be regenerated for configured checked-in public
  headers;
- generated reference and source pages exist for all 18 configured public
  headers after `make docs-check`;
- local-only staging policy is currently enforced by `make api-docs-freshness`.

Current evidence does not support these claims:

- hosted generated API publication;
- retained CI artifact publication;
- committed generated HTML output;
- generated HTML as release evidence;
- generated HTML completeness beyond the configured `include/*.h` input set;
- generated documentation for examples, tutorials, cookbook guidance,
  solver-selection guidance, install guidance, or maintainer policy;
- fail-closed Doxygen warning behavior through `WARN_AS_ERROR`.

## Day 3 Deliverables

- warning inventory
- page coverage matrix
- missing and stale page findings
- generated API claim-risk notes
- Day 3 warning and page-coverage artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Generated API readiness is supported by concrete warning and coverage data. | Complete | `make docs-check` and `make api-docs-freshness` passed with no warning lines observed and 18/18 header page coverage. |
| Publish blockers are separated from polish issues. | Complete | Blocker/polish table above separates current clean checks from decision-dependent hosting/committed-output work. |
| No publication decision is made without documented coverage evidence. | Complete | Day 3 records evidence but keeps all product-status options open for Day 5/Day 6. |
