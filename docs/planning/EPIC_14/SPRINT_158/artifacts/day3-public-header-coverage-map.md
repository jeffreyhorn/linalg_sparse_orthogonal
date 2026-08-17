# Day 3 Public Header Coverage Map

## Scope

Day 3 defines the generated API documentation coverage source set and maps the
current Doxygen output to checked-in public headers. It also records how the
generated installed `sparse_version.h` header is produced and why it is not
counted as a missing generated Doxygen page under the current configuration.

This artifact does not change Doxygen configuration, public headers, generated
HTML publication policy, or package install behavior.

## Coverage Source Set Decision

| Decision field | Day 3 decision |
| --- | --- |
| Primary Doxygen source set | Checked-in public headers matching `include/*.h`. |
| Expected checked-in headers | 18. |
| Required page for Day 3 coverage | One generated `*_8h.html` reference page per checked-in public header. |
| Supplemental page check | One generated `*_8h_source.html` source page per checked-in public header. |
| Generated `sparse_version.h` treatment | Separate generated installed-header policy row, not a missing checked-in public header page. |
| Publication implication | Coverage success does not publish `docs/api/html/`; generated output remains ignored local context until Day 5/6 decision. |

The coverage source set is intentionally limited to checked-in `include/*.h`
files because that is the configured `Doxyfile` input:

| Doxygen setting | Current value |
| --- | --- |
| `INPUT` | `include/` |
| `FILE_PATTERNS` | `*.h` |
| `RECURSIVE` | `NO` |
| `OUTPUT_DIRECTORY` | `docs/api` |
| `HTML_OUTPUT` | `html` |

## Header-To-Page Coverage Matrix

| Checked-in public header | Reference page | Source page | Status |
| --- | --- | --- | --- |
| `include/sparse_analysis.h` | `sparse__analysis_8h.html` | `sparse__analysis_8h_source.html` | Covered |
| `include/sparse_bidiag.h` | `sparse__bidiag_8h.html` | `sparse__bidiag_8h_source.html` | Covered |
| `include/sparse_cholesky.h` | `sparse__cholesky_8h.html` | `sparse__cholesky_8h_source.html` | Covered |
| `include/sparse_csr.h` | `sparse__csr_8h.html` | `sparse__csr_8h_source.html` | Covered |
| `include/sparse_dense.h` | `sparse__dense_8h.html` | `sparse__dense_8h_source.html` | Covered |
| `include/sparse_eigs.h` | `sparse__eigs_8h.html` | `sparse__eigs_8h_source.html` | Covered |
| `include/sparse_ic.h` | `sparse__ic_8h.html` | `sparse__ic_8h_source.html` | Covered |
| `include/sparse_ilu.h` | `sparse__ilu_8h.html` | `sparse__ilu_8h_source.html` | Covered |
| `include/sparse_iterative.h` | `sparse__iterative_8h.html` | `sparse__iterative_8h_source.html` | Covered |
| `include/sparse_ldlt.h` | `sparse__ldlt_8h.html` | `sparse__ldlt_8h_source.html` | Covered |
| `include/sparse_lu.h` | `sparse__lu_8h.html` | `sparse__lu_8h_source.html` | Covered |
| `include/sparse_lu_csr.h` | `sparse__lu__csr_8h.html` | `sparse__lu__csr_8h_source.html` | Covered |
| `include/sparse_matrix.h` | `sparse__matrix_8h.html` | `sparse__matrix_8h_source.html` | Covered |
| `include/sparse_qr.h` | `sparse__qr_8h.html` | `sparse__qr_8h_source.html` | Covered |
| `include/sparse_reorder.h` | `sparse__reorder_8h.html` | `sparse__reorder_8h_source.html` | Covered |
| `include/sparse_svd.h` | `sparse__svd_8h.html` | `sparse__svd_8h_source.html` | Covered |
| `include/sparse_types.h` | `sparse__types_8h.html` | `sparse__types_8h_source.html` | Covered |
| `include/sparse_vector.h` | `sparse__vector_8h.html` | `sparse__vector_8h_source.html` | Covered |

## Coverage Summary

| Check | Result |
| --- | --- |
| Checked-in public headers under `include/*.h` | 18 |
| Generated header reference pages | 18 |
| Generated header source pages | 18 |
| Missing checked-in public header reference pages | 0 |
| Missing checked-in public header source pages | 0 |
| Generated `sparse_version` page | 0; expected under current policy |

Day 3 therefore finds no missing generated pages for the checked-in public
header source set. This does not mean every symbol is fully documented. The Day
2 warnings remain real and must be triaged on Day 4 before any publication or
freshness claim can be made.

## Generated Version Header Treatment

`sparse_version.h` is generated, installed, and consumed by package users, but
it is not checked in as `include/sparse_version.h` and is not a current Doxygen
input.

| Surface | Owner | Day 3 interpretation |
| --- | --- | --- |
| Template | `include/sparse_version.h.in` | Source template for generated installed version metadata. |
| Version source | `VERSION` | Single source for major/minor/patch/string substitutions. |
| Make generation | `Makefile` `generate-version` target and `$(GENERATED_VERSION)` rule | Produces `build/include/sparse_version.h` with `sed` substitutions. |
| Make install | `Makefile` `install` target | Installs generated `sparse_version.h` beside checked-in public headers. |
| CMake generation | `CMakeLists.txt` `configure_file(...)` | Produces `${CMAKE_CURRENT_BINARY_DIR}/include/sparse_version.h`. |
| CMake install | `CMakeLists.txt` install rules | Excludes `*.h.in` from copied checked-in headers and installs generated `sparse_version.h` separately. |
| Current Doxygen visibility | `sparse_types.h` includes `sparse_version.h`; no separate `sparse_version` page is generated. | Generated version metadata is referenced but not documented as an expected Doxygen page. |

Day 3 policy:

- the page-coverage check should require pages for the 18 checked-in
  `include/*.h` inputs;
- generated `sparse_version.h` should be represented by a separate policy row
  in publication artifacts and maintainer guidance;
- absence of a `sparse_version` generated page is not a Day 3 coverage failure;
- changing this policy later requires a deliberate Doxygen input decision and
  documentation alignment.

## Warning And Exclusion Notes

Day 3 does not classify warnings; it carries the Day 2 warning inventory to Day
4:

| Warning owner | Warning count | Day 3 status |
| --- | ---: | --- |
| `include/sparse_lu_csr.h` | 5 | Requires Day 4 triage for unknown `\U` markup. |
| `include/sparse_types.h` | 4 | Requires Day 4 triage for undocumented typedef/macro definitions. |
| `include/sparse_iterative.h` | 1 | Requires Day 4 triage for undocumented `progress_user` member. |

No checked-in public header is excluded from Day 3 page coverage. Any future
exclusion must name the header, reason, owner, and support-tier effect.

## Day 4 Handoff

Day 4 should:

1. Normalize the 10 warning lines into stable warning categories.
2. Decide fix/defer/exclude/blocker status for each warning family.
3. Identify whether selected fixes are comment-only public-header edits or
   declaration/code edits.
4. Require `make format && make lint && make test` for any public header edit,
   even comment-only, per Sprint 157 Day 10 policy.
5. Preserve the Day 3 coverage decision unless a deliberate Doxygen policy
   change is selected.

## Completion Check

- The coverage source set is explicit: checked-in `include/*.h` headers.
- All 18 checked-in public headers have generated reference and source pages.
- Generated `sparse_version.h` behavior is documented separately from checked-
  in public header coverage.
- No generated HTML publication or freshness claim is made by this coverage
  map.
