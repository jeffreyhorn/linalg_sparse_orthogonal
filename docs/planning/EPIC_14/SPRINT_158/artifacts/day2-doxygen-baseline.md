# Day 2 Doxygen Baseline

## Scope

Day 2 runs the current generated API documentation command and records the
baseline before any publication decision, warning fix, page-coverage guard, or
generated-output tracking change.

Generated HTML remains ignored local context during Day 2. This artifact does
not promote `docs/api/html/` as source-controlled, hosted, complete, or fresh
public evidence.

## Command Baseline

| Field | Value |
| --- | --- |
| Command | `make docs 2>&1` |
| Exit status | `0` |
| Make target behavior | Prints `Generating API documentation with Doxygen...`, runs `doxygen Doxyfile`, and reports `Documentation generated in docs/api/html/`. |
| Doxygen binary | `/usr/local/bin/doxygen` |
| Doxygen version | `1.16.1` |
| Output directory | `docs/api/html/` |
| Tracking state | Ignored by `.gitignore` as `docs/api/` |

## Warning Summary

`make docs` completed successfully but emitted 10 warnings.

| Warning family | Count | Locations | Day 2 interpretation |
| --- | ---: | --- | --- |
| Unknown Doxygen command `\U` | 5 | `include/sparse_lu_csr.h`: 105, 152, 268, 286, 288 | Likely documentation markup escaping issue; requires Day 4 triage before publication. |
| Undocumented macro definitions | 3 | `include/sparse_types.h`: 45, 46, 47 | Public type/format macro docs are incomplete; requires Day 4 triage. |
| Undocumented typedef | 1 | `include/sparse_types.h`: 44 | `idx_t` needs documentation or explicit exclusion. |
| Undocumented struct member | 1 | `include/sparse_iterative.h`: 128 | `sparse_gmres_opts_t::progress_user` needs documentation or explicit exclusion. |

### Raw Warning Lines

```text
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_lu_csr.h:152: warning: Found unknown command '\U'
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_lu_csr.h:286: warning: Found unknown command '\U'
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_lu_csr.h:105: warning: Found unknown command '\U'
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_lu_csr.h:268: warning: Found unknown command '\U'
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_lu_csr.h:288: warning: Found unknown command '\U'
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_types.h:45: warning: Member IDX_MAX (macro definition) of file sparse_types.h is not documented.
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_types.h:46: warning: Member SPARSE_PRIDX (macro definition) of file sparse_types.h is not documented.
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_types.h:47: warning: Member SPARSE_SCNIDX (macro definition) of file sparse_types.h is not documented.
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_types.h:44: warning: Member idx_t (typedef) of file sparse_types.h is not documented.
/Users/jeff/experiments/linalg_sparse_orthogonal/include/sparse_iterative.h:128: warning: Member progress_user (variable) of struct sparse_gmres_opts_t is not documented.
```

The raw warning lines include the local absolute source path because Doxygen
prints diagnostic locations that way. The generated HTML itself was separately
checked for local absolute paths.

## Generated Output Inventory

| Output metric | Value |
| --- | ---: |
| Total files under `docs/api/html/` | 212 |
| Directories under `docs/api/html/` | 2 |
| Top-level `.html` files | 87 |
| Header reference pages matching `*_8h.html` | 18 |
| Header source pages matching `*_8h_source.html` | 18 |

Core index/navigation pages present:

- `index.html`
- `files.html`
- `annotated.html`
- `globals.html`

## Header Reference Pages

Generated header reference pages currently exist for all 18 checked-in public
headers configured by `Doxyfile`:

| Checked-in public header | Generated reference page |
| --- | --- |
| `include/sparse_analysis.h` | `sparse__analysis_8h.html` |
| `include/sparse_bidiag.h` | `sparse__bidiag_8h.html` |
| `include/sparse_cholesky.h` | `sparse__cholesky_8h.html` |
| `include/sparse_csr.h` | `sparse__csr_8h.html` |
| `include/sparse_dense.h` | `sparse__dense_8h.html` |
| `include/sparse_eigs.h` | `sparse__eigs_8h.html` |
| `include/sparse_ic.h` | `sparse__ic_8h.html` |
| `include/sparse_ilu.h` | `sparse__ilu_8h.html` |
| `include/sparse_iterative.h` | `sparse__iterative_8h.html` |
| `include/sparse_ldlt.h` | `sparse__ldlt_8h.html` |
| `include/sparse_lu.h` | `sparse__lu_8h.html` |
| `include/sparse_lu_csr.h` | `sparse__lu__csr_8h.html` |
| `include/sparse_matrix.h` | `sparse__matrix_8h.html` |
| `include/sparse_qr.h` | `sparse__qr_8h.html` |
| `include/sparse_reorder.h` | `sparse__reorder_8h.html` |
| `include/sparse_svd.h` | `sparse__svd_8h.html` |
| `include/sparse_types.h` | `sparse__types_8h.html` |
| `include/sparse_vector.h` | `sparse__vector_8h.html` |

No generated page matching `sparse_version` exists under the current
configuration. Day 3 must decide whether coverage is limited to checked-in
`include/*.h` inputs or whether generated installed-header behavior from
`include/sparse_version.h.in` needs a separate policy row.

## Non-Portable Metadata Check

The generated HTML was scanned for local absolute path fragments:

```text
rg -n "/Users/jeff|/Users/|/private/|/tmp/|/var/folders" docs/api/html || true
```

No matches were found.

The generated HTML was also scanned for timestamp-like strings:

```text
rg -n "[0-9]{4}-[0-9]{2}-[0-9]{2}|Generated on|timestamp|Timestamp" docs/api/html || true
```

Matches were limited to vendored JavaScript comments in `jquery.js`:

- `jQuery UI - v1.13.2 - 2022-08-01`
- `PowerTip v1.3.1 (2018-04-15)`

No per-run generated timestamp was identified in this focused scan.

## Tracking And Publication State

| Surface | Day 2 state |
| --- | --- |
| `docs/api/` | Ignored local generated-output directory. |
| `docs/api/html/` | Reproducible local Doxygen output from `make docs`, not source-controlled pass evidence. |
| `docs/api/html/index.html` | Present locally, ignored. |
| Publication decision | Not made on Day 2. |
| Freshness claim | Not made on Day 2. |

## Day 3 Handoff

Day 3 should build the coverage map from this baseline:

1. Use the 18 checked-in public headers as the initial expected source set.
2. Preserve separate treatment for generated `sparse_version.h` behavior.
3. Confirm whether each generated `*_8h.html` page should count as sufficient
   page coverage or whether source pages and symbols are also required.
4. Feed the 10 warnings into Day 4 warning triage.
5. Keep generated HTML local-only until Day 5/6 publication decision closes.

## Completion Check

- `make docs` ran successfully and generated `docs/api/html/`.
- Warning output was captured before any fix or publication decision.
- Generated output inventory and tracking state are recorded.
- No generated HTML is promoted as source-controlled, hosted, complete, or
  fresh public evidence.
