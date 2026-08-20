# Day 2: Generator And Output Inventory

## Purpose

Inventory generated API HTML inputs, commands, outputs, ignore rules, and
freshness assumptions before the Sprint 173 publication decision.

## Generator Command Inventory

| Surface | Current state | Notes |
| --- | --- | --- |
| Generator configuration | `Doxyfile` | Root-level Doxygen configuration. |
| Generator command | `make docs` | Runs `doxygen Doxyfile`. |
| Coverage command | `make api-docs-coverage` | Runs `python3 scripts/check_api_docs_coverage.py`. |
| Combined local check | `make docs-check` | Runs `docs` and then `api-docs-coverage`. |
| User-facing API entry point | `docs/api_reference.md` | Routes exact declarations back to public headers and local Doxygen HTML policy. |
| Maintainer policy | `docs/maintainer_guide.md` | States `docs/api/html/` is local-only, ignored, and not hosted/source-controlled evidence. |
| README command surface | `README.md` | Lists `make docs`, `make docs-check`, and `docs/api_reference.md`. |
| CI workflow surface | `.github/*` | No Day 2 references found for `docs-check`, Doxygen, `api-docs-coverage`, or `docs/api/html/`. |

## Doxygen Configuration Snapshot

`Doxyfile` currently defines:

| Setting | Value | Day 2 implication |
| --- | --- | --- |
| `INPUT` | `include/` | Checked-in public headers are the input root. |
| `FILE_PATTERNS` | `*.h` | Header files are the selected generated-doc input class. |
| `RECURSIVE` | `NO` | Only top-level headers under `include/` are selected. |
| `OUTPUT_DIRECTORY` | `docs/api` | Generated output is under the ignored docs API tree. |
| `GENERATE_HTML` | `YES` | HTML is generated. |
| `HTML_OUTPUT` | `html` | HTML output path is `docs/api/html/`. |
| `WARN_IF_UNDOCUMENTED` | `YES` | Undocumented public surfaces produce warnings. |
| `WARN_IF_DOC_ERROR` | `YES` | Documentation errors produce warnings. |
| `WARN_NO_PARAMDOC` | `YES` | Missing parameter docs produce warnings. |
| `WARN_AS_ERROR` | `NO` | Warnings are not currently fatal. |

## Tracked Input Inventory

The current checked-in Doxygen page-coverage input set is:

| Class | Files | Tracking status |
| --- | --- | --- |
| Doxygen config | `Doxyfile` | tracked |
| Make targets | `Makefile` | tracked |
| Coverage script | `scripts/check_api_docs_coverage.py` | tracked |
| API docs entry | `docs/api_reference.md` | tracked |
| Maintainer policy | `docs/maintainer_guide.md` | tracked |
| README command surface | `README.md` | tracked |
| Ignore policy | `.gitignore` | tracked |
| Public headers | `include/*.h` | 18 checked-in top-level headers |
| Installed version template | `include/sparse_version.h.in` | tracked, but outside current Doxygen page expectations |

The checked-in public headers are:

- `include/sparse_analysis.h`
- `include/sparse_bidiag.h`
- `include/sparse_cholesky.h`
- `include/sparse_csr.h`
- `include/sparse_dense.h`
- `include/sparse_eigs.h`
- `include/sparse_ic.h`
- `include/sparse_ilu.h`
- `include/sparse_iterative.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu.h`
- `include/sparse_lu_csr.h`
- `include/sparse_matrix.h`
- `include/sparse_qr.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_types.h`
- `include/sparse_vector.h`

## Generated Output Inventory

| Path | Day 2 state | Tracking/claim status |
| --- | --- | --- |
| `docs/api/` | ignored by `.gitignore` | local generated output root |
| `docs/api/html/` | present locally | local-only generated HTML |
| `docs/api/html/index.html` | present locally | ignored, not source-controlled |
| `docs/api/html/*_8h.html` | 18 files present | reference pages for checked-in headers |
| `docs/api/html/*_8h_source.html` | 18 files present | source pages for checked-in headers |
| `docs/api/html/search/` | present locally | ignored Doxygen search assets |
| `include/sparse_version.h` | ignored by `.gitignore` | generated installed header, not a Doxygen input page |

At Day 2 inspection time, `docs/api/html/` contained 214 local generated files.
This is useful local evidence only; it is not source-controlled, hosted, or
release evidence.

## Ignore And Staging Classification

| Surface | Classification | Action before Day 4 decision |
| --- | --- | --- |
| `Doxyfile` | tracked source input | Audit as freshness input. |
| `Makefile` docs targets | tracked source input | Audit as command/freshness input. |
| `scripts/check_api_docs_coverage.py` | tracked check input | Audit as freshness/check input. |
| `docs/api_reference.md` | tracked navigation input | Audit as documentation/freshness context. |
| `docs/maintainer_guide.md` | tracked policy input | Audit as claim-boundary input. |
| `README.md` | tracked navigation input | Audit as user-facing command input. |
| `include/*.h` | tracked generated-doc content input | Audit as primary freshness input. |
| `include/sparse_version.h.in` | tracked install-header template | Keep outside current Doxygen page expectations unless policy changes. |
| `docs/api/` | ignored generated output | Do not stage unless committed HTML is selected. |
| `docs/api/html/` | ignored generated output | Current only after local `make docs-check` passes. |

## Coverage Check Result

Day 2 ran:

```bash
python3 scripts/check_api_docs_coverage.py
```

Result:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

This proves only that expected pages exist in the current local generated tree.
It does not prove the generated tree is fresh relative to the tracked inputs.

## Freshness Map

| Input change | Should affect generated API freshness? | Current mechanical proof |
| --- | --- | --- |
| `include/*.h` comment/declaration changes | Yes | `make docs-check` regenerates; page coverage checks existence, not content freshness. |
| Adding/removing checked-in top-level `include/*.h` | Yes | Coverage script detects missing generated pages after generation. |
| `Doxyfile` input/output/warning changes | Yes | No dedicated freshness metadata; requires rerunning `make docs-check`. |
| `Makefile` docs target changes | Yes | No dedicated freshness metadata; requires command review and rerun. |
| `scripts/check_api_docs_coverage.py` changes | Yes | No dedicated freshness metadata; requires rerun. |
| `docs/api_reference.md` navigation/policy changes | Yes for user-facing interpretation | No generated HTML freshness proof; docs review required. |
| `docs/maintainer_guide.md` generated-doc policy changes | Yes for claim boundaries | Claim scans/deferral guards required if package/ABI/platform wording changes. |
| `README.md` docs command/navigation changes | Yes for user-facing discovery | Docs review required. |
| `include/sparse_version.h.in` changes | No under current Doxygen page policy | Owned by install/version validation, not Doxygen coverage. |

## Gaps And Ambiguities

1. **No source-to-output freshness metadata exists for generated API HTML.**
   The current check validates page coverage, not whether generated files match
   the current `include/*.h`, `Doxyfile`, Makefile target, coverage script, or
   docs navigation.

2. **Doxygen warnings are not fatal.** `WARN_AS_ERROR = NO`, so generated docs
   can complete with warnings unless a later gate captures and interprets them.

3. **No hosted or CI artifact lane is wired for generated API HTML.** Day 2
   found no `.github` workflow references to `docs-check` or Doxygen.

4. **Generated output is present locally but ignored.** This is consistent with
   the inherited local-only policy, but any later committed-output decision
   needs explicit ignore/staging changes.

5. **`sparse_version.h` remains an installed-header exception.** The coverage
   script intentionally excludes generated installed headers from current
   Doxygen expectations.

6. **Documentation navigation currently points to `docs/api_reference.md`, not
   generated HTML.** This is coherent with local-only generated HTML, but may
   need update if Day 4 selects hosted, committed, or artifact-only
   publication.

## Day 2 Completion Check

Day 2 completion criteria are met:

- generation inputs and outputs are visible;
- freshness behavior is understood before a publication decision;
- generated output remains ignored and was not staged unintentionally.

No `.c` or `.h` files changed on Day 2, so the full C quality gate is not
required for this day.
