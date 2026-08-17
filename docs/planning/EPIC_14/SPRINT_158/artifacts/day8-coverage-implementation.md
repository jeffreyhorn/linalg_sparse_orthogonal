# Day 8 Coverage Implementation

## Scope

Day 8 implements the generated API page-coverage guard designed on Day 7.

The implementation keeps generated API HTML local-only and ignored. It adds a
maintained command that checks local Doxygen output for expected public-header
pages, but it does not publish or commit `docs/api/html/`.

## Implemented Files

| File | Change |
| --- | --- |
| `scripts/check_api_docs_coverage.py` | New focused Python guard for generated API page coverage. |
| `Makefile` | Adds `api-docs-coverage` and `docs-check` targets near the existing `docs` target. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day7-page-coverage-check-design.md` | Corrects the documented Doxygen `.h` filename mapping from `_h` to `_8h`. |
| `docs/planning/EPIC_14/SPRINT_158/WORKING_NOTES.md` | Records Day 8 implementation and validation evidence. |

## Guard Behavior

`scripts/check_api_docs_coverage.py`:

- defaults to repository root based on the script location;
- inspects direct checked-in public headers under `include/*.h`;
- requires `docs/api/html/` to exist;
- requires `docs/api/html/index.html`;
- derives Doxygen page names from each checked-in header basename;
- requires both the generated reference page and generated source page for each
  checked-in public header;
- reports missing pages with both the source header and expected generated path;
- treats generated `sparse_version.h` as a separate installed-header policy
  row, not as an expected generated page;
- exits nonzero on missing generated output or missing pages.

The Doxygen filename mapping implemented by the script is:

| Header basename part | Doxygen output part |
| --- | --- |
| `_` | `__` |
| `.h` | `_8h` |
| reference page | `<escaped>_8h.html` |
| source page | `<escaped>_8h_source.html` |

Example:

| Header | Reference page | Source page |
| --- | --- | --- |
| `include/sparse_lu_csr.h` | `docs/api/html/sparse__lu__csr_8h.html` | `docs/api/html/sparse__lu__csr_8h_source.html` |

## Make Targets

| Target | Behavior |
| --- | --- |
| `make api-docs-coverage` | Runs `python3 scripts/check_api_docs_coverage.py` against the current local generated HTML tree. |
| `make docs-check` | Runs `make docs` and then `make api-docs-coverage`. |

`make docs` behavior is unchanged: it runs `doxygen Doxyfile` and writes
ignored local output under `docs/api/html/`.

## Validation Results

### Python Syntax Check

```text
python3 -m py_compile scripts/check_api_docs_coverage.py
```

Result: passed.

### Coverage Guard

```text
make api-docs-coverage
```

Result:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

### Combined Docs Check

```text
make docs-check
```

Result: passed the coverage guard after regenerating Doxygen output. Doxygen
still emitted the 10 known warnings from Day 4; those remain selected for Day 9
closure.

### Negative-Path Check

```text
python3 scripts/check_api_docs_coverage.py --html-dir /tmp/lso-missing-api-html 2>&1
```

Result:

```text
api-docs-coverage: FAIL: generated API HTML directory not found: /private/tmp/lso-missing-api-html; run `make docs` first
exit_status=1
```

An initial attempt to capture the negative-path exit status used the variable
name `status`, which is readonly in `zsh`; the validation was rerun with `rc`
and passed.

## Local-Only And Generated Version-Header Semantics

The guard validates ignored local generated output. It does not make
`docs/api/html/` source-controlled or hosted evidence.

The generated installed `sparse_version.h` remains outside the expected page
set because it is generated from `VERSION` and `include/sparse_version.h.in`
under build/install paths. Absence of a `sparse_version` generated page is not
a coverage failure under the Day 3 and Day 6 policy.

## Day 9 Handoff

Day 9 should fix or reclassify the selected warnings:

| Category | Owner | Expected implementation |
| --- | --- | --- |
| W158-01 | `include/sparse_lu_csr.h` | Escape or reword `L\U` prose so Doxygen no longer sees `\U`. |
| W158-02 | `include/sparse_types.h` | Add Doxygen comments for `idx_t`, `IDX_MAX`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX`. |
| W158-03 | `include/sparse_iterative.h` | Add member documentation for `sparse_gmres_opts_t::progress_user`. |

Because Day 9 will edit public headers, required validation should include:

- `make format && make lint && make test`;
- `make docs-check`;
- docs hygiene checks.

## Completion Check

- The guard can detect missing generated output and missing expected pages.
- The guard derives expectations from checked-in public headers.
- Generated `sparse_version.h` behavior matches the Day 3 policy.
- Generated HTML remains local-only and ignored.
