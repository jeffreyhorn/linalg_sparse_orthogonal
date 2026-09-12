# Day 2: Current Doxygen Baseline

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Reproduce the current local generated API output and freshness
checks without changing policy.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`

## Purpose

Day 2 records the current generated API HTML baseline before Sprint 204 makes
any publication or local-only policy decision. The goal is to prove the
existing Doxygen and API freshness path is reproducible locally and that the
generated output remains ignored local state.

## Environment Baseline

| Check | Result |
| --- | --- |
| `doxygen --version` | `1.16.1` |
| Generated output path | `docs/api/html/` |
| Doxygen output root | `docs/api/` |
| Current source-controlled API path | `docs/api_reference.md` plus checked-in public headers under `include/` |
| Current generated HTML support tier | Local-only generated output |

## Ignore And Tracking State

`git check-ignore -v docs/api docs/api/html docs/api/html/index.html` reported
all three paths as ignored by `.gitignore:44:docs/api/`.

After the Doxygen runs, `git status --short --ignored docs/api
docs/planning/EPIC_18/SPRINT_204` reported:

```text
?? docs/planning/EPIC_18/SPRINT_204/
!! docs/api/
```

The generated tree exists locally after validation, but it remains ignored.
There were no tracked, staged, or visible non-ignored untracked `docs/api`
files.

## Validation Commands

| Command | Result | Evidence |
| --- | --- | --- |
| `make docs-check` | Passed | Doxygen generated `docs/api/html/`; `api-docs-coverage` passed. |
| `make api-docs-freshness` | Passed | Doxygen and coverage passed; `api-docs-local-only` passed every ignore, tracking, Doxyfile, wording, and workflow check. |

## `make docs-check` Result

`make docs-check` completed successfully with this coverage summary:

| Metric | Count or disposition |
| --- | ---: |
| Checked-in public headers | 18 |
| Generated reference pages | 18 |
| Generated source pages | 18 |
| Generated `sparse_version.h` | Separate installed-header policy row; not an expected page |

No Doxygen warning lines were observed in the command output.

## `make api-docs-freshness` Result

`make api-docs-freshness` completed successfully and ran the same Doxygen and
coverage path plus the local-only guard. The guard confirmed:

- `docs/api`, `docs/api/html`, and `docs/api/html/index.html` are ignored;
- `Doxyfile` keeps the expected local-only contract for `INPUT`,
  `FILE_PATTERNS`, `RECURSIVE`, `OUTPUT_DIRECTORY`, `GENERATE_HTML`, and
  `HTML_OUTPUT`;
- no generated API files are tracked;
- no generated API files are staged;
- no generated API files are visible as non-ignored untracked files;
- README, `docs/api_reference.md`, and `docs/maintainer_guide.md` retain the
  current local-only generated API wording;
- workflows do not reference generated API output paths.

## Generated Output Inventory

After validation, the generated output directory contained:

| Inventory item | Count |
| --- | ---: |
| Files under `docs/api/html/` at max depth 1 | 156 |
| HTML files under `docs/api/html/` at max depth 1 | 88 |

The first observed HTML entry points included:

- `docs/api/html/index.html`
- `docs/api/html/files.html`
- `docs/api/html/globals.html`
- `docs/api/html/sparse__analysis_8h.html`
- `docs/api/html/sparse__analysis_8h_source.html`
- `docs/api/html/sparse__bidiag_8h.html`
- `docs/api/html/sparse__bidiag_8h_source.html`
- `docs/api/html/sparse__cholesky_8h.html`
- `docs/api/html/sparse__cholesky_8h_source.html`
- `docs/api/html/sparse__csr_8h.html`

## Interpretation

The current baseline supports the existing policy: generated API HTML is a
reproducible local convenience view for the configured checked-in public-header
input set, current only after `make api-docs-freshness`, and not a hosted,
retained-artifact, committed-output, release, package, ABI, platform-parity, or
state-of-the-art evidence surface.

This baseline does not decide Sprint 204 item 204.1. Hosted publication,
retained artifacts, committed generated output, and stronger local-only policy
remain open for Day 3-Day 5 evaluation.

## Completion Criteria Review

| Day 2 criterion | Status |
| --- | --- |
| Existing local generated API policy is backed by current command evidence or an explicit environment blocker. | Complete; `make docs-check` and `make api-docs-freshness` passed. |
| Any generated output visible to git is classified before decision work. | Complete; `docs/api/` is ignored only, with no tracked, staged, or visible non-ignored untracked files. |
| No publication path is inferred from local generated output alone. | Complete; this artifact records local-only generated output and leaves publication decisions open. |

## Changed Surfaces

- Added this Day 2 artifact.
- Updated `WORKING_NOTES.md` with the Day 2 validation and inventory record.

No source, public header, workflow, Makefile, Doxyfile, user-facing
documentation behavior, or generated-output policy changed on Day 2.
