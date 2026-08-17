# Day 12 Local Validation

## Scope

Day 12 validates the Sprint 159 hosted report-freshness changes after the
Day 10 normalizer implementation and Day 11 documentation alignment.

Changed files are workflow, docs, Python script, and Python focused tests.
No `.c` or `.h` files are modified, so the required C/header quality gate
(`make format && make lint && make test`) is not required for Day 12.

## Validation Commands

| Command | Result | Notes |
| --- | --- | --- |
| `make report-index-oracle-freshness` | Pass | Regenerated selected oracle outputs, checked required oracle freshness, and reported current selected rows as `fresh`. |
| `make report-index-comparison-freshness` | Pass | Regenerated selected QR minimum-norm comparison outputs, checked required comparison freshness, and reported six selected rows as `fresh`. |
| `python3 tests/test_normalize_report_index.py` | Pass | Focused normalizer tests, including selected comparison pass/missing/stale/duplicate/unexpected/fail/defer coverage. |
| `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py` | Pass | Python syntax compilation for changed executable script/test files. |
| `make docs-check` | Pass | Doxygen generation and API docs coverage check passed. |
| `git diff --check -- .github/workflows/ci.yml README.md docs/maintainer_guide.md docs/solver_selection.md scripts/normalize_report_index.py tests/corpus/README.md tests/test_normalize_report_index.py docs/planning/EPIC_13/SPRINT_159` | Pass | Diff whitespace hygiene passed. |
| `rg -n "[ \t]+$" ...` | Pass | No trailing whitespace found in changed Sprint 159 files. |

## Freshness Output Summary

Oracle freshness:

- selected gate command: `make report-index-oracle-freshness`;
- normalized row count: `54`;
- selected generated rows report `fresh`;
- no strict unchecked-warning wording appears on passing selected rows;
- command exits `0`.

Comparison freshness:

- selected gate command: `make report-index-comparison-freshness`;
- normalized row count: `7`;
- six selected generated comparison rows report `fresh`;
- optional dependency context remains separate from selected pass evidence;
- command exits `0`.

Docs check:

- Doxygen generated `docs/api/html/`;
- API docs coverage passed;
- checked-in public headers: `18`;
- generated reference pages: `18`;
- generated source pages: `18`;
- `sparse_version.h` remains governed by the separate installed-header policy
  row and is not an expected generated reference page.

## Issue And Follow-Up List

No local validation failures were observed.

Day 13 should focus on hosted readiness rather than new implementation:

- review `.github/workflows/ci.yml` job naming, timeout, summaries, and split
  artifact paths against the Day 12 local evidence;
- verify macOS and Windows workflows remain out of scope for Sprint 159
  report-index parity claims;
- prepare reviewer-facing notes that explain the local-only row metadata versus
  reviewed Linux hosted execution distinction;
- keep Sprint 160 QR comparison follow-up scoped to one selected comparison
  family.

## Completion Check

- Promoted freshness commands pass locally.
- Focused normalizer/report-index tests pass locally.
- Documentation generation and API coverage checks pass locally.
- Required quality checks are selected by changed-file type.
- No remaining Day 12 failures require closeout triage.
