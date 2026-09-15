# Sprint 204 Day 7: Policy Implementation Batch

## Purpose

Day 7 implements the selected stronger local-only generated API policy from
Day 5 and the tracking design from Day 6. The implementation deliberately
avoids hosted docs, retained artifacts, Pages deployment, committed generated
HTML, and Doxygen input broadening.

## Implemented Changes

| Surface | Change | Reason |
| --- | --- | --- |
| `scripts/check_api_docs_local_only.sh` | Added deterministic top-level workflow YAML scanning and a publication-semantics check. | Prevent workflows from combining `docs/api` output paths with artifact upload, Pages, `gh-pages`, or publication behavior while generated API HTML is local-only. |
| `tests/test_api_docs_local_only_guard.py` | Added a standalone regression suite with current-tree, passing-fixture, workflow-path, workflow-publication, and wording-failure cases. | Converts the shell guard behavior into reviewable regression evidence. |
| `Makefile` | Added the Python regression suite to `api-docs-local-only`. | Ensures `make api-docs-freshness` exercises the strengthened local-only policy guard. |

## Guard Behavior

The strengthened guard still proves the existing local-only invariants:

- `docs/api`, `docs/api/html`, and `docs/api/html/index.html` are ignored;
- generated API files under `docs/api/` are not tracked, staged, or visible as
  non-ignored untracked files;
- `Doxyfile` keeps the generated API input limited to `include/*.h` with
  output under `docs/api/html/`;
- README, API reference, and maintainer-guide wording continue to state the
  local-only generated-output boundary;
- workflows do not reference generated API HTML output paths.

Day 7 adds one stronger workflow invariant: a workflow file fails the guard if
it includes generated API output paths and any artifact, Pages, `gh-pages`, or
publication semantics in the same workflow. This prevents an accidental
half-publication lane from passing under the local-only policy.

## Regression Coverage

The new Python regression suite checks:

| Test | Covered behavior |
| --- | --- |
| `test_current_tree_passes_guard` | The real repository passes the strengthened shell guard. |
| `test_fixture_passes_guard` | A minimal local-only repository passes. |
| `test_workflow_generated_api_path_fails_clearly` | A workflow that references `docs/api/html` fails with the generated-output-path diagnostic. |
| `test_workflow_publication_semantics_fail_clearly` | A workflow that uploads `docs/api/html` fails with the publication/artifact/Pages diagnostic. |
| `test_missing_local_only_wording_fails_clearly` | Missing API-reference local-only wording fails clearly. |

## Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `bash -n scripts/check_api_docs_local_only.sh` | Passed | Shell syntax accepted. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Local-only guard reported all ignore, Doxyfile, tracking, wording, and workflow checks ok. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Regression suite completed without failures. |
| `make api-docs-freshness` | Passed | Doxygen regenerated local HTML; coverage reported 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages; strengthened local-only guard passed. |
| `git diff --check` | Passed | No whitespace errors. |

## Changed Surfaces

- `Makefile`
- `scripts/check_api_docs_local_only.sh`
- `tests/test_api_docs_local_only_guard.py`
- `docs/planning/EPIC_18/SPRINT_204/WORKING_NOTES.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day7-policy-implementation-batch.md`

## Non-Changed Surfaces

- No `.c` or `.h` files changed.
- No public API declarations changed.
- No workflow files changed.
- No `.gitignore` or `Doxyfile` changes were required.
- No generated files under `docs/api/` were made trackable or committed.
- No hosted, retained-artifact, Pages, or release publication support was
  added.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Apply selected local-only policy changes with minimal surface area. | Met. Only the guard, its regression test, Makefile wiring, and planning evidence changed. |
| Preserve existing docs freshness semantics unless the policy requires a documented change. | Met. `make api-docs-freshness` still runs Doxygen, coverage, and local-only validation; it now also runs the focused regression suite. |
| Keep generated API input limited to public headers. | Met. `Doxyfile` remains unchanged and the guard still checks `INPUT = include/` and `FILE_PATTERNS = *.h`. |
| Add guard coverage for accidental stale, staged, tracked, uploaded, or missing generated output behavior. | Met for Day 7 scope. Existing tracked/staged/ignored checks remain, and workflow upload/publication semantics now fail explicitly. |
| Run focused validation for modified policy surfaces. | Met. Shell syntax, direct guard, Python regression suite, API-doc freshness, and whitespace checks passed. |

## Day 7 Disposition

Item 204.2 is complete for the selected stronger local-only policy path. The
sprint should continue with Day 8 freshness and coverage review without
opening hosted generated API publication unless the product decision is
explicitly reopened.
