# Sprint 220 Day 14: Integrated Gate And Closeout

## Decision

The planned local structured-workflow guard is ready for hosted validation.
The generated Doxygen HTML decision remains **local-only**: a passing guard
does not approve committed HTML, Pages, release assets, or retained API-docs
artifacts. This is a local closeout, not a hosted CI pass or a new publication
claim.

## Gate Log

The local host used Doxygen 1.16.1 and an isolated Python environment at
`/tmp/codex-s220-reader-venv/bin/python` with `PyYAML==6.0.3` from
`requirements-api-docs.txt`.

| Check | Result | Scope |
| --- | --- | --- |
| `make api-docs-local-only API_DOCS_PYTHON=/tmp/codex-s220-reader-venv/bin/python` | Pass | Doxygen page coverage 18/18/18, current-workflow policy, frozen suite, mutation suite. |
| `make api-docs-routing API_DOCS_PYTHON=/tmp/codex-s220-reader-venv/bin/python` | Pass | Repeats local-only gate; routing checker covered seven documents and routing suite passed. |
| `API_DOCS_PYTHON=/tmp/codex-s220-reader-venv/bin/python bash scripts/check_api_docs_local_only.sh` | Pass | Git-state, ignore, Doxyfile, wording, and 110 job/step policy records. |
| Reader, input-adapter, policy, differential-harness suites | Pass after one test-only diagnostic update | 20, 7, 17, and 4 tests respectively. |
| Ten YAML edge fixtures | Pass | Expected parse, unsupported-syntax, dynamic-policy, and allowance stages. |
| Full frozen differential | Pass | 225 variants, zero fixture errors, zero current/structured workflow verdict differences. |
| `bash -n scripts/check_api_docs_local_only.sh`; `git diff --check` | Pass | Shell syntax and tracked-file whitespace. |

The initial input-adapter test run failed because it still expected the
retired `structured workflow extraction failed` wrapper. The guard correctly
reported `ci.yml:6:... parse: duplicate key 'run'`; the test now asserts that
source-located error and the current policy wrapper. The adapter suite, edge
matrix, and full differential passed on rerun. No guard behavior or frozen
verdict was changed for this diagnostic correction.

## Publication And Wiring Review

- `git check-ignore -v docs/api/html/index.html` resolves to the `docs/api/`
  ignore rule. `git status --short --ignored docs/api` shows only `!! docs/api/`;
  `git ls-files docs/api` and `git diff --cached --name-only -- docs/api`
  produce no paths.
- The three maintained workflows have nine `actions/upload-artifact` steps.
  Their configured paths are build or coverage outputs, not `docs/api/`.
  There are no maintained Pages, release-upload, or generated-API output-path
  references. The structured policy checked all 110 job/step records.
- Make keeps `docs-check` before `api-docs-local-only`, then the frozen and
  mutation suites, with `api-docs-routing` extending that route. Linux and
  Apple-Clang workflow validation install the pinned dependency and call the
  reader, adapter, policy, differential, frozen, mutation, and direct guard
  checks. Those changed hosted jobs have not run on this unpushed branch.
- PR #244 review corrected the differential harness to load the historical
  helper alongside the historical guard. The regenerated frozen report has
  no legacy discrepancies; the previously reported 43 false positives were
  import failures misclassified as policy rejections. Current and structured
  verdicts agree on all workflow rows.

## Residual Limits And Owners

| Limit | Owner / next evidence |
| --- | --- |
| No hosted result for the changed Linux/Apple-Clang workflow steps yet. | PR/CI owner: inspect both jobs after pushing the branch; do not label this a hosted pass before they finish. |
| `cmd` and unknown-shell staging are not general shell-analysis coverage. | Workflow guard maintainer: add command-specific rejecting and benign fixtures before relying on new shell publication forms. |
| Syntactically valid, unclassified third-party actions are not proof of non-publication. | Workflow reviewer: inspect action inputs and add a targeted fixture for a new publisher or path convention. |
| Any hosted or retained generated-HTML route remains unselected. | Product/docs owner: reopen the explicit publication decision with hosting, retention, freshness, routing, rollback, and security evidence. |

No `.c` or `.h` files changed during the sprint; C `make format`, `make lint`,
and `make test` are not applicable to this Python, shell, workflow, fixture,
and documentation change set. The planned Item 220.6 rollout allocation and
140-hour sprint scope are closed at the deliverable level, not as a claim
that 140 hours of wall-clock work were measured.
