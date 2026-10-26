# Sprint 220 Day 7: Structured Direct-Publication Policy

## Policy Boundary

`scripts/api_docs_workflow_policy.py` evaluates the Day 6 `PolicyInput`
records without concatenating workflow jobs or steps. Each rejection reports
the workflow file, source line/column, job, step index, and offending field.
The production guard invokes this checker after the legacy workflow scans.
That order preserves the existing staging diagnostics while adding direct
policy coverage for cases the text matcher missed. The legacy scans are not
retired until the Day 8 staging port and Day 9 differential.

| Input | Day 7 decision |
| --- | --- |
| Dynamic `uses` identity or malformed local/Docker/remote action form | Reject before treating the action as unrelated. The former dynamic-action bypass now fails in the production guard. |
| Direct `docs/api` reference in action identity, any `with` scalar, inherited workflow/job/step `env`, or active `run` text | Reject regardless of publisher classification. YAML comments are absent from decoded scalars; shell comments are removed from rendered commands using the effective shell family. |
| Path-bearing `with` value for upload-artifact, Pages/deploy, release, local, Docker, or unknown actions | Inspect every block-scalar path entry. Reject non-string/empty, dynamic, generated-API, docs-root, repository-root, and broad glob paths. Safe literal `build/` and `coverage/` paths remain allowed. |
| Known unrelated `actions/checkout` with a non-generated path | Allow; it is not classified as a publication action. Direct `docs/api` references still reject. |

Unknown actions with a path input are conservative: a safe literal path
passes, while an unresolved or broad docs/root path fails. A publisher with
no direct path is not automatically rejected because Pages deployment and
release actions may consume outputs staged elsewhere; Day 8 owns those
explicit relationships. No shell staging inference is made here.

Path checks operate on the adapter's canonical path entries, preserving the
original scalar and source mark. Command text is checked separately; the
stored YAML-decoded command is never rewritten. Case, separators, dot
segments, quoted path fragments, and globs are covered by focused tests.

## Focused Regression Matrix

`tests/test_api_docs_workflow_policy.py` covers:

- Real maintained workflows and safe `build/output` uploads.
- Direct generated references in `run`, `env`, `with.name`, and `with.path`.
- Quoted flow action/`with` mappings, aliased paths, and mixed-case Windows
  separators.
- Multiline path lists, root/docs globs, and unrelated checkout actions.
- Artifact, Pages, release, local, Docker, and unknown action families.
- Dynamic action/path expressions, invalid action forms, and typed paths.
- A full-guard mutation proving the former dynamic-action bypass now rejects
  with a job/step-specific diagnostic.

The reader and input-adapter suites remain separate owners of syntax,
source-location, scalar, and shell extraction. The frozen 190-function guard
corpus remains the compatibility oracle for existing verdicts. Day 8 must
address commands that stage docs into later upload paths; Day 9 must compare
all old/new cases before deleting the text matcher.

## Evidence

| Command | Result | Limit |
| --- | --- | --- |
| Pinned venv `tests/test_api_docs_workflow_policy.py` | 11 tests passed locally. | Focused Day 7 direct policy, including one full-guard bypass regression. |
| Pinned venv `scripts/api_docs_workflow_policy.py` | Passed on all three maintained workflows (110 step/job records). | Does not claim shell staging coverage. |
| One-pass frozen guard sweep | 190 test functions, zero mismatches. | Existing verdicts/diagnostics under the combined guard; Day 9 still owns variant-level old/new differential. |
| `make api-docs-local-only API_DOCS_PYTHON=<pinned-venv-python>` | Exit 0; Doxygen coverage, direct guard, and all 190 fixture functions passed. | Generated HTML stayed local and ignored. |
| Reader, policy-input, and routing suites | 20 reader, 7 adapter, and routing tests passed. | Local results; hosted CI is not claimed. |

No C or header files were edited; the C-code format/lint/test chain is not
required for this Python, workflow, fixture, and documentation change.
