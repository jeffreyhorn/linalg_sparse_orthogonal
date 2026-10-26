# Sprint 220 Day 6: Parser-To-Policy Handoff

## Scope And Boundary

`scripts/check_api_docs_local_only.sh` now invokes
`scripts/api_docs_workflow_policy_inputs.py` before its existing workflow
publication checks. This preflight composes and validates every `.yml` and
`.yaml` workflow through the Day 4-5 structured reader, then builds ordered,
source-located job/step policy inputs. Invalid or unsupported YAML fails
closed before the legacy text matcher runs. The Git-state, ignore, Doxyfile,
and product-wording checks still run first, and the Make target order remains
`docs-check` -> `api-docs-local-only` -> `api-docs-routing`.

The current publication/staging **verdicts still come from the legacy Bash
matcher**. Day 7 owns direct publication rules on these records; Day 8 owns
explicit staging links; Day 9 owns the complete differential. In particular,
the legacy diagnostic is still file-scoped. Source-located policy rejection
for every publication decision is not claimed on Day 6.

## Policy Input Contract

| Input | Representation and boundary |
| --- | --- |
| Origin | `workflow_path`, `job_key`, `step_index` (or `None` for reusable-job `uses`), and step/job `SourceLocation` survive extraction. Action and path fields retain their own source marks, including alias origin. |
| Action | Exact decoded `uses` scalar plus `action_dynamic`. Expressions and variable-shaped identities stay unresolved; no action is inferred from comments or unrelated steps. |
| Paths | Named `with` path inputs retain the original `ScalarValue` and one `PathEntry` per nonblank line. Literal entries are slash-normalized, dot-normalized, and casefolded; globs are preserved. Dynamic entries have no canonical path. Original spelling and source mark remain available for diagnostics. |
| Command/shell | `rendered_command` is the YAML-decoded `run` scalar, never path-normalized or shell-tokenized. `effective_shell` and `shell_family` come from reader precedence. PowerShell backslashes and backticks remain in command text. |
| Staging | `StagedOutputRelation` names a source, destination, producer step, and consumer step. No relation is inferred merely because a command and an upload appear in the same workflow. Day 8 must populate only proved same-step or explicit same-job links. |

`API_DOCS_PYTHON` selects the pinned-PyYAML interpreter for the guard and
`make api-docs-local-only`; the default is `python3`. A missing dependency
returns a nonzero setup diagnostic rather than falling back to the old
scanner. Linux and Apple-Clang CI use their existing isolated reader venv
for reader tests, adapter tests, and the direct guard.

## Maintained Workflow Comparison

| Workflow | Day 1 inventory | Day 6 extracted | Discrepancy |
| --- | --- | --- | --- |
| `ci.yml` | 10 jobs, five artifact uploads, Linux default shell | 10 jobs, 57 steps, five uploads, 42 POSIX `run` steps | None. All five uploads have step-local `with.path`. |
| `macos-ci.yml` | 6 jobs, two artifact uploads, macOS default shell | 6 jobs, 31 steps, two uploads, 23 POSIX `run` steps | None. Both block `with.path` lists retain individual entries. |
| `windows-ci.yml` | 5 jobs, two artifact uploads, explicit `pwsh`/`cmd` | 5 jobs, 22 steps, two uploads, nine PowerShell and six `cmd` `run` steps | None. Both uploads retain step-local block path lists. |

There are 21 jobs, 110 steps, and nine uploads in total; no reusable-job
`uses` appears in the maintained set. The Day 4 CI reader steps changed
command text but not job/step counts. No missing path field or effective-shell
mismatch was found against the frozen Day 1 inventory.

## Compatibility Discrepancies

A one-pass sweep invoked each of the 190 frozen legacy test functions against
the handoff. Five rejection-only functions expected an old policy diagnostic
but encountered a structured parse error first. No allow/reject verdict
changed:

| Fixture | Resolution |
| --- | --- |
| Dotted YAML anchor | Dotted anchor syntax is invalid; assert parse-stage rejection. |
| Multiline escaped quoted mapping key | The fixture is invalid YAML; assert parse-stage rejection. |
| Unquoted expression inside flow `with` mapping | Quote the expression so the valid fixture continues to test dynamic publication-path policy. |
| Windows indentless step with over-indented `with` | Correct child indentation; both variants continue to test PowerShell staging. |
| Flow sequence with a `name`-only step | Add `uses` to that step; both variants continue to test staging after an unrelated action. |

The frozen Day 2 verdicts remain intact. The two parse-stage diagnostic
changes are deliberate because syntactically invalid YAML must not reach
publication policy. Day 9 must still compare the full variant-level oracle,
including the known dynamic-action gap until Day 7 closes it.

## Evidence

| Command | Local result | Limit |
| --- | --- | --- |
| Pinned venv `tests/test_api_docs_workflow_reader.py` | 20 tests passed. | Extraction only. |
| Pinned venv `tests/test_api_docs_workflow_policy_inputs.py` | 7 tests passed. | Adapter, real inventory, canonical paths, source marks, shell and step boundaries. |
| `API_DOCS_PYTHON=<pinned-venv-python> bash scripts/check_api_docs_local_only.sh` | Passed on all three real workflows and unchanged non-YAML checks. | Legacy matcher still supplies publication verdicts. |
| One-pass 190-function fixture sweep | 185 unchanged function expectations; five diagnostic mismatches investigated above. All five pass in targeted rerun after fixture updates. | Differential at function granularity; Day 9 owns variant-level comparison. |
| `make api-docs-local-only API_DOCS_PYTHON=<pinned-venv-python>` | Exit 0 after the fixture updates; Doxygen generated 18 reference and 18 source pages for 18 public headers, and the direct guard plus all 190 fixture functions passed. | Local-only output remains ignored; no hosted CI result is claimed. |
| `python3 tests/test_api_docs_routing.py` | Exit 0. | Routing regression only; no hosted result. |
| `python3 -S scripts/api_docs_workflow_policy_inputs.py` | Nonzero with pinned PyYAML setup message. | Expected missing-dependency behavior. |

No C or header files were edited. The `make format && make lint && make test`
C-code chain is not required for this Python, workflow, Make, test, and
documentation change. No generated HTML was staged for publication.
