# Sprint 220 Day 1: Guard And Workflow Inventory

**Source revision:** `05dba39abaa7c0acb38ff064690cdf5f67acf352`
(`master` after PR #243, before Sprint 220 changes).
**Scope:** Inventory and baseline only. No workflow policy or generated-HTML
publication decision changes were made.

## Validation Route

| Surface | Current responsibility | Day 2-14 owner |
| --- | --- | --- |
| `Makefile` `api-docs-local-only` | Depends on `docs-check`, so Doxygen generation and API coverage run before the guard and its Python fixture suite. `api-docs-routing` depends on this target. | Preserve target order and CI reachability on Days 6 and 13-14. |
| `scripts/check_api_docs_local_only.sh` (2,085 lines) | Checks Git state, ignore rules, Doxyfile values, product wording, direct workflow references, and publication/staging semantics. Four embedded Python stages perform YAML comment stripping, path/anchor decoding, `run` folding, and dot-segment normalization. | Separate workflow extraction from policy on Days 3-9; retain non-workflow checks. |
| `scripts/api_docs_workflow_yaml_common.py` (246 lines) | Shared quote, YAML-comment, scalar, and shell-comment helpers for the embedded parsers. | Assess structured replacement and remove only proven-superseded helpers on Days 3-5 and 13. |
| `tests/test_api_docs_local_only_guard.py` (4,297 lines; 190 `test_` definitions) | Builds temporary Git fixture trees; exercises allowed and forbidden path, action, anchor, flow, scalar, staging, shell, PowerShell, and Git-state cases. | Freeze expected results on Days 2-3; run differential and mutation gates on Days 9-11. |
| `scripts/check_api_docs_routing.py` and routing tests | Validate the documentation and Make routing contract around the local-only guard. | Recheck during rollout on Days 13-14. |

`docs-check` is an alias of `api-docs-coverage`, whose prerequisite is `docs`.
The maintained route is therefore `docs` -> `api-docs-coverage` ->
`api-docs-local-only` -> `api-docs-routing`. The guard is also directly
invocable with `bash scripts/check_api_docs_local_only.sh`; the fixture suite
is directly invocable with `python3 tests/test_api_docs_local_only_guard.py`.
These direct commands do not replace the full Make route.

## Non-Workflow Contract

| Check | Current source | Regression owner |
| --- | --- | --- |
| `docs/api`, `docs/api/html`, and `index.html` remain ignored | `check_ignore_rules`, `.gitignore` | Existing guard fixture and Day 14 integrated gate |
| No staged, tracked, or visible unignored files under `docs/api` | `check_tracked_and_staged_absence`, Git index/worktree queries | Existing three negative Git-state fixtures |
| Doxygen still reads top-level public headers and writes local HTML to `docs/api/html` | `check_doxyfile_contract`, `Doxyfile` | Existing Doxyfile checks and `api-docs-coverage` |
| README, API reference, and maintainer guide still state local-only/no-publication status | `check_product_status_wording` | Existing wording fixtures and routing tests |

The structured YAML reader must not become the owner of these checks. They
continue to run even if no workflow directory is present.

## Tracked Workflow Inputs

`rg --files .github/workflows` finds exactly three tracked workflow files,
all `.yml`; no `.yaml` file is currently checked in. The guard scans both
extensions, and the fixture suite exercises `.yaml` inputs.

| Workflow | Jobs and shells | Publication-relevant inputs |
| --- | --- | --- |
| `.github/workflows/ci.yml` | 10 Linux jobs; `run` steps use the runner's default shell unless set in a step. No workflow/job `defaults.run.shell` declaration was found. | Five `actions/upload-artifact@v4` steps: oracle (`build/corpus/...`), selected comparison (`build/comparison/...`), selected performance (`build/bench-reports/canonical/...`), dead-code (`build/deadcode/...`), and coverage (`coverage/html/`). The uploads use `with.path`; coverage uses a single-line value, others block scalars. |
| `.github/workflows/macos-ci.yml` | 6 macOS jobs; no `defaults.run.shell` declaration found. | Two `actions/upload-artifact@v4` steps with block `with.path`: selected comparison under `build/comparison/...` and selected performance under `build/bench-reports/canonical/...`. |
| `.github/workflows/windows-ci.yml` | 5 Windows jobs. `run` steps explicitly select `pwsh` or `cmd`; one install/downstream `pwsh` block spans many lines. | Two `actions/upload-artifact@v4` steps with block `with.path`, both under `build/comparison/...`; no generated API HTML path. |

The real input shapes include literal `run: |` blocks, explicit Windows
shells, `uses` actions, `with.path` literal blocks, and a direct scalar path.
There are no `docs/api` references in these tracked workflow files. The
current passing guard is evidence for today's inputs, not proof that a future
workflow form will parse safely.

## Workflow Detection Boundaries

| Current stage | Input and result | Replacement concern |
| --- | --- | --- |
| `strip_yaml_comments` and `decode_yaml_publication_paths` | Produce normalized text from YAML lines, quotes, anchors, and path-bearing fields. | Preserve real YAML field values while excluding comments and inert text. |
| `fold_yaml_run_blocks` | Produces separately rendered `run` content, including folded and literal scalars and shell-sensitive word handling. | Retain scalar semantics and effective shell, especially PowerShell backslashes/backticks. |
| `normalize_workflow_dot_segments` | Canonicalizes selected path spellings after case/separator conversion. | Do not erase argument or path boundaries. |
| `check_no_workflow_publication_semantics` | Matches action, command, broad path, generated path, staging, archive, and dynamic-path patterns against whole-workflow normalized/folded text. | Keep publication decisions tied to the correct job/step and staged-output relationship. |
| `require_workflows_do_not_match` and `require_workflows_do_not_reference` | Reject direct generated-API references, regardless of the publication matcher. | Preserve the conservative direct-reference rule across both file extensions. |

Today's guard combines text from a workflow file before several policy
matches. A structured replacement needs to distinguish one step's path from
another step's action without losing legitimate cross-step staging/upload
relationships. Day 2 owns the expected verdict for each existing fixture.

## Baseline And Limits

On this local macOS worktree, Doxygen is available at
`/usr/local/bin/doxygen`, and `python3 --version` reports `3.14.5`.
`make api-docs-local-only` generates ignored HTML, checks 18 checked-in
public headers against 18 reference and 18 source pages, then runs the guard
and its 190-case fixture suite. The maintained target exited 0 locally;
`WORKING_NOTES.md` records the command and its scope.

No `requirements*.txt`, `pyproject.toml`, or existing PyYAML/ruamel import was
found in the repository. Day 3 must choose a structured parser with a clear
local and hosted dependency story; this inventory does not select one.

No C or header source was changed on Day 1. The Make C-format/lint/test chain
is not part of this inventory baseline; parser implementation days will run
the gates applicable to the files changed then. No hosted CI result is
claimed from this local run.
