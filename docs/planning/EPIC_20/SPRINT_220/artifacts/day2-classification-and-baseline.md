# Sprint 220 Day 2: Classification Corpus Freeze

**Frozen revision:** `05dba39abaa7c0acb38ff064690cdf5f67acf352`.
**Product rule:** Generated HTML under `docs/api/` remains ignored local
output, never a workflow artifact, hosted page, release asset, or staged
source for those outputs. The existing conservative guard also rejects direct
workflow references to `docs/api`, even without a known publisher.

## Case-Level Oracle

[day2-fixture-classification.tsv](day2-fixture-classification.tsv) names each
of the 190 test functions in `tests/test_api_docs_local_only_guard.py`, its
source line, expected verdict, and required diagnostic substring when a
rejection is asserted. The classification is frozen from the fixture's
written expectation and the local-only product rule, before a structured
parser exists. It is not an instruction to copy the old parser's mechanism.

| Function class | Count | Meaning |
| --- | ---: | --- |
| `REJECT` | 151 | Each invocation must exit nonzero and include its recorded diagnostic substring. Some functions loop over several rejected variants. |
| `ALLOW` | 35 | Each invocation must exit zero. Some functions loop over several allowed variants. |
| `MIXED` | 4 | The function contains both outcomes; the 26 concrete variants are enumerated below. |
| **Registered functions** | **190** | Every `test_` definition is called exactly once by `main()`; registration and manifest names agree. |

The four mixed functions contain 17 rejected and 9 allowed variants. The
manifest uses `MIXED` only for their parent functions; their variant rows
below are the actual expected outcomes. There are at least 212 scenario
invocations overall (186 single-verdict functions plus 26 mixed variants),
and possibly more because some single-verdict functions loop. The fixture
suite does not execute workflow `run` commands; it tests guard verdicts.

## Classification Rationale

| Fixture family | Expected decision and reason | Examples / diagnostic group |
| --- | --- | --- |
| Direct generated API path | Reject an active `docs/api` or `docs/api/html` reference, including `.yaml`, rooted/relative/dot-segment, escaped, quoted-fragment, and Windows spellings. This is the current direct-reference rule, not a claim that every command publishes HTML. | `test_workflow_generated_api_path_fails_clearly`, `test_yaml_workflow_generated_api_path_fails_clearly`; `generated API HTML output path` or `generated API output root`. |
| Publisher plus broad root | Reject `docs`, repository root, globs, or workspace paths in publication fields when they could contain ignored HTML. A named publisher with a safe `build/` path remains allowed. | `test_workflow_broad_docs_artifact_path_fails_clearly`, `test_workflow_unrelated_publisher_build_path_is_allowed`; `publishes docs or repository roots`. |
| Action identity | Reject known artifact, Pages, deploy, release, local, or Docker publication actions with generated/broad/dynamic paths; allow non-publishing actions even when a path-like field is present. | `test_workflow_local_action_broad_docs_path_fails_closed`, `test_workflow_non_publication_action_docs_path_is_allowed`; `publication, artifact, or Pages semantics`. |
| Anchors and flow forms | Resolve referenced path/`uses`/`run` aliases and flow mappings/sequences; reject active generated-doc publication, but allow quoted/commented/unreferenced anchor text. | `test_workflow_flow_anchored_docs_artifact_path_fails_clearly`, `test_workflow_aliased_docs_path_ignores_quoted_anchor_text`. |
| YAML scalars and comments | Preserve single/double quotes, escaped values, block/literal/folded content, chomping, and comment boundaries. Reject an executable generated path; allow a path that exists only in a YAML or shell comment. | `test_workflow_quoted_hash_before_generated_api_path_fails_clearly`, `test_workflow_folded_run_shell_comment_before_generated_path_is_allowed`. |
| POSIX staging and archives | Reject `cp`/`mv`/`rsync` or archive commands that put `docs` into a published artifact. Allow unrelated `build` staging, inert quoted text, or commands separated so they do not form the prohibited operation. | `test_workflow_7z_archived_docs_artifact_upload_fails_clearly`, `test_workflow_unrelated_build_copy_before_check_docs_upload_is_allowed`; `stages docs for publication` or `archives docs for publication`. |
| PowerShell staging | On an effective PowerShell shell, reject `Copy-Item`, `Move-Item`, or `Compress-Archive` from `docs`, including backtick continuation and `\.\docs\*` forms; allow corresponding `build` sources. Do not treat Bash's different backtick semantics as a PowerShell continuation. | Mixed variant matrix below; `stages docs`, `archives docs`, or `dynamic staging paths`. |
| Dynamic publication paths | Reject unresolved expressions/variables on publishing or staging paths because they could select local generated output; allow unrelated variables or a path field on a non-publishing action. | `test_workflow_dynamic_artifact_path_fails_closed`, `test_workflow_non_publication_action_dynamic_path_is_allowed`; `dynamic publication paths` or `dynamic staging paths`. |
| Git and product contract | Reject tracked, staged, or visible unignored generated files and loss of required local-only wording. Allow current/fixture trees that keep the output ignored and wording intact. | `test_tracked_generated_api_file_fails_clearly`, `test_fixture_passes_guard`; Git-state or wording diagnostic. |

These families cover both bypass mutations and false-positive controls. The
manifest's exact diagnostic substrings are compatibility evidence for the
current suite. A new implementation may improve wording only by updating
the test and documenting the changed diagnostic without weakening the
underlying allow/reject oracle.

The 151 rejection-only functions group by asserted diagnostic as follows;
the mixed variants have their own counts and are excluded here.

| Required substring in fixture | Functions |
| --- | ---: |
| `publishes docs or repository roots` | 54 |
| `stages docs for publication` | 42 |
| `publication, artifact, or Pages semantics` | 11 |
| `dynamic publication paths` | 10 |
| `archives docs for publication` | 8 |
| `generated API HTML output path` | 7 |
| `stages docs` | 6 |
| `generated API output root` | 4 |
| `dynamic staging paths` | 4 |
| `generated API output paths` | 1 |
| Tracked generated output diagnostic | 1 |
| Staged generated output diagnostic | 1 |
| Visible unignored output diagnostic | 1 |
| Missing local-only wording diagnostic | 1 |
| **Total** | **151** |

## Mixed Variant Matrix

All PowerShell cases below include a following `actions/upload-artifact@v4`
step for `artifact*` or `artifact/`; the columns show the source/command
distinction that determines whether generated docs can reach it.

### PowerShell Commands

`test_workflow_powershell_docs_publication_commands` uses a literal `run: |`
block with `shell: pwsh` on Windows.

| Variant | Source/command distinction | Verdict | Required diagnostic |
| ---: | --- | --- | --- |
| 1 | `Copy-Item -Recurse docs artifact` | REJECT | `stages docs` |
| 2 | `Copy-Item -Path ".\docs\*" ... -Recurse` | REJECT | `stages docs` |
| 3 | `Move-Item -Path docs -Destination artifact` | REJECT | `stages docs` |
| 4 | `Compress-Archive -Path docs ...` | REJECT | `archives docs` |
| 5 | `Compress-Archive -Path ".\docs\*" ...` | REJECT | `archives docs` |
| 6 | `Copy-Item` from `$env:DOCS_ROOT` | REJECT | `dynamic staging paths` |
| 7 | `Copy-Item` from `${env:DOCS_ROOT}` | REJECT | `dynamic staging paths` |
| 8 | `Move-Item` from `$source` | REJECT | `dynamic staging paths` |
| 9 | `Compress-Archive` from `$env:DOCS_ROOT` | REJECT | `dynamic staging paths` |
| 10 | `Copy-Item` from `do$($env:FRAGMENT)` | REJECT | `dynamic staging paths` |
| 11 | `Compress-Archive` from `${source}` | REJECT | `dynamic staging paths` |
| 12 | `Copy-Item` from literal `build` | ALLOW | - |
| 13 | `Write-Host` variable, then `Copy-Item` from `build` | ALLOW | - |
| 14 | `Move-Item` from literal `build` | ALLOW | - |
| 15 | `Compress-Archive` from literal `build` | ALLOW | - |

### Effective Shell And Backtick Continuation

`test_workflow_powershell_backtick_continuations` splits the `Copy-Item`
options from the source line with a backtick.

| Variant | Runner and effective shell | Source | Verdict | Required diagnostic |
| ---: | --- | --- | --- | --- |
| 1 | Windows, explicit step `pwsh` | `docs` | REJECT | `stages docs` |
| 2 | Windows, default shell | `docs` | REJECT | `stages docs` |
| 3 | Windows, default shell | `build` | ALLOW | - |
| 4 | Ubuntu, workflow default `pwsh` | `docs` | REJECT | `stages docs` |
| 5 | Windows, job default `powershell` | `docs` | REJECT | `stages docs` |
| 6 | Windows, explicit step `bash` | `docs` | ALLOW | - |

Variant 6 is a guard false-positive control, not a claim that the Bash
command is useful or that an arbitrary shell script is safe.

### Windows Path Separators

`test_workflow_powershell_backtick_windows_paths` uses the same continued
PowerShell command with Windows-style source paths.

| Variant | Source | Verdict | Required diagnostic |
| ---: | --- | --- | --- |
| 1 | `.\docs\*` | REJECT | `stages docs` |
| 2 | `.\build\*` | ALLOW | - |

### Folded Paragraph And Literal Block

`test_workflow_folded_blank_line_continuation_stages_docs` varies the YAML
scalar style and blank lines between `cp -R \` and `docs artifact/`.

| Variant | Scalar and blank lines | Verdict | Required diagnostic |
| ---: | --- | --- | --- |
| 1 | Folded `>` with one blank line | REJECT | `stages docs` |
| 2 | Folded `>` with two blank lines | ALLOW | - |
| 3 | Literal `|` with one blank line | ALLOW | - |

This distinction is about rendered command boundaries: folding can make the
source part of a continued copy, while a preserved line/paragraph break can
make it an unrelated command or argument. Day 5 must compare rendered YAML
scalars rather than strip shell comments before folding.

## Baseline Commands And Limits

| Command | Day 1/2 result | Scope |
| --- | --- | --- |
| `make api-docs-local-only` | Day 1 exit 0 on the same frozen source revision. | Doxygen, API coverage, direct guard, and all 190 registered fixture functions. Generated HTML was ignored. |
| `python3 tests/test_api_docs_local_only_guard.py` | Day 2 standalone exit 0; no assertion output. | All 190 registered test functions, including `test_current_tree_passes_guard`. |
| `bash scripts/check_api_docs_local_only.sh` | Day 2 standalone exit 0. | Git-state, Doxyfile, wording, real workflows, and direct-reference checks. |

Environment-sensitive inputs: Doxygen must be installed for the Make route;
the suite uses temporary Git repositories and platform shell processes; real
Windows `pwsh` commands are inspected as text, not executed on this macOS
host. One local passing run does not prove absence of flakiness. No fixture
was skipped or marked flaky in the current run, and no hosted CI result is
claimed. Day 3 will add malformed/duplicate-key and other unsupported-syntax
fixtures before choosing the structured parser.
