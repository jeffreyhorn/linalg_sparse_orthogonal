# Sprint 220 Day 4: Structured Workflow Reader

**Scope:** Implement item 220.2's first reader slice. The production
`check_api_docs_local_only.sh` policy and non-workflow checks are unchanged.

## Reader And Dependency

`scripts/api_docs_workflow_reader.py` is the single entry point for both
`.github/workflows/*.yml` and `*.yaml`. It uses the pinned
`requirements-api-docs.txt` (`PyYAML==6.0.3`), composes a complete YAML
document with `SafeLoader`, validates the node graph, then emits immutable
`WorkflowRecord`, `JobRecord`, and `StepRecord` values. The CLI prints a
workflow count on success and fails with file/line/column, stage, and reason
on reader errors. Absence of PyYAML is a clear nonzero failure; the reader
does not fall back to the old text normalizer or install packages at runtime.

| Record | Bounded contents |
| --- | --- |
| Workflow | Source path and location, workflow shell default and environment, ordered jobs. |
| Job | Job key/location, runner, job shell default, reusable `uses`/`with`, environment, ordered steps. |
| Step | Workflow/job/index/location, runner, explicit and effective shell, `uses`, `with`, `run`, environment. |
| Scalar | YAML-decoded value, tag, style, source span, and source location. |

Explicit step shell overrides job defaults, then workflow defaults, then a
known runner default (`pwsh` on Windows; `bash` on Ubuntu/macOS). Dynamic or
unknown runners yield `unknown`, not a guessed shell. Structured `run` values
retain YAML folding and chomping; shell interpretation remains the Day 5-8
policy work. Aliases resolve to values while job and step records keep their
own job keys and indexes. At Day 4, PyYAML composition still reported the
anchor's source mark at alias uses; Day 5 added separate use and definition
marks in the reader.

## Fail-Closed Boundary

The reader rejects malformed YAML, multiple documents, undefined or cyclic
aliases, duplicate keys anywhere in the workflow, empty/malformed job or
step collections, and wrong `uses`/`run` shapes as `parse` errors. It rejects
merge keys, custom/unknown or shape-incompatible tags, and non-scalar mapping
keys as `unsupported`. Invalid UTF-8 is a located `parse` error.
It checks both mapping keys and values, so `<<` cannot hide on a key node.
Typed scalar action inputs (for example, integer `retention-days`) are kept
with their tags instead of being misclassified as malformed. The later policy
layer will decide whether a typed or dynamic publication field is safe;
Day 4 does not substitute expressions or authorize uploads.

## Tests And CI

`tests/test_api_docs_workflow_reader.py` has eleven tests covering all three
maintained workflows, the Day 3 edge matrix's six parser/unsupported cases,
the four parseable policy/control cases, shell precedence, scalar style and
source spans, flow forms, job aliases, reusable jobs, malformed shapes,
extra documents, cycles, invalid UTF-8, and empty workflow directories. It
asserts nine real `upload-artifact` steps with retained `path` inputs and an integer
`retention-days` input.

Linux CI and the macOS Apple-Clang leg create isolated virtual environments,
install the pinned requirement, and run this reader suite. This is parser
coverage only; it is not yet proof that the new reader enforces the local-only
publication rule. The old guard remains the active guard until policy parity
and differential checks in later sprint days.

## Local Evidence

| Check | Result |
| --- | --- |
| Isolated venv install from `requirements-api-docs.txt` | PyYAML 6.0.3 installed. |
| `tests/test_api_docs_workflow_reader.py` in that venv | 11 tests passed. |
| `python3 scripts/api_docs_workflow_reader.py` with the isolated package | 3 maintained workflows parsed. |
| Reader without site packages (`python3 -S`) | Nonzero with dependency diagnostic. |
| `bash scripts/check_api_docs_local_only.sh` | Exit 0; existing local-only guard passed after CI workflow edits. |
| `python3 tests/test_api_docs_routing.py` | Exit 0; existing documentation routing suite passed. |

No C/H files changed, so the C format/lint/test sequence is not applicable
to this reader-only slice. Hosted CI results are pending and not claimed.
The [Day 5 evidence](day5-scalar-alias-shell-evidence.md) covers alias marks.
