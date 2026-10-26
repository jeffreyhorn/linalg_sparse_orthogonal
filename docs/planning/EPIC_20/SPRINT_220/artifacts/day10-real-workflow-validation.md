# Sprint 220 Day 10: Maintained Workflow Validation

## Scope And Record Parity

The structured reader parsed every `.yml` and `.yaml` file in
`.github/workflows/`. An independent `yaml.safe_load` traversal of each
workflow's `jobs.*.steps` gave the same step totals as the structured reader
and its policy-input adapter:

| Workflow | Jobs | Steps | `upload-artifact` steps |
| --- | ---: | ---: | ---: |
| `.github/workflows/ci.yml` | 10 | 57 | 5 |
| `.github/workflows/macos-ci.yml` | 6 | 31 | 2 |
| `.github/workflows/windows-ci.yml` | 5 | 22 | 2 |
| **Total** | **21** | **110** | **9** |

The adapter exposed 110 step-scoped policy records. There are no reusable
`jobs.*.uses` jobs requiring additional job-scoped records. It retained 65
POSIX, 9 PowerShell, and 6 `cmd` run steps, and all nine upload actions had
parsed `path` entries. No path entry selected `docs/` or `docs/api/`; the
existing uploads target build reports or `coverage/html/`. This is a complete
record-count check for the current three workflow files, not a claim about
arbitrary future action semantics.

The six `cmd` steps were inspected individually: they run PowerShell
validation, external-comparison generation, report-index validation, or the
static-install-contract test. None contains a shell publication command.
The policy receives these `cmd` step records and still applies direct
generated-path checks; shell staging recognition remains limited to POSIX
and PowerShell. This limit does not silently drop a current upload because
the two Windows upload actions are separate, parsed `uses` steps.

## Guard Result

With the pinned PyYAML 6.0.3 environment,
`API_DOCS_PYTHON=/tmp/codex-s220-reader-venv/bin/python bash scripts/check_api_docs_local_only.sh`
exited 0. Its structured preflight validated 110 records, its structured
publication policy checked 110 records, and the legacy direct workflow
checks passed. No maintained step was newly rejected, so no rejection was
waived or treated as safe without inspection.

The same guard reported passing checks for staged, tracked, and visible
untracked files under `docs/api/`; ignore rules for `docs/api/`,
`docs/api/html/`, and `docs/api/html/index.html`; six Doxyfile settings; and
the local-only product wording in README and the two API documentation files.
The frozen fixture corpus continues to exercise tracked, staged,
non-ignored untracked, and wording failures. A focused Day 10 regression
mutates `OUTPUT_DIRECTORY` and verifies that the Doxyfile contract still
rejects a non-local output location.

Generated HTML remains ignored local output. A local guard pass does not
establish hosted CI behavior or authorize artifact publication.
