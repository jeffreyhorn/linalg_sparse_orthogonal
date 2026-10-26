# Sprint 220 Day 13: Rollout And Legacy Parser Retirement

## Change And Boundary

`scripts/check_api_docs_local_only.sh` now delegates workflow publication
validation directly to `scripts/api_docs_workflow_policy.py`. The retired
whole-file YAML comment stripping, scalar decoding, run folding, dot-segment
normalization, and broad publication regexes no longer run. The shared Python
helper retains only `strip_shell_comment_stateful`, which the structured
direct and staging policies still import. Git-state/ignore, Doxyfile, product
wording, workflow-file discovery, and the local-only final status remain in
the shell guard.

The Day 11 false positive is closed: `actions/checkout@v4` with `path: docs/`
and a separate upload of only `build/output` now passes the combined guard.
This does not permit uploading `docs/` or `docs/api/html`.

## Frozen Oracle And Diagnostics

The frozen 225-variant differential passed with zero fixture errors and zero
current-versus-structured workflow verdict differences. It still reports 43
historical legacy false positives relative to the frozen oracle, not new
missed bypasses. The ten YAML edge fixtures passed, with two historical
legacy false positives. The 190-function guard suite and 13-case mutation
matrix passed separately.

The fixture verdict manifest was not changed. Assertions expecting the old
file-scoped phrases for generated paths or publication now check the
structured `generated API output path` detail. Dynamic publication fixtures
check source-located dynamic-path diagnostics. Three workspace-expression
paths are rejected as unresolved rather than guessed to be broad paths;
invalid dotted-anchor and quoted-key fixtures continue to reject at YAML
parse time. These are intentional diagnostic-stage or wording changes, not
allowance changes.

## Local And Hosted Route

`make api-docs-local-only` still depends on `docs-check`, then runs the
combined guard, frozen fixture suite, and mutation suite.
`make api-docs-routing` still extends it with the routing checker and suite.
The Linux and Apple-Clang CI validation steps install `PyYAML==6.0.3` in a
virtual environment and now run the frozen suite alongside reader, input,
policy, differential, mutation, and direct guard checks. Hosted execution
of the changed workflow has not yet been observed.

The maintainer guide now routes unsupported YAML to a reader fixture,
unresolved actions and paths to static declarations plus a policy fixture,
and unclassified third-party actions to input review. Future HTML
publication requires a new product decision and evidence for hosting or
retention, freshness, routing, rollback, and security. A guard bypass is not
an escalation path.

## Checks

| Command | Local result |
| --- | --- |
| `bash -n scripts/check_api_docs_local_only.sh` | Passed. |
| `API_DOCS_PYTHON=<venv-python> bash scripts/check_api_docs_local_only.sh` | Passed; 110 job/step records. |
| `<venv-python> tests/test_api_docs_local_only_guard.py` | Passed; 190 registered fixture functions. |
| `<venv-python> tests/test_api_docs_workflow_mutations.py` | Passed; three methods including the 13-case matrix and repaired false-positive probe. |
| `<venv-python> scripts/run_api_docs_workflow_differential.py --edges-only --output /tmp/s220-day13-edges.jsonl` | Passed; ten edge fixtures. |
| `<venv-python> scripts/run_api_docs_workflow_differential.py --output /tmp/s220-day13-differential-final.jsonl` | Passed; 225 variants, zero fixture errors. |
| `make api-docs-routing API_DOCS_PYTHON=<venv-python>` | Passed; Doxygen coverage 18/18/18, guard, frozen and mutation suites, and routing checks. |

`<venv-python>` was `/tmp/codex-s220-reader-venv/bin/python` on the local
macOS host, with `PyYAML==6.0.3` installed. Doxygen was available locally;
the differential used the frozen baseline revision from the Day 2 manifest.
These are local results only, not hosted CI evidence or publication approval.
No C or header files changed.
