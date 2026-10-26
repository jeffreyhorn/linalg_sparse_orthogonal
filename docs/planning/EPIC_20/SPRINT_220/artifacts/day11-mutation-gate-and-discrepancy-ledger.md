# Sprint 220 Day 11: Mutation Gate And Discrepancy Ledger

## Gate

`tests/test_api_docs_workflow_mutations.py` uses a temporary Git worktree per
case and asserts both the standalone structured policy and the combined
shell guard. Its 13-case matrix completed with no assertion failures:

| Mutation | Expected class | Observed result |
| --- | --- | --- |
| Add matching artifact upload after copying `docs/` into `artifact/` | Reject | Both reject the staged docs route. |
| Use an anchored `docs/` path in flow-form `with.path` | Reject | Both reject broad docs publication. |
| Move `docs/api/html` into an uploaded artifact directory | Reject | Both reject the generated API output reference. |
| Break a flow collection | Fail-closed parse error | Both reject malformed YAML. |
| Duplicate `with.path` | Fail-closed parse error | Both reject the duplicate key. |
| Use a dynamic upload path | Fail-closed policy error | Both reject; the legacy check reports first in the combined guard. |
| Use a dynamic action identity | Fail-closed policy error | Both reject the unresolved `uses` field. |
| Use a dynamic staging source before upload | Fail-closed policy error | Both reject the unresolved staging source. |
| Add an upload of unrelated `build/output` after staging docs | Allow | Both allow; staging does not contaminate unrelated output. |
| Put `docs/api/html` in a shell comment | Allow | Both ignore inert comment text. |
| Quote YAML anchor-looking text in an `echo` command | Allow | Both keep it inert. |
| Check out into `build/checkout` before a safe upload | Allow | Both allow the unrelated action. |
| Stage docs in one job, upload matching path in another job | Allow | Both preserve the job boundary. |

The suite has three test methods: matrix metadata, the 13-case matrix, and
the known legacy false-positive probe below. It is called by
`make api-docs-local-only` and by the pinned-PyYAML Linux and Apple-Clang
workflow validation steps. The Day 10 [frozen differential](day10-differential-reconciliation.md)
remains the verdict baseline; no Day 2 or Day 3 expectation was changed.

The integrated `make api-docs-local-only` target passed with the pinned
PyYAML environment after the new recipe line was added. Doxygen coverage
reported 18 public headers, 18 reference pages, and 18 source pages; the
current-workflow guard checked 110 structured records; the frozen fixture
command completed without an assertion failure; and the mutation suite
reported three passing methods. The separate documentation routing suite
also passed. These are local results, not hosted CI evidence.

## Discrepancy Disposition

The additional checkout probe is deliberately separate from the passing
matrix. `actions/checkout@v4` with `path: docs/`, followed by an artifact
upload of only `build/output`, is safe under the structured policy but is
rejected by the combined guard's legacy broad-path scan. That scan joins
`path: docs/` from the unrelated checkout with upload semantics elsewhere
in the same file. The test currently asserts both observations so the
over-rejection is visible rather than silently removed from coverage.

This is a **legacy false positive, not permission to publish docs**. Its
investigation is closed; remediation belongs to Day 13's planned retirement
of superseded text normalization. The probe must change to an allow
assertion only after the combined guard no longer calls the legacy scan and
the frozen and mutation suites still pass. No new mutation exposed a
structured-policy missed bypass or a newly rejected maintained workflow.

## Removal Map For Day 13

After parity and mutation checks are green on the structured route, the
workflow-only text normalizers in `scripts/check_api_docs_local_only.sh` are
removal candidates: `strip_yaml_comments`, `decode_yaml_publication_paths`,
`fold_yaml_run_blocks`, `normalize_workflow_dot_segments`,
`require_workflows_do_not_reference`, `require_workflows_do_not_match`, and
`check_no_workflow_publication_semantics`. The latter three are currently
called by `check_no_workflow_publication_path` before structured policy.

Preserve the Git-state/ignore checks, Doxyfile contract, product wording,
workflow-file discovery, and structured reader/policy invocation. Do not
remove `api_docs_workflow_yaml_common.py` wholesale: the structured direct
and staging policies still import `strip_shell_comment_stateful`. Audit its
other helper references only after the legacy consumers are gone. Day 13
must rerun the frozen, edge, and mutation gates and inspect any changed
diagnostics before deleting old code; Day 14 owns the integrated final gate.

This closes Item 220.4's planned 20 hours of differential investigation
(Days 10-11). The classified legacy false positive remains a scheduled
implementation task, not a waived safety verdict.
