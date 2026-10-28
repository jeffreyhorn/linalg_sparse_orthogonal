# Sprint 220 Day 10: Differential Corpus Reconciliation

## Frozen Corpus

The Day 2 manifest remains the independent verdict oracle. The Day 9
[machine-readable differential](day9-differential-report.jsonl) records 225
actual guard invocations from 190 registered fixture functions, including
all 26 variants of the four mixed functions. Of these, 219 are workflow
cases and six exercise the surrounding non-workflow guard. The expected
verdicts are 180 rejections and 45 allowances. No fixture was dropped or
had an execution error. Every current-guard verdict matched its frozen
expectation; every standalone structured-policy verdict matched the
workflow expectation; and the legacy baseline had no verdict disagreement
on this Day 2 corpus.

The corpus covers direct `docs/api` references, broad docs/root paths,
same-job staged artifacts, anchors and aliases, flow mappings and sequences,
quoted and folded scalars, shell comments, PowerShell backticks and Windows
paths, as well as benign unrelated-upload and inert-text controls. The six
non-workflow cases compare the legacy and current combined guards only;
`NOT_APPLICABLE` for standalone workflow policy is not counted as an allow.

Day 10 reran `scripts/run_api_docs_workflow_differential.py` over the full
frozen corpus into a temporary JSONL file: 225 rows, zero fixture errors,
zero classification labels. The new file was byte-for-byte identical to the
Day 9 report. A separate `--edges-only` rerun also matched the saved edge
report byte-for-byte, with only the three intentional legacy missed-bypass
labels below. Temporary report files were not added to the worktree.

## Diagnostic Changes

The baseline and combined guard agree on verdict but differ in rejection
diagnostic for 89 workflow invocations. In 87, the structured policy now
reports a workflow line/column and job/step rather than only a workflow
filename; the rejection kind remains covered by the fixture assertion. The
other two fail earlier at the parser stage:

| Fixture | Frozen Day 2 diagnostic | Current diagnostic | Reason |
| --- | --- | --- | --- |
| `test_workflow_dotted_anchor_docs_artifact_path_fails_clearly` | Broad docs publication | `parse: while scanning an anchor` | `&docs.path` is not an accepted YAML anchor token; publication analysis cannot safely interpret the document. |
| `test_workflow_multiline_escaped_quoted_path_key_docs_artifact_path_fails` | Broad docs publication | `parse: while scanning a simple key` | The quoted `path` key split after a backslash is malformed YAML; the reader rejects it before path extraction. |

Both cases still reject. The executable fixture assertions already name the
parse-stage diagnostics; the Day 2 manifest retains its historical
diagnostics as a frozen baseline, not an instruction to accept malformed
YAML or to restore text-normalizer error ordering. No expected verdict was
changed on Day 10.

## YAML Edge Corpus

The Day 3 [edge report](day9-edge-differential-report.jsonl) has ten cases:
eight expected rejections and two expected allowances. The structured
policy and combined guard match every expected verdict. Parser and policy
failure stages also match the independent edge matrix. The three actual
old/new verdict changes are:

| Fixture | Legacy | Structured/current | Resolution |
| --- | --- | --- | --- |
| `unknown_alias.yml` | Allow | Reject | An undefined alias cannot resolve a publication path; parse rejection is the intended fail-closed behavior. |
| `custom_tag.yml` | Allow | Reject | A custom YAML tag in a publication field is unsupported; rejecting it avoids interpreting untrusted nonstandard semantics. |
| `dynamic_action_identity.yml` | Allow | Reject | An unresolved `uses` identity cannot be classified as unrelated to publication; policy rejection is intended. |

The other seven edge cases preserve their old/new verdict. The differential
runner now exits nonzero for unexpected current or structured verdicts and
for an edge failure-stage mismatch. Intentional legacy missed bypasses
remain visible in the JSONL report but do not fail the gate. A focused
regression also proves that changing `Doxyfile`'s `OUTPUT_DIRECTORY` is
still rejected by the combined guard.

## Limits

This reconciliation concerns the frozen corpus and the current maintained
workflows. It does not establish complete shell-language or arbitrary
third-party action coverage. The [real-workflow validation](day10-real-workflow-validation.md)
records the current inventory and guard result; future command forms and
mutation gates remain subsequent sprint work.
