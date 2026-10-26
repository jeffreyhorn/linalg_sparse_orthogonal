# Sprint 220 Day 9: Policy Edge Cases And Differential Ledger

## Comparison Method

`scripts/run_api_docs_workflow_differential.py` loads the frozen Day 2
manifest and validates that all 190 registered fixture functions still match
it. It records each `run_guard()` invocation, including all 26 variants in
the four mixed functions. For each temporary fixture worktree it runs:

1. The current combined guard, returned to the existing fixture assertion.
2. The legacy guard from frozen revision
   `05dba39abaa7c0acb38ff064690cdf5f67acf352`.
3. The standalone structured reader and publication policy.

Only the temporary fixture's `scripts/` directory receives copies of the
baseline guard and its historical workflow-helper module; the checked-out
project is not modified by the runner. The one
`test_current_tree_passes_guard` control runs its baseline from a temporary
script with root lookup redirected to the checkout. The six non-workflow
contract cases compare the legacy and combined guards; the standalone
workflow policy is marked `NOT_APPLICABLE`, not silently counted as a pass.

Every JSONL row records fixture/variant, frozen expected verdict and
diagnostic, baseline revision, each observed verdict/diagnostic, workflow
files and source location when available, and explicit discrepancy labels.
The runner exits nonzero on fixture execution errors or incomplete mixed
variant capture; disagreement labels remain visible in the report rather
than being suppressed.

## Results

| Corpus | Scope | Result |
| --- | --- | --- |
| [Frozen fixture comparison](day9-differential-report.jsonl) | 225 invocations: 219 workflow and 6 non-workflow; 180 expected rejections and 45 allowances. | Zero execution errors, zero current-guard mismatches, zero baseline/structured workflow verdict differences. |
| [Day 3 YAML edge comparison](day9-edge-differential-report.jsonl) | Ten edge fixtures: eight rejections and two allowances. | Three legacy missed bypasses; structured and current guards matched all ten expectations. |

### Discrepancy Triage

| Fixture | Baseline | Structured/current | Classification | Reason |
| --- | --- | --- | --- | --- |
| `unknown_alias.yml` | Allow | Reject | Intended safety tightening; `legacy_missed_bypass` | Undefined YAML alias cannot safely resolve the publication path. |
| `custom_tag.yml` | Allow | Reject | Intended safety tightening; `legacy_missed_bypass` | Custom YAML tag in a publication field is outside the supported grammar. |
| `dynamic_action_identity.yml` | Allow | Reject | Intended safety tightening; `legacy_missed_bypass` | Unresolved `uses` identity cannot be classified as unrelated. |

No false positive, structured missed bypass, or new parser error was observed
in these two frozen corpora. This is local differential evidence, not a
claim that every possible shell command or future workflow is covered.
Implicit publisher inputs and additional command forms remain explicit Day 10
mutation-review targets.

During PR #244 review, the differential runner was found to load the frozen
guard with the current checkout's helper module. Import failures were then
misclassified as legacy policy rejections. Both checked-in JSONL reports were
regenerated with the guard and helper from the same frozen revision. The
corrected full corpus has no legacy discrepancies; the edge corpus retains
the three missed bypasses listed above. Earlier reports of 43 frozen-corpus
legacy false positives and two edge false positives were harness artifacts.

## Local Verification

- Focused structured policy: 17 tests passed.
- Differential harness: 3 tests passed, including a fixture-level smoke comparison.
- Full frozen comparison: 225 rows, zero fixture errors.
- YAML edge comparison: 10 rows, three labeled legacy misses.
- Current real-workflow local-only guard and documentation routing suite: passed.
- Both JSONL reports have no host temporary-directory paths.

No C or header files changed, so the C format/lint/test chain was not run.

## Policy Completion

Day 9 adds focused canonical-path and command tests for Windows separators,
normalized relative paths, docs globs, expression fragments, unknown actions,
direct shell upload paths, and inert command text. Explicit local inputs to
`gsutil` and `vercel deploy` are now recognized alongside the Day 8 shell
families. The structured direct and staged publication policy is complete
for Item 220.3's planned 28 hours (Days 7-9); Day 10 owns differential
reconciliation and further adversarial mutation review.
