# Sprint 224 Day 10: QR provenance and claim rejection ledger

## Scope and result

The selected Windows QR incompatible least-squares candidate now has negative fixtures for row, manifest, summary, hosted generator context and cross-target substitution. The local hosted QR fixture still passes selected-target freshness. These fixtures complete item 224.4's source-level rejection work; they do not certify a final-head hosted Windows artifact.

The normalizer now rejects a QR hosted summary that appends `no hosted CI proof`, `no Windows report freshness`, or an explicit broad/all-Windows QR claim elsewhere in the file while keeping its checked scope and non-claim line intact. Those statements contradict a passing, narrowly scoped `hosted_selected` record. The selected QR claim remains limited to the named incompatible 4×2 fixture and Windows MSVC job.

## Rejection observations

| Fixture mutation | Observed diagnostic or outcome |
| --- | --- |
| QR study row has Linux platform or wrong MSVC compiler | `comparison_hosted_metadata_mismatch: field=platform` or `field=compiler` |
| Row has wrong generator stage/configuration or `local_only` support | `field=configuration` or `field=support_tier` |
| Row changes baseline name, type, version, Python executable/version or command | Corresponding `field=baseline_*` mismatch against the hosted contract or manifest |
| Row changes project command, source branch or generation time | `field=project_probe_command`, `field=source_branch` or `field=generated_at_utc` |
| Row claims broad Windows QR support or stale hosted/Windows denials | `field=claim_scope` or `field=non_claims` |
| Manifest changes target/fixture, platform/compiler, baseline identity/helper/runtime or source branch | Corresponding `comparison_hosted_metadata_mismatch: field=...` |
| Summary changes its selected scope, non-claim line or provenance fields | `field=summary_scope_or_non_claims` or `field=summary_source_branch` / `field=summary_baseline_command` |
| Summary appends `no hosted CI proof` or `no Windows report freshness` after valid lines | `field=summary_scope_or_non_claims` |
| Summary appends an all-Windows QR solve or broad QR parity claim after valid lines | `field=summary_scope_or_non_claims` |
| Cholesky study bytes replace QR study bytes | QR selected validation rejects the substituted study; the independent Cholesky fixture still passes |
| Hosted generator context changes repository, workflow, Actions state, runner, event, branch, SHA or clean worktree | `hosted_context_mismatch` |
| QR generator is called under the Cholesky job | QR hosted tier is not selected |
| Windows workflow loses the QR job, artifact name, source SHA/branch or checkout reference | Selected workflow/PowerShell structural tests reject the drift |
| Windows upload adds a path, duplicates a path or broadens a glob | Exact six-file upload guard rejects the drift |
| Maintained documents make an unsupported broad Windows QR claim | Windows claim-boundary structural tests reject the wording |

The Day 9 ledger covers exact file membership, stale commits, dirty worktrees, row identity/order, Windows path variants and near matches. Together the Day 9 and Day 10 regressions cover the frozen Day 8 negative matrix. QR and Cholesky positive fixtures remain separate.

## Verification and remaining gate

`python3 tests/test_normalize_report_index.py`, `python3 tests/test_run_external_comparison.py`, `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_validate_windows_powershell.py`, `make support-docs-guard`, `make docs-check` and `git diff --check` passed after the Day 10 edits. No `.c` or `.h` file changed.

Day 11 must obtain a new hosted MSVC QR run and six-file artifact tied to the final changed source revision. No local fixture or historical intake ZIP promotes Windows QR freshness.
