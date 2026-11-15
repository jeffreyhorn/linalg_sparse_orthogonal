# Sprint 223 Day 10: Metadata and Claim Guards

Date: 2026-11-10. Item 223.4, 10 hours planned.

## Decision and implementation

The selected Cholesky row is still a candidate for Windows freshness. The normalizer now checks a `hosted_selected` Cholesky study **before** promoting its rows into the normalized index. It requires the exact Windows AMD64 platform, Visual Studio 2022 Release CMake probe identifier, hosted configuration, clean worktree, fixture-local hosted claim and complete non-claim text. A hosted signal in any row's tier, configuration, claim or Windows non-claim triggers the same check for every row, so a mixed local/hosted study cannot pass. Local-only studies remain valid without hosted metadata.

The guard also checks that `manifest.tsv` names the Cholesky target and fixture and agrees with the study on platform, compiler, configuration, source commit, branch, worktree state and generation time. `summary.md` must carry the bounded hosted scope and the same non-claims. Missing manifest or summary produces an explicit metadata diagnostic. The selected manifest and workflow guards continue to pin the exact Windows workflow, job and artifact. Other selected targets remain outside this hosted check.

## Negative-case ledger

| Mutation | Rejection |
| --- | --- |
| Linux platform in hosted row | `comparison_hosted_metadata_mismatch: field=platform` |
| Default or wrong CMake probe generator | `field=compiler` |
| Local tier mixed into hosted rows | `field=support_tier` |
| Local configuration mixed into hosted rows | `field=configuration` |
| Broad claim or missing non-claims | `field=claim_scope` or `field=non_claims` |
| Wrong manifest target, platform, compiler or source SHA | Corresponding manifest field mismatch |
| Wrong summary scope or non-claims | `field=summary_scope_or_non_claims` |
| Missing hosted manifest or summary | `field=manifest_missing` or `field=summary_missing` |
| Wrong Windows workflow, job or artifact in selected manifest | Exact tuple assertion in manifest suite |
| Wrong generated artifact path, missing/extra file, duplicate/foreign row, dirty hosted worktree | Day 9 guards remain active |

The unmutated six-row synthetic hosted bundle passes selected freshness. Existing local-only Cholesky and all other selected comparison fixtures continue to pass. The manifest suite still rejects Windows selection of other targets. API docs local-only policy files were not changed.

## Focused check transcript

All passed on the local branch:

```text
python3 tests/test_normalize_report_index.py                 test-normalize-report-index: ok
python3 tests/test_selected_report_targets_manifest.py       test-selected-report-targets-manifest: ok
python3 tests/test_selected_comparison_workflow.py           test-selected-comparison-workflow: ok
python3 tests/test_validate_windows_powershell.py            test-validate-windows-powershell: ok
python3 tests/test_run_external_comparison.py                test-run-external-comparison: ok
<temporary PyYAML 6.0.3 venv>/python tests/test_api_docs_workflow_policy.py   50 tests OK
python3 scripts/validate_corpus_schema.py                    validate-corpus-schema: ... ok
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
git diff --check
```

The first normalizer attempt failed because a legacy contract lacked `target_key`; the guard now keys on the existing Cholesky subfamily and the full suite passes. The first API docs check could not start because the active Python lacked pinned PyYAML; a temporary venv with `requirements-api-docs.txt` passed all 50 policy tests. Native MSVC/PowerShell execution and final-commit run, job and artifact verification belong to Days 11–12. This local test does not establish hosted freshness.
