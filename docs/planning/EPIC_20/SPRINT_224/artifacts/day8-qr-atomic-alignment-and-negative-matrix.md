# Sprint 224 Day 8: Atomic QR candidate and rejection matrix

## Candidate decision

Item 224.3's manifest, generator, Windows job, documentation, PowerShell/workflow guards and normalizer now describe **one** selected QR incompatible least-squares Windows candidate. The selected target is `SRT-COMP-QR-INCOMPATIBLE-LS` / `qr-incompatible-ls`, the fixture is `qr_overdetermined_incompatible_4x2`, and the report has six ordered comparison rows and six flat uploaded files. This is a coherent **source candidate**, not acceptance of a Windows freshness claim. The intake bundle from `44e3b84dab97a10b2727cc026e7124a9a5415421` predates these changes; Days 11–14 must audit a new artifact from the final changed head.

The normalizer previously applied strict hosted cross-file checks only to Cholesky. It now applies the same validation flow to the QR target, with its own exact fixture, generator configuration, baseline helper, claim scope, summary scope and non-claims. A Windows QR study triggers hosted validation even if a row tries to downgrade `support_tier` to `local_only`; other-platform local QR reports remain local. Cholesky's expected metadata remains separate. The hosted validator cross-checks row fields against the manifest and summary, including source commit/branch, clean worktree, platform, MSVC compiler, baseline identity, configuration and claims. The selected policy retains its six-row identity and required-file checks.

## Contract correspondence

| Surface | Exact selected value or boundary |
| --- | --- |
| Manifest row | Only QR incompatible row gains the third Windows workflow/job/artifact/platform tuple and `hosted_selected`; other selected rows are unchanged. |
| Windows job | `selected-qr-incompatible-comparison-freshness`, `windows-2022`, 20-minute timeout, explicit PR-head/push-SHA checkout and matching source SHA/branch environment. |
| Generator | Exact QR job/repository/workflow/platform/event/commit/branch/clean-tree gate; local runs retain `local_only`; Windows hosted output uses QR-only fixture-local wording. |
| Upload | `sprint209-windows-selected-comparison-qr-incompatible`, `if-no-files-found: error`, exactly `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv`. |
| Normalizer | `--family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`; hosted QR rows must agree with manifest and summary. |
| Public and maintainer docs | One conditional sentence matching the selected manifest. Broad QR, least-squares, Windows report, rank, package/ABI, performance and state-of-the-art claims remain excluded. |
| Unrelated scope | Cholesky selected proof stays separate. Sprint 220 generated API documentation remains local-only. |

## Frozen negative-fixture matrix for Days 9–10

The rows below specify the mutation, expected observable rejection and owner. An **existing** entry names a currently verified guard; a **planned** entry is the required test/guard work for Days 9–10 and is not counted as a passing gate today.

| Mutation | Expected rejection | Status / owner |
| --- | --- | --- |
| Missing one of six files | `comparison_required_files` with missing QR path, or hosted `field=manifest_missing` / `field=summary_missing` | Existing selected normalizer; Day 9 will add hosted mutations. |
| Renamed file or seventh unexpected file in uploaded/archive set | Exact six-file membership mismatch; reject before freshness promotion | Planned Day 9 artifact fixture; workflow upload guard already rejects extra upload path. |
| Stale or sentinel source SHA | `stale: source_commit does not match current HEAD`, or `comparison_hosted_metadata_mismatch: field=source_commit` | Existing freshness and hosted check; Day 9 mutation. |
| Wrong, duplicate, absent or extra study row ID | `comparison_selected_rows: row_set_mismatch` with `unexpected`, `duplicates` or `missing` | Existing selected policy; Day 9 hosted mutation. |
| Wrong Windows artifact path or a near-match directory | `comparison_artifact_path_mismatch` or selected artifact mismatch | Hosted normalizer path check; Day 9 mutation. |
| Wrong platform or MSVC compiler | `comparison_hosted_metadata_mismatch: field=platform` or `field=compiler` | Day 8 QR hosted regression passes. |
| Wrong baseline name/version/helper/runtime | `comparison_hosted_metadata_mismatch: field=baseline_name`, `field=baseline_version`, `field=baseline_helper_path` or runtime field | Hosted cross-file validator; Day 10 mutation. |
| Hosted Windows row downgraded to `local_only` or conflicting configuration | `comparison_hosted_metadata_mismatch: field=support_tier` or `field=configuration` | Day 8 QR hosted regression covers row downgrade; Day 10 completes bundle mutations. |
| Broad or stale hosted claim, non-claim or summary wording | `comparison_hosted_metadata_mismatch: field=claim_scope`, `field=non_claims` or `field=summary_scope_or_non_claims` | Day 8 QR claim regression; Day 10 cross-file mutation. |
| Wrong QR workflow job, source SHA/branch wiring or checkout ref | `missing selected QR token` / selected workflow `missing` diagnostic; generator `hosted_context_mismatch` | Existing Day 7 structural and source-context guards; Day 10 PR-head fixture. |
| Wrong upload artifact name, missing/duplicate/extra upload path or broad glob | `missing selected QR token`, `upload paths must match the exact selected QR six-file contract` or bounded-path diagnostic | Existing workflow/PowerShell guards; Days 9–10 exercise exact cases. |
| Cholesky evidence substituted for QR | QR selected row-set mismatch, target/fixture mismatch or wrong artifact path | Existing QR target filter and hosted metadata checks; Day 10 substitution fixture. |

## Verification and handoff

The QR hosted positive fixture and first five metadata mutations pass through the normalizer CLI. The selected manifest, generator, workflow, Windows structural and support-doc suites pass. The full normalizer suite and `git diff --check` are included in the Day 8 working-note result. No `.c` or `.h` file changed. Days 9–10 own the planned negative mutations and any discovered guard defects; the Windows QR claim remains conditional until the final changed-head hosted audit.
