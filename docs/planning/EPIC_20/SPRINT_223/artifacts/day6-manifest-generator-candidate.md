# Sprint 223 Day 6: Selected Manifest and Generator Candidate

## Decision and scope

The Day 5 decision permits a **candidate** for one selected Windows Cholesky comparison. This change aligns the source-controlled selected row with the bounded Windows CI lane and makes that lane's generated metadata consistent. It does not accept final Windows freshness: the Day 2 artifact came from the earlier `master` commit, and the final candidate still needs its own successful hosted run, six-file artifact and exact-commit audit on Days 11-12. Day 7 owns public and maintainer documentation alignment.

The only changed selected manifest row is `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` (`cholesky-spd-tridiag-5`). QR, partial SVD, LU, oracle, benchmark and other rows retain their prior Windows exclusions.

## Field-level contract diff

| Field | Before Day 6 | Candidate after Day 6 |
| --- | --- | --- |
| Manifest `workflow_file` | Linux CI; macOS CI | Same entries, then `.github/workflows/windows-ci.yml` |
| Manifest `workflow_job` | `generated-report-freshness`; `selected-comparison-freshness` | Same entries, then Windows `selected-comparison-freshness` |
| Manifest `workflow_artifact` | Linux and macOS selected-comparison artifacts | Same entries, then `sprint190-windows-selected-comparison-cholesky` |
| Manifest `workflow_platforms` | `linux;macos` | `linux;macos;windows`, aligned position by position with workflow/job/artifact |
| Manifest `support_tier` | `local_only` | `hosted_selected` for this selected target contract |
| Manifest `claim_scope` | Fresh for the named fixture against the dense helper | Fresh for the named fixture **on reviewed Linux, macOS and Windows hosted lanes** against the same source-controlled dense helper |
| Manifest Windows non-claim | `no Windows report freshness` | `no broad Windows report freshness` |
| Hosted Windows `study.tsv` tier and `configuration` | `local_only` in each of six rows and in the configuration suffix | `hosted_selected` in each row and configuration suffix, only in the exact Windows CI selected job context |
| Hosted Windows `study.tsv` claim, non-claims and `summary.md` | Fixture-local claim, “local generated study,” `no hosted CI proof`, `no Windows report freshness` | Fixture-local claim qualified to the selected Windows hosted lane; hosted summary sentence; remove the denial of this exact hosted CI proof; retain `no broad Windows report freshness` and all other exclusions |
| Local invocation and unrelated target output | `local_only` and local scope | Remains `local_only` and keeps existing local scope and non-claims |

The generator constructs hosted metadata from a copy of the Cholesky target definition; it does not mutate the global target table. It selects hosted semantics only when the target, GitHub job, Actions flag, repository, workflow name, Windows runner/platform and checked-out commit match the selected CI context. A partial or mismatched selected job context fails with `hosted_context_mismatch` before output is reset. These environment fields are **candidate metadata guards**, not independent authentication of a hosted run; the run/job/artifact audit remains mandatory.

The hosted artifact still excludes broad Cholesky and SPD correctness, reordering and factor-layout identity, CSC parity, fill superiority, external-library parity, package-manager and shared-library ABI proof, release proof, performance superiority, broad platform portability and state-of-the-art claims. Only the denial of the specific hosted CI proof and the absolute Windows freshness denial change for that exact lane.

## Contract and compatibility checks

- The six required filenames remain `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv`; the expected row count remains six and the six Cholesky row IDs retain their prior order.
- The Windows workflow already runs the selected Cholesky command, normalizer and bounded six-file upload. No workflow command or artifact name changed.
- The selected manifest tests enforce the exact Windows tuple, support tier, claim, non-claims and exclusion of other Windows selected rows. A mutation with an extra Windows artifact fails.
- Generator tests exercise exact selected hosted context, stale SHA and partial-context rejection, local/unrelated-target isolation and narrow hosted wording. The existing local comparison suite continues to pass.
- Corpus schema, workflow, PowerShell structural and normalizer tests pass on the candidate. `pwsh` is unavailable on this Mac, so native PowerShell parsing and actual MSVC execution remain hosted gates.

## Open gates

1. Day 7 must align public, maintainer and planning wording with the bounded candidate while preserving historical deferral records as history.
2. Days 8-10 must finish negative guards for missing/renamed/extra files, wrong path, wrong row set and mismatched generated metadata.
3. Days 11-12 must obtain a successful Windows selected job and downloadable six-file artifact from the exact final commit, then verify run SHA/branch/event, clean worktree, six hashes, all row fields and compiler/platform. A pull-request merge SHA must be distinguished from the branch head when auditing provenance.

Until those gates pass, `hosted_selected` in the candidate manifest is a proposed contract, not a released or publicly established Windows freshness claim. A contradiction in the final generated artifact returns the candidate to re-deferral.
