# Sprint 199 Day 4: Manifest Promotion Decision

## Purpose

Apply the Day 3 evidence threshold to
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` and decide whether the selected target
manifest can promote Windows for the bounded `cholesky-spd-tridiag-5`
comparison freshness lane.

## Decision

Day 4 re-defers manifest promotion.

`tests/corpus/manifests/selected_report_targets.tsv` remains unchanged for
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`:

- `workflow_platforms` remains `linux;macos`;
- `support_tier` remains `local_only`;
- `workflow_file` remains `.github/workflows/ci.yml;.github/workflows/macos-ci.yml`;
- `workflow_job` remains `generated-report-freshness;selected-comparison-freshness`;
- `workflow_artifact` remains
  `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness`;
- `non_claims` retains `no Windows report freshness`.

This is a deliberate re-deferral, not a rejection of the hosted Windows
artifact. The hosted artifact is credible evidence for one exact Windows
Cholesky selected-comparison run, but the source-controlled metadata and tests
do not yet satisfy the promotion threshold.

## Evidence Accepted

The following evidence is accepted as valid promotion input for later days:

| Evidence | Status |
| --- | --- |
| Hosted workflow run `34269219871` on `master` | Accepted |
| Commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` with clean worktree provenance | Accepted |
| Job `Windows selected Cholesky comparison freshness (MSVC)` succeeded | Accepted |
| Artifact `sprint190-windows-selected-comparison-cholesky` was present and non-expired during review | Accepted |
| Artifact contains the six expected files | Accepted |
| `study.tsv` contains the six expected Cholesky row IDs | Accepted |
| All six selected rows have `status=pass` | Accepted |
| Row platform is `windows-amd64` and compiler path is `cmake-probe:Visual Studio 17 2022:Release` | Accepted |

## Promotion Blockers

The selected manifest is not promoted because these blockers remain:

| Blocker | Why it blocks promotion | Owner days |
| --- | --- | --- |
| Windows artifact-path separators | Hosted rows use `build\comparison\cholesky_spd_tridiag_5\study.tsv`; selected manifest metadata uses forward slashes. Promotion should wait for explicit path-normalization tests proving selected artifact filtering handles Windows and mixed separators without false positives. | Days 5-7 |
| Generated `support_tier=local_only` | A manifest platform promotion to Windows would conflict with generated rows that still label the proof local-only unless support semantics are changed or explicitly bounded. | Day 7 and Day 11-Day 12 |
| Generated summary non-claims | The hosted summary still says `no hosted CI proof` and `no Windows report freshness`; public promotion would contradict generated evidence text unless the generator/summary semantics are reconciled. | Day 7 and Day 11-Day 12 |
| Missing/stale diagnostics not yet expanded | Day 4 has not yet added negative tests for missing rows, wrong targets, wrong platforms, stale artifacts, or near-match paths. | Days 5-7 |

## Manifest Impact

No manifest edit was made on Day 4.

The retained manifest state keeps the selected Cholesky comparison freshness
claim authoritative for Linux/macOS only. The Windows workflow remains a
guarded hosted path and does not become promoted selected freshness until the
remaining blockers are resolved in the same source-controlled claim surface.

## Required Future Promotion Shape

A future promotion should update the selected target row only when the
following surfaces agree:

1. `tests/corpus/manifests/selected_report_targets.tsv` lists Windows in the
   exact selected Cholesky row and names the Windows workflow job and artifact.
2. `scripts/normalize_report_index.py` and
   `tests/test_normalize_report_index.py` prove Windows artifact paths, mixed
   separators, missing rows, stale rows, wrong targets, and near-match paths.
3. Generated comparison summary/non-claim wording no longer contradicts the
   hosted Windows selected Cholesky claim.
4. README, INSTALL, corpus documentation, and maintainer guide all use the
   same narrow claim vocabulary.
5. Broad Windows report freshness, Windows selected oracle freshness, Windows
   selected benchmark freshness, other comparison targets, package/ABI,
   performance, release, and state-of-the-art claims remain explicit
   non-claims.

## Day 5 Handoff

Day 5 should start with Windows path-normalization tests for selected
comparison artifact filtering. The first required cases are:

- hosted row path with backslashes;
- manifest artifact pattern with forward slashes;
- mixed-separator path;
- absolute Windows path ending in the selected artifact;
- near-match artifact path that must not pass filtering.
