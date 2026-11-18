# Sprint 224 Day 3: QR Artifact Bytes and Row Traceability

## Audit scope and result

The [Day 2 Windows QR run](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057) produced [artifact 14181192246](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts/14181192246) on `master` commit `44e3b84dab97a10b2727cc026e7124a9a5415421`, which is the local `sprint-224` HEAD at intake. Day 3 reread the downloaded ZIP and its extracted files without changing their bytes. **The ZIP, six file identities, six ordered row IDs, row statuses, source fields, timestamps, artifact paths and observation-to-study values agree.** This structural audit does not promote selected Windows freshness: the rows remain `local_only`, the summary retains its Windows non-claim, and a final changed head needs a new hosted run.

## Archive and file hashes

The artifact API reports `sha256:09755ddbdc837e2b5587e15a200836c47c3ae6bc105fd3a25cdc8989a6baa69b` for the 4620-byte ZIP. The downloaded `/tmp/sprint224-day2-qr.zip` has the same SHA-256. Its six flat members are exactly the `SRT-COMP-QR-INCOMPATIBLE-LS` required-file set; there are no missing, duplicate, extra or wrong-subfamily names. Each extracted `/tmp/sprint224-day2-qr/` file matches its ZIP member byte for byte.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 492 | `a1ef353db22039bba57ba90ec5ec89d58c9f5ba26079179e492a3f0f7843e5c7` |
| `baseline_observations.tsv` | 502 | `eb43a2d6932783690c8fd5eabe7a8b54cca5d8c66c2ab1d3de5885fc0e880aa6` |
| `dependency_status.tsv` | 475 | `6da38e9d835253a3ac377403447c647b6674647ef58433a43ea08716055afc38` |
| `study.tsv` | 12632 | `5764b145ef9c5e80c47f73429cb11b724687c5a4c533b62a74a7707a354bb27d` |
| `summary.md` | 2441 | `d0fea4be3f97a00648a2ae31bba0e3781840c38d31f59e9b68f5331720b64e5e` |
| `manifest.tsv` | 1857 | `38a0d954a985bfebc5ee4e5b31aaa1452e93716af652448bb3386138a37acd71` |

## Exact row set and source fields

The six `comparison_row_id` values occur once each in the exact order in the source-controlled selected manifest. Every row has `report_family=comparison`, `subfamily=qr_incompatible_ls`, `fixture_key=qr_overdetermined_incompatible_4x2`, `operation=least_squares_solve`, and `status=pass`. Every row also has `source_commit=44e3b84dab97a10b2727cc026e7124a9a5415421`, `source_branch=master`, `worktree_state=clean`, `generated_at_utc=2026-11-15T21:18:11+00:00`, and `support_tier=local_only`.

Each row's `artifact_path` is `build\comparison\qr_incompatible_ls\study.tsv`. After normalizing Windows separators, it equals the selected manifest's `build/comparison/qr_incompatible_ls/study.tsv`. The generated timestamp matches `manifest.tsv` and precedes artifact creation at `2026-11-15T21:18:16Z`. The manifest names the same target, fixture, source commit, branch and clean worktree.

## Row-to-observation trace

The status and value checks below compare exact TSV strings. Both observation files have four rows, all with the selected fixture key and `status=pass`. The `project_vs_baseline_max_abs_delta` row intentionally reuses the two `solution_values` observation rows.

| Ordered study row ID | Study status reason | Project observation | Baseline observation | Exact study value match |
| --- | --- | --- | --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | `project_status_match` | `project_status`: `SPARSE_SUCCESS` | None; study baseline value empty | Yes |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | `baseline_status_success` | None; study project value empty | `baseline_status`: `success` | Yes |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | `project_baseline_residual_delta_within_tolerance` | `residual_norm`: `1.7320508075688772` | `baseline_residual_norm`: `1.7320508075688772` | Yes |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | `project_baseline_solution_norm_delta_within_tolerance` | `solution_norm`: `2.2360679774997894` | `baseline_solution_norm`: `2.2360679774997894` | Yes |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | `project_baseline_solution_values_delta_within_tolerance` | `solution_values`: `1.9999999999999996,-1.0000000000000002` | `baseline_solution_values`: `1.9999999999999998,-1` | Yes |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | `project_baseline_max_abs_delta_within_tolerance` | Same project `solution_values` | Same baseline `baseline_solution_values` | Yes |

All six row IDs match the source-controlled selected manifest in the order shown. Study status reasons match the metric-specific generator reasons. The observations are separate project and source-controlled dense-reference outputs; no Cholesky file contributes to this trace.

## Discrepancy and handoff log

| Question | Day 3 result |
| --- | --- |
| ZIP hash versus artifact API digest | Match. |
| Six required names and extracted bytes | Exact match; no missing, duplicate, extra or wrong-subfamily file. |
| Six row IDs and order versus selected manifest | Exact match; no missing, duplicate, extra or wrong-target row. |
| Row statuses, reasons, commit, branch, worktree, timestamp and path | All coherent for this `master` push run. |
| Row values versus project and baseline observation files | Exact string matches for all six rows. |
| Selected Windows promotion | **Still blocked:** rows say `local_only`, summary denies hosted/Windows freshness, and final changed head has no new artifact. |

Day 4 owns numerical tolerance recalculation, baseline/runtime and full manifest/summary provenance reconciliation. Day 5 owns the promotion decision. Day 3 made no source, manifest, workflow or public-claim change.
