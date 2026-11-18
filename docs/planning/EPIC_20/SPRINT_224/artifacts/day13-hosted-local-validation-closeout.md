# Sprint 224 Day 13: Hosted and local validation closeout

## Evidence status and disposition

Item 224.5's hosted recheck is complete for committed proposed head `07b7f831ffee04f90a25f6cd11182b141eb38f07`. The [Day 12 Windows run 43769185219](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219), [selected QR job 132981018706](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219/job/132981018706) and [artifact 14183482977](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219/artifacts/14183482977) are accessible and successful. The run, job, artifact API, manifest, all six study rows and summary name that SHA. The current committed local HEAD and draft PR #248 head also equal it. The artifact API reports `expired=false`, expiry `2027-02-13T22:25:11Z` and ZIP digest `sha256:36cdbf8ba494b1fa531d7ef4016317691e52da7d9266327cabf54d7e69050ec4`.

The named Windows QR incompatible claim is **eligible for draft approval at that committed SHA**, subject to Day 14's final exact-head gate. The Day 12 and Day 13 ledgers are uncommitted branch edits. Once committed, they change the PR head; the current artifact becomes historical for that newer SHA. The Day 13 disposition is therefore **not final approval**. A final-head hosted run, artifact inspection and unchanged QR-only claim check remain required.

## Retained six-file and row linkage

`gh run download 43769185219 -n sprint209-windows-selected-comparison-qr-incompatible` succeeded again on Day 13. Its six extracted files have exactly the SHA-256 hashes in the [Day 12 ledger](day12-final-proposed-head-qr-recheck.md), and no seventh file was present:

| File | Day 13 SHA-256 |
| --- | --- |
| `project_observations.tsv` | `a1ef353db22039bba57ba90ec5ec89d58c9f5ba26079179e492a3f0f7843e5c7` |
| `baseline_observations.tsv` | `eb43a2d6932783690c8fd5eabe7a8b54cca5d8c66c2ab1d3de5885fc0e880aa6` |
| `dependency_status.tsv` | `6da38e9d835253a3ac377403447c647b6674647ef58433a43ea08716055afc38` |
| `study.tsv` | `26673ba972db21c065de112431bd29164efedae8471d3f8e3ba697ef0a6f2d7e` |
| `summary.md` | `d81e8d67bfc677c7908e50702a1f3a10df6eba9b328f469683b08823f7d8a4d3` |
| `manifest.tsv` | `5d3b3ce15ba304322a82dee46442756d38aa7fbf5da539e3ecb8c196e5b1394b` |

The selected manifest requires the ordered six `comparison_qr_overdetermined_incompatible_4x2_{project_status,baseline_status,residual_norm,solution_norm,solution_values,project_vs_baseline_max_abs_delta}_v1` row IDs. The downloaded `study.tsv` contains those six exact IDs in that order, once each, all `pass` and `hosted_selected`. Manifest, study and summary source commits equal `07b7f831ffee04f90a25f6cd11182b141eb38f07`. The [Day 12 ledger](day12-final-proposed-head-qr-recheck.md) records all six IDs individually, file sizes, ZIP digest, row-to-observation trace and numerical audit. The re-downloaded bytes match that ledger, so the selected QR proof remains tied to this revision only.

## Local quality-gate transcript

| Command or check | Day 13 result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | PASS |
| `python3 tests/test_run_external_comparison.py` | PASS; guarded QR generator and local comparison fixture regressions |
| `python3 tests/test_selected_comparison_workflow.py` | PASS; named QR job, exact-head wiring and six-file upload contract |
| `python3 tests/test_normalize_report_index.py` | PASS; full normalizer and QR hosted mutation suite |
| `python3 tests/test_validate_windows_powershell.py` | PASS; Windows workflow structure and bounded claim checks |
| `make support-docs-guard` | PASS |
| `make docs-check` | PASS; Doxygen generated 18 public reference and 18 source pages, API docs coverage passed |
| `python3 scripts/normalize_report_index.py --build-root /tmp/sprint224-day12-audit-build --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | PASS, `freshness ok (17 rows)` with six fresh QR generated rows at committed HEAD |
| `gh run download 43769185219 -n sprint209-windows-selected-comparison-qr-incompatible -D /tmp/sprint224-day13-qr` and independent six-file SHA/row check | PASS; downloadable, same hashes and row/source contract |
| `git diff --check`; `git diff --name-only -- '*.c' '*.h'` | PASS; no whitespace defect and no C/header change, so `make format && make lint && make test` is not triggered by Day 13 |

The selected generator, manifest, workflow, normalizer and docs checks all pass. This local transcript supplements the hosted proof; no local check alone establishes Windows freshness.

## QR-only claim and non-claims

The single proposed sentence remains: “Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.” README, INSTALL, maintainer, corpus and schema surfaces label it conditional pending the final changed-head audit. The source-controlled selected manifest names only `SRT-COMP-QR-INCOMPATIBLE-LS` as the newly owned Windows QR comparison target; the compatible and minimum-norm QR targets remain local-only on Windows.

The retained artifact establishes neither broad QR or least-squares parity, raw QR basis identity, rank-threshold behavior, NumPy/SciPy or other external-package parity, broad Windows report freshness, package-manager or shared-library ABI support, performance superiority, nor state-of-the-art status. Cholesky's separate selected Windows proof is unchanged and is not inferred from this QR artifact. The optional NumPy/SciPy dependency rows are deferred and not pass evidence.

## Blockers and Day 14 gate

No local check, selected job, six-file hash or row check failed on Day 13. The remaining approval blocker is **future source-head drift**: committing the Day 12/13 evidence notes or final decision changes the SHA. Day 14 must apply the Sprint 208 promotion criteria to the final committed head, run the selected QR job for that SHA, download and verify its six-file artifact, and approve only the named target if every field and non-claim still agrees. If that exact-head evidence is unavailable or contradictory, re-defer and state the blocker.
