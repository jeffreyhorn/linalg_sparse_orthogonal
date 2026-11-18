# Sprint 224 Day 4: QR Numerical and Generator Provenance Audit

## Decision boundary

The [Day 2 Windows QR artifact](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts/14181192246) on source commit `44e3b84dab97a10b2727cc026e7124a9a5415421` passes this fixture's numerical and cross-file provenance audit. Its six rows are coherent with the two observation files, manifest, summary, source-controlled generator and dense reference helper. **This is a hosted collection result, not selected Windows freshness promotion:** every generated QR row is `local_only`, the summary says `no hosted CI proof` and `no Windows report freshness`, and the selected manifest still names Linux/macOS only. Day 5 must apply the promotion criteria and decide whether a QR-only candidate can proceed. A later changed head needs a new hosted artifact.

## Numerical worksheet

The source-controlled `QR_INCOMPATIBLE_LS_ENTRIES` and `qr_external_dense_reference.py` agree on

```text
A = [[1, 0], [0, 1], [1, 1], [2, -1]]
b = [1, -2, 2, 5]
selected expected x = [2, -1]
```

For the expected solution, `A x - b = [1, 1, -1, 0]`, so the residual norm is `sqrt(3) = 1.7320508075688772`, not zero. The columns of `A` are orthogonal to that residual (`Aᵀ(Ax-b)=0`), and the expected solution norm is `sqrt(5) = 2.2360679774997898`. The report's nonzero residual therefore matches the intentionally incompatible least-squares fixture; it is not a failed exact solve.

| Measurement | Project observation / study | Baseline observation / study | Recalculation and limit |
| --- | --- | --- | --- |
| Status | `SPARSE_SUCCESS` | `success` | Exact status values match their observation files and status-only rows. |
| Solution values | `[1.9999999999999996, -1.0000000000000002]` | `[1.9999999999999998, -1]` | Maximum componentwise project–baseline difference `2.2204460492503131e-16` ≤ `1e-10`; both are within `1e-10` of `[2, -1]`. |
| Residual norm | `1.7320508075688772` | `1.7320508075688772` | Recomputed `||Ax-b||₂` is `1.7320508075688772` on each side; reported cross-side delta `0` ≤ `1e-10`. |
| Solution norm | `2.2360679774997894` | `2.2360679774997894` | Recomputed `||x||₂` is `2.2360679774997894` on each side; difference from expected `sqrt(5)` is about `4.44e-16`; reported cross-side delta `0` ≤ `1e-10`. |
| Maximum solution delta row | Same two solution vectors | Same two solution vectors | Recomputed maximum absolute delta `2.2204460492503131e-16` equals the row's delta and is ≤ `1e-10`. |

The report uses `absolute` tolerance for residual norm, solution norm and maximum delta, and `absolute_per_component` for solution values. Status rows use `status_only`. The project and baseline observation TSV values match the corresponding study fields exactly as recorded in the [Day 3 row trace](day3-qr-artifact-row-audit.md). The summary's six metric rows repeat those study values and report six passing, zero non-passing selected rows. These checks establish consistency for this fixture and run only; they do not prove broad QR behavior or numerical ranking against an external package.

## Cross-file provenance matrix

| Field or relationship | Verified Day 4 evidence |
| --- | --- |
| Target and fixture | `manifest.tsv` names `qr-incompatible-ls` and `qr_overdetermined_incompatible_4x2`; `summary.md` repeats both, and all six study rows use the fixture. |
| Source revision and branch | Manifest, summary and every study row name `44e3b84dab97a10b2727cc026e7124a9a5415421` and `master`, matching the Day 2 push run and checkout log. |
| Worktree and timestamp | All three surfaces say `clean`; study and manifest share `2026-11-15T21:18:11+00:00`, before the artifact upload at `21:18:16Z`. |
| Platform and compiler | Manifest, summary and rows agree on `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release`; the selected job log shows the matching CMake/MSVC Release build. |
| Configuration | Manifest and all rows agree on `stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`, which equals `comparison_configuration` for this target's current tier. |
| Baseline identity | Manifest and rows agree on `source-controlled-dense-qr-reference`, `external-process-source-controlled-helper`, `qr_external_dense_reference.py`, its Windows helper path and the exact baseline command; summary repeats the command. This is a source-controlled helper, not NumPy/SciPy or another installed reference package. |
| Python runtime | Manifest and rows agree on the selected Python executable under `C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe` and Python `3.12.10` version string; the baseline command uses that executable. |
| Project command/version | Manifest's `project_probe_command` equals every row's `project_command` and the summary's project command; it names VS 2022 x64 Release and `qr_incompatible_ls_probe.exe`. Manifest and rows agree on project version `2.2.0`. |
| Five generated paths | Manifest's study, summary, project-observation, baseline-observation and dependency paths normalize to the exact five `build/comparison/qr_incompatible_ls/` files read in this bundle. The sixth required file is the manifest itself. |
| Claim fields | All rows exactly match the generator's fixture-local claim scope and non-claims. Summary exactly matches the generator's local scope and non-claims, including `no hosted CI proof` and `no Windows report freshness`. |

All fifteen shared generated provenance fields checked in the six rows match the manifest; the project command, generator configuration, five output paths and generated summary provenance match too. The QR job's default checkout is sufficient for this observed `master` push but does not yet prove source-head checkout semantics on a future pull-request run.

## Dependency and promotion disposition

| Dependency | Required? | Artifact status | Interpretation |
| --- | --- | --- | --- |
| `python3` | Yes | `pass` | Selected interpreter was available for this run; no package-manager inference. |
| `tests/qr_external_dense_reference.py` | Yes | `pass` | Source-controlled helper was available. |
| `numpy` | No | `defer` | Optional package baseline was not selected; no NumPy parity evidence. |
| `scipy` | No | `defer` | Optional package baseline was not selected; no SciPy parity evidence. |

**No numerical or cross-file provenance discrepancy was found for the current local-only artifact.** The unresolved promotion mismatch is intentional and explicit: the source-controlled selected manifest remains Linux/macOS-only; the generated Windows QR rows, configuration and summary still deny hosted/Windows selected freshness. Changing only one of those surfaces would create contradictory evidence. Day 5 must decide whether the successful intake supports an atomic, QR-only candidate; Days 11–14 must recheck the final proposed source revision. This artifact cannot be reused as exact-head proof after source edits.

## Audit commands

The audit parsed the downloaded `study.tsv`, project/baseline observations, `manifest.tsv`, `summary.md` and `dependency_status.tsv` as UTF-8. It reconstructed the source-controlled fixture from `scripts/run_external_comparison.py` and `tests/qr_external_dense_reference.py`, recalculated `Ax-b`, vector norms and project–baseline deltas with finite arithmetic, and compared the report's tolerances and field values. It also checked shared manifest/study fields, summary provenance, generated claim wording, source-controlled helper identity and required/optional dependency states. All Day 4 assertions passed; no source, workflow, manifest or public-claim file changed.
