# Sprint 223 Day 3: Cholesky Row and Artifact Traceability

## Audit disposition

**The downloaded Day 2 artifact passes the exact six-row and five-file cross-checks for its `master` source commit. Selected Windows freshness remains unpromoted.** This is a row and artifact audit of [run 42927368282](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282), [job 130217136498](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/job/130217136498) and [artifact 13814213527](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/artifacts/13814213527). It is not the final Sprint 223 commit. The generated `support_tier=local_only` and `no hosted CI proof` / `no Windows report freshness` wording still need Day 4's semantic audit and later atomic correction before any promotion.

## Byte custody and input identity

The six extracted files remain untouched in `/tmp/sprint223-day2-cholesky.csqbWpIx/`. Recomputed SHA-256 values match the [Day 2 intake table](day2-hosted-msvc-run-intake.md):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 470 | `1df661119e985947fb9ab890c81ea3b55554aad16254c15a3f376fa7cc9d9b99` |
| `baseline_observations.tsv` | 497 | `7806a9bf4eb8ada7e537ff53582dce8b5d0d3d35898a69e136db92f0fb2b5b8f` |
| `dependency_status.tsv` | 477 | `53bbd048f54089ef51a0c0de928fd99a51d5b375ac71dc59809d9554769c9726` |
| `study.tsv` | 12827 | `fe688714a23b27634cc282ceb5f62b1e73e84a757e33d91cb741888aa02b82f0` |
| `summary.md` | 2547 | `42cc909bc2dc9e036874e2f7028abad51d3cb61f2273d7f742e7638ca011b07a` |
| `manifest.tsv` | 1873 | `4847d4e01741674813539bae0e9c89ce9c510b154fbbcc99a5333e44b9d18cfa` |

The source-controlled input is the **named synthetic fixture**, not a `.mtx` file inside the artifact. The project generator defines `cholesky-spd-tridiag-5` using `CHOLESKY_SPD_TRIDIAG_5_ENTRIES`, a 5×5 system, RHS `[2,4,6,8,16]`, and oracle `[1,2,3,4,5]` in `scripts/run_external_comparison.py` (SHA-256 `cd6e432c6b736c094374e830c8259bec67904fa8eaa59ea7eb4ca259ea3c0689`). The baseline helper's `build_cholesky_spd_tridiag_5()` matrix is in `tests/chol_external_dense_reference.py` (SHA-256 `475f700795b525a29c9916dc45f52dccf401f998876532db9c3571ecfbc12447`). Both are source files at run head `7fc86594cb45477b58cd29706c5f87e005ca99c1`. The artifact contains summarized observations and command strings, not raw solver stdout/stderr or a source snapshot; the run SHA supplies their source-control identity.

## Exact study row audit

Parsed the selected manifest row `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` with a TSV reader and compared its six `expected_row_ids` **in order** with the six unique `study.tsv` IDs. There are no missing, duplicate, extra or wrong-target rows. Each selected row has `status=pass`, the expected metric-specific status reason, `source_commit=7fc86594cb45477b58cd29706c5f87e005ca99c1`, `source_branch=master`, `worktree_state=clean`, `fixture_key=cholesky_spd_tridiag_5` and `artifact_path=build\comparison\cholesky_spd_tridiag_5\study.tsv`. The Windows separators normalize to the selected path `build/comparison/cholesky_spd_tridiag_5/study.tsv`. All rows also say `platform=windows-amd64` and `compiler=cmake-probe:Visual Studio 17 2022:Release`; Day 4 will reconcile these with generator policy.

| Manifest-owned row suffix | Metric | Status reason | Observation source and check |
| --- | --- | --- | --- |
| `project_status_v1` | `project_status` | `project_status_match` | `project_observations.tsv`: `SPARSE_SUCCESS` equals study project value and expected value. |
| `baseline_status_v1` | `baseline_status` | `baseline_status_success` | `baseline_observations.tsv`: `success` equals study baseline value and expected value. |
| `residual_norm_v1` | `residual_norm` | `project_baseline_residual_delta_within_tolerance` | Project `residual_norm` and baseline `baseline_residual_norm` both equal study values `5.7560540319981793e-15`; study delta `0`. |
| `solution_norm_v1` | `solution_norm` | `project_baseline_solution_norm_delta_within_tolerance` | Project and baseline solution norms both equal study values `7.4161984870956648`; study delta `0`. |
| `solution_values_v1` | `solution_values` | `project_baseline_solution_values_delta_within_tolerance` | Both observation vectors equal their study values `1,2,3.0000000000000009,4.0000000000000009,5.0000000000000009`; study delta `0`. |
| `project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `project_baseline_max_abs_delta_within_tolerance` | Project and baseline vector strings match the two observation files; study delta `0`. |

The six full IDs are the prefix `comparison_cholesky_spd_tridiag_5_` plus the suffixes above. `project_observations.tsv` and `baseline_observations.tsv` each contain exactly four rows with `status=pass`. Their status and metric values match the corresponding `study.tsv` fields. The selected delta rows use the fixed `1e-10` tolerances shown in `study.tsv`; this audit checks report consistency and recorded status, rather than rerunning the Windows solver.

## Cross-file path and dependency trace

`manifest.tsv` contains 23 unique keys. Its `study_path`, `summary_path`, `project_observations_path`, `baseline_observations_path` and `dependency_status_path` normalize to the five files in the selected `build/comparison/cholesky_spd_tridiag_5/` output directory. Its source commit, branch, fixture, platform and compiler values match all six study rows. The manifest's `target` is `cholesky-spd-tridiag-5`; its baseline helper path is `tests/chol_external_dense_reference.py`. `summary.md` names the same target, source commit, branch, platform and compiler, lists the six study metrics, and reports six passing and zero non-passing selected rows.

`dependency_status.tsv` has two required `pass` dependencies (`python3` and the source-controlled dense Cholesky helper) and two optional `defer` dependencies (`numpy`, `scipy`). The deferred packages are not pass evidence and do not turn this into an external package comparison. The helper is the baseline input; `project_observations.tsv` and `baseline_observations.tsv` are the immediate per-side outputs; `study.tsv` is the six-row comparison; `manifest.tsv` links their paths; and `summary.md` narrates the same six rows. No unexpected dependency or artifact file was found.

## Discrepancy log and next gate

**Row/file discrepancies: none.** The observed contradiction is about **claim semantics**, not row freshness: all six rows and `manifest.tsv` still record `support_tier=local_only`, and `summary.md` still says `no hosted CI proof` and `no Windows report freshness` despite this successful hosted run. Those fields are accurately inherited from the current generator/manifest policy and must be reconciled deliberately on Day 4 and atomically on Days 6-8. The selected manifest still excludes Windows, and this `master` artifact does not cover future Sprint 223 source edits. No selected manifest, generator or public claim was changed on Day 3.

Audit method: a read-only Python `csv.DictReader` check compared the selected manifest IDs, all six study status/provenance/path fields, matching project/baseline observation values, unique manifest keys and file paths, and summary row count; it exited successfully. Separate SHA-256 checks recomputed all six extracted-file hashes. Day 4 must inspect generated support-tier/non-claim semantics against the workflow and platform evidence.
