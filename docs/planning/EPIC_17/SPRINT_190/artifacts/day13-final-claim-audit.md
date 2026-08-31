# Sprint 190 Day 13: Final Claim and Residual Audit

## Purpose

Audit the Sprint 190 implementation against the Day 4 decision record, project
plan, public claim boundaries, workflow contract, selected-target manifest
boundary, and residual queue before closeout.

## Decision Record Audit

| Day 4 Contract Surface | Day 13 Status | Evidence |
| --- | --- | --- |
| Target ID `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | Staged | Source manifest row remains Linux/macOS only; future exact Windows metadata allowlist is covered by tests. |
| Target key `cholesky-spd-tridiag-5` | Implemented | Windows workflow generator and freshness commands use the exact selected target. |
| Workflow file `.github/workflows/windows-ci.yml` | Implemented | New `selected-comparison-freshness` job is present in the Windows workflow. |
| Workflow job `selected-comparison-freshness` | Implemented | Job runs on `windows-2022` with `timeout-minutes: 20`. |
| CMake/MSVC build path | Implemented | Job configures with Visual Studio 17 2022, builds `sparse_lu_ortho`, and passes the `.lib` path to the generator. |
| Generator command | Implemented | Workflow runs `python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake` with Windows CMake probe arguments. |
| Freshness command | Implemented | Workflow runs target-specific normalization with `--selected-target cholesky-spd-tridiag-5`. |
| Workflow artifact `sprint190-windows-selected-comparison-cholesky` | Implemented | Upload step uses the exact artifact name with `if-no-files-found: error`. |
| Required files | Implemented | Upload path lists `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv` under the selected Cholesky artifact root. |
| Expected rows `6` | Implemented | Manifest row, report generator, normalizer tests, and Day 11/12 generated evidence all agree on six rows. |
| Manifest `windows` platform promotion | Not promoted | Source selected-target manifest still omits `windows`, preserving the staged promotion boundary until hosted evidence is reviewed. |
| Hosted Windows pass evidence | Pending | Local checks cannot observe GitHub-hosted `windows-2022` results. |

## Guard Audit

| Guard Surface | Status |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Allows only the selected Cholesky Windows workflow path and rejects unowned selected report freshness commands or artifacts. |
| `tests/test_validate_windows_powershell.py` | Covers hosted validation wiring, selected PowerShell snippets, local `pwsh` unavailable semantics, and strict `--require-pwsh` failure behavior. |
| `tests/test_selected_comparison_workflow.py` | Verifies exact Windows job, target, CMake probe mode, artifact name, required file list, timeout, and non-claim boundaries. |
| `tests/test_selected_report_targets_manifest.py` | Keeps current manifest `windows` metadata absent and tests the future exact Cholesky-only Windows allowlist. |
| `scripts/normalize_report_index.py` | Validates selected-target row sets, source freshness, required generated files, and stale-output failures. |
| `scripts/run_external_comparison.py` | Supports the CMake probe path used by the Windows workflow and local Day 11/12 evidence. |

## Documentation Claim Audit

The public and maintainer docs consistently state the bounded outcome:

- Windows has one bounded selected Cholesky comparison freshness workflow path
  for `cholesky-spd-tridiag-5`.
- The source manifest still does not list `windows` until selected Cholesky
  metadata, support tier, and claim boundaries are reviewed together.
- Broad Windows report freshness remains unsupported.
- Windows selected oracle freshness and selected benchmark freshness remain
  unsupported.
- The PowerShell validation lane is structural validation evidence, not
  generated report freshness evidence.
- Package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, broad platform parity, performance superiority, and
  state-of-the-art status remain non-claims.

No overbroad Windows freshness claim was found in README, INSTALL, maintainer
guide, corpus docs, schema docs, or Sprint 190 planning artifacts during the
Day 13 scan.

## Workflow Audit

The Windows workflow matches the selected contract:

- `selected-comparison-freshness` runs on `windows-2022`.
- The job has `timeout-minutes: 20`.
- CMake configure/build steps use `pwsh`.
- The report generator and freshness commands use `cmd`, avoiding Bash and
  Makefile assumptions.
- The generator is limited to `cholesky-spd-tridiag-5`.
- The freshness check is target-specific.
- Upload uses the exact Sprint 190 artifact name.
- Upload paths are explicit files, not broad directories.
- `if-no-files-found: error` is set.

## Residual Decision

`R186-WIN-REPORT-FRESHNESS` is **renewed and narrowed**, not closed.

Sprint 190 closes the product uncertainty around candidate selection and wires
the smallest credible hosted workflow path. It does not close the residual
because this local pass cannot observe hosted `windows-2022` execution and the
source selected-target manifest still omits `windows`.

The residual queue now records the narrowed closure target:

1. review hosted `selected-comparison-freshness` evidence for
   `cholesky-spd-tridiag-5`;
2. promote exactly the Cholesky selected-target row to `windows` metadata if
   hosted evidence passes and claim boundaries are aligned;
3. otherwise retain the staged boundary with the refreshed blockers from
   Sprint 190.

## Day 14 Closeout Checklist

1. Create the Sprint 190 closeout artifact.
2. Re-run the final Day 12 validation set.
3. Confirm local `pwsh` unavailable semantics or hosted evidence status.
4. Confirm no `.c` or `.h` files changed before deciding whether the full C
   gate is required.
5. Confirm generated `build/` comparison evidence remains ignored.
6. Confirm `R186-WIN-REPORT-FRESHNESS` remains renewed and narrowed in the
   residual queue.
7. Prepare retrospective inputs for commit, push, and pull request.

## Validation

Commands run during Day 13:

- `python3 tests/test_selected_comparison_workflow.py`
- `python3 scripts/validate_windows_powershell.py`
- `rg` audits across workflow, selected manifest, tests, public docs,
  maintainer docs, schema docs, and Sprint 190 artifacts.

`python3 tests/test_selected_comparison_workflow.py` passed.

`python3 scripts/validate_windows_powershell.py` completed all structural
checks and then reported `UNAVAILABLE: pwsh not found`; this is expected local
unavailable evidence and not hosted Windows pass evidence.

Day 13 changed planning documentation only. No `.c` or `.h` files were
modified, so the full `make format && make lint && make test` C gate is not
required.

