# Sprint 190 Day 3: Feasibility Probe

## Purpose

Probe whether the selected Cholesky comparison candidate can credibly become a
Windows selected report freshness lane, and decide whether Sprint 190 should
continue toward promotion or pivot to renewed deferral.

## Selected Candidate

| Field | Value |
| --- | --- |
| Manifest row | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Target key | `cholesky-spd-tridiag-5` |
| Generator | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| Expected rows | `6` |
| Report directory | `build/comparison/cholesky_spd_tridiag_5/` |
| Study file | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Claim scope | Fixture-local Cholesky SPD tridiagonal solve comparison only |

## Local Probe Result

Command:

```sh
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5
```

Result: passed locally.

Generated files:

| File | Role |
| --- | --- |
| `project_observations.tsv` | Project probe observations. |
| `baseline_observations.tsv` | Source-controlled dense Cholesky reference observations. |
| `dependency_status.tsv` | Required dependency status rows. |
| `study.tsv` | Six selected comparison rows. |
| `summary.md` | Human-readable comparison summary. |
| `manifest.tsv` | Source, platform, compiler, command, and artifact metadata. |

The generated `study.tsv` contains six selected rows plus the header, matching
the current manifest expectation.

## Positive Feasibility Evidence

- The Cholesky candidate has a precise, small output shape.
- The selected row count is six, not a broad family.
- The report bundle is contained under one comparison subdirectory.
- The baseline helper is source-controlled: `tests/chol_external_dense_reference.py`.
- The claim scope and non-claims are already narrow enough for a one-lane
  Windows decision if hosted evidence is added later.

## Blocking Evidence

The local generated manifest confirms the current project-probe path is not
Windows-safe:

- compiler defaults to `cc`;
- library fallback can call `make`;
- link input is `build/libsparse_lu_ortho.a`;
- link command uses `-lm`;
- output executable is extensionless;
- project command executes the extensionless probe directly.

The local run also records `platform=darwin-x86_64` and `worktree_state=dirty`,
so it is feasibility evidence only. It is not Windows freshness evidence.

## Windows Promotion Requirements

Promotion remains feasible only if Sprint 190 adds an exact Windows-safe
Cholesky probe path:

1. Configure and build with CMake/MSVC on `windows-2022`.
2. Generate only the Cholesky comparison target.
3. Compile the generated probe through a reviewed CMake/MSVC path.
4. Link against the reviewed static `.lib` output rather than the Unix archive.
5. Execute the generated probe with `.exe`-aware path handling.
6. Validate the six expected rows, expected row IDs, source commit, platform,
   and artifact path.
7. Upload only the selected Cholesky comparison bundle with
   `if-no-files-found: error`.

## Initial Artifact Contract

If promotion proceeds, use this provisional Windows artifact contract:

| Field | Provisional value |
| --- | --- |
| Workflow file | `.github/workflows/windows-ci.yml` |
| Workflow job | `selected-comparison-freshness` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Uploaded paths | The six files under `build/comparison/cholesky_spd_tridiag_5/` |
| Row count | `6` |
| Platform metadata | `windows` only for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |

## Decision Checkpoint

Continue toward promotion for one more implementation-design step.

This is not a promotion decision yet. The sprint should pivot to renewed
deferral if Day 4/Day 5 cannot define and implement a Windows-safe CMake/MSVC
generated-probe contract.

## Fallback Deferral Triggers

Renew the formal deferral if any of these remain unresolved:

- generated comparison probes still rely on `cc`, `make`, `.a`, or `-lm`;
- probe execution remains extensionless rather than `.exe` aware;
- hosted Windows workflow cannot produce and upload the exact Cholesky bundle;
- guards cannot allow exactly one Windows selected comparison lane without
  opening broad Windows report freshness.

## Validation

Commands run:

- `git status --short --branch`
- `python3 scripts/run_external_comparison.py --target
  cholesky-spd-tridiag-5`
- `sed -n '1,80p'
  build/comparison/cholesky_spd_tridiag_5/manifest.tsv`
- `sed -n '1,20p'
  build/comparison/cholesky_spd_tridiag_5/study.tsv`
- `wc -l build/comparison/cholesky_spd_tridiag_5/*.tsv
  build/comparison/cholesky_spd_tridiag_5/summary.md`

The probe generated ignored local report output under `build/`. Day 3 changed
only planning documentation, so `make format && make lint && make test` is not
required.
