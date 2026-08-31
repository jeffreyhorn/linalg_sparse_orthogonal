# Sprint 190 Day 5: Workflow Scaffold

## Purpose

Define the Windows workflow scaffold for the selected Cholesky comparison
freshness lane without adding a live freshness job before the Windows-safe
probe mode exists.

## Scaffold Decision

Day 5 intentionally does not edit `.github/workflows/windows-ci.yml`.

The current selected comparison generator still depends on Unix-shaped probe
behavior, so adding a live Windows selected freshness job now would create a
workflow that appears to promote evidence the branch cannot yet produce. The
correct scaffold for Day 5 is a source-controlled implementation contract that
Day 6 and Day 7 can turn into executable code and tests.

## Current Windows Workflow Boundary

The Windows workflow currently proves:

- CMake/MSVC configure, build, `ctest -N`, and full `ctest`;
- CMake install/downstream validation for the maintained static-first package
  surface;
- hosted PowerShell validation ownership for selected workflow snippets.

It still does not prove:

- selected comparison freshness;
- selected oracle freshness;
- selected benchmark freshness;
- broad generated report freshness;
- Windows Makefile parity;
- package-manager support;
- shared-library, dynamic ABI, or runtime-loader behavior.

## Proposed Job Shape

When the generator supports a reviewed Windows CMake/MSVC probe path, add this
single job to `.github/workflows/windows-ci.yml`:

```yaml
  selected-comparison-freshness:
    name: Windows selected Cholesky comparison freshness
    runs-on: windows-2022
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4

      - name: Configure reviewed Windows CMake build
        run: cmake -S . -B build -G "Visual Studio 17 2022" -A x64
        shell: pwsh

      - name: Build reviewed Windows CMake library
        run: cmake --build build --config Release
        shell: pwsh

      - name: Run selected Cholesky comparison freshness
        run: >
          python scripts/run_external_comparison.py
          --target cholesky-spd-tridiag-5
          --windows-cmake-build build
          --windows-config Release
        shell: cmd

      - name: Validate selected Cholesky comparison rows
        run: >
          python scripts/normalize_report_index.py
          --family comparison
          --require-generated comparison
          --check-freshness
        shell: cmd

      - name: Upload selected Windows Cholesky comparison freshness
        uses: actions/upload-artifact@v4
        with:
          name: sprint190-windows-selected-comparison-cholesky
          if-no-files-found: error
          path: |
            build/comparison/cholesky_spd_tridiag_5/project_observations.tsv
            build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv
            build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv
            build/comparison/cholesky_spd_tridiag_5/study.tsv
            build/comparison/cholesky_spd_tridiag_5/summary.md
            build/comparison/cholesky_spd_tridiag_5/manifest.tsv
```

The `--windows-cmake-build` and `--windows-config` flags are scaffolded names
for the future generator implementation. They are not present yet.

## Artifact Scope

| Rule | Contract |
| --- | --- |
| Name | `sprint190-windows-selected-comparison-cholesky` |
| Scope | One Cholesky comparison directory only. |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv`. |
| Missing files | Upload must fail with `if-no-files-found: error`. |
| Forbidden broad paths | `build/comparison/**`, `build/**`, and any other comparison subdirectory. |

## Guard Requirements

Before the job can become live:

1. `run_external_comparison.py` must support the Windows-safe Cholesky probe
   path.
2. `tests/test_selected_comparison_workflow.py` must validate the exact
   Windows job, command, artifact, files, and row count.
3. `tests/test_selected_report_targets_manifest.py` must allow `windows` only
   for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
4. `tests/test_validate_windows_powershell.py` must continue rejecting broad
   selected report generation and unowned PowerShell drift.
5. Claim-boundary docs must say this is one selected Cholesky comparison lane,
   not broad Windows report freshness.

## Day 5 Outcome

The scaffold is reviewable and exact, but intentionally not executable yet.
Day 6 should start with manifest/schema choices and decide whether the
freshness validator needs a target-specific comparison mode before the workflow
job can be safely added.

## Validation

Commands run:

- `git status --short --branch`
- `sed -n '166,198p' docs/planning/EPIC_17/SPRINT_190/PLAN.md`
- `sed -n '1,130p' .github/workflows/windows-ci.yml`
- `python3 tests/test_selected_comparison_workflow.py`

Day 5 changed only planning documentation. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

Run `git diff --check` after this artifact is added.
