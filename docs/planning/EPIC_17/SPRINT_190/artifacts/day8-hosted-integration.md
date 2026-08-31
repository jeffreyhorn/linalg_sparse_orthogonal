# Sprint 190 Day 8: Hosted Integration

## Purpose

Wire the selected Cholesky comparison freshness decision into hosted Windows CI
with bounded runtime and exact artifact semantics, while keeping all other
Windows report freshness surfaces deferred.

## Hosted Workflow Lane

Day 8 adds one Windows hosted job to `.github/workflows/windows-ci.yml`:

```yaml
selected-comparison-freshness:
  name: Windows selected Cholesky comparison freshness (MSVC)
  runs-on: windows-2022
  timeout-minutes: 20
```

The job is intentionally independent from broad Windows build/test claims. It
checks out the source, configures the existing CMake static library, builds the
`sparse_lu_ortho` target, generates one Cholesky comparison report, runs the
Day 7 target-specific freshness guard, and uploads only the approved Cholesky
artifact bundle.

## Command Contract

Generator command:

```sh
python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
```

Freshness command:

```sh
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

## Generator Portability

`scripts/run_external_comparison.py` now has an opt-in CMake probe mode. The
default Linux/macOS direct-compiler path remains unchanged. The CMake path is
selected explicitly in the Windows workflow and can also be selected locally
with:

```sh
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake
```

The CMake probe path:

- links the generated probe against an explicit static library path;
- avoids Makefile-only build assumptions;
- avoids Unix-only `-lm` and extensionless executable assumptions on MSVC;
- records the configure/build/run command in report provenance;
- keeps the dense baseline helper source-controlled and package-free.

## Artifact Policy

The only uploaded Windows selected report artifact is:

```text
sprint190-windows-selected-comparison-cholesky
```

The upload is fail-closed with `if-no-files-found: error` and includes exactly:

- `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv`
- `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv`
- `build/comparison/cholesky_spd_tridiag_5/study.tsv`
- `build/comparison/cholesky_spd_tridiag_5/summary.md`
- `build/comparison/cholesky_spd_tridiag_5/manifest.tsv`

Broad paths such as `build/comparison/**` remain forbidden.

## Guard Updates

`scripts/validate_windows_powershell.py` now permits only the exact Sprint 190
Cholesky lane. It still rejects:

- selected oracle freshness on Windows;
- broad selected comparison freshness on Windows;
- selected benchmark freshness on Windows;
- Linux/macOS selected artifact names on Windows;
- any artifact upload outside the selected Cholesky job;
- unowned Windows PowerShell steps.

The selected Cholesky job's CMake configure/build PowerShell snippets are now
owned by the PowerShell parser guard.

## Current Limitation

`tests/corpus/manifests/selected_report_targets.tsv` still omits `windows`.
That means source metadata is not yet promoted to hosted Windows selected
freshness. Day 8 creates the hosted execution path; Day 9 should add
deterministic manifest and drift tests before any metadata claim is widened.

Local success is not hosted Windows evidence. The hosted lane must pass in CI
before documentation can state that Windows selected Cholesky freshness is
reviewed.

## Validation

Commands run:

- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`
- `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`

All focused validation commands passed.

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` C gate is not required for Day 8.
