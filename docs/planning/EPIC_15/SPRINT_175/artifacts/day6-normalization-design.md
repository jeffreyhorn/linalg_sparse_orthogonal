# Day 6: Normalization Design

## Purpose

Convert the Day 5 path and execution audit into a concrete implementation
design for the selected macOS comparison freshness promotion lane, including
workflow changes, expected generated outputs, failure behavior, and focused
validation.

## Selected Implementation Strategy

Sprint 175 will implement **reviewed macOS selected comparison freshness** by
adding a bounded hosted macOS workflow job that runs:

```sh
make report-index-comparison-freshness
```

The same implementation batch should reconcile the existing Linux hosted
selected comparison summary and artifact upload inventory so both Linux and
macOS hosted selected comparison lanes cover the same four selected targets:

- `qr-minnorm`;
- `qr-compatible-ls`;
- `partial-svd-diag6-k2`;
- `lu-nonsym-square-5`.

No changes are required to `scripts/run_external_comparison.py`,
`scripts/normalize_report_index.py`, or the Make target before adding the
macOS workflow lane. Day 5 found the selected runner already uses platform-safe
Python `Path` handling, argv-list subprocess calls, `sys.executable` helper
execution, UTF-8 writes, and LF TSV line endings for the selected macOS lane.

## Workflow Design

### New macOS Job

Add a new job to `.github/workflows/macos-ci.yml`:

```yaml
selected-comparison-freshness:
  name: macOS reviewed selected comparison freshness
  runs-on: macos-latest
  timeout-minutes: 15
  steps:
    - uses: actions/checkout@v4
    - name: Run reviewed selected comparison freshness
      run: make report-index-comparison-freshness
    - name: Summarize reviewed selected comparison freshness
      run: |
        python3 - <<'PY'
        ...
        PY
    - name: Upload reviewed selected comparison freshness artifacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: sprint175-macos-selected-comparison-freshness
        retention-days: 7
        if-no-files-found: error
        path: |
          ...
```

The job should not install LLVM, cppcheck, CMake, Doxygen, lcov, or package
tools. The selected comparison freshness command needs Make, Python 3, the
default C compiler, and the repository sources. Those are already consistent
with the existing Apple Clang reviewed path on `macos-latest`.

### Summary Script Design

The summary script should use the same four-target inventory as the Make
target:

| Label | Directory | Expected selected rows |
| --- | --- | ---: |
| `qr-minnorm` | `build/comparison/qr_minnorm` | 6 |
| `qr-compatible-ls` | `build/comparison/qr_compatible_ls` | 6 |
| `partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2` | 10 |
| `lu-nonsym-square-5` | `build/comparison/lu_nonsym_square_5` | 6 |

The script should:

1. read each target's `study.tsv`, `dependency_status.tsv`, and `manifest.tsv`;
2. count selected rows and pass rows;
3. count dependency statuses;
4. print target label, fixture, selected-row count, pass-row count, dependency
   pass/defer counts, source commit, source branch, and platform metadata;
5. print aggregate `selected_targets=4`, `total_selected_rows=28`, and
   `total_pass_rows=28`.

Failure policy:

- missing `study.tsv`, `dependency_status.tsv`, or `manifest.tsv` should fail
  the summary step naturally through file open errors;
- unexpected row counts should fail the summary step explicitly;
- pass-row counts below expected row counts should fail the summary step
  explicitly;
- missing manifest `platform` or source metadata should fail the summary step
  explicitly.

### Artifact Upload Design

Upload exactly the selected generated comparison files for all four target
directories:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

Use `if-no-files-found: error` and `retention-days: 7`.

The artifact name should identify platform and scope, for example:

```text
sprint175-macos-selected-comparison-freshness
```

## Linux Reconciliation Design

Update `.github/workflows/ci.yml` so the existing Linux reviewed selected
comparison freshness job:

- names all four selected comparison targets in comments;
- includes `("lu-nonsym-square-5", Path("build/comparison/lu_nonsym_square_5"))`
  in the summary target list;
- uploads all six `build/comparison/lu_nonsym_square_5/*` selected files;
- prints aggregate `selected_targets=4`, `total_selected_rows=28`, and
  `total_pass_rows=28` after the Sprint 174 LU addition.

This is a consistency reconciliation, not the selected Sprint 175 platform
promotion. The selected promotion remains the new macOS hosted lane.

## Documentation Design

Update maintained docs after workflow implementation:

| File | Required update |
| --- | --- |
| `README.md` | State that selected comparison freshness is local plus reviewed Linux and macOS hosted selected artifact evidence after the new lane passes. |
| `docs/maintainer_guide.md` | Add macOS selected comparison freshness to the report-index workflow/platform sections and preserve non-claims. |
| `tests/corpus/README.md` | Clarify that selected comparison freshness has reviewed Linux and macOS hosted lanes only for selected artifacts. |
| `benchmarks/README.md` | Keep report-index handoff wording bounded if it mentions hosted selected comparison freshness. |

Do not update package-manager, shared-library ABI, runtime-loader, broad
platform, release, performance, or state-of-the-art claims except to preserve
explicit non-claims.

## Expected Generated Output And Freshness Behavior

After implementation:

- local `make report-index-comparison-freshness` still regenerates four
  selected local comparison families and checks 32 selected comparison rows
  total: four source-controlled contract rows plus 28 generated rows;
- Linux hosted selected comparison freshness summarizes and uploads all four
  selected comparison families;
- macOS hosted selected comparison freshness summarizes and uploads all four
  selected comparison families;
- generated files remain ignored under `build/comparison/*` in source control;
- hosted artifacts are retained by workflow artifact upload, not committed.

## Unsupported Platform And Deferral Policy

Windows selected report freshness remains unsupported by this design.

The implementation must keep these non-claims visible:

- no Windows report freshness;
- no Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- no hosted publication of all generated reports;
- no hosted generated API HTML;
- no broad report-index freshness;
- no unselected comparison family freshness;
- no broad solver correctness or external-library parity;
- no package-manager support;
- no shared-library ABI support;
- no runtime-loader behavior;
- no release evidence;
- no performance superiority;
- no state-of-the-art status.

If a future sprint promotes Windows report freshness, it should design a
CMake/PowerShell-native or otherwise reviewed Windows-safe lane separately.

## Focused Validation Plan

Run locally after implementation:

```sh
make report-index-comparison-freshness
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/run_external_comparison.py --self-check
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

Run workflow/static checks:

```sh
python3 - <<'PY'
from pathlib import Path
for path in [
    ".github/workflows/ci.yml",
    ".github/workflows/macos-ci.yml",
]:
    text = Path(path).read_text(encoding="utf-8")
    for needle in [
        "qr_minnorm",
        "qr_compatible_ls",
        "partial_svd_diag6_k2",
        "lu_nonsym_square_5",
    ]:
        if needle not in text:
            raise SystemExit(f"{path} missing {needle}")
PY
```

Run claim/deferral checks if docs are changed:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

If `.c` or `.h` files are modified, run:

```sh
make format && make lint && make test
```

## Day 6 Completion Record

- Implementation choices are documented before workflow edits.
- Promotion and remaining deferral outcomes are enforceable.
- The validation scope matches the selected macOS comparison freshness lane.
- Linux hosted selected comparison LU reconciliation is part of the same
  implementation batch for consistency.
