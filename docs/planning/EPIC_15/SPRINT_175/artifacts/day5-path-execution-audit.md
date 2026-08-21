# Day 5: Path And Execution Assumption Audit

## Purpose

Audit the selected macOS comparison freshness lane before implementation:
command trace, path handling, shell/executable assumptions, temporary probe
behavior, generated output staging, text output behavior, and minimal
normalization required for a reviewed hosted lane.

## Selected Lane

| Field | Value |
| --- | --- |
| Platform | macOS hosted CI |
| Workflow file | `.github/workflows/macos-ci.yml` |
| Command | `make report-index-comparison-freshness` |
| Selected targets | `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, `lu-nonsym-square-5` |
| Generated root | `build/comparison/` |
| Primary proof artifact | each target's `study.tsv` |

## Execution Trace

The selected Make target expands to:

```sh
make report-index-comparison-freshness
```

That target depends on the static library and runs:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
python3 scripts/run_external_comparison.py --target qr-compatible-ls
python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2
python3 scripts/run_external_comparison.py --target lu-nonsym-square-5
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

For each `run_external_comparison.py` target, the runner:

1. resolves the repository root and selected output directory;
2. resets existing files in that output directory;
3. ensures `build/libsparse_lu_ortho.a` exists, building it through `make` if
   needed;
4. builds and runs a temporary C project probe;
5. runs the source-controlled dense-reference helper with the active Python
   executable;
6. writes project observations, baseline observations, dependency status,
   study rows, summary, and manifest files under the selected output
   directory;
7. validates selected row membership and pass status;
8. lets `normalize_report_index.py` require selected generated comparison rows
   and check freshness against source metadata.

## Path Handling Audit

| Surface | Current behavior | macOS interpretation |
| --- | --- | --- |
| Repository root | `Path(__file__).resolve().parents[1]` | portable on hosted macOS. |
| Output directories | target metadata uses `REPO_ROOT / "build" / "comparison" / ...` | portable; all selected outputs are under ignored `build/`. |
| Output reset | creates the target directory and unlinks files/symlinks directly inside it | sufficient for current flat generated output; no recursive cleanup risk. |
| Project probe source | temporary directory via `tempfile.mkdtemp(prefix="sparse-comparison-")` | portable on macOS; path may include `/var/folders/...` but Python passes it as argv. |
| Project probe binary | temp path without extension | acceptable on POSIX/macOS; no `.exe` handling required for selected macOS lane. |
| Include paths | `root / "include"` and `root / "build" / "include"` passed as separate argv elements | portable for paths with spaces because `subprocess.run` uses argv lists. |
| Static library path | default `build/libsparse_lu_ortho.a` | native static archive for macOS Make build; already used by local target. |
| Artifact upload paths | workflow paths must enumerate `build/comparison/<target>/*` files | implementation must include all four target directories, including Sprint 174 LU. |
| Manifest relative paths | uses `Path.relative_to(root)` where possible | portable on macOS; generated manifest stores POSIX-style relative paths because macOS paths are POSIX. |

## Shell And Executable Assumption Audit

| Surface | Current behavior | macOS interpretation |
| --- | --- | --- |
| Make target | standard Make recipe invokes `python3` and static library target | macOS CI already uses Make in reviewed paths; no new shell family needed. |
| Python executable | Make invokes `python3`; runner invokes helpers via `sys.executable` | macOS hosted image includes Python; using `sys.executable` keeps helper invocation consistent inside the runner. |
| Compiler | runner uses `CC` env via `shlex.split(os.environ.get("CC", "cc"))`, defaulting to `cc` | hosted macOS default `cc` is Apple Clang and is already reviewed in macOS workflow. |
| Compiler identity | runs `<cc> --version` and records first line | Apple Clang supports this; identity is metadata only. |
| Probe execution | `subprocess.run([str(binary)], cwd=root, ...)` | valid on macOS POSIX executable path. |
| Baseline helpers | `tests/qr_external_dense_reference.py`, `tests/svd_external_dense_reference.py`, `tests/lu_external_dense_reference.py` invoked as Python scripts | no executable bit or shebang dependency because runner calls `sys.executable helper fixture`. |
| Bash/PowerShell | selected lane does not need Bash-specific script execution beyond Make's default shell | no PowerShell needed; Windows assumptions remain out of scope. |
| CMake | selected comparison freshness does not use CMake | no CMake report path required for macOS selection. |

## Text Output And Newline Audit

| Surface | Current behavior | macOS interpretation |
| --- | --- | --- |
| TSV writing | `path.open("w", encoding="utf-8", newline="")` and `lineterminator="\n"` | deterministic LF TSV output on macOS. |
| TSV reading in summaries | Python `csv.DictReader(..., delimiter="\t")`; existing Linux summary uses `newline=""` | macOS workflow summary should use the same reader pattern. |
| Markdown summary | runner writes UTF-8 Markdown with generated values | portable; not used as parser input for freshness. |
| Helper output | plain `OK <n>` and one numeric value per line | portable; parsed with `splitlines()`. |
| Manifest output | TSV key/value rows with LF terminators | portable; source commit/platform/compiler metadata are text-only. |
| Make output | echo lines are informational | not part of freshness parsing. |

## Generated Output Staging Audit

Generated outputs remain ignored local or hosted artifact outputs:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`
- `build/comparison/qr_compatible_ls/project_observations.tsv`
- `build/comparison/qr_compatible_ls/baseline_observations.tsv`
- `build/comparison/qr_compatible_ls/dependency_status.tsv`
- `build/comparison/qr_compatible_ls/study.tsv`
- `build/comparison/qr_compatible_ls/summary.md`
- `build/comparison/qr_compatible_ls/manifest.tsv`
- `build/comparison/partial_svd_diag6_k2/project_observations.tsv`
- `build/comparison/partial_svd_diag6_k2/baseline_observations.tsv`
- `build/comparison/partial_svd_diag6_k2/dependency_status.tsv`
- `build/comparison/partial_svd_diag6_k2/study.tsv`
- `build/comparison/partial_svd_diag6_k2/summary.md`
- `build/comparison/partial_svd_diag6_k2/manifest.tsv`
- `build/comparison/lu_nonsym_square_5/project_observations.tsv`
- `build/comparison/lu_nonsym_square_5/baseline_observations.tsv`
- `build/comparison/lu_nonsym_square_5/dependency_status.tsv`
- `build/comparison/lu_nonsym_square_5/study.tsv`
- `build/comparison/lu_nonsym_square_5/summary.md`
- `build/comparison/lu_nonsym_square_5/manifest.tsv`

Sprint 175 should upload these as CI artifacts for the selected hosted macOS
lane but must not commit them.

## Minimal Normalization Required

No Python runner path normalization is required before implementing the macOS
lane. The selected runner already uses argv-list subprocess calls, `Path`
objects, UTF-8 writes, LF TSV output, and `sys.executable` for helpers.

Implementation should focus on:

1. adding a macOS workflow job that runs
   `make report-index-comparison-freshness`;
2. setting `CC=cc` explicitly or relying on default Apple Clang while
   documenting the compiler identity captured in generated manifests;
3. summarizing all four selected comparison targets;
4. uploading all six generated files for each of the four selected target
   directories;
5. reconciling the existing Linux hosted selected comparison summary and
   upload list so it also includes `lu-nonsym-square-5`;
6. preserving local-only source control staging for `build/comparison/*`.

## Risks To Carry Into Day 6

| Risk | Mitigation |
| --- | --- |
| macOS job runtime grows because it builds the static library and runs four comparison targets | keep timeout bounded and use only selected comparison freshness, not broader report-index checks. |
| hosted summary diverges between Linux and macOS | use the same four-target inventory in both jobs. |
| artifact upload misses the Sprint 174 LU directory | enumerate all four directories explicitly or use a reviewed bounded glob for selected directories only. |
| documentation overstates macOS support | update docs to say reviewed macOS selected comparison freshness only, not broad macOS platform parity. |
| Windows support gets implied | preserve explicit Windows non-claims and keep Windows out of this selected lane. |

## Day 5 Completion Record

- Every command invoked by the selected freshness lane is traced.
- Path, shell, executable, temporary directory, newline, and generated-output
  assumptions are recorded.
- Minimal normalization work is scoped before workflow implementation.
- Generated output remains ignored local output unless uploaded by the selected
  reviewed hosted macOS lane.
