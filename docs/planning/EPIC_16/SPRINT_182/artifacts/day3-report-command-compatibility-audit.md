# Sprint 182 Day 3: Report Command Compatibility Audit

## Purpose

Day 3 audits selected report freshness command internals for Windows execution
risk. The audit classifies each selected command as a promotion candidate,
possible with refactor, deferral candidate, or out-of-scope before Sprint 182
selects a Windows report freshness path.

## Selected Command Compatibility Matrix

| Command | Current role | Windows compatibility findings | Classification |
| --- | --- | --- | --- |
| `make report-index-oracle-freshness` | Selected Linux oracle freshness wrapper. | Depends on Makefile execution, `$(LIB)`, `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`, and `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`. Windows does not currently claim Makefile parity. | Possible with refactor; not directly promotable. |
| `scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Underlying selected oracle generator. | Uses `pathlib`, explicit `newline=""`, and LF TSV output, which are favorable. Solver probes still default to `CC` or `cc`, link `build/libsparse_lu_ortho.a`, pass `-lm`, emit extensionless temp executables, and compile with Unix-style compiler arguments. | Possible with CMake/MSVC-aware probe refactor. |
| `make report-index-comparison-freshness` | Selected Linux/macOS comparison freshness wrapper. | Depends on unreviewed Windows Makefile behavior but delegates generation to four direct Python target invocations plus report-index normalization. | Possible with refactor; wrapper itself is not directly promotable. |
| `scripts/run_external_comparison.py --target qr-minnorm` | Selected QR minimum-norm comparison generator. | Source-controlled dense QR helper and Python TSV writing are favorable. Project probe still assumes `cc`, Unix archive, `-lm`, extensionless executable, and fallback `make` build of the library. | Leading possible-with-refactor candidate. |
| `scripts/run_external_comparison.py --target qr-compatible-ls` | Selected QR compatible least-squares comparison generator. | Same Python/direct-helper advantages and the same Unix project-probe assumptions as `qr-minnorm`. | Leading possible-with-refactor candidate. |
| `scripts/run_external_comparison.py --target partial-svd-diag6-k2` | Selected partial-SVD diagonal top-k comparison generator. | Same project-probe assumptions; baseline helper remains source-controlled Python rather than optional NumPy/SciPy. | Possible with refactor. |
| `scripts/run_external_comparison.py --target lu-nonsym-square-5` | Selected LU nonsymmetric square-solve comparison generator. | Same project-probe assumptions; baseline helper remains source-controlled Python rather than optional external packages. | Possible with refactor. |
| `make bench-canonical-report-freshness` | Selected Linux canonical benchmark freshness wrapper. | Depends on Makefile benchmark builds, Bash report generation, benchmark executable paths, and Python checker. | Deferral candidate for Sprint 182 unless a Windows-native report generator is created. |
| `scripts/bench_canonical_report.sh` | Canonical benchmark report generator. | Requires Bash, `set -euo pipefail`, POSIX redirection, `date`, `git`, `uname`, `${CC:-cc} --version`, `head`, `otool`/`ldd`, `grep`, `basename`, here-doc manifest output, and extensionless benchmark binaries. | Deferral candidate. |
| `scripts/check_bench_canonical_freshness.py` | Canonical benchmark freshness checker. | Uses Python/pathlib and manifest/TSV checks, so the checker is more portable than the generator. It still depends on generated benchmark artifacts existing first. | Support tool only; not a standalone promotion path. |

## Shell And Executable Risks

| Risk | Affected commands | Windows implication |
| --- | --- | --- |
| Makefile wrapper | Oracle, comparison, benchmark selected freshness wrappers. | Current Windows workflow explicitly does not claim Makefile parity. |
| POSIX shell script | Benchmark generator. | `scripts/bench_canonical_report.sh` is not portable to the current PowerShell lane without rewrite or Bash enablement. |
| Default `cc` compiler discovery | Oracle and comparison Python probes, benchmark metadata. | Windows CMake lane proves MSVC through CMake, not a Unix-like `cc` command. |
| Unix static archive path | Oracle and comparison Python probes. | Probes expect `build/libsparse_lu_ortho.a`; Windows install/build proof produces `.lib` artifacts through CMake. |
| `-lm` link flag | Oracle and comparison Python probes. | MSVC does not use the Unix math-library link model. |
| Extensionless temp executables | Oracle and comparison Python probes, benchmark binaries. | Windows command execution normally needs `.exe` paths unless the invocation layer handles suffixes deliberately. |
| POSIX metadata commands | Benchmark generator. | `uname`, `otool`, `ldd`, `grep`, `head`, and `/proc`-style assumptions do not map to the current Windows runner contract. |

## Path And Newline Findings

The Python report writers are generally favorable for portability:

- `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, and
  `scripts/normalize_report_index.py` use `pathlib` for repository-relative
  paths.
- TSV writers use `newline=""` and `lineterminator="\n"` for stable LF output.
- `scripts/run_external_comparison.py` writes source-controlled outputs under
  deterministic `build/comparison/...` directories and removes previous files
  from those directories before generation.
- `scripts/check_bench_canonical_freshness.py` resolves relative report
  directories beneath the repository root and checks required artifacts by
  manifest-derived names.

The remaining path risk is not TSV writing; it is the command layer that
builds and executes probes or benchmarks.

## Dependency Findings

| Dependency | Current command behavior | Windows decision impact |
| --- | --- | --- |
| Python interpreter | Selected hosted lanes use `python3`; comparison helpers use `sys.executable` internally. | Windows must prove the exact executable name or use a setup step before promotion. |
| Source-controlled dense helpers | Comparison targets use QR, SVD, and LU helper scripts from `tests/`. | Favorable for Windows because they avoid external baseline packages. |
| NumPy/SciPy | Comparison dependency rows mark these optional packages as `defer`, not pass evidence. | Favorable; Windows promotion should preserve no external-library parity claim. |
| C compiler/linker | Oracle/comparison probes assume Unix-style compiler arguments. | Requires refactor to CMake/MSVC-aware build or Windows-specific probe command. |
| Benchmark executables | Benchmark path builds and runs four benchmark binaries. | Higher runtime and portability risk than selected comparison. |

## Advisory And Out-Of-Scope Surfaces

`tests/corpus/manifests/report_families.tsv` includes report-index,
coverage, dead-code, sentinel, guardrail, package, CI, documentation, and
runtime-backend rows. These rows should not be treated as Windows report
freshness promotion candidates in Sprint 182.

| Surface | Day 3 classification |
| --- | --- |
| report-index generated/missing rows | Guard and normalization support only. |
| coverage | Out of scope for Windows selected report freshness. |
| dead-code report | Out of scope; Windows dead-code flow remains staged/deferred. |
| sentinel and guardrail reports | Out of scope unless separately selected in a future sprint. |
| package/static install | Already covered by Windows CMake install/downstream proof; not generated report freshness. |
| CI and documentation | Claim-boundary surfaces only. |
| runtime-backend governance | Documentation/advisory surface only. |

## Preliminary Candidate Ranking

| Rank | Path | Rationale |
| --- | --- | --- |
| 1 | Selected comparison direct Python target, likely one QR target first. | Best chance because dependencies are source-controlled, artifacts are already exact, and the row scope is small. Needs probe build refactor or Windows-specific invocation. |
| 2 | Selected oracle freshness. | Similar Python/TSV strengths, but broader selected row surface and solver probe assumptions make it heavier than one comparison target. |
| 3 | Formal Windows deferral. | Valid if Day 4-6 confirm that refactoring a Windows-safe probe path exceeds Sprint 182 scope. |
| 4 | Selected benchmark freshness. | Highest risk due to Bash generator, metadata commands, benchmark runtime, and performance-claim sensitivity. |

## Day 3 Decisions

- Do not promote any current Makefile wrapper directly to Windows.
- Carry selected comparison direct Python invocation into deeper evaluation as
  the leading possible promotion path.
- Treat benchmark freshness as a deferral candidate unless a Windows-native
  generator is explicitly implemented.
- Keep advisory and unselected report-family rows outside Sprint 182 Windows
  promotion selection.

## Day 4 Handoff

Day 4 should audit generated artifact and data semantics for the leading
paths:

- selected comparison direct Python outputs under `build/comparison/...`;
- selected oracle outputs under `build/corpus/oracle/`,
  `build/corpus-reports/`, and `build/report-index/`;
- benchmark canonical outputs under `build/bench-reports/canonical/`;
- manifest `platform`, compiler, artifact, required-file, expected-row, and
  newline behavior on any Windows candidate.

## Validation

Day 3 is documentation-only. Validation:

- `python3 -m py_compile scripts/run_corpus_oracle.py scripts/run_external_comparison.py scripts/normalize_report_index.py scripts/check_bench_canonical_freshness.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every current selected report freshness command has a Windows risk assessment. | Complete | Selected command compatibility matrix. |
| Advisory or unselected report families remain separate from promotion candidates. | Complete | Advisory and out-of-scope surface table. |
| At least one promotion or deferral path is ready for deeper evaluation. | Complete | Preliminary ranking carries selected comparison direct Python invocation forward and keeps formal deferral open. |
