# Day 2: Generated Report Family Inventory

## Purpose

Inventory generated report families, commands, output paths, freshness checks,
CI references, staging status, and current platform assumptions before Sprint
175 selects a cross-platform report freshness promotion or deferral lane.

## Generated Report Command Inventory

| Command | Owner surface | Primary generated output | Freshness check | Current publication boundary |
| --- | --- | --- | --- | --- |
| `make report-index-oracle-freshness` | `Makefile`, `docs/maintainer_guide.md`, `tests/corpus/manifests/report_families.tsv` | `build/corpus/oracle/corpus.oracle.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | local generated output; selected gate mirrored by reviewed Linux hosted CI |
| `make report-index-comparison-freshness` | `Makefile`, `docs/maintainer_guide.md`, `tests/corpus/manifests/report_families.tsv` | `build/comparison/{qr_minnorm,qr_compatible_ls,partial_svd_diag6_k2,lu_nonsym_square_5}/` | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | local generated output; reviewed Linux hosted CI exists but has stale target/upload inventory for the Sprint 174 LU addition |
| `make bench-canonical-report-freshness` | `Makefile`, `benchmarks/README.md`, `docs/maintainer_guide.md` | `build/bench-reports/canonical/` | `python3 scripts/check_bench_canonical_freshness.py --report-dir "$(BENCH_CANONICAL_REPORT_DIR)" --mode local` | selected local benchmark freshness; reviewed Linux hosted selected-performance lane |
| `make api-docs-freshness` | `Makefile`, `README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | `docs/api/html/` under ignored `docs/api/` | `docs-check` plus `bash scripts/check_api_docs_local_only.sh` | guarded local-only generated API HTML; not report freshness promotion evidence |
| `make coverage` | `Makefile`, `tests/corpus/manifests/report_families.tsv` | `coverage/coverage-src.info`, `coverage/html/` | advisory `normalize_report_index.py` coverage freshness checks | local advisory output only |
| `make deadcode-report` | `Makefile`, `tests/corpus/manifests/report_families.tsv` | `build/deadcode/report.tsv` | advisory `normalize_report_index.py` deadcode freshness checks | local advisory output only |
| `make performance-sentinels` | `Makefile`, `tests/corpus/manifests/report_families.tsv` | `build/bench-reports/sentinels/*.tsv` | sentinel report checks and normalized report-index context | local sentinel output only |
| `make large-matrix-guardrails` | `Makefile`, `tests/corpus/manifests/report_families.tsv` | `build/bench-reports/large-matrix-guardrails/index.tsv` | guardrail report checks and normalized report-index context | local guardrail output only |

## Report-Family Manifest Cross-Reference

| Manifest family | Subfamily | Row origin | Support tier | Generator | Artifact pattern | Sprint 175 interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle` | `generated_reference` | `generated_local` | `local_only` | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | selected local oracle rows; Linux hosted mirror exists for selected artifacts |
| `oracle` | `solver_backed` | `generated_local` | `local_only` | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | selected solver-backed oracle rows; Linux hosted mirror exists for selected artifacts |
| `benchmark` | `canonical` | `generated_local` | `local_only` | `make bench-canonical-report` | `build/bench-reports/canonical/index.tsv` | local measurements; selected freshness has a Linux hosted selected-performance lane |
| `sentinel` | `runtime` | `generated_local` | `local_only` | `make performance-sentinels` | `build/bench-reports/sentinels/sentinels.tsv` | local bounded wall-check context, not cross-platform report freshness proof |
| `sentinel` | `advisory` | `generated_local` | `local_only` | `make performance-sentinels` | `build/bench-reports/sentinels/*.tsv` | advisory local measurement context |
| `guardrail` | `large_matrix` | `generated_local` | `local_only` | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/index.tsv` | local structural/guardrail report context |
| `deadcode` | `report` | `generated_local` | `local_only` | `make deadcode-report` | `build/deadcode/report.tsv` | local maintainer-classification report |
| `coverage` | `src` | `generated_local` | `local_only` | `make coverage` | `coverage/coverage-src.info` | local coverage context only |
| `report_index` | `missing_generated` | `generated_local` | `local_only` | `python3 scripts/normalize_report_index.py` | `build/report-index/normalized-index.tsv` | absent-generated signal, not pass evidence |
| `comparison` | `qr_minnorm` | `generated_local` | `local_only` | `python3 scripts/run_external_comparison.py --target qr-minnorm` | `build/comparison/qr_minnorm/study.tsv` | selected local comparison report; Linux hosted mirror uploads artifacts |
| `comparison` | `qr_compatible_ls` | `generated_local` | `local_only` | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | `build/comparison/qr_compatible_ls/study.tsv` | selected local comparison report; Linux hosted mirror uploads artifacts |
| `comparison` | `partial_svd_diag6_k2` | `generated_local` | `local_only` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | selected local comparison report; Linux hosted mirror uploads artifacts |
| `comparison` | `lu_nonsym_square_5` | `generated_local` | `local_only` | `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5` | `build/comparison/lu_nonsym_square_5/study.tsv` | selected local comparison report added in Sprint 174; Linux hosted mirror inventory has not been reconciled yet |
| `ci` | `reviewed_lanes` | `source_controlled` | `reviewed_cross_platform` | `GitHub Actions` | `.github/workflows/*.yml` | CI lane definitions only; absent logs do not prove freshness |

## Generated Output Staging Map

| Output path | Source-controlled? | Ignored/local? | Hosted artifact lane? | Notes |
| --- | --- | --- | --- | --- |
| `build/corpus/oracle/*.tsv` | no | yes | Linux selected oracle artifacts uploaded | Generated by oracle freshness; source-controlled proof is command, tests, manifest, and docs. |
| `build/corpus-reports/*` | no | yes | Linux selected oracle artifacts uploaded | Split oracle reports are reviewer artifacts in Linux hosted lane. |
| `build/comparison/qr_minnorm/*` | no | yes | Linux selected comparison artifacts uploaded | Current hosted inventory includes this target. |
| `build/comparison/qr_compatible_ls/*` | no | yes | Linux selected comparison artifacts uploaded | Current hosted inventory includes this target. |
| `build/comparison/partial_svd_diag6_k2/*` | no | yes | Linux selected comparison artifacts uploaded | Current hosted inventory includes this target. |
| `build/comparison/lu_nonsym_square_5/*` | no | yes | not yet uploaded in existing Linux hosted inventory | Local selected freshness includes this target after Sprint 174; CI summary/upload list needs reconciliation before hosted claim can include it. |
| `build/bench-reports/canonical/*` | no | yes | Linux selected-performance artifacts uploaded | Hosted lane is threshold-free selected metadata, not portable timing proof. |
| `docs/api/html/*` | no | yes | no | Guarded local-only generated API HTML. |
| `coverage/coverage-src.info` | no | yes | no selected hosted freshness lane | Local coverage context only. |
| `build/deadcode/report.tsv` | no | yes | no selected hosted freshness lane | Local maintainer advisory report. |
| `build/report-index/normalized-index.tsv` | no | yes | no broad hosted freshness lane | Navigation/index output, not release proof. |

## CI References

### Linux Reviewed Hosted Oracle/Comparison Freshness

`.github/workflows/ci.yml` has a `generated-report-freshness` job on
`ubuntu-latest`. It runs:

- `make report-index-oracle-freshness`;
- an oracle summary script;
- artifact upload for selected oracle reports;
- `make report-index-comparison-freshness`;
- a selected comparison summary script;
- artifact upload for selected comparison reports.

The current local comparison target includes four selected comparison families:

- `qr-minnorm`;
- `qr-compatible-ls`;
- `partial-svd-diag6-k2`;
- `lu-nonsym-square-5`.

The existing hosted comparison summary/upload inventory still lists only the
three pre-Sprint-174 targets:

- `qr-minnorm`;
- `qr-compatible-ls`;
- `partial-svd-diag6-k2`.

This is a Day 2 inventory gap. It does not break the local freshness command,
but it means the hosted comparison artifact set and summary wording are stale
relative to the Sprint 174 LU addition.

### Linux Reviewed Hosted Selected Performance Freshness

`.github/workflows/ci.yml` has a `hosted-performance-freshness` job on
`ubuntu-latest`. It uses hosted metadata variables and runs selected canonical
benchmark freshness in hosted mode. This lane does not promote broad benchmark
publication, raw timing superiority, package support, ABI support, or broad
platform support.

## Platform Assumptions By Report Path

| Report path | Linux assumptions | macOS assumptions | Windows assumptions | Day 2 risk |
| --- | --- | --- | --- | --- |
| Oracle freshness | POSIX shell, Make, Python 3, C compiler/static library build, generated `build/` paths | likely feasible but not reviewed as a macOS report-freshness lane | Make and POSIX shell assumptions likely block direct parity; CMake-first Windows evidence is separate | medium |
| Comparison freshness | POSIX shell, Make, Python 3, C compiler/static library build, temporary C probes, helper scripts, generated `build/` paths | likely feasible but not reviewed as a macOS report-freshness lane | Make, POSIX shell, temporary C probe compiler invocation, executable suffix/path behavior, and helper invocation need audit | high |
| Canonical benchmark freshness | POSIX shell script, Make, benchmark binaries, CPU metadata, generated `build/bench-reports/` paths | likely feasible but timing/metadata semantics differ; not reviewed as macOS selected-performance lane | Make/POSIX shell and benchmark executable behavior need audit; portable timing claims excluded | high |
| Generated API freshness | Doxygen, Python 3, Bash local-only guard, ignored `docs/api/` tree | likely feasible if Doxygen/Bash available; local-only only | Bash/Doxygen availability and path semantics need audit; not report freshness evidence | medium |
| Coverage | lcov/gcov/gcovr, compiler-specific coverage flags, generated `coverage/` paths | backend differs; maintainer guide already documents Apple Clang/GCC differences | not selected; compiler/coverage tooling differs | high |
| Deadcode | local analysis tools and generated `build/deadcode/` paths | not reviewed as report freshness lane | not reviewed as report freshness lane | medium |
| Report index normalization | Python 3, generated local artifacts under `build/` | platform-neutral when required generated artifacts exist | platform-neutral when required generated artifacts exist | low |

## Candidate Lanes Made Visible By Day 2

1. **Linux hosted comparison reconciliation for Sprint 174 LU.**
   - Value: closes stale hosted summary/upload inventory after Sprint 174.
   - Risk: low to medium; still Linux, not cross-platform beyond Linux.
   - Limitation: does not satisfy macOS or Windows promotion by itself.

2. **macOS selected comparison freshness supplemental/reviewed lane.**
   - Value: likely closest cross-platform report freshness promotion because
     macOS already has reviewed static-first package/install evidence.
   - Risk: medium; needs Make, compiler, Python helper, temp probe, and artifact
     path validation on macOS.

3. **Windows selected comparison freshness deferral or CMake-first promotion
   design.**
   - Value: high if closed, because Windows currently has CMake-first evidence
     but no selected report freshness lane.
   - Risk: high; Make/POSIX shell and temporary C probe assumptions likely need
     normalization or formal blocker documentation.

4. **macOS or Windows oracle freshness.**
   - Value: useful selected report path.
   - Risk: medium to high; requires generated oracle command and report-index
     freshness audit on the chosen platform.

5. **Generated API freshness CI lane.**
   - Value: documentation freshness.
   - Risk: medium; not a report freshness promotion and should not be selected
     unless Sprint 175 explicitly changes scope.

## Day 2 Completion Record

- Generated report commands, Make targets, scripts, and CI references are
  visible.
- Generated output paths are classified as ignored local output unless an
  explicit Linux hosted selected lane exists.
- The report-family manifest has an owner, command, artifact pattern, support
  tier, and non-claim record for every selected generated report family.
- Platform assumptions are listed before Sprint 175 selects a promotion or
  deferral lane.
