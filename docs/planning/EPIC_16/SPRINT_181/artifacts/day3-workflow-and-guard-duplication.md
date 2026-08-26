# Sprint 181 Day 3: Workflow And Guard Duplication Audit

## Purpose

Day 3 turns the Day 2 surface inventory into a precise duplication audit. The
goal is to identify which selected report target facts should move to a
manifest, and which facts should remain owned by guards because they describe
workflow structure rather than selected target metadata.

## Scope

Inspected owners:

- `Makefile`
- `scripts/normalize_report_index.py`
- `scripts/check_bench_canonical_freshness.py`
- `tests/test_normalize_report_index.py`
- `tests/test_selected_comparison_workflow.py`
- `tests/test_bench_canonical_freshness.py`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `tests/corpus/manifests/report_families.tsv`
- `tests/corpus/schemas/report_index_fields.md`
- README, maintainer guide, and benchmark docs

## Selected Oracle Duplication

| Duplicated fact | Current owners | Candidate manifest ownership |
| --- | --- | --- |
| Selected command | `Makefile`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, README, maintainer guide, report-index schema docs, Linux workflow | Manifest should own canonical selected oracle target and remediation command. Makefile may remain the executable wrapper. |
| Generator command | `Makefile`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, maintainer guide, report-index schema docs | Manifest should own the command string: `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`. |
| Expected total rows | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, maintainer guide, report-index schema docs | Manifest should own `52`. Normalizer should validate against manifest value. |
| Solver-family row counts | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, maintainer guide, report-index schema docs | Manifest should own partial-SVD `26`, QR `23`, and unknown `3` if kept as selected count contract. |
| Selected fixture keys | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, maintainer guide | Manifest should own selected fixture-key set or a child table keyed by selected oracle target. |
| Artifact paths | `tests/corpus/manifests/report_families.tsv`, `Makefile`, `scripts/normalize_report_index.py`, tests, docs, Linux workflow upload block | Manifest should own generated artifact pattern and required files; workflow guard should keep exact upload-block validation. |

## Selected Comparison Duplication

| Duplicated fact | Current owners | Candidate manifest ownership |
| --- | --- | --- |
| Selected target keys | `Makefile`, `tests/test_selected_comparison_workflow.py`, Linux/macOS workflow inline Python, README, maintainer guide, benchmark docs, report-index schema docs | Manifest should own `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`. |
| Target directories/subfamilies | `scripts/normalize_report_index.py`, `tests/test_selected_comparison_workflow.py`, Linux/macOS workflow upload blocks, benchmark docs, maintainer guide | Manifest should own target key to subfamily/path mapping. |
| Expected rows by target | `tests/test_selected_comparison_workflow.py`, Linux/macOS workflow inline Python, maintainer guide, report-index schema docs | Manifest should own `6`, `6`, `10`, and `6`. |
| Selected row IDs | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | Manifest should own row IDs directly or derive them from target and metric definitions. |
| Required uploaded files | `tests/test_selected_comparison_workflow.py`, Linux/macOS workflow upload blocks | Manifest can own required file names, but guard should continue proving exact upload block scope and `if-no-files-found: error`. |
| Support tiers and non-claims | `tests/corpus/manifests/report_families.tsv`, `tests/test_normalize_report_index.py`, README, maintainer guide, benchmark docs, report-index schema docs | Manifest should own target support tier and non-claim strings or validate against report-family row policy. |

## Selected Performance Duplication

| Duplicated fact | Current owners | Candidate manifest ownership |
| --- | --- | --- |
| Selected artifact | `scripts/check_bench_canonical_freshness.py`, `tests/test_bench_canonical_freshness.py`, README, maintainer guide, benchmark docs, Linux workflow inline Python | Manifest should own `bench_refactor_csc`. |
| Selected command/workload | `scripts/check_bench_canonical_freshness.py`, benchmark docs, maintainer guide, Linux workflow comments | Manifest should own `tests/data/suitesparse/nos4.mtx --repeat 1` and `nos4.mtx`. |
| Expected relative path | `scripts/check_bench_canonical_freshness.py`, benchmark docs, Linux workflow upload block | Manifest should own `bench_refactor_csc.csv`; guard should keep upload block checks exact. |
| Methodology fields | `scripts/check_bench_canonical_freshness.py`, tests, benchmark docs, maintainer guide | Manifest should own stable selected values such as repeat semantics, threshold, baseline, warmup, and variance only if Day 4 chooses to include performance row metadata. |
| Hosted support fields | Linux workflow env, `scripts/check_bench_canonical_freshness.py`, tests, README, maintainer guide, benchmark docs | Manifest should own selected hosted support tier and claim boundary; workflow should remain the executable source for CI env injection. |

## Workflow Job And Upload Scope Notes

| Workflow | Exact current scope | Guard-owned structure |
| --- | --- | --- |
| `.github/workflows/ci.yml` | `generated-report-freshness` runs selected oracle freshness and selected comparison freshness; uploads `sprint159-oracle-freshness` and `sprint175-linux-selected-comparison-freshness`. `hosted-performance-freshness` runs selected canonical performance freshness and uploads `sprint168-selected-performance-freshness`. | Job names, step names, `actions/upload-artifact@v4`, `if-no-files-found: error`, retention, exact upload block boundaries, and absence of unselected report promotion should remain guard-owned. |
| `.github/workflows/macos-ci.yml` | `selected-comparison-freshness` runs selected comparison freshness and uploads `sprint175-macos-selected-comparison-freshness`. | Job name, step name, upload action, fail-closed upload behavior, selected comparison-only claim wording, and macOS non-claims should remain guard-owned. |
| `.github/workflows/windows-ci.yml` | No selected report freshness lane. Windows remains CMake-first with explicit non-claims for Makefile parity, pkg-config execution parity, package-manager behavior, runtime-loader behavior, and broad Windows parity. | Guard should keep Windows report freshness absence/non-claim as a structural/platform boundary until Sprint 182 changes it. |

## Guard/Test Embedded Expectations

| Owner | Embedded expectations | Risk |
| --- | --- | --- |
| `tests/test_selected_comparison_workflow.py` | Selected comparison target tuples, expected rows, required uploaded files, artifact names, workflow job placement, summary failure strings, and macOS non-claim wording. | High drift risk because target facts are copied from workflow inline Python and docs. |
| `tests/test_normalize_report_index.py` | Selected oracle fixture keys, oracle row counts, comparison row IDs, comparison subfamilies, support tiers, freshness policies, artifact patterns, CI row claims, and diagnostic strings. | High drift risk because target identity and normalizer behavior are mixed in one test file. |
| `tests/test_bench_canonical_freshness.py` | Selected benchmark artifact, local/hosted support tiers, claim boundaries, and selected methodology values. | Medium drift risk; scope is only one selected performance row, but constants duplicate checker and docs. |
| `scripts/normalize_report_index.py` | Selected oracle constants, selected comparison row IDs, comparison artifact diagnostics, canonical remediation text, strict/advisory freshness policy sets. | High drift risk and primary Day 8-9 refactor target. |
| `scripts/check_bench_canonical_freshness.py` | Selected performance artifact, command, fixture, relative path, repeat semantics, support tiers, claim boundaries, required columns, and required artifacts. | Medium drift risk; likely needs manifest lookup or manifest validation in later sprint days. |
| `Makefile` | Freshness target commands for selected oracle/comparison/performance. | Low to medium drift risk; may remain executable entry point but should validate against manifest-owned generator commands. |

## Documentation Duplication

| Doc owner | Repeated facts |
| --- | --- |
| README | Selected freshness command names; selected comparison family names; hosted Linux/macOS lane meanings; selected canonical benchmark command; Windows report freshness non-claim. |
| Maintainer guide | Selected oracle expected outputs and row counts; selected comparison target paths and row counts; common freshness commands; diagnostics; selected performance workload and non-claims. |
| Benchmark docs | Selected canonical benchmark artifact/workload; selected comparison artifact directories; report-index handoff paths; hosted performance support tier and claim boundary. |
| Report-index schema docs | Report-index fields; selected oracle and comparison freshness commands; expected selected rows; artifact paths; strict failure semantics; non-claims. |
| INSTALL | Package proof-owner rows and package-manager/shared-library non-claims; should not become a selected generated report proof surface. |

## Duplication Classification

| Classification | Examples | Manifest candidate? |
| --- | --- | --- |
| Target identity | Oracle target, comparison target keys, selected benchmark artifact | Yes |
| Expected counts | Oracle total/family rows; comparison target row counts | Yes |
| Artifact paths | `build/corpus/oracle/*.tsv`, comparison study paths, benchmark canonical files | Yes |
| Required artifact files | Comparison `project_observations.tsv`, `study.tsv`, `manifest.tsv`, benchmark `index.tsv`/`manifest.txt` | Yes, with workflow guard preserving exact upload block scope |
| Generator command | Oracle, comparison, and benchmark freshness commands | Yes |
| Support tier | `local_only`, `reviewed_cross_platform`, `hosted_selected` | Yes |
| Claim scope/non-claims | Selected evidence claim and unsupported interpretations | Yes, or manifest validates existing report-family rows |
| Workflow structure | Job names, step names, upload action, `if-no-files-found`, placement outside validated lane | No, guard-owned |
| Platform boundary | Linux selected oracle/comparison/performance, macOS selected comparison, Windows report non-claim | Partly: manifest can own platform scope; workflow guard owns YAML enforcement |
| Diagnostic wording | Missing rows, stale rows, duplicate rows, upload failures | Partly: manifest can name targets; scripts/tests own diagnostic format |

## Day 4 Handoff

Day 4 should design a schema that can own target facts without flattening
workflow structure into data. The minimum useful schema needs:

- stable target key;
- family and subfamily;
- generator command;
- artifact pattern and required files;
- expected row count;
- support tier and freshness policy;
- hosted platform/job/artifact metadata where selected;
- claim scope and non-claims;
- owner and introduction source;
- duplicate-key rules.

The schema should avoid making docs, workflow comments, or report-family
summary rows substitute for generated proof.

## Validation

Day 3 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| All known duplicate selected target lists have owner files. | Complete | Oracle, comparison, performance, workflow, guard/test, and docs tables above. |
| YAML guard scope boundaries are explicit before refactor. | Complete | Workflow job and upload scope notes distinguish exact YAML structure from manifest-owned data. |
| Manifest-owned fields are distinguished from guard-owned structural checks. | Complete | Duplication classification table and Day 4 handoff separate target facts from workflow structure. |
