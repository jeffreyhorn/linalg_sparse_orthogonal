# Sprint 166 Day 2: Generated API And Report Evidence Inventory

## Purpose

Day 2 inventories the Epic 14 generated documentation, generated API, hosted
oracle/comparison, and report-index evidence surfaces. The goal is to identify
source-backed evidence owners and claim boundaries before Sprint 166 runs final
validation, reconciles hosted CI, or recalibrates public claims.

## Source Inputs

| Source | Day 2 use |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_157/RETROSPECTIVE.md` | Baseline/evidence-contract source and quality-surface owner. |
| `docs/planning/EPIC_14/SPRINT_158/RETROSPECTIVE.md` | Generated API HTML publication closure and local-only Doxygen policy. |
| `docs/planning/EPIC_14/SPRINT_159/RETROSPECTIVE.md` | Hosted oracle/comparison freshness promotion and hosted Linux evidence lane. |
| `docs/planning/EPIC_14/SPRINT_160/RETROSPECTIVE.md` | QR comparison family expansion to the second selected QR comparison family. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day10-quality-surface-map.md` | Validation-by-touched-surface baseline. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day11-claim-target-register.md` | Accepted and rejected Epic 14 claim-target baseline. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day14-closeout-handoff.md` | Generated API closeout and Sprint 159 hosted-report handoff. |
| `docs/planning/EPIC_14/SPRINT_159/artifacts/day14-closeout.md` | Hosted freshness closeout and artifact publication evidence. |
| `docs/planning/EPIC_14/SPRINT_160/artifacts/day14-closeout.md` | Two-family selected QR comparison closeout. |
| `Makefile` | Current generated docs/report command owners. |
| `.github/workflows/ci.yml` | Current reviewed hosted oracle/comparison freshness lane. |
| `tests/corpus/manifests/report_families.tsv` | Source-controlled report-family ownership, support tier, freshness policy, claim scope, and non-claims. |

## Sprint 157 Evidence Contract Inventory

| Evidence area | Sprint 157 close state | Sprint 166 use |
| --- | --- | --- |
| Baseline inventory | Inventoried code, public API, tests, CI, documentation, generated artifacts, package, platform, ABI, quality, residual, target, claim, and risk surfaces. | Use as the top-level evidence inventory baseline. |
| Quality surface map | Defines validation by touched surface and requires the full C gate for `.c` and public `.h` changes. | Use to decide Sprint 166 Day 4-6 validation scope. |
| Claim target register | Separates accepted target claims from rejected unsupported claim families. | Use as the initial public-claim audit baseline. |
| Generated artifact boundary | Distinguishes checked-in metadata from ignored local generated outputs. | Preserve during generated API/report claim recalibration. |
| Carry-forward targets | Names Sprints 158-166 complete-gap targets. | Use for project-plan reconciliation. |

Sprint 157 does not provide generated API or hosted report proof itself. It
provides the evidence contract and stop conditions for later generated
documentation/report work.

## Sprint 158 Generated API Inventory

| Surface | Evidence owner | Current status | Claim boundary |
| --- | --- | --- | --- |
| Doxygen generated API HTML | `make docs`, `Doxyfile`, ignored `docs/api/` output | Local-only generated output. | Not committed, not hosted, not release evidence. |
| Page coverage guard | `scripts/check_api_docs_coverage.py`, `make api-docs-coverage` | Checks generated reference/source pages for configured checked-in public headers. | Covers configured checked-in public-header input set only. |
| Aggregate docs check | `make docs-check` | Runs Doxygen and page coverage. | Current only for the branch/checkout where the command passed. |
| Public API authority | checked-in public headers and `docs/api_reference.md` | Source-header-first API ownership. | Generated HTML does not replace source declarations. |
| Generated version header | `include/sparse_version.h.in`, install validation | Install-artifact policy row. | Not an expected Doxygen page under current checked-in-header input set. |

Sprint 158 closed the ambiguity around generated API HTML by selecting
local-only ignored output with deterministic regeneration and coverage checks.
It did not publish hosted Doxygen HTML.

## Sprint 159 Hosted Oracle/Comparison Inventory

| Surface | Evidence owner | Current status | Claim boundary |
| --- | --- | --- | --- |
| Selected oracle freshness | `make report-index-oracle-freshness` | Regenerates selected QR/partial-SVD oracle output and runs strict oracle freshness normalization. | Fixture-local selected hosted/local freshness only; no broad solver correctness or external parity. |
| Oracle generator | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Maintained generator command behind the Make target. | Generator output alone is not pass evidence without strict freshness normalization. |
| Oracle normalizer | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | Strict selected oracle row-state check. | Missing, stale, failed, skipped, deferred, duplicate, unexpected, or incomplete selected rows fail. |
| Hosted Linux lane | `.github/workflows/ci.yml`, job `Linux reviewed hosted oracle/comparison freshness` | Reviewed hosted lane for selected oracle and comparison freshness. | Linux reviewed hosted evidence only; no macOS/Windows report-index parity. |
| Split oracle artifacts | `.github/workflows/ci.yml`, artifact `sprint159-oracle-freshness` | Uploads bounded oracle files with strict missing-file behavior. | Uploaded artifacts are selected evidence plus generated-reference context, not broad report-index publication. |
| Selected comparison freshness | `make report-index-comparison-freshness` | Current target regenerates selected comparison output and runs strict comparison freshness normalization. | Local selected generated comparison freshness; hosted claim must match actual artifact coverage. |
| Comparison artifacts | `.github/workflows/ci.yml`, artifact `sprint159-comparison-qr-minnorm` | Hosted artifact name and summarizer remain QR-minnorm-oriented. | Needs Sprint 166 reconciliation against current multi-family selected comparison target. |

Sprint 159 promoted a reviewed Linux hosted freshness lane for selected
oracle/comparison evidence. It did not promote broad report-index freshness,
macOS/Windows report-index parity, optional NumPy/SciPy pass evidence,
package/ABI evidence, performance evidence, release proof, or state-of-the-art
claims.

## Sprint 160 QR Comparison Inventory

| Surface | Evidence owner | Current status | Claim boundary |
| --- | --- | --- | --- |
| `qr-minnorm` comparison | `scripts/run_external_comparison.py --target qr-minnorm`, `build/comparison/qr_minnorm/study.tsv` | Selected QR minimum-norm fixture comparison. | Fixture-local only; no broad QR or external-library parity. |
| `qr-compatible-ls` comparison | `scripts/run_external_comparison.py --target qr-compatible-ls`, `build/comparison/qr_compatible_ls/study.tsv` | Selected QR compatible least-squares fixture comparison. | Fixture-local only; no raw basis, orientation, global rank-threshold, or broad solve claim. |
| Descriptor-backed runner | `scripts/run_external_comparison.py` | Shared target descriptor model for selected comparison targets. | Runner targets are selected evidence, not broad parity. |
| Comparison freshness gate | `make report-index-comparison-freshness` | Current Make target runs `qr-minnorm`, `qr-compatible-ls`, and later `partial-svd-diag6-k2` before strict freshness normalization. | Local selected comparison freshness unless hosted lane/artifacts prove the same selected surface. |
| Normalizer tests | `tests/test_normalize_report_index.py` | Covers selected row complete, missing, unexpected, duplicate, stale, fail, skip, and defer behavior. | Row-state correctness only, not solver broadness. |
| Runner tests | `tests/test_run_external_comparison.py` | Covers CLI target dispatch, generated files, row IDs, metadata, support tier, and optional dependency context. | External process comparison harness behavior only. |

Sprint 160 added one QR comparison family and kept both selected QR comparison
families fixture-local and `local_only`.

## Current Command Owner Map

| Command | Owner surface | Generated artifacts | Evidence level |
| --- | --- | --- | --- |
| `make docs-check` | generated API docs | ignored Doxygen HTML under `docs/api/` | local-only generated API freshness/page coverage |
| `make api-docs-coverage` | generated API page coverage | generated Doxygen index/reference/source pages | local-only page-coverage check |
| `make report-index-oracle-freshness` | selected oracle freshness | `build/corpus/oracle/*.tsv`, `build/corpus-reports/*` | local selected oracle freshness; reviewed Linux hosted lane runs this command |
| `make report-index-comparison-freshness` | selected comparison freshness | `build/comparison/*` selected study outputs | local selected comparison freshness; hosted lane needs reconciliation with current selected surface |
| `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | oracle row-state normalizer | optional normalized diagnostics | strict selected oracle freshness semantics |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | comparison row-state normalizer | optional normalized diagnostics | strict selected comparison freshness semantics |
| `python3 tests/test_normalize_report_index.py` | normalizer regression tests | none committed | local Python test evidence |
| `python3 tests/test_run_external_comparison.py` | external comparison runner regression tests | temporary generated outputs | local Python test evidence |

## Report-Family Source-Controlled Rows

`tests/corpus/manifests/report_families.tsv` is the current source-controlled
owner for report-family semantics. Relevant Day 2 rows include:

| Family | Subfamily | Generator command | Artifact pattern | Support tier | Key non-claim |
| --- | --- | --- | --- | --- | --- |
| `oracle` | `generated_reference` | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | `local_only` | no hosted CI proof by metadata alone; no broad corpus or external parity |
| `oracle` | `solver_backed` | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` | `local_only` | no broad QR or partial-SVD correctness |
| `comparison` | `qr_minnorm` | `python3 scripts/run_external_comparison.py --target qr-minnorm` | `build/comparison/qr_minnorm/study.tsv` | `local_only` | no broad QR, NumPy, SciPy, LAPACK, SuiteSparse, Eigen, hosted, release, platform, package, ABI, performance, or state-of-the-art claim |
| `comparison` | `qr_compatible_ls` | `python3 scripts/run_external_comparison.py --target qr-compatible-ls` | `build/comparison/qr_compatible_ls/study.tsv` | `local_only` | no broad QR parity, raw QR basis identity, Q sign/orientation, global rank-threshold, broad solve, hosted, release, package, ABI, performance, or state-of-the-art claim |
| `comparison` | `partial_svd_diag6_k2` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2/study.tsv` | `local_only` | no broad SVD correctness, raw singular-vector identity, external-library parity, hosted, release, platform, package, ABI, performance, or state-of-the-art claim |
| `report_index` | `missing_generated` | `python3 scripts/normalize_report_index.py` | `build/report-index/normalized-index.tsv` | `local_only` | no pass evidence, freshness proof, completeness claim, or release proof |

The source-controlled rows define ownership and claim boundaries. They are not
generated pass evidence by themselves.

## Stale Or Missing Evidence Links For Reconciliation

| Item | Evidence observed | Sprint 166 action |
| --- | --- | --- |
| Hosted comparison artifact naming | `.github/workflows/ci.yml` still names the hosted comparison step and artifact around QR minimum-norm (`sprint159-comparison-qr-minnorm`) while `make report-index-comparison-freshness` now regenerates QR min-norm, QR compatible LS, and partial-SVD selected comparison output. | Day 7 CI reconciliation should decide whether hosted summaries/artifacts need renaming or explicit scope text before final claims cite hosted comparison evidence. |
| Hosted comparison artifact contents | The workflow uploads only `build/comparison/qr_minnorm/*` files in the Sprint 159 artifact block. | Day 7 should verify whether current hosted evidence covers only QR min-norm artifacts or whether later workflow changes are needed for QR compatible LS and partial-SVD artifacts. |
| Generated API HTML hosted publication | Sprint 158 selected local-only ignored Doxygen HTML. | Final claim audit must keep generated API HTML as local-only unless a new hosted publication lane is added. |
| Generated `sparse_version.h` API page | Sprint 158 records it as install-artifact policy, not a Doxygen input page. | Final API evidence inventory must avoid claiming generated-version-header Doxygen page coverage. |
| Broad report-index freshness | Sprint 159 rejected broad `normalize_report_index.py --check-freshness` as a hosted promotion target without selected family filters. | Final validation may run broad checks for diagnostics, but public claims should cite selected gates or explicitly advisory rows. |
| macOS/Windows report-index parity | Sprint 159 kept report-index hosted evidence Linux reviewed only. | Hosted CI reconciliation must avoid claiming macOS/Windows report-index parity from Linux hosted evidence. |

## Day 3 Handoff

Day 3 should continue the final evidence inventory with partial-SVD comparison,
Windows package decision, methodology-bound performance publication,
public-header/API coherence, and static-first package-boundary evidence. It
should carry forward the hosted comparison artifact-scope mismatch as a CI
reconciliation item for Day 7.

## Validation Notes

Day 2 changed only Sprint 166 planning artifacts. No `.c`, `.h`, source,
script, workflow, or public documentation files were modified, so the full C
quality gate and generated report commands were not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Generated API and report-publication evidence has source-backed owners. | Complete | Inventory maps Doxygen, API coverage, oracle, comparison, normalizer, report-family, and workflow owners. |
| Hosted versus local-only generated evidence is distinguished. | Complete | Hosted Linux oracle/comparison lane is separated from local-only generated rows and source-controlled advisory metadata. |
| Missing evidence is recorded before claim audit begins. | Complete | Reconciliation register records hosted comparison artifact-scope, generated API hosted publication, generated version header, broad report freshness, and macOS/Windows report-index parity gaps. |
