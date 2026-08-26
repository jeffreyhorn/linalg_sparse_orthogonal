# Sprint 181 Day 14: Closeout And Handoff

## Purpose

Day 14 reconciles Sprint 181 deliverables, validation records, residual risks,
retained duplication exceptions, and Sprint 182 handoff inputs.

Sprint 181 goal: centralize selected report target metadata so workflows,
guards, docs, and freshness checks stop duplicating target lists by hand.

## Project-Plan Reconciliation

| Item | Status | Evidence |
| --- | --- | --- |
| 181.1 Target Inventory | Complete | Days 1-3 inventory report target fields, current report surfaces, workflow scopes, docs claim surfaces, and duplicated owner files. |
| 181.2 Manifest Schema | Complete | Days 4-6 define, prototype, validate, and document `tests/corpus/manifests/selected_report_targets.tsv`. |
| 181.3 Guard Refactor | Complete | Days 7-9 move selected oracle/comparison normalizer expectations and selected benchmark checker expectations behind the manifest. |
| 181.4 Workflow Scope Checks | Complete | Day 10 refactors workflow guards to consume manifest workflow fields while preserving exact job and upload-block checks. |
| 181.5 Documentation Alignment | Complete | Day 11 updates README, maintainer guide, benchmark docs, report-index schema docs, and corpus docs to name the selected-target manifest as authority. |
| 181.6 Validation | Complete | Days 12-13 harden diagnostics and run the integrated validation sweep. |

## Final Deliverables

- Canonical selected report target manifest:
  `tests/corpus/manifests/selected_report_targets.tsv`.
- Manifest parser/schema validation in `scripts/validate_corpus_schema.py`.
- Manifest-backed selected oracle/comparison normalizer checks in
  `scripts/normalize_report_index.py`.
- Manifest-backed selected benchmark artifact/support checks in
  `scripts/check_bench_canonical_freshness.py`.
- Manifest-backed workflow guard checks in
  `tests/test_selected_comparison_workflow.py`.
- Selected target manifest regression tests in
  `tests/test_selected_report_targets_manifest.py`.
- Updated report-index, maintainer, README, benchmark, and corpus docs.
- Sprint 181 working notes and Day 1-14 artifacts.

## Authority Model

`tests/corpus/manifests/report_families.tsv` remains the broad report-family
authority for family-level row meanings, freshness policies, default support
tiers, claim scopes, and non-claims.

`tests/corpus/manifests/selected_report_targets.tsv` is now the selected
target authority for:

- selected target identity;
- selected generator commands;
- artifact patterns and required files;
- expected rows and expected row IDs;
- selected workflow files, jobs, platforms, and upload artifact names;
- selected support tiers and freshness policies;
- selected claim scopes, non-claims, owners, and provenance.

## Retained Duplication Exceptions

Some duplication remains intentional:

| Exception | Reason |
| --- | --- |
| Make target names in docs | They are user entry points and should remain directly visible. |
| High-level non-claim wording | Public and maintainer docs need readable support boundaries even though row-level boundaries live in manifests. |
| Exact YAML structure checks | Workflow job boundaries, upload action placement, and `if-no-files-found: error` are guard-owned structure. |
| Oracle solver-family bucket counts | The selected-target manifest owns total rows and fixture keys, not per-solver-family bucket counts. |
| Benchmark methodology fields | The manifest does not yet model workload command, matrix size, repeat semantics, warmup, variance, baseline, threshold, or methodology notes as typed fields. |
| Generated row-name summaries | Docs keep row-name summaries to help maintainers interpret generated `study.tsv` output. |

## Unsupported Claims Preserved

Sprint 181 does not claim:

- broad report-index freshness;
- selected oracle freshness on macOS;
- Windows report freshness;
- unselected oracle/comparison/benchmark freshness;
- package-manager support;
- shared-library ABI support;
- release proof;
- broad platform parity;
- broad external-library parity;
- performance superiority;
- state-of-the-art status.

## Sprint 182 Handoff

Sprint 182 starts from these Windows report freshness inputs:

- Windows report freshness remains explicitly unselected in Sprint 181.
- `.github/workflows/windows-ci.yml` stays CMake-first and package/static
  install scoped; it does not run selected report freshness commands.
- `tests/test_selected_comparison_workflow.py` now rejects selected oracle,
  comparison, and benchmark freshness commands or selected upload artifact
  names in the Windows workflow.
- The selected-target manifest has no Windows `workflow_platforms` value.
- If Sprint 182 promotes one Windows-safe report freshness path, it should add
  or update a selected-target manifest row with explicit Windows workflow
  metadata, exact artifacts, freshness policy, support tier, claim scope, and
  non-claims.
- If Sprint 182 defers Windows report freshness, it should preserve the
  current guard rejection and add a formal deferral record.

## Retrospective Inputs

What worked:

- TSV matched existing corpus manifest conventions and kept review diffs small.
- Incremental guard migration reduced risk: normalizer first, benchmark
  second, workflow guard third, docs last.
- Manifest-backed workflow tests kept exact YAML block checks without broad
  scans.
- Day 13 exposed that benchmark Make freshness and benchmark regression tests
  must run sequentially because they share `build/bench-reports/canonical/`.

What remains:

- Benchmark methodology details need a future schema decision if they should
  move out of `scripts/check_bench_canonical_freshness.py`.
- Oracle per-solver-family bucket counts need a future schema decision if they
  should move out of normalizer compatibility checks.
- Sprint 182 must decide whether Windows gets one selected report freshness
  lane or remains formally deferred.

## Final Validation

Day 13 integrated validation passed:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_normalize_report_index.py`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `make bench-canonical-report-freshness`
- `python3 tests/test_bench_canonical_freshness.py`
- `bash scripts/static_package_deferral_check.sh`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_corpus_schema.py scripts/check_bench_canonical_freshness.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_bench_canonical_freshness.py`
- `git diff --check`

Day 14 closeout validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 181 deliverables are validated and reconciled. | Complete | All six project-plan items are reconciled above; Day 13 integrated validation passed. |
| Selected report target manifest is the documented target-list authority. | Complete | README, maintainer guide, benchmark docs, corpus docs, schema docs, normalizer tests, benchmark tests, and workflow tests reference the selected-target manifest. |
| Sprint 182 can begin from clear Windows report freshness decision inputs. | Complete | Windows remains unselected with guard coverage; promotion or formal deferral inputs are listed. |
