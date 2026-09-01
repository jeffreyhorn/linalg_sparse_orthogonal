# Sprint 191 Day 14 Closeout and Handoff

## Scope Closed

Sprint 191 closes with exactly one additional bounded external comparison
family: `qr-incompatible-ls`.

The family compares the fixture-local
`qr_overdetermined_incompatible_4x2` least-squares solve against the
source-controlled dense QR reference helper. It is local-only evidence and does
not expand QR platform, package, ABI, performance, release, or broad
least-squares support claims.

## Implemented Surfaces

| Surface | Closeout state |
| --- | --- |
| Runner target | `scripts/run_external_comparison.py` owns `qr-incompatible-ls` with nonzero expected residual semantics. |
| Fixture evidence | The target uses `qr_overdetermined_incompatible_4x2` and records project, baseline, residual, solution norm, solution values, and max-delta rows. |
| Aggregate freshness | `make report-index-comparison-freshness` regenerates the QR incompatible artifacts with the other selected local comparison families. |
| Normalizer | Target-specific freshness diagnostics include the six QR incompatible generated rows. |
| Manifests | `report_families.tsv` and `selected_report_targets.tsv` record the family, target, artifact path, and expected row set. |
| Workflows | Linux and macOS selected comparison artifact uploads include exact QR incompatible paths; Windows remains outside this target. |
| Tests | Runner, normalizer, selected workflow, selected target manifest, schema, docs guard, and QR owner checks cover the added family. |
| Docs | README, cookbook, maintainer guide, solver selection, corpus README, schema docs, and docs guard describe the expanded fixture-local comparison set without broader claims. |

## Final Generated Evidence

Final local generation wrote the expected QR incompatible artifacts:

- `build/comparison/qr_incompatible_ls/project_observations.tsv`
- `build/comparison/qr_incompatible_ls/baseline_observations.tsv`
- `build/comparison/qr_incompatible_ls/dependency_status.tsv`
- `build/comparison/qr_incompatible_ls/study.tsv`
- `build/comparison/qr_incompatible_ls/summary.md`
- `build/comparison/qr_incompatible_ls/manifest.tsv`

The final `study.tsv` contained six rows, all `pass`:

| Metric | Result |
| --- | --- |
| `project_status` | `SPARSE_SUCCESS` matched expected project status. |
| `baseline_status` | Source-controlled helper reported `success`. |
| `residual_norm` | Project and baseline both reported `1.7320508075688772`. |
| `solution_norm` | Project and baseline agreed within tolerance. |
| `solution_values` | Project and baseline agreed within tolerance for `2,-1`. |
| `project_vs_baseline_max_abs_delta` | Maximum absolute solution delta was within `<=1e-10`. |

The final manifest identified:

- `target=qr-incompatible-ls`;
- `fixture_key=qr_overdetermined_incompatible_4x2`;
- `baseline_helper_path=tests/qr_external_dense_reference.py`;
- `baseline_type=external-process-source-controlled-helper`;
- `configuration=stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`;
- `source_branch=sprint-191`;
- `worktree_state=dirty`.

The dependency status file kept optional package baselines deferred:

| Dependency | Status | Required | Interpretation |
| --- | --- | --- | --- |
| `python3` | `pass` | `yes` | Selected interpreter available. |
| `tests/qr_external_dense_reference.py` | `pass` | `yes` | Source-controlled dense reference helper available. |
| `numpy` | `defer` | `no` | Optional package baseline was not selected and is not pass evidence. |
| `scipy` | `defer` | `no` | Optional package baseline was not selected and is not pass evidence. |

## Final Validation

Commands run on Day 14:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
make build/test_qr_solve
./build/test_qr_solve
python3 tests/test_run_external_comparison.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
python3 tests/test_normalize_report_index.py
bash scripts/check_qr_header_docs_guard.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
rg -n 'five fixture|five selected|five generated|minimum-norm and compatible|QR minimum-norm and compatible|compatible least-squares rows from|selected generated comparisons for `qr_underdetermined_minnorm_2x4` and `qr_overdetermined_compatible_5x3`' README.md INSTALL.md docs/maintainer_guide.md docs/solver_selection.md docs/cookbook.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/check_qr_header_docs_guard.sh
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- direct QR incompatible generator run passed;
- QR solve owner test passed with 19 tests, 0 failures, and 1104 assertions;
- external comparison runner tests passed;
- target-specific QR incompatible freshness passed with the six generated rows
  marked fresh;
- normalizer tests passed;
- QR docs guard passed;
- corpus schema validation passed;
- selected target manifest test passed;
- selected comparison workflow guard passed;
- aggregate selected comparison freshness passed with 46 normalized rows;
- Python syntax compilation passed;
- active-doc stale wording scan returned only the intended cookbook phrase and
  guard assertion;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for this sprint closeout.

An initial parallel invocation of the target-specific freshness command raced
the generator before `study.tsv` existed. The same command was rerun after
generation and passed, so no freshness defect remains from that ordering issue.

## Retained Non-Claims

Sprint 191 does not claim:

- broad QR external-library parity;
- broad least-squares parity;
- Windows selected QR incompatible freshness;
- package-manager dependency proof for NumPy or SciPy;
- ABI, packaging, release, or performance support;
- state-of-the-art sparse linear algebra coverage.

## Residual Queue

1. Windows QR incompatible selected freshness remains deferred until an MSVC
   proof and corresponding manifest/workflow ownership are added.
2. Optional NumPy/SciPy package baselines remain advisory deferred dependencies,
   not pass evidence.
3. Broader QR least-squares comparison families remain future work beyond this
   single incompatible fixture.
4. Generated local comparison artifacts remain ignored build outputs and must be
   regenerated for review evidence.
5. Future external comparison families should reuse the Sprint 191 pattern:
   bounded fixture, explicit non-claims, exact manifest rows, target-specific
   freshness diagnostics, and focused failure coverage.

## PR-Ready Handoff

The branch is ready for retrospective drafting and PR preparation with the
following summary:

- added one local-only QR incompatible least-squares external comparison family;
- kept dependency and support claims source-controlled and fixture-scoped;
- extended selected report manifests and freshness diagnostics;
- added workflow artifact coverage for Linux and macOS;
- calibrated active docs and guard checks to include incompatible
  least-squares evidence without implying broader support;
- validated runner, normalizer, schema, workflow, docs, QR owner, and aggregate
  selected freshness behavior.
