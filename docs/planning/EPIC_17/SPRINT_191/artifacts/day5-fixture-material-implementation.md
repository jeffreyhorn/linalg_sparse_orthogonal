# Sprint 191 Day 5: Fixture Material Implementation

## Purpose

Implement the selected deterministic fixture material for `qr-incompatible-ls`
without duplicating already owned source-controlled fixture data or committing
generated scratch output.

## Implemented Fixture Material

Day 5 added the selected fixture to `scripts/run_external_comparison.py` as a
runner-owned target descriptor:

| Field | Implemented value |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Entry constant | `QR_INCOMPATIBLE_LS_ENTRIES` |
| Matrix shape | 4 rows by 2 columns |
| Matrix entries | `(0,0)=1`, `(1,1)=1`, `(2,0)=1`, `(2,1)=1`, `(3,0)=2`, `(3,1)=-1` |
| RHS | `[1.0, -2.0, 2.0, 5.0]` |
| Expected solution | `[2.0, -1.0]` |
| Expected solution norm | `2.2360679774997898` |
| Expected residual norm | `1.7320508075688772` |
| Output directory | `build/comparison/qr_incompatible_ls` |
| Baseline value count | `3` |

The fixture remains handwritten and deterministic. No generated fixture file
or external data source was added.

## Runner Coherence Change

The selected fixture is intentionally inconsistent, so a valid solve has a
nonzero residual. Day 5 added target-level `expected_residual_norm` handling
for solve-style project and baseline observation rows:

- existing compatible solve targets continue to default to expected residual
  `0.0`;
- `qr-incompatible-ls` observation rows pass when residuals match
  `1.7320508075688772` within `1e-10`;
- study rows continue to compare project residual against baseline residual;
- the residual study row now reports the nonzero expected residual value for
  targets that define `expected_residual_norm`.

This keeps `project_observations.tsv`, `baseline_observations.tsv`, and
`study.tsv` internally coherent for an intentionally incompatible
least-squares fixture.

## Target Naming

| Surface | Name |
| --- | --- |
| CLI target | `qr-incompatible-ls` |
| Subfamily | `qr_incompatible_ls` |
| Fixture | `qr_overdetermined_incompatible_4x2` |
| Artifact directory | `build/comparison/qr_incompatible_ls` |
| Summary title | `QR Incompatible Least-Squares External Comparison Study` |

The name is deliberately parallel to `qr-compatible-ls` while clearly marking
the residual-bearing incompatible case.

## Fixture Coherence Checks

Day 5 added `test_qr_incompatible_ls_fixture_contract()` in
`tests/test_run_external_comparison.py`. The test asserts:

- target descriptor entries, row count, and column count;
- RHS values;
- expected solution;
- expected solution norm;
- expected nonzero residual norm;
- baseline value count;
- project observation rows pass when residual equals the expected nonzero
  residual;
- baseline observation rows pass when residual equals the expected nonzero
  residual.

The existing selected-target generation test now also includes
`qr-incompatible-ls`, but report-family metadata checking is deferred until the
manifest/report-index integration days add those source-controlled rows.

## Generated Output Policy

The Day 5 generator smoke test wrote scratch output under:

```text
build/comparison/qr_incompatible_ls/
```

`build/` is ignored by `.gitignore`, and no generated comparison artifacts
were added to source control.

## Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 5.

## Day 6 Handoff

Day 6 should focus on the reference execution path and failure coverage:

- assert `tests/qr_external_dense_reference.py` is the required helper for the
  new target;
- add or extend tests for missing helper, malformed helper output, and
  baseline command failure if not already covered sufficiently;
- confirm dependency rows for `python3`, `tests/qr_external_dense_reference.py`,
  NumPy, and SciPy match the Day 4 policy;
- decide whether any additional parser tests are needed before manifest
  integration.
