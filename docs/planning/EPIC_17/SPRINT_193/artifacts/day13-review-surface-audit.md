# Day 13 Review-Surface Audit Artifact

## Scope

Day 13 audited the selected QR external-reference rank/nullspace/threshold
cluster after extraction into `tests/test_qr_external_ref_helpers.h`.

The audit checked whether the selected review surface is smaller and clearer
without changing public APIs, production sources, QR tolerances, or external
reference success-path behavior.

## Before/After Metrics

| Measure | Before Sprint 193 branch edits | Current branch state | Result |
| --- | ---: | ---: | --- |
| `tests/test_qr.c` line count | 3970 | 3040 | Main QR proof owner reduced by 930 lines |
| Selected helper line count | 0 | 1004 | Selected cluster isolated in a family-local helper |
| `test_qr` registered tests | 77 | 79 | Existing selected tests preserved; 2 reader failure tests added |
| Selected helper-owned tests/readers | 0 | 8 moved selected tests, 2 reader helpers, and 2 reader failure tests | Reviewable boundary is explicit |
| Production source changes under `src/` | 0 | 0 | No production algorithm change |
| Public header changes under `include/` | 0 | 0 | No API or ABI surface change |
| Library source-list count | 49 | 49 | Source manifest ownership unchanged |
| New guard ownership | 0 | `make qr-external-ref-helper-guard` plus Python fixtures | Boundary is mechanically checked |

## Boundary Consistency

| Boundary | Audit Result |
| --- | --- |
| QR proof owner | `tests/test_qr.c` still owns `main`, `RUN_TEST(...)` registration, and the selected economy body |
| Extracted helper | `tests/test_qr_external_ref_helpers.h` owns selected rank/nullspace/threshold reader helpers and selected moved tests |
| Build ownership | `test_qr` remains the build target; the helper is not added to Make/CMake library source lists |
| Maintainer docs | `docs/maintainer_guide.md` records the same helper/proof-owner split and forced rebuild caveat |
| Guard checks | `scripts/check_qr_external_ref_helper_guard.sh` checks helper presence, selected registrations, source-list absence, and docs markers |
| Guard tests | `tests/test_qr_external_ref_helper_guard.py` covers positive and negative guard behavior |

## Diff-Risk Register

| Risk Area | Audit Result |
| --- | --- |
| Public API/ABI | No `include/` changes were present |
| Production behavior | No `src/` changes were present |
| QR numeric tolerances | No tolerance-policy or rank-threshold policy change was introduced |
| Selected success-path tests | Selected rank/nullspace/threshold tests remain registered through `test_qr` |
| Reader diagnostics | Invalid-argument and unsupported-fixture reader paths are now explicitly tested |
| Build-system churn | Makefile change only adds the helper guard target; CMake test registration remains unchanged |
| Generated artifacts | No generated report or build artifacts were added |
| Broad refactor risk | Extraction stayed within the selected QR external-reference cluster |

## Validation Evidence

Day 11 integrated validation passed:

- `make source-list-check`
- `python3 tests/test_qr_external_ref_helper_guard.py && make qr-external-ref-helper-guard`
- forced rebuild and execution of `./build/test_qr`
- `make quality-review-cmake-compile`

Day 12 full quality gate passed:

- `make format && make lint && make test`
- follow-up `make source-list-check`
- follow-up guard regression tests and `make qr-external-ref-helper-guard`
- `git diff --check`

Day 13 reran the guard and whitespace checks after adding this audit record.

## Review-Surface Reduction

The selected cluster is now reviewable as a named, family-local helper instead
of being embedded inside the large QR proof owner. Reviewers can inspect the
selected external-reference reader helpers and rank/nullspace/threshold tests in
one file while still checking registration from `tests/test_qr.c`.

The extraction reduces the size of `tests/test_qr.c` by 930 lines and adds a
mechanical guard that keeps the selected boundary from drifting back into the
main QR test file or into library source manifests.

## Residuals and Deferred Candidates

- `test_qr_external_dense_reference_economy_projector_5x3` remains in
  `tests/test_qr.c` by Sprint 193 scope decision.
- Other large QR/economy/sparse-mode/refinement clusters remain candidates for
  future review-surface sprints only after separate boundary review.
- Header-only QR helper edits still require a forced rebuild for focused
  `test_qr` validation; the maintainer guide and guard documentation record
  that caveat.
- Day 14 still owns final closeout and handoff confirmation.
