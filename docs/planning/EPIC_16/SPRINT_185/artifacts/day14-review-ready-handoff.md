# Sprint 185 Day 14: Review-Ready Handoff

## Purpose

Close Sprint 185 by packaging the final LDLT CSC review-surface reduction for
retrospective creation and PR review.

## Project-Plan Outcome

| Item | Sprint 185 outcome |
| --- | --- |
| 185.1 Cluster Selection | Selected exactly one large review surface: `tests/test_ldlt_csc.c`. |
| 185.2 Extraction Design | Designed three family-local helper-header boundaries and a no-behavior-change validation contract. |
| 185.3 Mechanical Extraction | Extracted supernode, fixture/setup, and oracle/native-wrapper helpers into family-local headers included by `tests/test_ldlt_csc.c`. |
| 185.4 Drift Guard Update | Added `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard`. |
| 185.5 Maintenance Note | Drafted `day10-maintenance-invariants.md` and promoted the guidance into `docs/maintainer_guide.md`. |
| 185.6 Validation | Ran focused selected-cluster validation and the full C quality gate. |

## Final File Layout

| File | Final role |
| --- | --- |
| `tests/test_ldlt_csc.c` | Registered LDLT CSC proof-owner binary, public test bodies, `main`, `RUN_TEST(...)` ordering, external dense-reference state, and remaining broad/stateful helpers. |
| `tests/test_ldlt_csc_supernode_helpers.h` | Family-local supernode fixture, snapshot, dense-SPD, and factor-state comparison helpers. |
| `tests/test_ldlt_csc_fixtures.h` | Family-local KKT, scaled-KKT, and analysis-backed two-pass fixture/setup helpers. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Family-local dense-oracle, symmetric-swap, and native-wrapper comparison helpers. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Selected-cluster guard for helper presence, include ownership, and registration boundaries. |
| `docs/maintainer_guide.md` | Discoverable maintainer guidance for the new helper ownership split. |

## Final Review-Surface Size

| Path | Final lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

`tests/test_ldlt_csc.c` was reduced from the Day 3 baseline of 3915 lines to
3469 lines, a 446-line reduction in the selected proof-owner file.

## Final Validation Evidence

Day 12 focused validation passed:

- forced rebuild of `build/test_ldlt_csc`;
- `make build/test_ldlt_csc`;
- `./build/test_ldlt_csc`: 100 tests, 0 failures, 0 skips, 3556 assertions;
- `make ldlt-csc-helper-guard`;
- `make source-list-check`: PASS, 49 library sources;
- `git diff --check`.

Day 13 full validation passed:

- `make format`;
- `make lint`;
- `make test`, ending with `All tests passed.`;
- `make ldlt-csc-helper-guard`;
- `make source-list-check`: PASS, 49 library sources;
- `git diff --check`.

Day 14 closeout validation passed:

- `make ldlt-csc-helper-guard`;
- `make source-list-check`;
- `git diff --check`.

## Scope Review

No production source, public API, internal solver API, CMake registration,
library source manifest, or new test binary was added. The only Makefile
change is the selected-cluster guard target.

The accumulated diff is limited to:

- the selected LDLT CSC test cluster and its family-local helper headers;
- the selected-cluster guard script and Make target;
- maintainer guidance;
- Sprint 185 planning artifacts and working notes.

No stale TODOs, unresolved open questions, untracked generated files, or
accidental scope expansion were found during Day 14 closeout. Deferred
candidates remain explicitly outside Sprint 185 scope.

## Deferred Follow-Up Candidates

| Candidate | Deferred reason |
| --- | --- |
| `tests/test_qr.c` | Large review surface, but recent QR work and existing QR proof-owner extraction make it a separate future decision. |
| `tests/test_svd.c` | Strong helper-header precedent, but broader full/partial SVD ownership should be planned separately. |
| `tests/test_graph.c` | Existing fixture seam, but graph/FM environment behavior is not part of the selected direct-solver cluster. |
| `tests/test_integration.c` | Broad cross-solver owner with high behavior-preservation risk. |
| `tests/test_iterative.c` | Allocation-failure proof ownership remains sensitive. |
| `src/sparse_ldlt_csc.c` | Implementation extraction would require library source registration and a separate behavior-risk review. |

## Retrospective Inputs

- Primary win: the selected proof-owner file is smaller while keeping the
  existing `test_ldlt_csc` binary, test names, test order, fixture values, and
  tolerances stable.
- Most useful guardrail: `make ldlt-csc-helper-guard` now mechanically protects
  the helper-header layout.
- Main residual risk: future contributors could still add one-off helpers back
  into `tests/test_ldlt_csc.c`; the maintainer guidance now documents where
  reusable fixture/oracle/supernode helpers belong.
- Validation baseline: focused selected-cluster validation and the full C gate
  both passed before closeout.

## PR Review Notes

Reviewers can verify the sprint with:

```sh
make format
make lint
make test
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

For a faster focused check of the selected cluster:

```sh
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```
