# Day 14 Closeout Artifact

## Scope Closed

Sprint 193 closed one selected review-surface reduction claim: the selected QR
external-reference rank/nullspace/threshold cluster was extracted from
`tests/test_qr.c` into `tests/test_qr_external_ref_helpers.h` while preserving
`test_qr` as the proof-owner executable.

No public API, production source, QR tolerance, rank policy, solver behavior, or
generated report surface was intentionally changed.

## Changed Files

| File | Purpose |
| --- | --- |
| `tests/test_qr.c` | Retains QR proof-owner registration, `main`, selected `RUN_TEST(...)` entries, and the scoped economy test body |
| `tests/test_qr_external_ref_helpers.h` | Owns selected QR external-reference readers, selected moved rank/nullspace/threshold tests, and reader failure-path tests |
| `Makefile` | Adds `qr-external-ref-helper-guard` |
| `scripts/check_qr_external_ref_helper_guard.sh` | Mechanically checks the helper boundary and source-list absence |
| `tests/test_qr_external_ref_helper_guard.py` | Covers positive and negative guard behavior |
| `docs/maintainer_guide.md` | Documents the helper/proof-owner split, formatter-stable QR/solver helper include dependencies, and forced focused rebuild caveat |
| `docs/planning/EPIC_17/SPRINT_193/*` | Records the Sprint 193 plan, working notes, and day-by-day artifacts |

## Final Metrics

| Measure | Before Sprint 193 branch edits | Final branch state | Result |
| --- | ---: | ---: | --- |
| `tests/test_qr.c` line count | 3970 | 3040 | Main QR proof owner reduced by 930 lines |
| Selected helper line count | 0 | 1004 | Selected cluster isolated in a family-local helper |
| `test_qr` registered tests | 77 | 79 | Existing selected tests preserved; 2 reader failure tests added |
| Production source changes under `src/` | 0 | 0 | No production algorithm change |
| Public header changes under `include/` | 0 | 0 | No API or ABI surface change |
| Library source-list count | 49 | 49 | Source manifest ownership unchanged |
| CMake/Makefile test-count parity | 59/59 | 59/59 | Test registration parity preserved |

## Final Validation

Final Day 14 command:

```sh
make source-list-check && \
python3 tests/test_qr_external_ref_helper_guard.py && \
make qr-external-ref-helper-guard && \
make quality-review-cmake-compile && \
make format && \
make lint && \
make test
```

Result: passed.

Observed validation details:

- `make source-list-check`: passed with 49 library sources.
- `python3 tests/test_qr_external_ref_helper_guard.py`: passed.
- `make qr-external-ref-helper-guard`: passed, including required files,
  proof-owner registration, helper boundary, selected cluster ownership,
  header-only registration, and maintainer docs.
- `make quality-review-cmake-compile`: passed configure, clean rebuild,
  `ctest -N`, and Makefile/CMake parity.
- CMake tests: 59.
- Makefile tests: 59.
- `make format`: passed.
- `make lint`: passed strict warnings, clang-tidy, and cppcheck.
- `make test`: passed with final `All tests passed.`
- `test_qr`: 79 tests, 0 failures, 0 skips, 976 assertions.
- `test_reorder_nd`: 35 tests, 0 failures, 1 skip.
- `test_reorder_amd_qg`: 7 tests, 0 failures, 0 skips, 2068 assertions.
- `git diff --check`: passed after the final gate.

## Retrospective Inputs

What worked:

- A narrow, selected QR external-reference cluster made the review-surface
  reduction measurable and behavior-preserving.
- The guard target gives future reviewers a fast way to detect helper boundary
  drift.
- Day 12 and PR review exposed and fixed formatter-stability issues by making
  the helper own its `test_qr_helpers.h` and `test_solver_helpers.h`
  dependencies.

What was constrained:

- Header-only test helper edits still need forced focused rebuild validation
  when not running the full suite.
- The economy external-reference test remained in `tests/test_qr.c` by explicit
  scope choice.
- Other large QR clusters were left untouched to avoid broad review churn.

Closed claim:

- Sprint 193 reduced one selected large QR review surface by extracting the
  selected external-reference rank/nullspace/threshold cluster into a
  mechanically guarded family-local helper.

## Residuals and Handoff

- `test_qr_external_dense_reference_economy_projector_5x3` remains in
  `tests/test_qr.c`.
- Other large QR/economy/sparse-mode/refinement clusters remain future
  candidates only after separate boundary review.
- Future header-only QR helper edits should either force-rebuild `build/test_qr`
  before focused execution or run the full Makefile gate.
- The branch is ready for the Sprint 193 retrospective, commit, push, and pull
  request creation.
