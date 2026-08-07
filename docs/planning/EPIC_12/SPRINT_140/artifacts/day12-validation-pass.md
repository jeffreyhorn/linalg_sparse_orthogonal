# Day 12 Validation Pass

## Scope

Day 12 validates the Sprint 140 partial-SVD edge-case and convergence residual
closure work across the focused corpus lane, source/build registration surfaces,
documentation hygiene, and the full C quality gate required for changed `.c`
and `.h` files.

No production source behavior is changed by this artifact.

## Command Log

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | PASS | Corpus manifests and expected TSV files validated successfully. |
| `python3 scripts/run_corpus_oracle.py --include-partial-svd` | PASS | Generated the combined QR and opt-in partial-SVD oracle outputs under ignored `build/` paths. |
| `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus` | PASS | `test_svd_partial_corpus` ran 5 tests, 0 failures, and 107 assertions. |
| `make source-list-check` | PASS | Source list parity passed with 49 library sources. |
| `make quality-review-cmake-compile` | PASS | CMake configured, rebuilt cleanly, registered `test_svd_partial_corpus`, and reported 59 CMake tests matching 59 Makefile tests. |
| `make format && make lint && make test` | PASS | Required full gate passed after formatting, clang-tidy/cppcheck linting, and the full Make test suite. |
| `python3 -m py_compile scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py` | PASS | Oracle and schema scripts compile; generated `scripts/__pycache__` was removed afterward. |
| `git diff --check` | PASS | No whitespace errors in the working diff. |
| Targeted trailing-whitespace scan | PASS | No trailing blanks found across touched docs, corpus, scripts, build files, source, or tests. |
| Targeted TSV width check | PASS | Corpus manifest, expected fixture, oracle, and report-index TSV rows have stable column widths. |
| Targeted markdown link check | PASS | Touched documentation and Sprint 140 artifacts have resolvable local links. |
| Generated-artifact status check | PASS | Generated oracle/report/CMake outputs remain under ignored `build/` paths and are not promoted. |

## Focused Partial-SVD Evidence

The focused `test_svd_partial_corpus` owner exercises the Sprint 140 generated
clustered/repeated 8x6 diagonal fixture with `k = 3`:

- default-budget success;
- top-3 singular-value checks;
- left and right top-k subspace-projector checks;
- triplet residual checks;
- orthogonality checks;
- tight-budget fail-closed behavior; and
- recovery after a tight-budget failure.

The opt-in oracle command also emits generated partial-SVD rows for
`partial_svd_clustered_repeated_diag8x6_k3_v1` under the local build report
paths.

## Skips and Non-Claims

- No hosted CI proof was run locally.
- No generated oracle, report, skip, manifest, or CMake build artifacts are
  promoted from `build/`.
- No performance, platform parity, package, ABI, external-library parity,
  broad repeated-spectrum, broad partial-SVD correctness, partial-result, or
  state-of-the-art claim is added by this validation pass.

## Rerun Requirements

Rerun this Day 12 validation set after changes to:

- `src/sparse_svd_partial.c`;
- `include/sparse_svd.h`;
- `tests/test_svd_partial_corpus.c`;
- `tests/test_svd_partial_helpers.h`;
- `tests/test_svd_partial_shared_helpers.h`;
- `tests/corpus/**/*.tsv`;
- `scripts/validate_corpus_schema.py`;
- `scripts/run_corpus_oracle.py`;
- `Makefile`;
- `CMakeLists.txt`; or
- SVD/corpus documentation that changes maintained evidence wording.
