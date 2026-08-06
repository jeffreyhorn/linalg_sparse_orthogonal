# Day 12 Focused Validation

## Scope

Day 12 executed the required focused QR, corpus, documentation, source-list,
generated-artifact, and full C quality gates for Sprint 139. This validation
covered the Day 7 Python oracle changes, Day 9 C/H/build-system proof owner,
and Day 10-11 documentation updates.

## Focused QR and Corpus Checks

Passed:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
cmake -S . -B build/qr-corpus-proof && cmake --build build/qr-corpus-proof --target test_qr_corpus && ./build/qr-corpus-proof/test_qr_corpus
```

Focused Make and CMake QR proof results:

- `test_qr_corpus` ran 4 tests.
- 0 tests failed.
- 0 tests were skipped.
- 83 assertions passed.
- Solver-produced normalized nullspace residual was approximately
  `2.220e-16`.
- Deterministic reference-direction residual was `0.000e+00`.

The opt-in solver QR oracle/report command wrote:

- `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

Generated output facts:

- oracle TSV line count: 7 lines, meaning 1 header plus 6 rows.
- report index line count: 8 lines, meaning 1 header plus 7 rows.
- skip TSV line count: 2 lines, meaning 1 header plus 1 optional-data row.
- manifest line count: 15 lines.
- manifest `oracle_row_count=6`.
- manifest `solver_families=qr,unknown`.
- manifest `solver_qr_row_count=3`.
- manifest command:
  `scripts/run_corpus_oracle.py --include-solver-qr`.

Oracle row inspection confirmed all six rows passed:

- `qr_rank_deficient_6x4_nullspace_v1_nullity`, `unknown`, `pass`, observed
  `1`.
- `qr_rank_deficient_6x4_nullspace_v1_projector_residual`, `unknown`, `pass`,
  observed `0`.
- `qr_rank_deficient_6x4_nullspace_v1_rank`, `unknown`, `pass`, observed `3`.
- `qr_rank_deficient_6x4_nullspace_v1_qr_nullity`, `qr`, `pass`, observed
  `1`.
- `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual`, `qr`, `pass`,
  observed approximately `2.2204460492503131e-16`.
- `qr_rank_deficient_6x4_nullspace_v1_qr_rank`, `qr`, `pass`, observed `3`.

## Source-List, Script, and Artifact Hygiene

Passed:

```sh
python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py
rg -n "test_qr_corpus" Makefile CMakeLists.txt
git diff --check
```

Source-list parity confirmed:

- `Makefile` lists `$(TESTDIR)/test_qr_corpus.c`.
- `CMakeLists.txt` registers `add_sparse_test(test_qr_corpus)`.

Passed trailing-whitespace scans for edited public docs, corpus docs, Sprint
139 planning artifacts, the oracle script, and QR corpus test/helper files.

Passed focused relative Markdown link validation for:

- `README.md`
- `docs/algorithm.md`
- `docs/cookbook.md`
- `docs/maintainer_guide.md`
- `docs/solver_selection.md`
- `examples/README.md`
- `tests/corpus/README.md`
- Sprint 139 plan, working notes, and artifacts

Generated corpus outputs remain ignored build artifacts:

- `git status --short --ignored build/corpus build/corpus-reports` reports the
  generated `build/` tree as ignored.
- `git ls-files build/corpus build/corpus-reports` reports no tracked files.

The Python compile check produced local bytecode cache files under
`scripts/__pycache__/`; those generated files were removed after validation.

## Full Required Quality Gate

Because Sprint 139 modified `.c` and `.h` files, Day 12 ran the full required
quality gate:

```sh
make format && make lint && make test
```

Result: passed. The final test summary reported `All tests passed.`

The full gate included:

- clang-format over source, test, benchmark, example, and public header files.
- benchmark/example tooling build.
- strict warning compile.
- clang-tidy.
- cppcheck.
- full Make test suite, including `test_qr_corpus`.

## Reruns and Non-Issues

Two report-inspection commands were rerun with corrected TSV field names after
using stale column names from memory. These were inspection-command mistakes,
not oracle-generation or validation failures. The corrected inspection passed
and is recorded above.

No supplemental checks were skipped. Generated oracle/report files were not
promoted into source control.
