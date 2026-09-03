# Day 2 Adoption Friction Audit

## Objective

Rank documentation and example friction by user impact, duplication, staleness
risk, evidence coupling, and overclaim risk before implementation begins.

## Inputs

Day 2 used:

- Sprint 194 plan and Day 1 adoption intake notes;
- `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`,
  `docs/solver_selection.md`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`, `examples/README.md`, `benchmarks/README.md`,
  and `tests/corpus/README.md`;
- `tests/corpus/manifests/selected_report_targets.tsv`;
- installed consumer proof owners `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, `sparse.pc.in`,
  `cmake/SparseConfig.cmake.in`, and `examples/cmake_example/`;
- public headers under `include/` for diagnostic and Doxygen narrative
  surface.

## Ranking Method

Scores use a 1-5 scale where 5 is highest. The weighted judgment favors
complete closure of a small number of high-value adoption gaps:

- user impact;
- duplication;
- staleness risk;
- evidence coupling;
- accidental overclaim risk;
- validation availability.

## Ranked Friction Table

| Rank | Friction | User impact | Duplication | Staleness | Evidence coupling | Overclaim | Validation availability | Recommended action |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Support/readiness truth is repeated across user, maintainer, corpus, benchmark, and planning surfaces. | 5 | 5 | 5 | 5 | 5 | 4 | Create a compact matrix before any prose reduction. |
| 2 | Static-first install and downstream-consumer guidance is split across README, INSTALL, tutorial, examples, templates, and install tests. | 5 | 5 | 4 | 4 | 5 | 5 | Consolidate minimal Make/`pkg-config` and CMake installed-consumer paths under INSTALL. |
| 3 | Diagnostics vocabulary is repeated across README, examples, cookbook, solver-selection, tutorial, and headers. | 5 | 4 | 4 | 3 | 3 | 4 | Make solver-selection the canonical diagnostic handoff and link from other surfaces. |
| 4 | Selected comparison/performance evidence caveats are too detailed in first-use docs. | 4 | 5 | 5 | 5 | 5 | 5 | Summarize selected evidence in user docs and link to manifest/corpus/benchmark owners. |
| 5 | README carries both first-use routing and maintainer/evidence command density. | 5 | 4 | 4 | 4 | 4 | 3 | Defer rewrite until support matrix and tutorial owner decisions exist. |
| 6 | Public headers include some broad workflow narrative alongside API-local contracts. | 3 | 3 | 3 | 3 | 4 | 5 | Audit comments after diagnostics wording is canonical; keep declarations unchanged. |
| 7 | Maintainer guide is large and repeats package/platform/report proof detail. | 2 | 4 | 4 | 5 | 3 | 4 | Keep as proof owner; only add cross-links or ownership summaries needed by matrix. |
| 8 | Benchmark/report handoff appears in multiple first-use docs. | 3 | 3 | 4 | 4 | 4 | 4 | Keep benchmark command detail in `benchmarks/README.md`; route from README by need. |
| 9 | Historical sprint links appear near current user truth. | 2 | 3 | 4 | 5 | 3 | 2 | Keep only when they explain current limitations or decisions. |

## Adoption Path Comparison Against Evidence

| Adoption path | Current user route | Evidence owner | Audit result |
| --- | --- | --- | --- |
| Local build and first solve | README -> `make` -> `make examples` -> `examples/README.md` | Makefile, examples sources, `make examples-build` | Clear route, but README surrounds it with advanced command and evidence material. |
| Local build-tree program | Tutorial and README quick-start/link commands | Makefile build output and examples | Keep separate from installed-prefix consumer instructions. |
| Unix static install | INSTALL Makefile quick start | `tests/test_install.sh`, `sparse.pc.in` | Strong proof owner; user docs should point here instead of copying static-first detail. |
| Unix `pkg-config` consumer | INSTALL `pkg-config` section | `tests/test_install.sh` | Strongly validated on Unix side; Windows must stay metadata-only. |
| Installed CMake consumer | INSTALL CMake section and `examples/cmake_example/` | `tests/test_cmake_install.sh`, `cmake/SparseConfig.cmake.in` | Good proof owner; tutorial should be minimal and should match the checked example. |
| Windows CMake-first consumer | INSTALL Windows/MSVC section | `.github/workflows/windows-ci.yml`, Windows CMake install/downstream lane, PowerShell guard | Support is narrower than broad Windows parity and must remain row-specific. |
| Solver chooser | README, solver-selection, cookbook, examples | tests, selected oracle/comparison targets, headers | Solver-selection is the best owner for diagnostic escalation and workflow caveats. |
| Benchmark/report reader | README, benchmarks README, corpus README, maintainer guide | selected target manifest, report scripts/tests, benchmark freshness checks | User docs should summarize; detail belongs to benchmark/corpus/manifest owners. |

## User Truth Versus Historical Evidence

| Topic | Active user truth should be | Historical/proof owner should be |
| --- | --- | --- |
| Platform support | Compact support/readiness matrix in user docs, likely INSTALL. | CI workflows and maintainer guide. |
| Static package shape | INSTALL static-first contract and installed files table. | Install scripts, CMake/package templates, maintainer guide. |
| Package-manager support | Not currently provided; use source install via Make/CMake. | Homebrew proof artifacts, package deferral scripts, Sprint 188 records. |
| Windows report freshness | One bounded selected Cholesky comparison lane only. | Sprint 190 artifacts, selected target manifest, Windows workflow, maintainer guide. |
| Selected comparison evidence | Fixture-local selected comparison summary by workflow. | `selected_report_targets.tsv`, corpus README, comparison scripts/tests. |
| Selected performance evidence | Threshold-free methodology/freshness summary. | benchmarks README, selected target manifest, benchmark freshness scripts/tests. |
| API docs | Source-controlled API reference plus public headers; generated HTML is local-only. | Doxyfile, docs check scripts, Sprint 179/186 decisions. |
| Diagnostics | Canonical user handoff in solver-selection. | Public headers for exact return-code/struct contracts and tests for behavior. |

## Cleanup Boundaries For Upcoming Days

1. Design the support/readiness matrix first. It must preserve non-claims for
   Windows breadth, package-manager breadth, shared libraries, dynamic ABI,
   broad report freshness, selected oracle/benchmark freshness on Windows,
   portable performance, and state-of-the-art status.
2. Do not rewrite README support sections until the matrix owner and link
   policy are established.
3. Keep installed consumer tutorial content aligned with install tests,
   package templates, and `examples/cmake_example/`.
4. Use `docs/solver_selection.md#diagnostics-handoff` as the target canonical
   diagnostics owner.
5. Treat public-header cleanup as comment-only unless a later sprint explicitly
   changes API, which Sprint 194 currently lists as a non-goal.
6. Keep selected target exact row identities and artifact metadata in
   `tests/corpus/manifests/selected_report_targets.tsv` rather than copying
   them through first-use docs.

## Initial Overclaim Risks

| Risk | Guardrail |
| --- | --- |
| Broad Windows parity from a Windows CMake row | Name the exact Windows CMake subset and retained non-claims. |
| Windows `pkg-config` command support from installed `sparse.pc` metadata | Separate metadata inspection from command execution. |
| Package-manager support from Homebrew proof material | State package-manager support is not currently provided. |
| Shared-library or dynamic ABI support from static install docs | Keep static-first package contract and deferred shared-library row explicit. |
| External-library parity from selected comparison rows | Use fixture-local selected comparison wording. |
| Portable performance from selected benchmark freshness | State methodology/freshness only, no timing threshold or portability claim. |
| Hosted API publication from Doxygen freshness | State generated HTML is local-only. |

## Day 3 Recommendation

Proceed with a support/readiness matrix contract. Recommended owner:
`INSTALL.md`, with README, tutorial, cookbook, solver-selection, API reference,
examples, benchmarks, corpus docs, and maintainer guide linking to the matrix
only where the reader needs current support status. Maintainer guide should
remain the proof semantics owner behind the matrix, not the primary first-user
matrix page.

## Validation

Day 2 changed planning documentation only.

```sh
git diff --check
```

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required for this day.
