# Day 1 Adoption Intake

## Objective

Establish Sprint 194 scope, owner files, user-facing adoption paths, and
support/readiness truth sources before any documentation rewrite begins.

## Sprint 194 Scope Map

| Item | Owner surfaces identified on Day 1 | First Day 2 follow-up |
| --- | --- | --- |
| 194.1 Adoption Audit | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md`, `docs/maintainer_guide.md` | Rank duplicated adoption and claim wording by reader impact. |
| 194.2 Support Matrix | `INSTALL.md#supported-platforms`, `docs/maintainer_guide.md`, `.github/workflows/*.yml`, `tests/corpus/manifests/selected_report_targets.tsv`, package/install tests | Choose one compact user-facing matrix owner and link policy. |
| 194.3 Installed Consumer Tutorial | `INSTALL.md`, `examples/cmake_example/`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, `tests/test_install.sh`, `tests/test_cmake_install.sh` | Compare documented commands with install proof behavior. |
| 194.4 Diagnostics Coherence | `docs/solver_selection.md`, `examples/README.md`, `docs/cookbook.md`, `README.md`, `include/*.h`, solver examples | Select canonical diagnostic vocabulary and link owners. |
| 194.5 Header Narrative Cleanup | Public headers under `include/`, especially longer workflow comments in solver and constructor headers | Identify comments that are narrative rather than API-local contracts. |
| 194.6 Validation | `Makefile`, docs/Doxygen scripts, install scripts, selected target/report tests, package guards, Windows PowerShell validator | Build a validation matrix for each planned edit type. |

## Adoption Workflows

| Workflow | Current route | Evidence owner | Day 1 concern |
| --- | --- | --- | --- |
| Source checkout first solve | README -> `make` -> `make examples` -> `./build/example_basic_solve` | `README.md`, `examples/README.md`, example sources | First-use path is clear, but nearby command/evidence sections make README long. |
| Build-tree local link | Tutorial local `cc -Iinclude -Lbuild` example | `docs/tutorial.md`, Makefile build output | Ensure it remains separate from installed package guidance. |
| Static installed archive | `make install` or `cmake --install` | `INSTALL.md`, install scripts, CMake package template, `sparse.pc.in` | Static-first contract is repeated in multiple docs. |
| Unix `pkg-config` downstream | `pkg-config --cflags --libs sparse` after Make install | `INSTALL.md`, `sparse.pc.in`, `tests/test_install.sh` | Needs crisp platform boundary because Windows only inspects metadata. |
| Installed CMake downstream | `find_package(Sparse)` and `Sparse::sparse_lu_ortho` | `INSTALL.md`, `examples/cmake_example/`, `tests/test_cmake_install.sh` | Needs a minimal tutorial that stays aligned with the checked example. |
| Solver choice | Choose by matrix shape and diagnostic result | `docs/solver_selection.md`, cookbook, examples, headers | Diagnostics wording repeats and should converge on one vocabulary owner. |
| Maintainer/release review | Quality, package, report, API, and platform checks | `docs/maintainer_guide.md`, Makefile, workflows, selected manifests | Maintainer proof details should not dominate first-use docs. |

## Current Truth-Source Candidates

The strongest candidates for active user truth are:

- `README.md` for the shortest entry route and links;
- `INSTALL.md` for install, downstream consumers, static package shape, and
  platform support summary;
- `docs/solver_selection.md` for solver choice and diagnostic vocabulary;
- `docs/api_reference.md` plus `include/*.h` for exact public declarations;
- `benchmarks/README.md` for benchmark/report interpretation;
- `docs/maintainer_guide.md` for maintainer policy and proof semantics;
- `tests/corpus/manifests/selected_report_targets.tsv` for selected report
  target metadata and claim scopes.

Historical sprint artifacts should remain provenance and residual context, not
the first place a new user must read for current support.

## Initial Duplication Findings

| Finding | Evidence from inventory | Candidate cleanup direction |
| --- | --- | --- |
| Support/platform truth is copied broadly. | README, INSTALL, maintainer guide, corpus docs, selected target manifest, and benchmark docs all restate parts of the support story. | Add a compact current-support matrix and link to proof owners. |
| Package non-claims are repeated. | Static-first, shared-library, dynamic ABI, package-manager, Homebrew, and Windows `pkg-config` boundaries appear in README, INSTALL, API reference, examples, and maintainer docs. | Keep full installed-consumer detail in INSTALL and reduce other docs to summary links. |
| Selected evidence caveats are heavy in first-use docs. | README and solver-selection include detailed selected oracle/comparison/performance non-claims also owned by corpus/benchmark docs. | Preserve claim boundaries but make selected manifest and evidence docs the detailed owners. |
| Diagnostic vocabulary is distributed. | README, examples README, cookbook, solver-selection, and headers all discuss status, residual, convergence, rank, and error handoff. | Use solver-selection as the canonical user-facing diagnostic handoff. |
| Header comments carry some broad workflow explanation. | Public headers include API-local contracts plus longer narrative around defaults, diagnostics, backend behavior, and repeated workflows. | Move only non-contract narrative after Doxygen coverage owners are known. |
| Installed-consumer tutorial is split. | INSTALL has commands; examples have `cmake_example`; tests prove install behavior; README/tutorial summarize routes. | Consolidate into a small tutorial path backed by existing proof tests. |

## Risk Register

| Risk | Impact | Day 1 mitigation |
| --- | --- | --- |
| Simplification turns into support expansion. | Users may read unearned Windows, package-manager, shared-library, ABI, report, or performance claims. | Keep non-claims explicit and matrix rows evidence-bound. |
| User docs lose maintainer provenance. | Future maintainers may not know why a claim is narrow. | Link user matrix rows to maintainer/corpus/benchmark proof owners. |
| Header cleanup changes API behavior or Doxygen output. | Public contract or generated docs could regress. | Treat declaration preservation and Doxygen checks as mandatory for header edits. |
| Tutorial commands drift from proof scripts. | Downstream consumers could get commands that CI does not prove. | Use install tests and example CMake project as executable owners. |
| Links become stale during consolidation. | The simplified adoption path may route users to broken anchors. | Include link/anchor inspection in later validation. |

## Day 2 Audit Questions

1. Should the compact support/readiness matrix live in `INSTALL.md` or a new
   dedicated docs page linked from README and INSTALL?
2. Which repeated support claims can become links without weakening visible
   non-claims?
3. Which docs are the first cleanup targets for installed-consumer guidance?
4. Which diagnostic terms need exact canonical wording before edits begin?
5. Which public header narratives are genuinely relocatable, and which are
   API-local call-site contracts that must stay in headers?

## Validation

Day 1 is documentation/planning only. The validation gate is:

```sh
git diff --check
```

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` gate is not required for this day.
