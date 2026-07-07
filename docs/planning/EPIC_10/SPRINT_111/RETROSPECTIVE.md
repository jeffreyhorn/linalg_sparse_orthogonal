# Sprint 111 Retrospective

**Sprint:** 111 - API Usability, Documentation & Example Coherence
**Duration:** 14 days (Days 1-14 landed on branch `sprint-111`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 111 started from Sprint 110 Matrix Market, builder-boundary, and
      proof-owner follow-through without claiming a public Matrix I/O module
      or public Matrix builder API.
- [x] README, tutorial, examples, install docs, benchmark docs, public headers,
      and planning artifacts were audited from a first-time user perspective.
- [x] maintainer-only proof surfaces were separated from user adoption
      surfaces in Sprint 111 working notes and follow-up docs.
- [x] a concise solver-selection guide was added at
      `docs/solver_selection.md`.
- [x] compressed-first CSR and CSC workflows were documented and demonstrated
      with public APIs only.
- [x] Matrix Market load/save behavior was documented in terms of public
      functions, ownership, errors, duplicate handling, final-zero elision,
      pattern entries, symmetric expansion, and runtime behavior.
- [x] benchmark interpretation docs now frame reports as local,
      configuration-sensitive measurement evidence rather than portable
      performance proof.
- [x] README, tutorial, examples README, solver-selection guide, Matrix Market
      docs, benchmark docs, and public header comments agree on public API
      names and user-facing boundaries.
- [x] example and public-header changes were validated with focused example
      builds and the final full quality gate.
- [x] final validation passed:
  - `make examples`
  - `./build/example_compressed_input`
  - `./build/example_matrix_market`
  - `cmake -S . -B cmake-build`
  - `cmake --build cmake-build --target example_compressed_input example_matrix_market`
  - `./cmake-build/example_compressed_input`
  - `./cmake-build/example_matrix_market`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scans over touched docs, examples, public header, and
    Sprint 111 artifacts
- [x] residual deferred debt is dependency-ordered for Sprint 112 or later.

## What Went Well

1. **The sprint made the adoption path concrete.**
   The new solver-selection guide gives users one place to decide between
   CSR/CSC input, Matrix Market loading, one-shot direct solves, repeated
   direct solves, iterative handles, eigensolvers, SVD, reorder/fill workflows,
   and benchmark handoffs.

2. **Compressed-first examples became copyable public workflows.**
   `examples/example_compressed_input.c` now demonstrates caller-owned CSR and
   CSC arrays, independent matrix construction, cleanup, solves, and residual
   checks. The example README points users to the compressed-input path without
   relying on private helpers or maintainer proof scaffolding.

3. **Matrix Market behavior is now documented as public behavior, not source
   ownership.**
   `docs/matrix_market.md`, `include/sparse_matrix.h`, the solver guide,
   tutorial, and `examples/example_matrix_market.c` agree on
   `sparse_load_mm(...)` / `sparse_save_mm(...)`, caller ownership,
   `SPARSE_ERR_IO` and `sparse_errno()` behavior, parse errors, duplicate
   last-entry-wins semantics, final-zero elision, pattern defaults, and
   symmetric expansion.

4. **Benchmark claims became more defensible.**
   README, the solver guide, and `benchmarks/README.md` now describe benchmark
   results as local measurements tied to workload, branch, compiler, platform,
   backend, build options, matrix corpus, and thread settings. That keeps
   user-facing docs useful without implying universal performance proof.

5. **Maintainer proof language moved away from first-contact docs.**
   README and tutorial now route users toward guides and examples first, while
   evidence boundaries and reviewed-quality details remain in maintainer-facing
   documentation and Sprint artifacts.

6. **Validation matched the changed surface.**
   The branch touched documentation, examples, CMake registration, and a public
   header comment. It ran focused example checks, CMake example checks,
   Markdown link validation, final diff hygiene, and the full
   `make format && make lint && make test` chain.

## What Didn't Go Well

1. **The adoption surface is still broad.**
   README, tutorial, examples, solver guide, Matrix Market docs, benchmark
   docs, public headers, and maintainer docs all need to stay aligned. Sprint
   111 improved routing, but future feature work can still create drift if it
   updates only one surface.

2. **External references were not network-checked.**
   The sprint validated local relative Markdown links, but external Matrix
   Market, SuiteSparse, and related reference URLs remain a future
   documentation QA task.

3. **Benchmark docs remain detailed.**
   `benchmarks/README.md` now has a clearer interpretation entry point, but it
   still carries live lane names and report mechanics. That detail is useful,
   but future work should split or index it if scanability declines.

4. **The guide is intentionally evidence-bounded.**
   Sprint 111 did not claim shared-library support, a public Matrix I/O
   module, a public builder API, universal benchmark superiority, or broader
   external-oracle coverage. That restraint is correct, but it means future
   product expansions still need implementation and proof before docs can
   advertise them.

## Final Metrics

### Validation

| Metric | Sprint 111 close state |
|---|---:|
| Makefile examples build | passed, 14 example binaries built |
| compressed-input example | passed through Makefile and CMake builds |
| Matrix Market example | passed through Makefile and CMake builds |
| CMake configure/build for touched examples | passed |
| relative Markdown link existence check | passed on README, solver guide, Matrix Market docs, tutorial, benchmark README, and examples README |
| full branch-level gate | `make format && make lint && make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched docs, examples, public header, and Sprint artifacts |
| public API declaration drift | 0 declarations |
| public header comment updates | 1 header |
| new public module or builder API claims | 0 |

### Adoption Surface Movement

| surface | Sprint 111 close state |
|---|---:|
| new solver-selection guide | 204 lines |
| Matrix Market documentation | 190 lines |
| benchmark documentation | 569 lines |
| examples README | 320 lines |
| compressed-input example | 124 lines |
| Matrix Market example | 119 lines |
| public Matrix Market header comments updated | yes |
| CMake example registration updates | yes |

### Sprint 111 Artifact Package

| Metric | Sprint 111 close state |
|---|---:|
| artifact files under `SPRINT_111/artifacts/` | 13 |
| planning and working-note files | 2 |
| retrospective files | 1 |

Notes:

- inventory and guide artifacts:
  - `day1-user-journey-inventory.md`
  - `day2-documentation-gap-audit.md`
  - `day3-solver-guide-outline.md`
- examples and workflow artifacts:
  - `day5-compressed-first-example-audit.md`
  - `day6-csr-csc-construction-example.md`
  - `day7-solver-workflow-example-alignment.md`
  - `day8-advanced-and-matrix-market-examples.md`
- documentation coherence artifacts:
  - `day9-matrix-market-behavior-docs.md`
  - `day10-header-tutorial-coherence.md`
  - `day11-benchmark-interpretation.md`
  - `day12-audience-boundary-split.md`
- validation and closeout artifacts:
  - `day13-integrated-validation.md`
  - `day14-closeout-and-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Future documentation QA should network-check external Matrix Market,
  SuiteSparse, and related reference URLs.
- README quality and CI wording should stay compact; avoid growing the README
  into a maintainer handbook.
- Benchmark documentation should keep live lane names and report mechanics, but
  future work should split or index it if users can no longer scan it easily.
- `docs/algorithm.md` should be reviewed only if it becomes a public adoption
  or reference surface; otherwise it can remain technical background.
- README and guide performance wording should stay tied to measured local
  evidence, not universal speed claims.

Still consciously constrained rather than silently solved:

- no public Matrix I/O module claim;
- no public Matrix builder API claim;
- no public API declaration change;
- no shared-library/ABI or platform-support expansion claim;
- no universal benchmark/performance claim;
- no maintainer proof artifact promoted as the first adoption path.

Not carried forward as unresolved Sprint 111 debt:

- user-journey audit;
- solver-selection guide creation;
- compressed-first CSR/CSC example path;
- Matrix Market example path;
- Matrix Market public behavior documentation;
- benchmark interpretation documentation;
- README/tutorial/examples audience-boundary cleanup;
- integrated example, CMake, Markdown, and full quality validation.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-user-journey-inventory.md](./artifacts/day1-user-journey-inventory.md)
- [day2-documentation-gap-audit.md](./artifacts/day2-documentation-gap-audit.md)
- [day3-solver-guide-outline.md](./artifacts/day3-solver-guide-outline.md)
- [day5-compressed-first-example-audit.md](./artifacts/day5-compressed-first-example-audit.md)
- [day6-csr-csc-construction-example.md](./artifacts/day6-csr-csc-construction-example.md)
- [day7-solver-workflow-example-alignment.md](./artifacts/day7-solver-workflow-example-alignment.md)
- [day8-advanced-and-matrix-market-examples.md](./artifacts/day8-advanced-and-matrix-market-examples.md)
- [day9-matrix-market-behavior-docs.md](./artifacts/day9-matrix-market-behavior-docs.md)
- [day10-header-tutorial-coherence.md](./artifacts/day10-header-tutorial-coherence.md)
- [day11-benchmark-interpretation.md](./artifacts/day11-benchmark-interpretation.md)
- [day12-audience-boundary-split.md](./artifacts/day12-audience-boundary-split.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
