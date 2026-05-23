# Code Review

**Date:** 2026-05-21  
**Reviewer:** Codex  
**Scope:** Full-project source review of the current `linalg_sparse_orthogonal` tree after Epic 3 closeout, with emphasis on code correctness, maintainability, efficiency, documentation, and usability.

## Executive Summary

The repository is in strong engineering shape compared with a typical C numerical codebase:

- the project has broad algorithm coverage,
- the public headers are heavily documented,
- the test surface is large,
- and Epic 3 left behind a real quality contract rather than a one-off cleanup.

The next meaningful improvements are no longer “add missing infrastructure.” They are structural:

- reduce hidden state and caller footguns in the API,
- split monolithic implementation hotspots,
- centralize repeated safety logic,
- lower heap churn in repeated numeric workloads,
- and simplify the quality/documentation contract so it is easier to maintain.

I did **not** find a clear immediate release-blocking correctness bug during this review. The main risks are architectural and operational: the code works, but several core surfaces are getting harder to reason about safely.

## Findings

### 1. High: the mutable matrix-state model is still too easy to misuse, and too much correctness is delegated to caller discipline

The central `SparseMatrix` type still plays too many roles:

- original input matrix,
- reordered matrix,
- in-place LU/Cholesky factor container,
- and object carrying hidden `factored` / permutation state.

Representative references:

- `README.md:240`
- `README.md:551`
- `include/sparse_lu.h:60`
- `include/sparse_lu.h:87`
- `include/sparse_cholesky.h:114`
- `include/sparse_cholesky.h:163`
- `include/sparse_svd.h:69`
- `include/sparse_svd.h:248`

Why this matters:

- Many APIs require callers to know whether a matrix is still in its original state, has identity permutations, has already been factored, or must be copied before reuse.
- The cancellation contract for in-place factorization is especially subtle: even immediate cancellation can mutate cached norms and factor-state flags before the first callback is observed.
- This is documented, but documentation is compensating for a state model that is easy to misuse.

Primary improvement:

- move toward explicit analysis / factor / solve handles rather than overloading `SparseMatrix` with lifecycle state.

### 2. High: `src/sparse_graph.c` has become a monolithic algorithm host, which now raises both correctness and maintainability risk

The graph / nested-dissection surface is carrying too many responsibilities in one file:

- multilevel hierarchy construction,
- heavy-edge matching,
- multiple bisection strategies,
- FM refinement,
- separator lifting strategies,
- environment-driven behavior overlays,
- and partition orchestration.

Representative references:

- `src/sparse_graph.c:1`
- `src/sparse_graph.c:1992`
- `src/sparse_graph.c:3171`
- `src/sparse_graph.c:3462`

The file is over 3,500 lines, and its hottest single routines are large enough to resist focused review:

- `graph_refine_fm(...)` is more than 500 lines
- `graph_edge_separator_to_vertex_separator(...)` is roughly 275 lines

Why this matters:

- It is increasingly difficult to change one heuristic without risking another.
- The environment-variable overlays are load-bearing behavior switches embedded inside the same implementation unit as the core partitioning logic.
- The file is now large enough that performance work, bug fixing, and behavior auditing all compete inside the same context.

Primary improvement:

- split this subsystem into smaller modules with explicit ownership boundaries:
  - coarsening,
  - bisectors,
  - FM refinement,
  - separator lifting,
  - and runtime/config parsing.

### 3. High: allocation and overflow hardening is present, but the project still implements it in too many inconsistent local forms

The repository has clearly invested in integer-overflow defense, but the implementation pattern is fragmented.

Representative local helpers:

- `src/sparse_etree.c:5`
- `src/sparse_dense.c:8`
- `src/sparse_svd.c:12`
- `src/sparse_eigs.c:153`

Representative unchecked auxiliary allocations:

- `benchmarks/bench_main.c:108`
- `examples/example_analysis.c:122`

Repository-wide signal from this review pass:

- `src/` alone currently contains hundreds of allocation sites
- `tests/` contains well over a thousand raw allocation calls

Why this matters:

- The core library already knows how to do careful size validation, but the discipline is not centralized.
- That increases drift risk: newer files tend to be hardened better than older or auxiliary ones.
- Tooling, examples, and benchmarks are part of the supported engineering surface; they should not lag far behind the library’s own safety standard.

Primary improvement:

- introduce a shared internal allocation/size utility layer and migrate modules to it instead of maintaining parallel helper idioms.

### 4. Medium: repeated-solve and eigensolver workloads still pay avoidable heap churn because workspaces are one-shot

The iterative and eigensolver paths allocate substantial temporary workspaces inside each call.

Representative references:

- `src/sparse_iterative.c:189`
- `src/sparse_iterative.c:381`
- `src/sparse_iterative.c:1027`
- `src/sparse_eigs.c:1082`
- `src/sparse_eigs.c:2174`
- `src/sparse_eigs.c:2644`

Why this matters:

- For a single solve, this is acceptable.
- For repeated solves in a time loop, repeated RHS batches, or repeated eigensolver runs with the same dimension profile, this creates unnecessary allocator traffic and cache disruption.
- The API gives advanced users no clean way to reuse scratch storage or pin working buffers.

Primary improvement:

- add optional reusable workspace/context objects for iterative solvers and eigensolvers while preserving the current simple one-shot APIs as wrappers.

### 5. Medium: the quality contract is powerful, but it is over-distributed across Makefile logic, shell/Python helpers, and README prose

Epic 3 successfully created a real quality workflow, but that workflow now lives in several places at once:

- wrapper/entrypoint semantics in `Makefile`
- artifact generation and platform handling in shell/Python helpers
- operator/readiness semantics in `README.md`

Representative references:

- `Makefile:487`
- `Makefile:600`
- `README.md:730`
- `README.md:763`
- `scripts/deadcode_workflow.sh:1`

Why this matters:

- A maintenance change to one reviewed path often requires coordinated edits in multiple files.
- It raises the odds of “documentation is honest, script behavior drifted” or “Makefile changed, README language lagged.”
- The quality system is now important enough that it would benefit from stronger single-source-of-truth boundaries.

Primary improvement:

- reduce duplication by defining clearer ownership:
  - Makefile owns commands,
  - scripts own machine behavior,
  - a smaller maintainer doc owns policy,
  - README links rather than repeats.

### 6. Medium: benchmark/developer CLI surfaces are inconsistent and still weaker than the library APIs they expose

`bench_main` is still using manual command parsing and weak numeric conversion:

- `atoi(...)` for numeric flags
- ad hoc string matching
- incomplete reorder exposure in the usage/parser path

Representative references:

- `benchmarks/bench_main.c:4`
- `benchmarks/bench_main.c:631`

Notable current mismatch:

- the file advertises `none|rcm|amd|nd`
- the enum and library support `SPARSE_REORDER_COLAMD`
- but the parser path still does not accept `colamd`

Why this matters:

- The library API surface is more capable than one of its main engineering tools.
- Manual `atoi` parsing is brittle compared with the stronger `strtol`/range-checked style already used in newer tooling such as `bench_eigs`.
- Developer tools are part of usability; if they lag, users learn the wrong operational model.

Primary improvement:

- centralize CLI parsing helpers for benchmarks/examples and bring `bench_main` up to the same input-validation standard as the more recent tooling.

### 7. Medium: documentation is rich, but the README has become an overloaded document that mixes user guide, maintainer guide, CI contract, readiness policy, and API caveats

The README currently owns:

- the command map,
- cross-platform CI contract,
- readiness checklist,
- test-surface policy,
- warning authority notes,
- dead-code contract,
- and general API guidance.

Representative references:

- `README.md:730`
- `README.md:763`
- `docs/tutorial.md:146`
- `include/sparse_lu.h:60`

Why this matters:

- The project now has enough operational maturity that one document should not need to teach both end users and maintainers everything.
- Some of the most important behavioral constraints are repeated in headers, tutorial prose, and README sections because there is no lighter-weight maintainer standard surface.
- That increases review churn and makes doc ownership harder to reason about.

Primary improvement:

- keep README user-facing and concise, and move maintainer-specific quality/state-model policy into a shorter dedicated maintainer guide.

## Category Assessment

### Correctness

The codebase looks materially healthier than it did before Epic 3. I did not find an obvious current algorithmic defect that should block normal use.

The biggest correctness risks now are structural:

- hidden mutable matrix state,
- caller-managed copy-before-use semantics,
- and inconsistent auxiliary overflow/allocation discipline.

### Maintainability

This is now the dominant improvement category.

The main pain points are:

- monolithic implementation files,
- giant test files,
- repeated safety helpers,
- and duplicated quality-contract logic across build/docs/scripts.

### Efficiency

The main efficiency improvement is not obviously a new numerical kernel. It is reducing avoidable overhead around:

- repeated workspace allocation,
- benchmark/tool duplication,
- and large multi-purpose implementation units that are harder to optimize locally.

### Documentation

Documentation breadth is good, but ownership is diffuse.

The next improvement is not “write more docs.” It is:

- shorten the README,
- define a clearer maintainer-policy home,
- and reduce repeated behavioral caveats.

### Usability

The main usability issue is that the public library surface is stronger than some of the surrounding engineering tools:

- stateful matrix semantics are still easy to misuse,
- benchmark CLIs are inconsistent,
- and some contracts are only obvious after reading README + tutorial + headers together.

## Validation

This review was based on direct source inspection of the current `main`-derived tree, including:

- public headers under `include/`
- core numeric implementations under `src/`
- benchmark/examples/tooling surfaces
- existing Epic 3 review artifacts and quality-contract documentation

I also started a full local `make quality-review-full` pass during the review to confirm the current project is still using the reviewed baseline rather than stale ad hoc commands.

## Bottom Line

The codebase has crossed the point where the highest-value work is structural refinement, not raw feature addition.

The next major improvements should be:

1. make matrix/factor lifecycle state more explicit,
2. split monolithic algorithm hosts,
3. centralize allocation/overflow discipline,
4. add reusable workspaces for repeated numeric workloads,
5. simplify the quality-contract ownership model,
6. modernize benchmark CLIs,
7. and separate user-facing docs from maintainer-facing policy.
