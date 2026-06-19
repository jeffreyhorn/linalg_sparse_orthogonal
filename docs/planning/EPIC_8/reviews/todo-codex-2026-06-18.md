# Epic 8 Gap-Closure Todo — 2026-06-18

This todo converts the review findings into a concrete closure sequence. The
order is deliberate: it starts with architecture and proof surfaces that change
the ceiling, then moves to capability breadth, maintainability, platform, and
final integration.

Sprint 80 Days 2-8 tightened how this sequence should be read:

- the strongest reviewed baseline remains `make quality-review-full`
- the first maintained external correctness lane is bounded to a CHOLMOD-class
  SPD direct-solver comparison
- BLAS/LAPACK-class references are performance-reference support, not a broad
  maintained correctness contract
- canonical benchmark reporting remains threshold-free
- the epic is explicitly fenced against fake platform parity, shared-library
  maturity, broad capability-genericity, or “rewrite the whole library”
  interpretations

The sequence below now reads against that contract.

## Stage 1: Freeze the competitive target and measurement model

### 1. Reconfirm the strongest local baseline

- rerun the full reviewed baseline
- capture the reviewed CMake parity count
- capture install/export proof status
- capture canonical benchmark outputs

**Done when:** the project has one fresh Epic 8 baseline artifact that names
the reviewed test count, install/export status, and canonical benchmark bundle.

### 2. Define the actual competitive target

- decide which comparison class the library is aiming at:
  - educational / research-grade
  - production scientific sparse toolkit
  - high-performance direct-solver platform
- write down which claims Epic 8 will and will not try to earn

**Done when:** Epic 8 has an explicit “state-of-the-art enough” target instead
of a vague aspiration.

### 3. Add an external reference-oracle contract

- decide which external stacks to compare against where feasible:
  - CHOLMOD-class / SuiteSparse-family SPD direct-solver references first
  - LAPACK / BLAS for dense subproblems as performance-reference support
  - optional additional references where practical
- define which comparisons are:
  - correctness-only
  - performance-only
  - advisory, not shipping requirements

**Done when:** the repo has a bounded external comparison policy that does not
inflate portability claims.

## Stage 2: Fix the core storage and workflow ceiling

### 4. Audit all remaining linked-list-first workflow costs

- measure how often one-shot direct paths still start from
  `SparseMatrix`-centric mutation assumptions
- identify the highest-cost conversion and publication seams
- rank by user value, not by file size

**Done when:** the top conversion and shell-ownership seams are ranked with
clear evidence.

### 5. Design a compressed-first public workflow

- define the future role of the orthogonal linked-list shell
- define the compressed-first construction/import path
- keep a bounded mutable-construction story for callers that need it

**Done when:** there is one explicit product model where linked lists are a
bounded surface rather than the conceptual center of the library.

### 6. Add compressed-first construction/import entry points

- provide direct construction/import for CSR/CSC-backed workflows
- reduce needless round-trips through the linked-list shell
- preserve current callers through compatibility shims where needed

**Done when:** large direct workflows can begin in compressed form without the
linked-list shell being the default way in.

### 7. Tighten repeated-run lifecycle ownership

- reduce one-shot/repeated-run ambiguity in direct solver entry points
- make lifecycle-valid and shell-compatibility states easier to reason about
- ensure mutated shells cannot silently masquerade as solve-ready assets

**Done when:** the public workflow story is simpler and less surprising.

## Stage 3: Raise the dense/backend performance ceiling

### 8. Profile the dense-kernel consumers first

- identify which Cholesky / LDL^T / QR / SVD paths spend the most time in dense
  helpers
- quantify panel/update workloads before touching abstractions

**Done when:** the backend plan is driven by measured hotspots rather than
generic “use BLAS” intuition.

### 9. Design an optional backend layer

- define a narrow dense-kernel ABI
- keep the builtin backend as the default self-contained surface
- allow optional external acceleration without rewriting the product around it

**Done when:** the code can support external dense kernels without pretending
to be a giant plugin framework.

### 10. Integrate BLAS/LAPACK-class acceleration on the highest-value paths

- start with the densest supernodal/direct lanes
- keep the fallback builtin path fully working
- make runtime/backend observability explicit

**Done when:** benchmarked paths can actually exceed the current scalar kernel
ceiling on realistic workloads.

### 11. Refresh benchmark measurability

- expose backend selection and major kernel path in benchmark output
- add before/after comparison guidance without changing the threshold-free
  reading of the canonical report surface or creating fake portable timing
  gates

**Done when:** backend-aware performance changes become reviewable artifacts.

## Stage 4: Expand the capability envelope

### 12. Choose the next capability targets explicitly

- rank:
  - complex scalar support
  - mixed precision
  - wider index maturity
  - broader eigensolver surface
  - additional reusable iterative lifecycles

**Done when:** Epic 8 works on the highest-value capability gaps first instead
of widening everything shallowly.

### 13. Generalize the scalar contract beyond real-only

- introduce the minimum viable internal/public abstraction for additional
  scalar families
- avoid fake “generic everywhere” claims until real support exists

**Done when:** at least one bounded non-real or broader scalar lane is truly
supported, tested, and documented.

### 14. Mature the index-width contract

- review all remaining 32-bit assumptions
- strengthen 64-bit build and package confidence
- ensure overflow and allocation behavior stays explicit

**Done when:** larger-index support reads as a supported product lane, not just
as a compile-time seam.

### 15. Re-rank eigensolver and iterative breadth

- decide whether to broaden unsymmetric eigensolvers, iterative handles, or
  preconditioner combinations next
- avoid broadening all fronts at once

**Done when:** the next algorithm-surface expansion is narrow, tested, and
worth shipping.

## Stage 5: Strengthen assurance with external and generative proof

### 16. Add maintained differential tests for direct solvers

- compare selected direct-solver outputs and residuals against external
  references where feasible
- begin with the most stable SPD / LDL^T / QR seams

**Done when:** the project has at least one maintained external oracle lane for
core direct solves.

### 17. Expand seeded property testing beyond the current bounded lanes

- grow the large-`n` property model beyond the current Cholesky/LDL^T lifecycle
  subset
- keep seeds fixed and artifacts reviewable

**Done when:** property coverage is broader without becoming non-deterministic.

### 18. Add failure-path and cancellation differential checks

- test cancellation and error paths against stronger invariants
- include reorder/no-reorder, shell/public-lifecycle, and repeat-run cases

**Done when:** lifecycle and cancellation contracts are harder to regress.

### 19. Decide which assurance lanes can become reviewed cross-platform

- separate “great local proof” from “reviewed platform proof”
- expand only where evidence and runtime cost support it
- keep Linux as the strongest reviewed truth unless later evidence truly
  broadens that contract

**Done when:** Windows/macOS proof growth is bounded and truthful.

## Stage 6: Reduce maintainability concentration

### 20. Finish the next source decomposition pass

- start with `src/sparse_iterative.c`
- then address the highest-value remaining large direct-solver files
- extract helpers by ownership boundary, not by arbitrary line counts

**Done when:** the largest mixed-role sources are structurally easier to review.

### 21. Finish the next giant-test architecture pass

- start with `tests/test_ldlt_csc.c`
- then `tests/test_qr.c`
- keep family-local helper surfaces local instead of widening shared harnesses

**Done when:** proof ownership is clearer and `main()` registration walls are
smaller.

### 22. Remove stale chronology and sprint-history residue

- scrub permanent comments and support surfaces where old sprint naming still
  leaks into technical explanation
- keep durable technical rationale

**Done when:** permanent code/docs read more like product surfaces and less like
planning artifacts.

## Stage 7: Reduce the runtime long pole

### 23. Profile `test_reorder_nd` and reviewed runtime concentration

- measure where reviewed validation time is going
- decide what is algorithmic cost versus proof-surface organization cost

**Done when:** the long pole is decomposed into actionable causes.

### 24. Reduce ND/reordering reviewed runtime without weakening proof

- optimize hotspots where justified
- split or rebalance fixtures if the proof organization is the bigger issue
- keep parity and correctness stronger than speed-only wins

**Done when:** the reviewed baseline is materially faster or more scalable.

### 25. Add scalable performance-comparison artifacts for the reorder lane

- capture before/after measurements
- keep claims bounded and machine-class aware

**Done when:** reorder/runtime work can be justified with artifact evidence.

## Stage 8: Converge packaging, ABI, and cross-platform quality

### 26. Design a real shared/static product matrix

- decide whether the library will stay static-first permanently or support a
  maintained shared-library lane
- define the ABI promise level explicitly
- do not assume a shared lane will be added unless proof and packaging support
  it credibly

**Done when:** `BUILD_SHARED_LIBS` no longer reads like a rejected request
unless that is still the deliberate long-term decision.

### 27. Strengthen install/export and downstream consumer proof

- broaden `find_package(Sparse)` and `pkg-config` consumer coverage
- validate the installed surface on all maintained platforms where feasible
- keep reviewed-platform claims narrower than local convenience proof where the
  workflows remain intentionally asymmetric

**Done when:** downstream consumption is a stronger, more portable story.

### 28. Expand Windows/macOS proof only where it is sustainable

- add reviewed install or property lanes only where the toolchain/runtime is
  stable enough
- avoid performative parity claims

**Done when:** platform quality grows in credible, durable steps.

## Stage 9: Simplify front-door usability and public coherence

### 29. Redesign the README around user decisions, not policy density

- keep the front door focused on:
  - what the library is
  - which workflow to choose
  - how to build and consume it
  - what the strong and weak claims are
- move deep policy text down-stack where possible

**Done when:** a new user can answer the first three adoption questions quickly.

### 30. Simplify workflow teaching surfaces

- tighten tutorial, examples, and examples README
- separate adoption guidance from benchmark/proof governance

**Done when:** examples teach workflows without carrying excessive policy load.

### 31. Reduce internal-policy leakage in public headers

- move benchmark-derived thresholds and deep heuristics out of user-facing
  narrative where they are not part of the real contract
- keep advanced docs available elsewhere

**Done when:** headers are clearer, smaller, and more contract-focused.

## Stage 10: Final integration and claim calibration

### 32. Re-run the entire code/product review near the end of Epic 8

- repeat the same efficiency / maintainability / usability / coherence review
- compare against the Epic 8 opening assessment

**Done when:** the epic can prove improvement, not just report activity.

### 33. Benchmark against the chosen reference class

- compare correctness, usability, package shape, and performance against the
  bounded target competitors chosen in Stage 1
- state clearly where the library now matches, exceeds, or still trails
- do not widen this into “compare against everything” theater

**Done when:** the “state-of-the-art” claim is calibrated by evidence.

### 34. Close the residual queue honestly

- name what Epic 8 solved
- name what remains deferred
- keep non-claims explicit

**Done when:** Epic 8 ends with one truthful post-epic carry-forward queue.

## Sequencing Summary

If work has to be cut, do not cut from the front. The highest-value closure
order is:

1. baseline and external-oracle contract
2. storage/workflow ceiling
3. dense/backend ceiling
4. capability breadth
5. assurance expansion
6. maintainability hotspots
7. reviewed runtime concentration
8. packaging/platform convergence
9. front-door usability simplification
10. final comparison and claim calibration
