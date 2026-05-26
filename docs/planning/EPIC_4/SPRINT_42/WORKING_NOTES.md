# Sprint 42 Working Notes

## Day 1

**Objective:** Turn the Sprint 42 project-plan scope plus the Sprint 40
lifecycle/handle-model contract and the Sprint 41 closeout into a concrete
baseline/setup package by confirming the preserved internal-first,
compatibility, and validation constraints, naming the Sprint 42 workstreams
explicitly, and defining the authoritative seam/input surfaces before
lifecycle-handle implementation begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 42 plan and main prerequisite artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day14-architecture-contract-synthesis.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day14-closeout-and-handoff.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`

### Day 1 Findings

#### 1. Sprint 42 starts from a preserved Sprint 40/Sprint 41 baseline, not from quality-contract repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized
- Sprint 41 already left behind a shared internal arithmetic/allocation seam:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`

Interpretation:

- Sprint 42 is not reopening the quality baseline
- Sprint 42 is a lifecycle-handle and compatibility-groundwork sprint layered
  on top of a preserved reviewed baseline and the Sprint 41 helper package

#### 2. The Sprint 42 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's eight workstreams directly from the plan:

- lifecycle seam inventory refresh
- internal handle scaffolding
- matrix-state guard helpers
- factor-path normalization
- cancellation-contract normalization
- compatibility bridge planning
- focused lifecycle tests
- validation closeout

Interpretation:

- the sprint front half should stay audit/design first:
  - seam inventory
  - handle scaffolding design
  - matrix-state guard design
- implementation should then land through bounded internal seams rather than
  broad public-contract churn

#### 3. Sprint 40's lifecycle and migration contracts are load-bearing prerequisites, not just context

Sprint 42 inherits the following architecture rules directly from Sprint 40:

- LU and Cholesky are the first major lifecycle-handle landing targets
- `sparse_factors_t` remains the main bridge normalization seam
- wrapper preservation is load-bearing for direct-factorization families
- cancellation and copy-before-reuse remain first-class migration risks
- `README.md`, `docs/tutorial.md`, examples, and lifecycle-sensitive headers are
  the main public migration-sensitive surfaces
- any `*.c` / `*.h` refactor should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial refactors should still default to:
  - `make quality-review-full`

Interpretation:

- Sprint 42 implementation must preserve compatibility while changing
  internals
- lifecycle groundwork is not permission to reshape public contracts
  opportunistically

#### 4. Sprint 41's shared-helper layer is now an implementation prerequisite for Sprint 42

Sprint 41 closed with a validated shared safety seam and a proven migration
pattern:

- shared helper seam:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- proven migration pattern already landed in:
  - `src/sparse_etree.c`
  - `src/sparse_analysis.c`
  - `src/sparse_iterative.c`
- explicit exception/defer classes remain:
  - symbolic accumulation choreography may stay local
  - file-specific cleanup choreography may stay local
  - specialized hotspot work like `src/sparse_graph.c` remains separate

Interpretation:

- Sprint 42 should reuse the shared helper seam by default
- Sprint 42 does not need to rediscover the internal-first execution rules that
  Sprint 41 already validated

#### 5. The highest-risk Day 1 lifecycle seam cluster is explicit before code changes begin

The main Day 1 high-risk seams already handed forward are:

- LU factorization internals
- Cholesky factorization internals
- `sparse_factors_t`
- lifecycle-sensitive precondition checks across:
  - QR
  - SVD
  - analysis
  - direct factorization paths

Interpretation:

- the first implementation cluster is lifecycle-boundary work, not general
  maintainability cleanup
- Sprint 42 should begin with seam refresh and handle/guard design before
  attempting factor-path code movement

#### 6. The Day 1 preserve-not-reopen boundary is clear

Sprint 42 is an internal-handle and compatibility-scaffolding sprint. Day 1
confirms that it should not reopen:

- broad public API redesign
- public explicit-handle rollout beyond compatibility scaffolding
- cross-platform contract changes
- dead-code topology changes
- unrelated hotspot decomposition work such as the larger graph queue

Interpretation:

- the correct Sprint 42 shape is:
  - inventory
  - design
  - bounded internal landing work
  - focused tests
  - validation
- broader public-handle enrichment remains later Epic 4 work
