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

## Day 2

**Objective:** Refresh the lifecycle seam inventory against the current live
LU / Cholesky / LDLT / analysis / QR / SVD surfaces so Sprint 42's first
implementation batches are driven by the actual hidden-mutation and
precondition seams still present in code, headers, and user-facing docs.

### Commands Run

1. Re-read the Sprint 42 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/PLAN.md`
2. Re-read the Sprint 40 lifecycle baseline artifacts:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day5-lifecycle-inventory-lu-cholesky-ldlt.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day6-lifecycle-inventory-qr-svd-analysis-iterative-eigs.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day8-lifecycle-contract-map.md`
3. Sweep the current headers, implementations, and user-facing docs for
   lifecycle-sensitive seams:
   - `rg -n "identity permutations|original matrix|factored|cancel|progress|perm|row_perm|col_perm|sparse_factors_t|typedef struct sparse_factors|sparse_lu_factor|sparse_cholesky_factor|sparse_ldlt_factor|sparse_qr_factor|sparse_svd_compute|sparse_analyze|sparse_factor_numeric|sparse_refactor_numeric" include src README.md docs/tutorial.md`
4. Re-read the key lifecycle-sensitive header contracts:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
   - `sed -n '1,260p' include/sparse_qr.h`
   - `sed -n '1,260p' include/sparse_svd.h`

### Day 2 Findings

#### 1. LU and Cholesky remain the only high-risk matrix-as-factor-handle seams

The live header and implementation contracts still make LU and Cholesky the
clearest lifecycle outliers:

- factorization mutates the same `SparseMatrix` that later becomes the solve
  handle
- permutation state remains owned directly by the matrix object
- solve-readiness still depends on matrix-local state such as `factored`
- cancellation can leave the matrix in a non-original state before the first
  callback returns

Interpretation:

- LU and Cholesky are the strongest immediate Sprint 42 internal-handle
  insertion targets
- these are the seams where hidden lifecycle overloading is still the most
  severe

#### 2. LDLT and the analysis path are the main bridge class rather than direct mutation targets

The current LDLT / analysis family remains structurally split:

- LDLT already takes `const SparseMatrix *A` and writes into `sparse_ldlt_t`
- `sparse_analyze()` and `sparse_factor_numeric()` already expose explicit
  analysis/factor handles
- but `sparse_factors_t` still stores a matrix-centric payload:
  - `SparseMatrix *F`
  - plus LDLT side arrays

Interpretation:

- Sprint 42 should treat `sparse_factors_t` as a bridge-normalization seam,
  not as a full public redesign target
- the highest-value work here is payload ownership cleanup and compatibility
  scaffolding, not headline API replacement

#### 3. QR and SVD are already handle-oriented, but their lifecycle friction is precondition drift

The live QR/SVD family still fits the Sprint 40 pattern:

- factor/result state is externalized into:
  - `sparse_qr_t`
  - `sparse_svd_t`
- input matrices are read-only
- cancellation is clean relative to LU/Cholesky

The remaining friction is strict eligibility:

- identity permutations required
- original/unfactored matrix view required in practice
- implementation and docs enforce these rules through repeated bespoke checks

Interpretation:

- QR and SVD are primary Day 4 / Day 6 guard-helper adoption targets
- they do not need early internal-handle payload work on the same level as LU
  and Cholesky

#### 4. The main Day 2 seam split is now explicit: hidden mutation vs strict eligibility

The current live lifecycle queue reduces to two fundamentally different seam
classes:

- hidden mutable lifecycle overloading:
  - LU
  - Cholesky
- explicit-handle but strict eligibility burden:
  - LDLT
  - analysis / `sparse_factors_t`
  - QR
  - SVD

Interpretation:

- Sprint 42 should not treat every lifecycle-sensitive family as the same kind
  of refactor
- Day 3 handle design should center on the hidden-mutation class
- Day 4 guard design should center on the strict-eligibility class

#### 5. The immediate Sprint 42 landing order is now concrete

The refreshed landing order is:

1. LU and Cholesky internal ownership seams
2. shared original-state / identity-permutation / factored-state guard helpers
3. bounded adoption in LDLT / analysis / QR / SVD entry paths
4. initial `sparse_factors_t` bridge normalization
5. focused tests around misuse and cancellation

Interpretation:

- Sprint 42 should start where internal ownership ambiguity is highest, then
  spread the shared guard layer across the already handle-oriented paths
- this preserves the Sprint 40 internal-first rule while avoiding premature
  public-handle churn

## Day 3

**Objective:** Define the first concrete internal handle scaffolding for Sprint
42 by turning the Day 2 seam split into explicit ownership boundaries for LU,
Cholesky, and the `sparse_factors_t` bridge, while preserving current public
API behavior and leaving broader public explicit-handle work for later Epic 4
phases.

### Commands Run

1. Re-read the Sprint 42 Day 3 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/PLAN.md`
2. Re-read the Sprint 40 handle-model design inputs:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day9-handle-model-design-1.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day10-handle-model-design-2-and-migration-strategy.md`
3. Re-read the refreshed Sprint 42 lifecycle seam inventory:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_42/artifacts/day2-lifecycle-seam-refresh-inventory.md`

### Day 3 Findings

#### 1. Sprint 42 only needs three first-phase internal handle families

The Day 2 seam inventory supports a narrow first-phase internal object set:

- LU numeric payload handle
- Cholesky numeric payload handle
- `sparse_factors_t` bridge payload normalization seam

Interpretation:

- Sprint 42 does not need a broad internal object explosion
- the first handle work should stay focused on the seams where ownership is
  currently ambiguous or matrix-centric

#### 2. The immediate architectural goal is payload separation, not public API shape change

The first handle layer should change who owns numeric factor state internally,
not what callers call publicly:

- existing one-shot LU and Cholesky entry points remain the public wrapper
  surface
- existing analyze-once entry points remain the public bridge surface
- internal payloads become the true owners of factor-state data

Interpretation:

- Sprint 42 should land internal handle boundaries under the current public
  APIs
- wrapper preservation is load-bearing throughout Phase 1

#### 3. The stable keep/move split is now concrete enough for implementation

Keep on `SparseMatrix` during Sprint 42:

- coefficient/value storage semantics
- structural editing semantics
- matrix query/arithmetic behavior
- current compatibility-facing wrapper role where public APIs still accept
  matrix objects

Move behind first-phase internal handles or bridge payloads:

- LU/Cholesky numeric factor payload ownership
- factor-local solve-readiness state
- factor-local permutation/telemetry ownership where internal seams permit
- bridge-owned numeric payload inside `sparse_factors_t`

Interpretation:

- Sprint 42 now has a practical ownership split for implementation
- later public explicit-handle work can build on this split instead of having
  to invent it again

#### 4. `sparse_factors_t` should evolve as a preserve-and-normalize bridge, not be replaced

The bridge rule is now explicit:

- keep `sparse_factors_t` as the public factor-handle wrapper for the
  analyze-once workflow
- normalize what it owns internally before changing how callers reason about
  it
- treat it as the main compatibility scaffold between current matrix-centric
  internals and later explicit numeric-factor payloads

Interpretation:

- the analyze-once workflow remains one of the cleanest public lifecycle
  surfaces in the repo
- Sprint 42 should protect that public shape while reducing its matrix-centric
  internal coupling

#### 5. The Day 5 / Day 6 implementation boundary is now explicit

The first implementation batches should now divide cleanly:

- Day 5:
  - initial LU/Cholesky internal payload seam
- Day 6:
  - shared matrix-state guard helper layer and early adoption

That ordering matters because the handle seam and the guard seam solve
different lifecycle problems:

- handle seam:
  - ownership ambiguity
  - hidden mutation overloading
- guard seam:
  - eligibility drift
  - repeated precondition checks

Interpretation:

- Sprint 42 should not try to solve both seams with one abstraction
- Day 3 now fixes the ownership half before Day 4 defines the guard half
