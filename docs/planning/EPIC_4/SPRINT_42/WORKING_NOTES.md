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

## Day 4

**Objective:** Define the shared internal matrix-state guard helper layer for
Sprint 42 by turning the repeated factored/non-identity/original-state checks
into a bounded internal validation seam, while leaving algorithm-specific
numerical and structural checks local to their current factorization families.

### Commands Run

1. Re-read the Sprint 42 Day 4 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/PLAN.md`
2. Re-read the Day 2 and Day 3 Sprint 42 design inputs:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_42/artifacts/day2-lifecycle-seam-refresh-inventory.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_42/artifacts/day3-internal-handle-scaffolding-design.md`
3. Sweep the current codebase for duplicated lifecycle-sensitive precondition
   checks:
   - `rg -n "identity permutations|non-identity|factored|original.*matrix|must have identity|has identity|has non-identity|A->factored|has_identity_perms|sparse_row_perm\\(|sparse_col_perm\\(" include src`
4. Re-read representative duplicated guard implementations:
   - `sed -n '100,180p' src/sparse_analysis.c`
   - `sed -n '20,70p' src/sparse_ic.c`
   - `sed -n '20,80p' src/sparse_ilu.c`
   - `sed -n '550,590p' src/sparse_qr.c`
   - `sed -n '1000,1035p' src/sparse_svd.c`

### Day 4 Findings

#### 1. The repeated seam is small enough for one shared internal guard layer

The code sweep confirms that Sprint 42 does not need a broad lifecycle
validation framework. The repeated seam is narrowly concentrated around:

- factored-state rejection
- identity row/column permutation checks
- "original matrix / original-state required" as a semantic wrapper over those
  concrete checks

Interpretation:

- Day 4 should define one small internal guard layer, not a large policy
  framework
- the strongest value is reducing duplicated bespoke loops and ad hoc guard
  wording drift

#### 2. The natural design split is shared lifecycle-state helpers plus local algorithm checks

The live code already suggests the right boundary:

- shared checks:
  - matrix not factored
  - identity row/column permutations
  - common "original-state required" gate
- local checks:
  - symmetry / SPD checks
  - square/rectangular shape checks
  - reorder-enum validation
  - algorithm-specific numerical or structural assumptions

Interpretation:

- Sprint 42 should centralize only the lifecycle-state checks that are
  semantically common
- it should not hide algorithm-specific validation behind a generic helper
  layer

#### 3. The first shared helper targets are now concrete enough to name

The Day 4 guard layer should be designed around a small private seam such as:

- identity-permutation predicate/helper
- "matrix must be in original state" validator
- "matrix must already be factored" validator
- compatibility-consistent error-return helpers for touched families

Interpretation:

- the design target is a small internal helper family, likely in a private
  `src/` header/source pair
- the layer should be usable both by direct-factor families and by the already
  handle-oriented QR/SVD/analysis families

#### 4. The first adoption matrix is now explicit

Primary Day 6 adoption targets:

- LU
- Cholesky
- LDLT
- analysis
- QR
- SVD

Near-adjacent likely follow-ons once the helper exists:

- ILU / ILUT / IC
- selected CSC factor backends where they currently mirror the same entry
  guards

Interpretation:

- Day 6 can land useful adoption immediately without needing a repo-wide
  migration sweep
- the helper layer is broad enough to justify its existence but still bounded

#### 5. The guard layer must preserve current user-visible semantics, not reinterpret them

The current families mostly agree on outward behavior already:

- bad lifecycle state returns `SPARSE_ERR_BADARG`
- matrix-shape and matrix-class errors remain separate
- original-state requirements remain part of the public semantic contract even
  when implemented through factored/permutation checks

Interpretation:

- the guard layer should preserve current error semantics
- its purpose is to normalize implementation and wording drift, not change the
  public contract in Sprint 42

## Day 5 - Internal handle scaffolding batch 1

### What I changed

Day 5 moved from design to the first real internal lifecycle-handle landing.
The batch stayed intentionally narrow and internal-first:

- added a private factor-state seam in:
  - `src/sparse_factor_state_internal.c`
  - `src/sparse_matrix_internal.h`
- added private factor-state storage to `SparseMatrix`
- wired matrix create / copy / free / invalidation to the new seam in
  `src/sparse_matrix.c`
- migrated the linked-list LU path in `src/sparse_lu.c` onto the seam
- migrated the linked-list Cholesky path in `src/sparse_cholesky.c` onto the
  seam
- added the new helper source to:
  - `Makefile`
  - `CMakeLists.txt`

The important boundary held:

- public APIs did not change
- `factored` and `factor_norm` remain compatibility-visible matrix fields
- the new seam is internal ownership scaffolding, not a public handle rollout

### What the new seam does

The private layer now provides:

- LU / Cholesky factor-state binding
- compatibility-preserving publication of:
  - factored state
  - factor norm
- clone support for `sparse_copy()`
- clear/reset support for touched matrix mutation paths

Interpretation:

- Sprint 42 now has a real first-phase handle seam instead of another design
  note
- later guard-helper and bridge-normalization work can build on a live internal
  ownership object

### Validation and one important operational caveat

Because `*.c` / `*.h` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

The first `make test` pass failed in `test_chol_csc` writeback round-trip
coverage, but the failure was not a real Day 5 logic regression.

Root cause:

- Day 5 changed the private `SparseMatrix` layout
- the local incremental build did not rebuild every consumer of
  `src/sparse_matrix_internal.h`
- `src/sparse_chol_csc.c` was still linked against the pre-Day-5 layout during
  that first pass

I verified this directly, then reran the authoritative validation from a clean
tree:

- `make clean`
- `make format`
- `make lint`
- `make test`

Authoritative result:

- all passed

Interpretation:

- the Day 5 code batch is valid
- the only snag was stale-object local validation after a private struct-layout
  change
- Sprint 42 should treat the clean rebuild result as authoritative

### Day 5 conclusion

Day 5 landed the intended Sprint 42 ownership seam:

- LU and Cholesky now publish factor-state through a shared private layer
- matrix lifecycle operations understand that layer
- the Sprint 40 architecture contract is still preserved:
  - internal-first
  - compatibility-preserving
  - no premature public lifecycle/API churn

## Day 6 - Matrix-state guard helper implementation

### What I changed

Day 6 implemented the shared matrix-state guard seam designed on Day 4 and
landed the first live adoption set.

Added:

- `src/sparse_matrix_state_internal.h`

The new private helper seam provides:

- `sparse_matrix_has_identity_row_col_perms(...)`
- `sparse_matrix_has_identity_perms(...)`
- `sparse_matrix_require_original_row_col_state(...)`
- `sparse_matrix_require_original_state(...)`
- `sparse_matrix_require_factored_state(...)`

The first-wave adoption set landed in:

- `src/sparse_analysis.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_bidiag.c`
- `src/sparse_ilu.c`
- `src/sparse_ic.c`
- `src/sparse_cholesky.c`
- `src/sparse_lu.c`

The batch removed repeated bespoke checks for:

- original matrix required
- identity row/column permutations required
- in the touched Cholesky path, full original permutation state required
- factored matrix required in touched solve-side LU / Cholesky paths

### Boundary that stayed intentionally local

Day 6 did **not** try to genericize all validation.

The shared helper seam owns only:

- original-state required
- identity-permutation required
- factored-state required

Algorithm-specific checks remain local:

- symmetry / SPD
- shape/dimension compatibility
- reorder options
- numeric thresholds
- symbolic/storage assumptions

Interpretation:

- the new seam reduces lifecycle guard drift
- it does not create a fuzzy catch-all validation layer

### Validation

Because `*.c` / `*.h` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

Unlike Day 5, this batch did not need a clean-tree rerun. The standard gate
passed directly after the Day 6 implementation work.

### Day 6 conclusion

Sprint 42 now has a real shared matrix-state guard layer in live code:

- the first-wave lifecycle-sensitive families no longer each interpret the
  same original-state rules independently
- touched factored-state checks now align better with the Day 5 private
  factor-state seam
- the Sprint 40 compatibility contract is still preserved:
  - internal-first
  - no public API churn
  - stable `SPARSE_ERR_BADARG` lifecycle failures

## Day 7 - Factor-path landing audit

### What I audited

Day 7 turned the live Sprint 42 code into a bounded landing order for Days 8
through 10 instead of treating every lifecycle-sensitive family as the same
kind of problem.

I re-read the current factor-entry and bridge surfaces in:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

The key Day 7 question was not "where are factors involved?" It was:

- where the Day 5 private factor-state seam is already strong enough to
  support direct normalization
- where a small bridge adapter is still needed first
- which families are already sufficiently guard-complete for Sprint 42

### Main classification result

The current factor-path set now separates cleanly into three groups.

#### 1. Ready for direct normalization

- LU one-shot matrix path
- Cholesky one-shot matrix path

Why:

- both now have the Day 5 private factor-state seam
- both now have Day 6 factored/original-state guard normalization in the
  touched entry points
- both still preserve the public one-shot matrix API, so Sprint 42 can
  tighten internal ownership/publication without public churn

Interpretation:

- Day 8 should target LU and Cholesky first

#### 2. Bridge paths needing minor local adapters

- `sparse_factors_t` analyze-once bridge in `src/sparse_analysis.c`
- LDLT analyze-once / CSC bridge follow-ons where they support the above
- Cholesky CSC writeback/publication seam

Why:

- `sparse_factors_t` is already the public compatibility bridge
- but it still packages a matrix-centric payload:
  - `SparseMatrix *F`
  - LDLT-specific side arrays
- `sparse_factor_numeric`, `sparse_factor_solve`, and `sparse_factor_free`
  still do ad hoc bridge assembly/reconstruction work
- the CSC writeback path still republishes factor state adjacent to, rather
  than fully through, the Day 5 private seam

Interpretation:

- Day 9 should center on bounded `sparse_factors_t` normalization
- LDLT and CSC follow-ons should only enter as small adapters that make that
  bridge cleaner

#### 3. Guard-complete or lower-priority for Sprint 42

- QR
- SVD
- symbolic `sparse_analyze` entry

Why:

- Day 6 already gave QR and SVD the shared original-state seam they needed
- both already externalize results/handles rather than overloading
  `SparseMatrix` the way LU/Cholesky do
- `sparse_analyze` itself is no longer the main lifecycle problem; the bridge
  around `sparse_factor_numeric` is

Interpretation:

- QR/SVD are important preserved surfaces, but not the main Day 8/9 ownership
  targets

### `sparse_factors_t` readiness result

Day 7 confirms that Sprint 42 can safely begin bounded bridge normalization
around `sparse_factors_t` because:

- the public bridge object already exists
- the Day 5 factor-state seam is now live
- the Day 6 lifecycle-state seam is now live
- the current bridge logic is concentrated in `src/sparse_analysis.c`

The important limit is equally explicit:

- Sprint 42 should normalize implementation-side ownership and handoff
- it should not redesign the installed public shape of `sparse_factors_t`

### Day 8-10 landing order fixed

Day 7 now fixes the main implementation order:

- Day 8:
  - LU / Cholesky direct path normalization
- Day 9:
  - bounded `sparse_factors_t` bridge normalization
  - small LDLT / CSC adapter follow-ons only if they directly support that
    bridge cleanup
- Day 10:
  - cancellation / mutation contract normalization across the touched direct
    and bridge paths

### Day 7 conclusion

Sprint 42's next lifecycle batches are now bounded correctly:

- LU and Cholesky are the direct normalization targets
- `sparse_factors_t` is the main bridge seam
- LDLT is mostly a follow-on adapter surface, not the primary ownership
  rewrite target
- QR and SVD can remain stable while the higher-pressure lifecycle seams are
  cleaned up

## Day 8 - Factor-path normalization batch 1

### What I changed

Day 8 landed the first direct normalization batch from the Day 7 landing map:

- LU one-shot matrix path
- Cholesky one-shot matrix path
- bounded Cholesky CSC writeback/publication alignment

The main code change was to strengthen the private factor-state seam as the
touched authoritative internal publication path.

Added to the private helper layer in:

- `src/sparse_matrix_internal.h`
- `src/sparse_factor_state_internal.c`

New helpers:

- `sparse_factor_state_begin_lu(...)`
- `sparse_factor_state_begin_cholesky(...)`
- `sparse_factor_state_replace_reorder_perm(...)`
- `sparse_factor_state_publish_factored(...)`

Then adopted them in:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`

### What normalized

The Day 8 batch removed the touched publication drift in three places.

#### LU

- factor entry now starts through `sparse_factor_state_begin_lu(...)`
- touched reorder-permutation replacement now routes through
  `sparse_factor_state_replace_reorder_perm(...)`

#### Cholesky

- linked-list factor entry now starts through
  `sparse_factor_state_begin_cholesky(...)`
- touched reorder-permutation replacement now routes through
  `sparse_factor_state_replace_reorder_perm(...)`
- the CSC dispatch path now also binds/resets the private Cholesky seam before
  symbolic/numeric CSC work begins

#### Cholesky CSC writeback

- writeback precondition now uses the shared Day 6 original-state helper
- empty and non-empty writeback completion now publish through
  `sparse_factor_state_publish_factored(...)`
  instead of setting:
  - `reorder_perm`
  - `factor_norm`
  - `factored`
  by hand

Interpretation:

- the CSC Cholesky path no longer bypasses the Day 5 private seam on its final
  publication step
- LU and Cholesky now look more like one internal lifecycle family in the
  touched entry/publication paths

### What stayed intentionally unchanged

Day 8 did **not** widen into:

- public API changes
- `sparse_factors_t` bridge normalization
- broader LDLT ownership work
- QR/SVD ownership churn
- cancellation-contract rewriting

Interpretation:

- the batch stayed exactly in the Day 7 scope
- Day 9 can now focus on the analyze-once bridge rather than reopening direct
  LU/Cholesky entry work

### Validation

Because `*.c` / `*.h` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

Authoritative result:

- all passed

One compile issue surfaced on the first pass:

- `sparse_cholesky_factor_opts(...)` reused `payload_err` in the CSC path
  without declaring it locally

I fixed that immediately and reran the full required gate from the top. The
rerun passed completely.

### Day 8 conclusion

Sprint 42 now has a cleaner direct LU / Cholesky lifecycle publication path:

- factor entry starts through dedicated private seam helpers
- touched reorder-permutation ownership routes through one helper
- CSC Cholesky writeback now publishes through the same internal seam instead
  of bypassing it

## Day 9 - Factor-path normalization batch 2

### What I changed

Day 9 landed the bounded analyze-once bridge normalization batch chosen on Day
7:

- `sparse_factors_t` bridge cleanup in `src/sparse_analysis.c`
- small LDLT bridge normalization where it directly supported that cleanup
- no public shape changes in `include/sparse_analysis.h`

The batch stayed intentionally local to `src/sparse_analysis.c`.

Added new private bridge helpers:

- `sparse_factors_init_payload(...)`
- `sparse_factors_take_matrix_factor(...)`
- `sparse_factors_take_ldlt_factor(...)`
- `sparse_factors_make_ldlt_view(...)`

I also brought working-copy sanitation onto the Day 8 private factor-state seam
by replacing direct `reorder_perm` cleanup with:

- `sparse_factor_state_replace_reorder_perm(...)`

### What normalized

The Day 9 batch removed the main ad hoc bridge assembly/reconstruction drift in
three places.

#### Working-copy sanitation

- `sanitize_working_copy(...)` now clears owned reorder-permutation state
  through `sparse_factor_state_replace_reorder_perm(...)`

Interpretation:

- working-copy cleanup now routes through the same ownership seam already used
  by the touched LU / Cholesky publication paths

#### LU / Cholesky analyze-once factor handoff

- `sparse_factor_numeric(...)` now initializes bridge payload state through
  `sparse_factors_init_payload(...)`
- LU and Cholesky factor ownership transfer now routes through
  `sparse_factors_take_matrix_factor(...)`

Interpretation:

- bridge payload setup is no longer open-coded at each factorization case
- the analyze-once LU / Cholesky path now pulls factor norm from the private
  factor-state seam rather than duplicating local handoff logic

#### LDLT bridge assembly and solve-view reconstruction

- LDLT ownership transfer now routes through
  `sparse_factors_take_ldlt_factor(...)`
- `sparse_factor_solve(...)` now rebuilds the temporary LDLT solve view through
  `sparse_factors_make_ldlt_view(...)` instead of open-coding each field

Interpretation:

- the LDLT bridge remains compatibility-preserving, but its implementation-side
  packaging is now centralized
- the analyze-once bridge no longer spreads touched LDLT handoff logic across
  multiple independent blocks

### What stayed intentionally unchanged

Day 9 did **not** widen into:

- public `sparse_factors_t` redesign
- installed-header changes in `include/sparse_analysis.h`
- broader LDLT API churn
- QR / SVD ownership changes
- cancellation / mutation contract rewriting

Interpretation:

- the batch stayed exactly on the Day 7 Day 9 target
- Day 10 can now focus on cancellation / mutation normalization rather than
  bridge ownership drift

### Validation

Because `*.c` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

Authoritative result:

- all passed

### Day 9 conclusion

Sprint 42 now has a cleaner analyze-once bridge path without changing the
public bridge object:

- working-copy permutation cleanup now routes through the private factor-state
  seam
- LU / Cholesky bridge payload setup is centralized
- LDLT bridge ownership transfer and solve-view reconstruction are centralized
- `sparse_factors_t` remains compatibility-preserving while the implementation
  seam becomes more uniform

## Day 10 - Cancellation and mutation contract normalization

### What I changed

Day 10 landed the bounded contract-normalization batch Sprint 42 planned for
the touched lifecycle paths:

- direct LU cancellation / pre-mutation failure cleanup
- direct and CSC Cholesky state-entry timing cleanup
- analyze-once bridge output-commit cleanup
- focused lifecycle-contract regression coverage

The batch stayed intentionally narrow:

- no public API redesign
- no broader lifecycle rewrite
- no QR / SVD / LDLT ownership expansion
- no broad README/tutorial churn

### What normalized

The Day 10 batch reduced drift in three main places.

#### 1. LU now restores compatibility state on pre-mutation exits

Touched files:

- `src/sparse_matrix_internal.h`
- `src/sparse_factor_state_internal.c`
- `src/sparse_lu.c`

Day 10 changes:

- private factor-state payloads now snapshot prior compatibility mirrors:
  - previous `factored`
  - previous `factor_norm`
- added `sparse_factor_state_restore_compat(...)`
- LU now restores those mirrors on pre-mutation exits such as:
  - immediate cancellation before any in-loop mutation
  - pre-mutation singular / early error exits

Interpretation:

- the direct LU path is now more internally consistent with its actual mutation
  boundary
- immediate cancellation no longer leaves the factor-state mirrors drifted if
  the matrix body was still untouched by the inner factor path
- this still does **not** undo any fill-reducing reorder already applied by
  `sparse_lu_factor_opts(...)`

#### 2. Cholesky now delays state-entry until the path is truly crossing its mutation seam

Touched file:

- `src/sparse_cholesky.c`

Day 10 changes:

- linked-list Cholesky now validates symmetry, computes `||A||_inf`, and
  allocates its local work buffers before entering the private factor-state
  seam
- CSC Cholesky now delays `sparse_factor_state_begin_cholesky(...)` until after
  successful symbolic / CSC working-format preparation and numeric CSC
  elimination, just before writeback to `mat`
- linked-list cancellation comments now say directly that the upper triangle is
  already stripped before the first callback emission

Interpretation:

- Cholesky still has its load-bearing in-place mutation contract
- but the code now enters the compatibility-state seam closer to the real
  lifecycle boundary instead of mutating that state too early on preparatory
  failures

#### 3. The analyze-once bridge now commits output only on success

Touched files:

- `src/sparse_analysis.c`
- `tests/test_etree.c`

Day 10 changes:

- `sparse_factor_numeric(...)` now factors into a local `new_factors` payload
  first
- the caller-provided `sparse_factors_t` is only freed/replaced after full
  success
- added focused regression coverage proving that an existing successful factor
  object remains usable after a later `sparse_factor_numeric(...)` failure

Interpretation:

- the analyze-once bridge now has the same success-only commit shape already
  used by `sparse_refactor_numeric(...)`
- failure no longer leaves a partially rewritten bridge object behind

#### 4. Focused lifecycle-contract assertions are now explicit in tests

Touched files:

- `tests/test_integration.c`
- `tests/test_etree.c`

Day 10 additions:

- LU cancel-at-step-0 regression now asserts the cancelled matrix is rejected
  by solve
- Cholesky cancel-at-step-0 regression now asserts the cancelled matrix is
  rejected by solve
- analyze-once failure regression now asserts the old factors still solve
  correctly after a failed replacement attempt

Interpretation:

- the touched lifecycle semantics are now exercised directly rather than living
  only in comments

### What stayed intentionally unchanged

Day 10 did **not** widen into:

- public lifecycle-header redesign beyond small contract wording cleanup
- broader README/tutorial reconciliation
- full public `sparse_factors_t` redesign
- QR / SVD / LDLT cancellation work
- reorder rollback or full bit-identical restoration after already-mutating
  paths

Interpretation:

- the batch stayed on the Sprint 42 Day 10 target
- remaining broader lifecycle wording or public-handle work still belongs to
  later Epic 4 phases

### Validation

Because `*.c` / `*.h` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

Authoritative result:

- all passed

### Day 10 conclusion

Sprint 42's touched lifecycle paths now express a more consistent internal
contract:

- LU restores compatibility-state mirrors on pre-mutation exits
- Cholesky enters the private factor-state seam closer to its actual mutation
  boundary
- the analyze-once bridge now commits factor output only on success
- focused tests now assert the touched cancel/failure semantics directly
