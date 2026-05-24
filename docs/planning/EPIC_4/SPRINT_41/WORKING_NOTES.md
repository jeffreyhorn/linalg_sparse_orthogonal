# Sprint 41 Working Notes

## Day 1

**Objective:** Turn the Sprint 41 project-plan scope plus the Sprint 40
architecture contract and Epic 4 remediation plan into a concrete
baseline/setup package by confirming the preserved internal-first and
validation constraints, naming the Sprint 41 helper-consolidation workstreams
explicitly, and defining the authoritative hotspot/input surfaces before code
migration begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 41 plan and the main prerequisite artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/artifacts/day14-architecture-contract-synthesis.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day11-quality-contract-ownership-map.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
3. Re-read a representative prior Epic 4 Day 1 structure:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/artifacts/day1-baseline-and-scope.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day1-authoritative-inputs.txt`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
6. Confirm the Day 1 named hotspot modules exist in-tree:
   - `ls src/sparse_dense.c src/sparse_svd.c src/sparse_eigs.c src/sparse_etree.c`

### Day 1 Findings

#### 1. Sprint 41 starts from a preserved Epic 3/Sprint 40 baseline, not from missing quality infrastructure

The inherited starting contract remains stable and explicit:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- the Sprint 40 architecture contract is already documented:
  - internal-first groundwork
  - lifecycle/state taxonomy
  - handle-model migration strategy
  - quality-truth ownership map
  - validation anchor
  - public migration-risk boundaries

Interpretation:

- Sprint 41 is not rebuilding the reviewed-quality baseline
- Sprint 41 is a bounded internal helper/safety consolidation sprint layered on
  the preserved Sprint 40 contract

#### 2. The Sprint 41 workstreams are explicit and implementation order is already bounded

Day 1 confirms the sprint's seven workstreams directly from the plan:

- helper-pattern inventory
- shared utility design
- first core-module migration
- broader `src/` migration
- auxiliary-surface alignment
- prep-rule documentation
- validation closeout

Interpretation:

- the front half of the sprint should stay audit/design first:
  - pattern inventory
  - shared utility design
  - first bounded migration batch
- later work should remain scoped to helper consolidation rather than
  lifecycle/public API churn

#### 3. Sprint 40's ownership and validation contracts are load-bearing prerequisites, not optional context

Sprint 41 must preserve the following inherited rules:

- commands/wrapper truth remains owned by `Makefile`
- machine behavior remains owned by scripts
- CI matrix truth remains owned by workflow YAML
- concise operator summaries remain owned by `README.md`
- API/lifecycle semantics remain owned by headers/tutorial/examples
- any `*.c` / `*.h` refactor should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial refactors should still default to:
  - `make quality-review-full`
- dead-code execution remains serialized

Interpretation:

- helper consolidation cannot silently rewrite contract ownership
- Sprint 41 implementation must respect the validation floor that Sprint 40
  already anchored

#### 4. The first hotspot migration cluster is explicit and consistent across the plan and remediation review

The Day 1 named helper hotspots are:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

These match the first local helper-copy migration cluster already called out in
the Epic 4 remediation plan.

Interpretation:

- Sprint 41 starts from measured, pre-identified consolidation targets rather
  than an ad hoc repo-wide sweep
- the first migration cluster is narrow enough to stay behavior-preserving and
  internal-first

#### 5. The Day 1 preserve-not-reopen boundary is now clear

Sprint 41 is helper/safety groundwork, not early lifecycle-handle landing
work. Day 1 confirms that the sprint should not reopen:

- public migration-risk surfaces in:
  - `README.md`
  - `docs/tutorial.md`
  - lifecycle-sensitive installed headers
- cross-platform contract changes
- dead-code topology changes
- new reviewed-quality wrapper semantics

Interpretation:

- the correct Sprint 41 shape is:
  - inventory
  - design
  - internal consolidation
  - validation
- explicit handle enrichment, bridge normalization, and public doc
  reconciliation remain later Epic 4 work

## Day 2

**Objective:** Inventory the local allocation/overflow helper patterns in the
first Sprint 41 hotspot modules, classify them into explicit consolidation
buckets, and separate truly shared safety helpers from the file-specific logic
that should remain local during the first migration wave.

### Commands Run

1. Re-read the Sprint 41 Day 2 plan section and current working notes:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Sweep the Day 2 hotspot modules for allocation/overflow idioms:
   - `rg -n "overflow|SIZE_MAX|IDX_MAX|malloc|calloc|realloc|sizeof\\(|bytes|count|capacity|alloc" src/sparse_dense.c src/sparse_svd.c src/sparse_eigs.c src/sparse_etree.c`
3. Measure hotspot size for context:
   - `wc -l src/sparse_dense.c src/sparse_svd.c src/sparse_eigs.c src/sparse_etree.c`
4. Re-read the helper definitions and representative allocation sites:
   - `sed -n '1,120p' src/sparse_dense.c`
   - `sed -n '1,120p' src/sparse_svd.c`
   - `sed -n '140,220p' src/sparse_eigs.c`
   - `sed -n '1,120p' src/sparse_etree.c`
5. Re-read representative specialized cases:
   - `sed -n '150,240p' src/sparse_svd.c`
   - `sed -n '1080,1135p' src/sparse_eigs.c`
   - `sed -n '250,330p' src/sparse_etree.c`
   - `sed -n '560,640p' src/sparse_etree.c`
6. Re-sweep exact helper-family signals:
   - `rg -n "size_mul_overflow|alloc_would_overflow|SIZE_MAX -|> SIZE_MAX / sizeof|> SIZE_MAX /" src/sparse_dense.c src/sparse_svd.c src/sparse_eigs.c src/sparse_etree.c`

### Day 2 Findings

#### 1. The strongest direct-consolidation seam is the repeated `size_mul_overflow` family

Three of the four Day 2 hotspot modules carry their own local multiplication
guard with the same core semantics:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`

All three implement the same basic contract:

- inputs:
  - `size_t a`
  - `size_t b`
  - `size_t *out`
- return:
  - `0` on success
  - nonzero on overflow
- guard:
  - `a != 0 && b > SIZE_MAX / a`

Interpretation:

- this is the highest-confidence shared helper candidate for Day 3 design and
  Day 4 implementation
- the helper already exists as an implicit repository-wide idiom; Sprint 41's
  job is to stop carrying it as repeated file-local code

#### 2. The hotspot modules reduce cleanly into four helper-pattern buckets

Day 2's four planned buckets are now grounded in actual code:

- size multiplication overflow checks:
  - repeated `size_mul_overflow(...)`
  - repeated `count > SIZE_MAX / sizeof(T)` style guards
- `idx_t` / `size_t` representability checks:
  - `sparse_etree.c` cast-back validation from accumulated `size_t` totals
  - count/nnz values that must fit both allocation arithmetic and `idx_t`
- count-to-bytes conversions:
  - `elems -> bytes` derivation in SVD/eigs workspaces
  - direct `n * sizeof(T)` / `m*n*sizeof(T)` guards in dense/etree
- common allocation/free/reset helpers:
  - repeated allocate-many-then-free-on-failure blocks
  - repeated zeroing/init patterns after overflow validation

Interpretation:

- Sprint 41 is not just consolidating one multiplication helper
- it is consolidating a small safety-helper family, with clear subtypes that
  later design work can model separately

#### 3. `sparse_etree.c` is the main specialized branch, not a direct clone of the dense/SVD/eigs pattern

`src/sparse_etree.c` does carry repeated allocation-safety logic, but its main
patterns are different:

- single-dimension guard:
  - `alloc_would_overflow(idx_t n, size_t elem_size)`
- cumulative prefix-sum overflow checks:
  - `total_nnz > SIZE_MAX - cj`
  - `u_total > SIZE_MAX - cj`
- cast-back representability checks:
  - `(size_t)sym->col_ptr[j + 1] != total_nnz`
- zero-safe/nonzero-safe row-index allocation shapes:
  - `sym_U->nnz > 0 ? sym_U->nnz : 1`

Interpretation:

- `sparse_etree.c` should not be treated as a pure `size_mul_overflow`
  migration
- it likely needs:
  - a shared one-dimensional count-to-bytes helper
  - a shared accumulation/representability helper or documented keep-local
    decision
- its symbolic-structure accumulation rules are semantically different from
  dense workspace sizing and should remain explicit in the Day 3 API design

#### 4. `sparse_svd.c` and `sparse_eigs.c` share the strongest multi-buffer workspace pattern

The most reusable allocation-shape cluster is the workspace-pack style used in:

- `src/sparse_svd.c`
- `src/sparse_eigs.c`

Common structure:

- derive element counts with `size_mul_overflow`
- derive bytes from element counts with `size_mul_overflow(..., sizeof(T), ...)`
- allocate several sibling buffers together
- free the full sibling set on any failure

Representative examples include:

- SVD:
  - `mt*k`, `nt*k`, `m*k`, `n*k`, `m*m`, `n*n`, `kk*n`
- eigs:
  - `n*m_cap`, `m_cap*m_cap`, `n*k`, `n*block_size`

Interpretation:

- Day 3 should consider not only a low-level multiply helper, but also whether
  one or two tiny convenience helpers for:
  - `count -> bytes`
  - matrix/workspace element-count derivation
  are justified
- the free-on-failure blocks themselves are repeated, but they may not all
  belong in a single generic allocation helper if that would obscure current
  error/cleanup semantics

#### 5. `sparse_dense.c` is the best example of near-duplicate logic that still needs cleanup even where it does not use `size_mul_overflow`

`src/sparse_dense.c` mixes two styles today:

- local `size_mul_overflow(...)`
- older quotient-based overflow checks:
  - `mn / n != m`
  - `n > SIZE_MAX / sizeof(double)`

Interpretation:

- Day 2 confirms that Sprint 41 must normalize not only repeated helper
  definitions but also repeated manual arithmetic idioms
- `sparse_dense.c` is a strong first migration target because it contains both:
  - an identical shared-helper candidate
  - older manual patterns that should collapse onto the same utility layer

## Day 3

**Objective:** Turn the Day 2 duplication map into a concrete internal utility
API proposal by choosing the target private file layout, splitting helpers into
inline vs normal-function tiers, and defining the explicit keep-local boundary
for specialized symbolic/zero-size logic before Day 4 implementation begins.

### Commands Run

1. Re-read the Sprint 41 Day 3 plan section and current working notes:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Re-read the Day 2 inventory artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day2-helper-pattern-inventory.md`
3. Sweep the current `src/` surface and exact helper-copy sites:
   - `rg --files src | sort`
   - `rg -n "static int size_mul_overflow|static inline int alloc_would_overflow|size_mul_overflow\\(|alloc_would_overflow\\(" src`
4. Re-read representative internal helper/layout conventions:
   - `sed -n '1,220p' src/sparse_errno_internal.h`
   - `sed -n '1,220p' src/sparse_matrix_internal.h`
   - `sed -n '1,180p' src/sparse_bicgstab_internal.h`
   - `sed -n '1,120p' src/sparse_qr.c`
5. Reconfirm build-surface implications for a new internal helper header/source:
   - `rg -n "LIB_SRCS|src/.*\\.c|sparse_qr.c|sparse_dense.c|sparse_eigs.c|sparse_etree.c|sparse_svd.c" Makefile CMakeLists.txt`
6. Re-read the relevant Epic 4 remediation/project-plan guidance:
   - `rg -n "helper|allocation|overflow|size_mul_overflow|alloc_would_overflow|representability" docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md docs/planning/EPIC_4/PROJECT_PLAN.md docs/planning/EPIC_4/SPRINT_40/artifacts/day14-architecture-contract-synthesis.md`

### Day 3 Findings

#### 1. The utility layer should be a private `src/` helper pair, not a public API surface

The cleanest Day 4 landing shape is:

- private header:
  - `src/sparse_alloc_internal.h`
- private implementation file:
  - `src/sparse_alloc_internal.c`

Reasons:

- the work is explicitly internal-first
- current private naming conventions already support focused `*_internal.h`
  surfaces
- the new source file can be added cleanly to both:
  - `Makefile` `LIB_SRCS`
  - `CMakeLists.txt` library source list

Interpretation:

- Sprint 41 should not overload `sparse_matrix_internal.h` with another large
  mixed-purpose utility section
- a dedicated private helper layer keeps the Day 4 implementation bounded and
  easy to audit

#### 2. The API should be tiered: inline arithmetic helpers in the header, allocation wrappers in the source

The strongest split is:

- `static inline` helpers in `sparse_alloc_internal.h` for tiny pure
  arithmetic/bounds operations
- normal functions in `sparse_alloc_internal.c` only where a shared allocation
  wrapper materially improves consistency

Recommended header tier:

- `sparse_size_mul_overflow(size_t a, size_t b, size_t *out)`
- `sparse_size_add_overflow(size_t a, size_t b, size_t *out)`
- `sparse_count_bytes_overflow(size_t count, size_t elem_size, size_t *bytes)`
- `sparse_idx_count_bytes_overflow(idx_t count, size_t elem_size, size_t *bytes)`
- `sparse_size_to_idx_checked(size_t value, idx_t *out)`

Recommended source-tier candidates:

- `sparse_malloc_array(size_t count, size_t elem_size, void **out)`
- `sparse_calloc_array(size_t count, size_t elem_size, void **out)`

Interpretation:

- the repeated `size_mul_overflow` family belongs in header-inline form
- small shared alloc wrappers are justified only where they preserve current
  semantics cleanly and avoid another wave of ad hoc `malloc/calloc` error
  translation

#### 3. Sprint 41 should avoid a macro-heavy interface

Day 3 does not support a macro-first design.

Why:

- arithmetic helpers have clear typed signatures already
- macros would obscure evaluation and debugging for no real gain
- the repo already uses `static inline` successfully for small internal helper
  logic

Design decision:

- no public-facing macros
- no typed allocation macros as the primary interface
- use normal C helpers with explicit types/signatures instead

Interpretation:

- Sprint 41's goal is consolidation and clarity, not a preprocessor layer
- the helper API should remain inspectable and easy to call-site-diff during
  migration

#### 4. Zero-size policy, symbolic prefix sums, and file-local cleanup choreography should remain caller-owned

Day 2's specialization split is load-bearing here. The shared helper layer
should not try to hide:

- zero-size object policy:
  - e.g. `NULL` data for empty dense matrices
  - explicit `calloc(1, ...)` for empty symbolic column-pointer storage
  - `nnz > 0 ? nnz : 1` sentinel allocation shapes
- symbolic prefix-sum accumulation:
  - `total_nnz > SIZE_MAX - cj`
  - `u_total > SIZE_MAX - cj`
- module-specific sibling-buffer cleanup sequences

Interpretation:

- the shared layer should centralize arithmetic and byte-derivation truth
- it should not erase legitimate differences in symbolic-storage semantics or
  lifecycle-specific cleanup ownership

#### 5. Day 4 and the broader Sprint 41 queue now have concrete insertion points

First-wave Day 4 / Day 5 / Day 6 insertion points remain:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

But the Day 3 design also now explicitly anticipates the broader queue:

- `src/sparse_qr.c` already carries the same `size_mul_overflow` idiom
- later broader `src/` migration should be able to adopt the same utility layer
  without redesign

Interpretation:

- the Sprint 41 utility layer should be designed for first-wave use now and
  broader `src/` reuse shortly after
- Day 3's design is therefore deliberately small but not one-off

## Day 4

**Objective:** Implement the new shared internal helper layer from Day 3,
wire it into the build, and prove it is practical with a narrow low-risk
integration batch before the broader Sprint 41 migration days begin.

### Commands Run

1. Re-read the Day 3 design and current Sprint 41 notes:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/artifacts/day3-shared-utility-api-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Re-read the current low-risk proof candidates and build-surface locations:
   - `sed -n '1,120p' src/sparse_dense.c`
   - `sed -n '1,120p' src/sparse_qr.c`
   - `sed -n '35,80p' Makefile`
   - `sed -n '72,105p' CMakeLists.txt`
3. Sweep the helper-copy and allocation call sites used for the first proof
   batch:
   - `rg -n "size_mul_overflow|malloc\\(|calloc\\(" src/sparse_qr.c src/sparse_dense.c`
4. Implement the shared helper layer and wire it into both build systems:
   - `src/sparse_alloc_internal.h`
   - `src/sparse_alloc_internal.c`
   - `Makefile`
   - `CMakeLists.txt`
5. Replace the first low-risk local helper uses:
   - `src/sparse_dense.c`
   - `src/sparse_qr.c`
6. Re-sweep the touched source surface:
   - `rg -n "size_mul_overflow|sparse_size_mul_overflow|sparse_calloc_array|sparse_malloc_array|sparse_alloc_internal" src/sparse_dense.c src/sparse_qr.c src/sparse_alloc_internal.h src/sparse_alloc_internal.c`
7. Run the required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 4 Findings

#### 1. The shared helper layer now exists as a concrete private `src/` module

Day 4 landed the planned private helper pair:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

The header now owns the shared arithmetic/bounds tier:

- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

The source now owns the initial allocation-wrapper tier:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`

Interpretation:

- Sprint 41 has moved from design to a real shared internal helper layer
- the landing shape matches the Day 3 private-header/private-source model

#### 2. The new helper layer is wired into both maintained build surfaces

Day 4 added `src/sparse_alloc_internal.c` to:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` library source list

Interpretation:

- the helper layer is not a local experiment in one command path
- it is now part of the maintained library build in both direct and CMake
  flows

#### 3. The first proof batch stayed narrow and behavior-preserving

The Day 4 proof batch intentionally touched only two low-risk integration
surfaces:

- `src/sparse_dense.c`
  - `dense_create()` now uses the shared helper path for:
    - `rows * cols` validation
    - dense storage allocation through `sparse_calloc_array(...)`
  - one existing local multiplication call site was switched to
    `sparse_size_mul_overflow(...)`
- `src/sparse_qr.c`
  - removed the file-local `size_mul_overflow(...)`
  - switched existing overflow checks to `sparse_size_mul_overflow(...)`

Interpretation:

- Day 4 proved both helper tiers in live code:
  - inline arithmetic helpers
  - source-backed allocation wrapper
- it did so without opening the broader Day 5/6 first-wave migration scope

#### 4. The implementation needed two small safety-style cleanups during validation, both inside the new helper source

The first `make lint` pass surfaced two helper-layer issues:

- clang-analyzer rejected a zero-byte `calloc(...)` path
- cppcheck flagged a redundant post-zero-size overflow branch

Day 4 fixed both by tightening the shared helper behavior:

- zero-size requests now return `SPARSE_OK` with `*out = NULL` without calling
  `malloc` / `calloc`
- `sparse_calloc_array(...)` now uses the same validated-bytes path as the
  malloc wrapper and allocates with `calloc(1, bytes)`

Interpretation:

- the Day 4 gate did real work on the new helper layer rather than just
  rubber-stamping it
- the resulting helper surface is stricter and cleaner than the first draft

#### 5. The required Day 4 code-quality gate passed completely

After the two helper-source cleanups, the full required gate passed:

- `make format`
- `make lint`
- `make test`

Interpretation:

- the new private helper module builds cleanly in the maintained library
  surfaces
- the narrow proof batch did not introduce behavior drift in the full direct
  test suite
- Sprint 41 can now move into the first planned hotspot migration days from a
  validated helper baseline

## Day 5

**Objective:** Apply the new shared helper layer to the first real hotspot
pair by removing duplicated overflow-multiplication helpers from
`src/sparse_svd.c` and `src/sparse_eigs.c`, migrating their shared
count-to-bytes guard sites onto the Day 4 utility layer, and explicitly
leaving their larger multi-buffer workspace cleanup choreography local.

### Commands Run

1. Re-read the Sprint 41 plan and current working notes:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Audit the first hotspot pair for duplicated helper definitions and live
   call sites:
   - `rg -n "size_mul_overflow|SIZE_MAX / sizeof|SIZE_MAX /|malloc\\(|calloc\\(" src/sparse_svd.c src/sparse_eigs.c`
   - `sed -n '1,120p' src/sparse_svd.c`
   - `sed -n '140,220p' src/sparse_eigs.c`
3. Recheck the migration state after the first edit pass:
   - `git status --short --branch`
   - `sed -n '1,60p' src/sparse_eigs.c`
   - `rg -n "size_mul_overflow|sparse_size_mul_overflow" src/sparse_svd.c src/sparse_eigs.c`
4. Run the required code-quality gate because `*.c` changed:
   - `make format`
   - `make lint`
   - `make test`
5. Capture the final code delta and write the Day 5 notes:
   - `git diff -- src/sparse_svd.c src/sparse_eigs.c`
   - `git diff --stat`

### Day 5 Findings

#### 1. The first real hotspot pair was the right Day 5 target

Day 2 had already identified the strongest shared migration pair as:

- `src/sparse_svd.c`
- `src/sparse_eigs.c`

Day 5 confirmed why:

- both still carried their own local `size_mul_overflow(...)`
- both used that helper repeatedly for:
  - element-count multiplication
  - count-to-bytes derivation
  - multi-buffer workspace sizing
- both were large enough that removing the duplicate helper meaningfully
  shrinks local safety-code repetition without forcing a broader redesign

Interpretation:

- this was the first migration batch with real leverage
- it stayed within Sprint 41's bounded helper-consolidation scope

#### 2. Day 5 removed the duplicated local multiply guards from both hotspot modules

The code migration batch landed the same shared-helper pattern in both files:

- removed the file-local `size_mul_overflow(...)` helper from:
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
- moved their shared overflow guard sites to:
  - `sparse_size_mul_overflow(...)`

Day 5 also completed the integration seam cleanly in `src/sparse_eigs.c` by:

- adding the private helper include:
  - `#include "sparse_alloc_internal.h"`
- updating one retained explanatory comment to name the shared helper rather
  than the removed local helper

Interpretation:

- Day 4's helper layer is now proven in the first major hotspot pair, not just
  in low-risk proof files
- the shared arithmetic helper is now an actual consolidation tool rather than
  unused internal infrastructure

#### 3. The migrated call sites are the shared arithmetic seam, not the broader workspace choreography

The migrated sites in `src/sparse_svd.c` cover the repeated shared arithmetic
cases:

- economy/full U/V workspace sizes
- bidiagonal diagonal/superdiagonal byte sizing
- Lanczos partial-SVD workspace sizing
- sigma/output allocation size guards
- low-rank / pseudoinverse dense-buffer size checks

The migrated sites in `src/sparse_eigs.c` cover the same seam in eigensolver
workspaces:

- Lanczos vector/work buffer bytes
- thick-restart `V`, `Y`, and `K×K` dense scratch sizing
- locked-vector growth sizing
- LOBPCG block workspace sizing

Interpretation:

- the migrated code is the true shared arithmetic seam Day 2 measured
- Day 5 intentionally did not try to genericize:
  - multi-buffer sibling allocation ordering
  - per-file cleanup labels / goto choreography
  - file-specific ownership and cancellation cleanup paths

#### 4. The residual local-helper boundary is now cleaner and more explicit

After Day 5, the remaining specialized local logic in these files is no longer
“each file owns its own generic multiply-overflow helper.” The remaining local
ownership is narrower:

- SVD-specific:
  - bidiagonal / Lanczos workspace composition
  - low-rank dense reconstruction policy
  - output-shape-dependent cleanup sequencing
- eigs-specific:
  - Lanczos/thick-restart/LOBPCG workspace family composition
  - restart-state and block-method buffer ownership
  - algorithm-specific cleanup/error propagation sequencing

Interpretation:

- later Sprint 41 work can now focus on truly file-specific allocation
  choreography and representability helpers
- the generic arithmetic safety layer is no longer a hotspot-local concern in
  the first migrated pair

#### 5. The required Day 5 gate passed completely

Because `src/sparse_svd.c` and `src/sparse_eigs.c` changed, the full required
gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- the first real hotspot migration batch preserved behavior
- the shared helper seam is safe in large solver modules, not just in small
  proof integrations
- Sprint 41 can move into the broader `src/` migration work from a validated
  first-wave landing

## Day 6

**Objective:** Finish the planned first-wave hotspot migration set by moving
the remaining named hotspot module and the lingering manual byte-count drift
onto the shared helper layer, then record the broader `src/` gap list that
still belongs to later Sprint 41 work.

### Commands Run

1. Re-read the Sprint 41 plan and the Day 5 handoff state:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '582,760p' docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Re-audit the remaining first-wave surfaces and their helper drift:
   - `rg -n "alloc_would_overflow|SIZE_MAX -|SIZE_MAX / sizeof|size_mul_overflow|sparse_size_mul_overflow|malloc\\(|calloc\\(" src/sparse_etree.c src/sparse_dense.c src/sparse_svd.c src/sparse_eigs.c src/sparse_qr.c`
   - `sed -n '1,220p' src/sparse_alloc_internal.h`
   - `sed -n '1,220p' src/sparse_etree.c`
   - `sed -n '220,760p' src/sparse_etree.c`
   - `sed -n '1,140p' src/sparse_dense.c`
3. Recheck the final Day 6 helper surface after editing:
   - `rg -n "alloc_would_overflow|SIZE_MAX / sizeof|SIZE_MAX -|sparse_idx_count_bytes_overflow|sparse_count_bytes_overflow|sparse_malloc_array|sparse_calloc_array" src/sparse_etree.c src/sparse_dense.c`
   - `git diff -- src/sparse_dense.c src/sparse_etree.c`
4. Run the required code-quality gate because `*.c` changed:
   - `make format`
   - `make lint`
   - `make test`
5. Capture the broader post-Day-6 gap list:
   - `rg -n "SIZE_MAX / sizeof|SIZE_MAX -|malloc\\(\\(size_t\\)|calloc\\(\\(size_t\\)|sparse_size_mul_overflow|sparse_malloc_array|sparse_calloc_array" src | sed -n '1,240p'`
   - `git diff --stat`

### Day 6 Findings

#### 1. Day 6 completed the planned first-wave hotspot set

The Sprint 41 first-wave hotspot list from Day 1 was:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

Status after Day 6:

- `src/sparse_svd.c`:
  - migrated in Day 5
- `src/sparse_eigs.c`:
  - migrated in Day 5
- `src/sparse_dense.c`:
  - initial proof landing in Day 4, reconciled in Day 6
- `src/sparse_etree.c`:
  - migrated in Day 6

Interpretation:

- the planned first-wave hotspot set is now complete
- Day 7 can correctly shift to the broader `src/` audit rather than carrying
  unfinished first-wave work

#### 2. `src/sparse_etree.c` now uses the shared helper layer for its generic allocation-safety seam

Day 6 removed the file-local `alloc_would_overflow(...)` helper and replaced
the generic n-based allocation seam with the shared helper layer:

- added the private helper include:
  - `#include "sparse_alloc_internal.h"`
- removed the local overflow helper entirely
- migrated repeated array allocation sites to:
  - `sparse_malloc_array(...)`
  - `sparse_calloc_array(...)`
- migrated count/bytes and accumulation checks to:
  - `sparse_count_bytes_overflow(...)`
  - `sparse_idx_count_bytes_overflow(...)`
  - `sparse_size_add_overflow(...)`
  - `sparse_size_to_idx_checked(...)`

The migrated `etree` families include:

- core etree and postorder work arrays
- child-list and marker/tmp arrays
- symbolic Cholesky `col_ptr` / `row_idx` sizing
- symbolic LU bridge arrays and U-structure accumulation sizing

Interpretation:

- the remaining named hotspot module is no longer carrying its own generic
  overflow/allocation helper
- Day 2's “shared arithmetic/bytes helpers, keep-local symbolic semantics”
  split held up cleanly in live code

#### 3. `src/sparse_dense.c` reconciled the remaining manual byte-count drift

Day 4 had already proven the helper layer in `dense_create()`, but `dense_gemm`
and `dense_gemv` still carried hand-written byte-count logic. Day 6 aligned
those with the shared helper style:

- `dense_gemm(...)` now uses:
  - `sparse_size_mul_overflow(...)`
  - `sparse_count_bytes_overflow(...)`
  for both the zero-sized matrix fast path and the normal output buffer path
- `dense_gemv(...)` now uses:
  - `sparse_count_bytes_overflow(...)`
  for `y` byte sizing
- `tridiag_qr_eigenpairs(...)` now uses:
  - `sparse_malloc_array(...)`
  for its sort/permutation scratch buffers

Interpretation:

- the first-wave hotspot list now has a more consistent helper-ownership model
- Day 6 was not just “migrate etree”; it also closed the lingering style/semantics
  mismatch inside the already-touched dense hotspot

#### 4. The keep-local boundary is still explicit after the first-wave completion

Day 6 did **not** flatten specialized logic that still belongs inside
`src/sparse_etree.c` or `src/sparse_dense.c`, including:

- symbolic-structure algorithms
- etree/postorder traversal semantics
- symbolic LU/Cholesky bridge logic
- dense eigensolver/tridiagonal algorithm structure
- file-specific cleanup/error-propagation choreography

Interpretation:

- first-wave migration is complete without turning Sprint 41 into a broad
  allocator-framework rewrite
- the shared helper layer now owns the generic safety seam, while algorithmic
  ownership remains local

#### 5. Day 6 produced a concrete broader `src/` gap list for Day 7

The post-Day-6 broader sweep leaves a cleaner next queue, led by:

- `src/sparse_ic.c`
- `src/sparse_iterative.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `src/sparse_graph.c`

The remaining broader pressure points are now clearer:

- direct `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))`
  families
- manual `SIZE_MAX / sizeof(...)` guards
- a few remaining `SIZE_MAX - ...` accumulation checks
- modules that already use `sparse_size_mul_overflow(...)` but have not yet
  adopted the source-backed shared wrappers

Interpretation:

- Day 7 can now separate easy direct substitutions from true specialized keeps
- the first-wave migration no longer obscures the broader audit queue

#### 6. The required Day 6 gate passed completely

Because `src/sparse_dense.c` and `src/sparse_etree.c` changed, the full
required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- Sprint 41 now has a validated completed first-wave hotspot migration
- the remaining Sprint 41 work can focus on the broader source-tree audit and
  migration passes rather than reopening the initial hotspot set

## Day 7

**Objective:** Turn the post-Day-6 residual helper queue into an explicit
broader `src/` migration map by separating easy direct substitutions from
moderate adapter-heavy cases and true specialized keep/defer cases, then
record the next concrete migration order for Days 8 and 9.

### Commands Run

1. Re-read the Sprint 41 plan's Day 7 scope and the latest Day 6 handoff:
   - `rg -n "Day 7" -A20 docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `tail -n 120 docs/planning/EPIC_4/SPRINT_41/WORKING_NOTES.md`
2. Reconfirm the broader `src/` duplication map after Day 6:
   - `rg -n "SIZE_MAX / sizeof|SIZE_MAX -|malloc\\(\\(size_t\\)|calloc\\(\\(size_t\\)|sparse_size_mul_overflow|sparse_malloc_array|sparse_calloc_array" src | sed -n '1,320p'`
3. Reconfirm size concentration among the larger source modules:
   - `wc -l src/*.c | sort -nr | sed -n '1,20p'`
4. Inspect representative next-queue modules to classify migration difficulty:
   - `sed -n '1,220p' src/sparse_ic.c`
   - `sed -n '1,260p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_iterative.c`
   - `sed -n '1,260p' src/sparse_graph.c`

### Day 7 Findings

#### 1. The broader `src/` queue is real, but it is not one flat migration class

After the first-wave hotspot completion, the broader pressure is still
concentrated in familiar patterns:

- direct `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))`
- manual `SIZE_MAX / sizeof(...)` guards
- manual `SIZE_MAX - ...` accumulation checks
- modules that already use some shared helpers but still own large local
  allocation seams

But the representative file reads make the migration shape clearer:

- some modules are now near-mechanical substitutions
- some modules need helper adoption tied to larger local workspace choreography
- some modules are still too algorithmically dense for Sprint 41's
  “shared safety seam only” boundary

Interpretation:

- Day 7 should produce a priority map, not a generic “migrate more files”
  note
- Days 8 and 9 can now target the highest-value safe modules first

#### 2. `src/sparse_ic.c` is the clearest next direct-substitution candidate

`src/sparse_ic.c` still carries a compact local seam:

- paired `SIZE_MAX / sizeof(...)` guards for `double` and `idx_t`
- `calloc((size_t)n, sizeof(double))`
- `malloc((size_t)n * sizeof(idx_t))`
- `calloc((size_t)n, sizeof(char))`

The file is otherwise structurally simple in the relevant area:

- one main factorization entry point
- straightforward n-based workspace ownership
- local cleanup that does not depend on a custom allocator framework

Interpretation:

- `src/sparse_ic.c` is the strongest Day 8 candidate
- it should be handled as an easy direct substitution, not deferred behind
  larger hotspot files

#### 3. `src/sparse_analysis.c` is also a strong near-term migration target

`src/sparse_analysis.c` still has repeated manual n-based allocation logic:

- repeated `SIZE_MAX / sizeof(idx_t)` checks
- repeated `malloc((size_t)n * sizeof(idx_t))`
- scratch/permutation arrays with clear count semantics

The key distinction from `sparse_graph.c` or `sparse_iterative.c` is that
these allocation sites are still mostly narrow and structurally obvious:

- permutation arrays
- etree/postorder arrays
- compact analysis-side work arrays

Interpretation:

- `src/sparse_analysis.c` belongs in the same near-term batch as
  `src/sparse_ic.c`
- it is the best second proof that the shared helper layer scales beyond the
  initial hotspot set without requiring helper-layer redesign

#### 4. `src/sparse_iterative.c` is high-value, but it is a moderate adapter-heavy case

`src/sparse_iterative.c` is one of the largest remaining source files and
still carries many manual safety/allocation patterns:

- `SIZE_MAX / sizeof(...)` guards
- `SIZE_MAX - total` accumulation logic
- large packed workspaces for Krylov methods
- per-solver scratch ownership interleaved with progress/cancel/result logic

This is no longer a “simple malloc replacement” module. The allocation seam is
deeply tied to:

- packed workspace layout
- solver-specific cleanup paths
- progress/cancel behavior
- residual-history and stagnation helper state

Interpretation:

- `src/sparse_iterative.c` should remain a top Sprint 41 target
- but it should be scheduled after the easier `ic` / `analysis` pair
- it belongs in Day 9 or later within the sprint, not the first broader batch

#### 5. `src/sparse_graph.c` is the clearest specialized keep/defer case

`src/sparse_graph.c` remains the largest file in `src/` and carries a
multi-algorithm allocation surface:

- graph/subgraph construction
- coarsening hierarchies
- partition/refinement state
- separator/workspace arrays
- dense comment/history context around multiple algorithm families

Even in the first representative slice, the file already shows:

- parent/child graph build ownership
- custom mapping arrays
- CSR-style structural construction
- multiple locally meaningful scratch lifetimes

Interpretation:

- `src/sparse_graph.c` should not be used as a routine Day 8 migration target
- it is the strongest “specialized keep/defer” example in the broader queue
- later work there should likely be bundled with a more focused maintainability
  or decomposition pass, not just helper substitution

#### 6. Day 7 produces a concrete Days 8-9 order instead of a generic backlog

The broader `src/` queue now separates cleanly into:

- easy direct substitutions:
  - `src/sparse_ic.c`
  - `src/sparse_analysis.c`
- moderate helper-adapter cases:
  - `src/sparse_iterative.c`
  - `src/sparse_qr.c`
  - likely follow-ons such as `src/sparse_lu.c`, `src/sparse_lu_csr.c`,
    `src/sparse_chol_csc.c`, and `src/sparse_ldlt_csc.c`
- specialized keep/defer cases:
  - `src/sparse_graph.c`
  - selected reorder/symbolic-heavy files where allocation meaning is tightly
    bound to algorithm structure

Interpretation:

- Day 8 should target:
  - `src/sparse_ic.c`
  - `src/sparse_analysis.c`
- Day 9 should target:
  - `src/sparse_iterative.c`
  - optionally `src/sparse_qr.c` if the live batch remains bounded
- `src/sparse_graph.c` should stay out of the routine Sprint 41 migration path

#### 7. Day 7 is intentionally docs-only and does not require the code-quality gate

No `*.c` or `*.h` files changed today. The work was:

- broader `src/` helper-duplication classification
- next-batch prioritization
- keep/defer boundary documentation

Interpretation:

- the full `make format` / `make lint` / `make test` gate was not required
- the right output for Day 7 is the migration decision package that keeps
  Days 8-9 bounded and behavior-preserving

## Day 8

**Objective:** Land the first broader `src/` migration batch by moving the
highest-value low-risk pair from Day 7 — `src/sparse_ic.c` and
`src/sparse_analysis.c` — onto the shared helper layer while keeping the
batch behavior-preserving and confirming whether the Day 4 helper API needs
extension.

### Commands Run

1. Re-read the Sprint 41 Day 8 scope and the Day 7 priority map:
   - `sed -n '220,280p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day7-broader-src-migration-audit.md`
2. Inspect the two Day 8 target modules and the current shared helper layer:
   - `sed -n '1,260p' src/sparse_ic.c`
   - `sed -n '1,360p' src/sparse_analysis.c`
   - `sed -n '360,520p' src/sparse_analysis.c`
   - `sed -n '1,240p' src/sparse_alloc_internal.h`
   - `sed -n '1,240p' src/sparse_alloc_internal.c`
3. Reconfirm the live manual allocation/overflow sites in the target files:
   - `rg -n "SIZE_MAX / sizeof|malloc\\(|calloc\\(" src/sparse_ic.c`
   - `rg -n "SIZE_MAX / sizeof|malloc\\(|calloc\\(" src/sparse_analysis.c`
4. Run the required full validation gate after the code changes:
   - `make format`
   - `make lint`
   - `make test`
5. Review the resulting source diff and final branch state:
   - `git diff -- src/sparse_ic.c src/sparse_analysis.c`
   - `git status --short --branch`

### Day 8 Findings

#### 1. Day 8 landed the planned first broader `src/` migration pair directly

The broader Day 7 priority order held up cleanly in live code:

- migrated:
  - `src/sparse_ic.c`
  - `src/sparse_analysis.c`
- did not mix in:
  - `src/sparse_iterative.c`
  - `src/sparse_qr.c`
  - `src/sparse_graph.c`

Interpretation:

- Sprint 41 stayed on the bounded Day 7 migration order
- Day 8 is a real broader `src/` consolidation pass, not a reopened
  first-wave hotspot batch

#### 2. `src/sparse_ic.c` was a true direct-substitution success case

Day 8 migrated the compact IC(0) allocation seam onto the shared helper layer:

- added:
  - `#include "sparse_alloc_internal.h"`
- replaced the local workspace guards/allocations for:
  - `val`
  - `pattern`
  - `in_pat`
- adopted:
  - `sparse_calloc_array(...)`
  - `sparse_malloc_array(...)`

What did **not** change:

- factorization algorithm
- error propagation
- cleanup choreography
- solve semantics

Interpretation:

- the Day 7 classification was correct
- `src/sparse_ic.c` was a strong low-risk broader proof target

#### 3. `src/sparse_analysis.c` also fit the shared helper layer without redesign

Day 8 migrated the remaining direct analysis-side allocation seams:

- added:
  - `#include "sparse_alloc_internal.h"`
- replaced manual n-based allocation in:
  - `apply_supernodal_postorder(...)`
  - permutation storage
  - `etree` / `postorder` arrays
  - `cc` scratch storage
  - `b_perm`
  - `x_tmp`
- adopted:
  - `sparse_malloc_array(...)`

What stayed local:

- reorder dispatch
- symbolic-analysis ownership
- factor-type dispatch
- solve semantics and permutation meaning

Interpretation:

- the shared helper layer now covers a bridge/lifecycle-sensitive module, not
  just the first-wave hotspot files
- Day 8 broadened helper ownership without blurring Sprint 41 into lifecycle
  refactor work

#### 4. Day 8 did not require a helper-layer extension

One explicit Day 8 question was whether the live broader migration would prove
the Day 4 helper API too narrow. It did not.

The existing helper set was sufficient:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

Interpretation:

- the current shared helper layer is still the right size for Sprint 41
- Day 9 can stay focused on broader migration, not helper-API redesign

#### 5. Day 8 preserves the keep-local boundary from Sprint 41's design

The migration moved only the generic safety seam. It did **not** attempt to
rewrite:

- IC(0) symbolic/factor ownership
- analysis/factor dispatch
- permutation meaning
- symbolic etree/postorder semantics
- solve/result orchestration

Interpretation:

- Sprint 41 remains an internal helper consolidation sprint
- the batch is consistent with the Sprint 40 architecture contract and the
  Day 3 helper-layer design

#### 6. Day 9 is now narrowed further rather than expanded

After the Day 8 batch, the strongest remaining broader mainline target is:

- `src/sparse_iterative.c`

The next optional follow-on remains:

- `src/sparse_qr.c`

And the specialized defer remains:

- `src/sparse_graph.c`

Interpretation:

- Day 8 did not surface a missing-helper emergency that would reorder the
  broader queue
- Day 9 can stay centered on `src/sparse_iterative.c`

#### 7. The required Day 8 gate passed completely

Because `src/sparse_ic.c` and `src/sparse_analysis.c` changed, the full
required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- the first broader `src/` migration batch is validated
- Sprint 41 can proceed to the next broader pass from a clean state

## Day 9

**Objective:** Finish the main broader `src/` helper-consolidation pass by
landing the next mainline target from the Day 8 handoff, confirming whether
the shared helper layer still covers the live migration seam cleanly, and
recording what remains intentionally local or deferred after the broader pass.

### Commands Run

1. Re-read the Sprint 41 Day 9 scope and the Day 8 handoff:
   - `sed -n '244,320p' docs/planning/EPIC_4/SPRINT_41/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day8-broader-src-migration-batch1.md`
2. Reconfirm the residual manual allocation/overflow surface in the Day 9 queue:
   - `rg -n "SIZE_MAX / sizeof|SIZE_MAX -|malloc\\(|calloc\\(" src/sparse_iterative.c src/sparse_qr.c`
3. Inspect the full iterative-solver seam across its main workspace families:
   - `sed -n '1,420p' src/sparse_iterative.c`
   - `sed -n '420,780p' src/sparse_iterative.c`
   - `sed -n '780,1160p' src/sparse_iterative.c`
   - `sed -n '1160,1440p' src/sparse_iterative.c`
   - `sed -n '1440,1560p' src/sparse_iterative.c`
4. Recheck the post-edit residual allocation surface in the main target:
   - `rg -n "SIZE_MAX / sizeof|SIZE_MAX -|malloc\\(|calloc\\(" src/sparse_iterative.c`
5. Run the required full validation gate after the code changes:
   - `make format`
   - `make lint`
   - `make test`

### Day 9 Findings

#### 1. Day 9 completed the main broader `src/` target rather than widening the batch

The live Day 9 decision held to the Day 8 handoff:

- migrated:
  - `src/sparse_iterative.c`
- explicitly did **not** include:
  - `src/sparse_qr.c`
  - `src/sparse_graph.c`

Interpretation:

- Sprint 41 finished the main broader pass without turning Day 9 into a second
  sprawling mixed-module batch
- the bounded Day 7/8/9 order held up cleanly in live code

#### 2. `src/sparse_iterative.c` absorbed the shared helper layer across all main workspace families

Day 9 moved the remaining generic allocation/overflow seam in
`src/sparse_iterative.c` onto the shared helper layer, including:

- stagnation tracker allocation
- CG packed workspaces
- matrix-free CG packed workspaces
- GMRES fast-path and initial-residual scratch buffers
- GMRES Hessenberg/Arnoldi packed workspace allocation
- block-CG per-column and packed workspaces
- MINRES packed workspace allocation

The batch adopted:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`

Interpretation:

- the main Day 9 target was completed at the generic safety seam level
- the file no longer carries the broad manual `malloc` / `calloc` /
  `SIZE_MAX` drift that Day 7 identified

#### 3. The shared helper layer still did not require redesign

Day 9 was the strongest remaining test of the Day 4 helper API because
`src/sparse_iterative.c` mixes:

- packed workspaces
- multi-solver ownership
- matrix-free paths
- block-solver paths
- callback/progress/cancellation behavior

Even with that broader pressure, the current helper layer was still enough:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`

Interpretation:

- Sprint 41 can now say the shared helper layer is sufficient for the current
  intended coverage set
- no Day 9 helper-interface drift surfaced that would justify redesign inside
  this sprint

#### 4. Day 9 preserved the local algorithm/workspace boundary

The migration did **not** rewrite or normalize:

- Krylov method structure
- solver/result semantics
- progress/cancel behavior
- residual-history logic
- block-solver orchestration
- matrix-free dispatch meaning

Only the generic safety seam moved.

Interpretation:

- Sprint 41 remained within the internal-first helper-consolidation contract
- Day 9 did not drift into workspace-API redesign or algorithm refactoring

#### 5. The resulting Sprint 41 local-specialization keep/defer list is now explicit

After Day 9, the remaining named broader surfaces separate into:

- still local / deferred for later bounded work:
  - `src/sparse_qr.c`
  - `src/sparse_graph.c`

The reasons differ:

- `src/sparse_qr.c`
  - still a real helper-alignment candidate
  - but broader and more mixed than the bounded Day 9 target
- `src/sparse_graph.c`
  - still the clearest specialized keep/defer case because allocation meaning
    is tightly bound to graph-structure and multilevel algorithm ownership

Interpretation:

- the main Sprint 41 source-tree consolidation target is complete
- what remains local is now intentional rather than accidental

#### 6. The required Day 9 gate passed completely

Because `src/sparse_iterative.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- the broader mainline `src/` consolidation pass is validated
- Sprint 41 can now move on to the auxiliary-surface audit from a clean state
