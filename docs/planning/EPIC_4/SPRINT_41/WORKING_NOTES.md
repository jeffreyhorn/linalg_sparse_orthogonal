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
