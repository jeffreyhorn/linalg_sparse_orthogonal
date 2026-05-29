## Sprint 47 Day 9: Example Safety Audit & Batch Design

### Objective

Audit the example surface for unchecked arithmetic, weak helper patterns, and
stale auxiliary conventions after the Day 8 benchmark-front-half landing, then
choose a bounded Day 10 cleanup batch instead of carrying a generic "example
cleanup" backlog.

### Commands Run

1. Re-read the Sprint 47 Day 9-10 plan section:
   - `sed -n '269,340p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the touched small-example surfaces and the current example allocation
   helper seam:
   - `sed -n '1,260p' examples/example_eigs.c`
   - `sed -n '1,240p' examples/example_iterative.c`
   - `sed -n '1,220p' examples/example_matrix_free.c`
   - `sed -n '1,220p' examples/example_alloc_helpers.h`
3. Refresh the broader example queue and raw-allocation markers:
   - `rg --files examples`
   - `rg -n "malloc\\(|calloc\\(|strtol|strtod|atoi|example_[mc]alloc_array|argc|argv|SPARSE_ERR_ALLOC|sqrt\\(\\(double\\)n\\)" examples`
4. Re-read the strongest remaining raw-allocation candidates and the already
   aligned COLAMD example:
   - `sed -n '1,220p' examples/example_analysis.c`
   - `sed -n '1,220p' examples/example_condition.c`
   - `sed -n '1,280p' examples/example_ic_minres.c`
   - `sed -n '220,360p' examples/example_eigs.c`
   - `sed -n '1,220p' examples/example_colamd.c`
5. Re-read the current example README contract:
   - `sed -n '1,220p' examples/README.md`

### Findings

#### 1. The example surface is not one uniform cleanup target

The examples now fall into three practical classes:

- already aligned to the current helper/safety direction
- clear direct shared-helper adoption targets
- larger raw-allocation examples that would turn Day 10 into a broad rewrite

Interpretation:

- the right Day 10 move is a narrow helper-adoption batch
- the wrong move is to treat every example with a raw `malloc` or `calloc` as
  an equal-priority Sprint 47 target

#### 2. `example_iterative`, `example_matrix_free`, and `example_colamd` are already aligned

These examples already use `examples/example_alloc_helpers.h` where dynamic
scratch is part of the public example story:

- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`

They already match the current small-example pattern well enough:

- checked `idx_t` count handling via `example_malloc_array(...)` /
  `example_calloc_array(...)`
- no new Sprint 47 parsing drift
- no obvious helper duplication that justifies churn

Interpretation:

- these should stay intentionally untouched on Day 10 unless a very small
  follow-on falls out for free

#### 3. `example_eigs.c` is the strongest direct Day 10 target

`example_eigs.c` still repeats the same raw allocation pattern several times:

- `calloc((size_t)n * (size_t)k, sizeof(double))`
- `malloc((size_t)n * sizeof(double))`
- the same shape repeated again for the KKT and LOBPCG sections

Why it is the best direct target:

- the file is visible and high-signal
- the cleanup is mostly helper adoption, not algorithm redesign
- Sprint 41/47 helper conventions already provide the right seam
- the repeated bundle pattern gives Day 10 a coherent "before/after" result

Interpretation:

- `example_eigs.c` should be the primary Day 10 landing

#### 4. The remaining raw-allocation examples are real, but they are not the right first batch

The largest remaining raw-allocation examples are:

- `example_ic_minres.c`
- `example_analysis.c`
- `example_condition.c`

They do contain improvement opportunities:

- unchecked `(size_t)n` / `(size_t)n * (size_t)nrhs` allocation math
- repeated local scratch allocation patterns
- some opportunities to adopt the example helper seam

But they are weaker first targets for Sprint 47:

- `example_ic_minres.c` is substantially larger and spans three sub-demos
- `example_analysis.c` mixes timing/refactor workflow demonstration with local
  scratch ownership
- `example_condition.c` is tiny and pedagogical, so the gain from churn is low

Interpretation:

- these belong in the "real later cleanup" bucket, not the immediate bounded
  Day 10 batch

#### 5. Several examples should remain intentionally untouched in Sprint 47

The following examples do not currently justify Day 10 code churn:

- `example_basic_solve.c`
- `example_least_squares.c`
- `example_minnorm.c`
- `example_svd_lowrank.c`
- `example_ldlt.c`
- `examples/cmake_example/main.c`

Reason:

- they did not surface a strong shared-helper adoption need in this audit
- they are primarily compact public-usage references rather than auxiliary
  safety hotspots

Interpretation:

- leaving these alone is part of keeping Sprint 47 honest

### Day 10 Target Set

Primary target:

- `examples/example_eigs.c`

Allowed small follow-on only if it stays obviously narrow:

- one tiny helper-seam adoption in `example_condition.c`

Explicit non-targets for Day 10:

- `example_ic_minres.c`
- `example_analysis.c`
- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`
- broad example README churn

### Bottom Line

Sprint 47 does not need a broad example sweep. The example queue is now
concrete:

- primary direct shared-helper adoption target:
  - `example_eigs.c`
- already aligned and keep-as-is:
  - `example_iterative.c`
  - `example_matrix_free.c`
  - `example_colamd.c`
- real later cleanup surfaces, but not the right first batch:
  - `example_ic_minres.c`
  - `example_analysis.c`
  - `example_condition.c`

That gives Day 10 a bounded safety cleanup target instead of another generic
"touch the examples" day.
