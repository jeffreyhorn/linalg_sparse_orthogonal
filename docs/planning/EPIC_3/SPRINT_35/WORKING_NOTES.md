# Sprint 35 Working Notes

## Day 1

**Objective:** Convert the Sprint 35 project-plan items into a concrete public-surface audit baseline by reconfirming the Sprint 34 enforced-state invariants, inventorying installed headers / README / tutorial / example surfaces, and identifying the highest-signal documentation drift before any rewrite work begins.

### Commands Run

1. Read the Sprint 35 scope and Sprint 34 closeout inputs:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/RETROSPECTIVE.md`
2. Inventory current public-surface files:
   - `find include -maxdepth 1 -name '*.h' | sort`
   - `find examples -maxdepth 1 -name '*.c' | sort`
   - `find docs -maxdepth 2 -type f \( -name '*.md' -o -name '*.txt' \) | sort`
   - `ls INSTALL* 2>/dev/null || true`
3. Inspect likely public-usage and stale-pattern hotspots:
   - `rg -n "sparse_.*opts_t|reorder|quality-review|deadcode|designated|example" include README.md docs examples INSTALL* benchmarks/README.md -g '!build/**'`
   - `rg -n "sparse_.*opts_t opts = \\{|sparse_.*opts_t [A-Za-z_]+ = \\{|\\.reorder =|quality-review|deadcode-check|make examples|make tooling-build" include README.md docs/tutorial.md docs/algorithm.md examples/README.md examples/*.c -g '!build/**'`
   - `rg -n "typedef struct .*sparse_cg_opts_t|sparse_cg_opts_t|sparse_ilu_opts_t|sparse_iter_opts_t|sparse_gmres_opts_t" include src docs/tutorial.md README.md examples -g '!build/**'`
   - `sed -n '150,260p' docs/tutorial.md`
   - `sed -n '1,120p' include/sparse_iterative.h`
   - `sed -n '1,120p' include/sparse_ilu.h`

### Day 1 Baseline Findings

- Sprint 35 starts from the enforced Sprint 34 close exactly as intended:
  - reviewed local quality wrappers already in force
  - reviewed CMake parity wrappers already in force
  - Linux CI phase-1 reviewed enforcement already in force
  - authoritative active suite count remains `53`
  - dead-code/operator command map is already documented and validated
- Current branch head at Day 1 baseline capture: `9f2fe79`

### Current Public-Surface File Inventory

- installed public headers: `18`
- shipped top-level example programs: `12`
- primary public docs in immediate Sprint 35 scope:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/algorithm.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `INSTALL.md`

Interpretation:

- Sprint 35 is not starting from an amorphous doc backlog
- the public-surface rewrite is concentrated in a bounded set of installed headers, top-level examples, and six high-value documentation files

### Highest-Signal Day 1 Drift

The Day 1 audit did **not** show a broad leftover positional-initializer backlog in installed headers. Most currently visible public header examples already use designated initializers.

The highest-signal public drift instead appears to be **consistency and truthfulness** across tutorial/example prose and type names:

- `docs/tutorial.md` still names stale iterative/ILU option types:
  - `sparse_cg_opts_t` at lines `175` and `327`
  - `sparse_ilu_opts_t` at line `220`
- current public headers define:
  - `sparse_iter_opts_t` in `include/sparse_iterative.h`
  - `sparse_gmres_opts_t` in `include/sparse_iterative.h`
  - ILU(0) usage in `include/sparse_ilu.h` without a matching `sparse_ilu_opts_t`
- public-facing designated-initializer examples are already present across:
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_analysis.h`
  - `include/sparse_reorder.h`
  - `include/sparse_iterative.h`
  - `include/sparse_svd.h`
  - `docs/algorithm.md`
  - `README.md`
  - multiple shipped examples

Interpretation:

- Sprint 35 Day 1 does **not** justify assuming a large header-only initializer rewrite queue
- the stronger likely queue is:
  - tutorial / README / examples consistency
  - public type-name truthfulness
  - reorder/precondition wording alignment
  - maintainer rule unification for the public example style that is already partially in place

### Likely First Implementation Surfaces

1. High-priority public-doc truthfulness:
   - `docs/tutorial.md`
   - `README.md`
2. Installed-header example/style consistency pass:
   - `include/sparse_iterative.h`
   - `include/sparse_reorder.h`
   - `include/sparse_svd.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   - `include/sparse_analysis.h`
3. Shipped example consistency surfaces:
   - `examples/example_iterative.c`
   - `examples/example_matrix_free.c`
   - `examples/example_ic_minres.c`
   - `examples/example_ldlt.c`
   - `examples/example_colamd.c`
   - `examples/example_analysis.c`
4. Supporting public-doc polish:
   - `examples/README.md`
   - `benchmarks/README.md`
   - `INSTALL.md`

### Day 1 Interpretation

- Sprint 35 begins from a clean enforcement baseline, not from validation debt.
- The public-surface problem is narrower and more truthful than a generic “rewrite all examples” backlog:
  - public designated-initializer adoption is already widespread
  - the biggest immediate inconsistency is stale tutorial/API-usage language, especially around iterative-solver option types
- That changes the likely sprint shape:
  - Day 2 should audit installed-header examples carefully rather than presuming mass rewrites
  - Day 6 will likely be load-bearing because README/tutorial/example prose consistency may be the dominant queue

### Day 1 Outputs

- `artifacts/day1-public-doc-baseline.md`
- `artifacts/day1-public-surface-inventory.txt`

## Day 2

**Objective:** Audit the installed-header example surface directly, determine which public headers still teach brittle or outdated patterns, and distinguish real header cleanup debt from the broader README/tutorial/example consistency queue identified on Day 1.

### Commands Run

1. Re-read the Sprint 35 Day 1 baseline and Day 2 scope:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
   - `sed -n '40,95p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
2. Read the first-pass installed headers named on Day 1:
   - `sed -n '1,140p' include/sparse_iterative.h`
   - `sed -n '1,140p' include/sparse_reorder.h`
   - `sed -n '1,140p' include/sparse_svd.h`
   - `sed -n '1,140p' include/sparse_lu.h`
   - `sed -n '1,140p' include/sparse_cholesky.h`
   - `sed -n '1,140p' include/sparse_ldlt.h`
   - `sed -n '1,140p' include/sparse_analysis.h`
3. Search for stale example/style and wording signals inside the audited headers:
   - `rg -n "= \\{[^.][^\\n]*\\}|not implemented|SPARSE_ERR_BADARG|reorder|precondition|default:|NULL for defaults|With fill-reducing reordering|Usage pattern|Workflow" include/sparse_iterative.h include/sparse_reorder.h include/sparse_svd.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_analysis.h`
   - `python3 - <<'PY' ... scan audited headers for *_opts_t example lines ... PY`
4. Cross-check the likely stale wording against current public declarations:
   - `sed -n '1,120p' include/sparse_lu.h`
   - `sed -n '1,120p' include/sparse_cholesky.h`
   - `sed -n '1,120p' include/sparse_svd.h`

### Day 2 Audit Findings

#### 1. The installed-header initializer surface is already mostly aligned

Across the seven high-priority headers audited on Day 2:

- `include/sparse_iterative.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`

all currently visible public options-struct examples already use designated initializers.

Interpretation:

- Sprint 35 does **not** inherit a large installed-header positional-initializer rewrite queue.
- Day 4 and Day 5 should therefore be scoped as targeted header cleanup and wording reconciliation, not as a broad mechanical conversion pass.

#### 2. The strongest true header debt is internal contradiction in `include/sparse_svd.h`

`include/sparse_svd.h` contains two public-surface inconsistencies:

- `sparse_svd_opts_t::economy` documents that Sprint 29 Day 3 enabled full mode when `economy = 0`
- but the `sparse_svd_compute()` return-code docs still say:
  - `compute_uv is set without economy (full SVD not implemented)`
- and the `sparse_svd_partial()` docs say:
  - singular vectors are recovered when `opts->compute_uv` is set
  - while the `@param opts` text still says singular vectors are not computed

Interpretation:

- this is a real installed-header truthfulness issue
- it is more important than initializer syntax cleanup because it can misstate actual supported API behavior to downstream users

#### 3. The other audited headers are primarily `keep`, not `rewrite`

Current Day 2 classification:

- `include/sparse_lu.h`: keep, already aligned on designated initializer usage
- `include/sparse_cholesky.h`: keep, already aligned on designated initializer usage
- `include/sparse_ldlt.h`: keep, already aligned on designated initializer usage
- `include/sparse_analysis.h`: keep, already aligned on designated initializer usage
- `include/sparse_reorder.h`: keep, example style aligned; wording may need later cross-doc consistency review only
- `include/sparse_iterative.h`: keep, example style aligned; larger truthfulness drift is still in `docs/tutorial.md`, not here
- `include/sparse_svd.h`: update, due to the full-SVD / partial-SVD wording contradictions above

#### 4. The real Day 2 result narrows the sprint

The strongest remaining public-example debt is now split clearly:

- header-level direct rewrite queue:
  - small
  - led by `include/sparse_svd.h`
- cross-surface truthfulness queue:
  - larger
  - still led by `docs/tutorial.md`
  - likely followed by `README.md` and selected shipped examples

### Day 2 Interpretation

- Day 2 confirmed that Day 1's caution was correct: the installed-header surface is not where most of the remaining mechanical initializer debt lives.
- The header work is now narrow enough that Day 3 should focus on a public-facing style/wording standard, not a broad conversion policy.
- `include/sparse_svd.h` is the clearest concrete header file to carry into the first implementation batch because it contains actual contradictory user-facing behavior docs.

### Day 2 Outputs

- `artifacts/day2-header-example-audit.md`
