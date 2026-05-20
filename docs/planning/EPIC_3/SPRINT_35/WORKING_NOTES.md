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

## Day 3

**Objective:** Define the maintainer-facing public example standard that Sprint 35 will apply across installed headers, README/tutorial snippets, shipped examples, and explanatory test references so later edits converge on one stable API-usage contract instead of making local one-off wording decisions.

### Commands Run

1. Re-read the Sprint 35 baseline, Day 2 audit, and Day 3 scope:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
   - `sed -n '60,120p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
2. Inspect current public-facing example patterns across docs, headers, and examples:
   - `rg -n "_opts_t [A-Za-z_][A-Za-z0-9_]* = \\{|_opts_t\\s+[A-Za-z_][A-Za-z0-9_]*\\s*=\\s*\\{0\\}|NULL for defaults|defaults|designated|\\.compute_|\\.reorder\\s*=|\\.pivot_tol\\s*=|\\.max_iter\\s*=" README.md docs/tutorial.md docs/algorithm.md examples include tests -g '!build/**'`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_iterative.h`
   - `sed -n '1,220p' README.md`
   - `sed -n '150,260p' docs/tutorial.md`
   - `sed -n '1,220p' examples/example_iterative.c`
   - `sed -n '1,220p' tests/test_framework_optin.c`

### Day 3 Style-Contract Findings

#### 1. Public material already uses three distinct teaching modes

The current public-facing surface is not inconsistent because designated initializers are rare. It is inconsistent because it mixes three different patterns without a stated rule for when each should be used:

- **Designated initializer examples** for non-default tuning:
  - installed headers such as `include/sparse_iterative.h` and `include/sparse_lu.h`
  - `README.md`
  - `docs/algorithm.md`
  - multiple shipped examples
- **`NULL` option-pointer examples** to teach the pure-default path:
  - function parameter docs throughout installed headers
- **Stale type-name / workflow examples** in a few public docs:
  - especially `docs/tutorial.md` still using `sparse_cg_opts_t` and `sparse_ilu_opts_t`

Interpretation:

- Sprint 35 needs a public example **selection rule**, not just a syntax rule.
- The main failure mode to prevent is not “someone used braces wrong”; it is “different public surfaces teach different API contracts.”

#### 2. The stable public rule should be “designated init for non-defaults, NULL for pure defaults”

The repo already points toward a coherent standard:

- use a designated initializer whenever an example is trying to teach one or more meaningful non-default fields
- use `NULL` only when the point of the example is explicitly “take the library defaults”

This aligns with:

- Sprint 31 / Sprint 34 designated-initializer cleanup
- the trailing-field back-compat notes embedded in multiple public structs
- the existing header idiom `@param opts ... NULL for defaults`

Interpretation:

- public examples should not use positional struct literals
- public examples should also avoid spelling out a pseudo-default struct when the example really just wants default behavior
- this rule is more truthful and shorter for readers than treating all examples as full struct declarations

#### 3. Acceptable exceptions are narrow

The Day 3 audit did not justify a broad exception bucket. The only useful public-facing exceptions are:

- **pure-default call sites**: pass `NULL` instead of inventing an options struct
- **single-line compact snippets**: designated initializers may stay on one line when readability is still good
- **tests as explanatory references**: they may remain denser than README/header prose, but any snippet copied into public docs should still follow the public rule

Non-exceptions:

- stale historical type names
- hypothetical option structs that no longer exist
- positional options-struct literals in public docs
- zero-init sentinels presented as if they were the recommended public style

#### 4. The style rule also needs a wording contract

Day 2 already showed that syntax alone is not enough: `include/sparse_svd.h` is a real public-surface problem even though its example syntax is already modern.

The paired wording contract should therefore be:

- examples and prose must name the **current shipped types**
- examples and prose must describe the **current shipped behavior**
- when defaults are discussed, they should describe what the current implementation actually does
- reorder/precondition guidance should name only the modes or paths the shown API surface really supports

Interpretation:

- Day 4 / Day 5 header cleanup must include wording reconciliation where needed, not only snippet cleanup
- Day 6 onward should treat stale type names and stale behavior claims as the primary public-doc truthfulness queue

### Day 3 Maintainer Contract

Sprint 35 should apply the following cross-surface rule:

1. In installed headers, README snippets, tutorial snippets, and example-facing docs, show option structs with **designated initializers** whenever the example teaches any non-default configuration.
2. When an example intends to teach the default path only, prefer passing `NULL` to the options parameter rather than declaring an all-default options struct.
3. Use the **current public type names and field names only**.
4. Keep snippets minimal: show only the fields relevant to the example instead of restating implicit defaults.
5. Treat explanatory tests as an implementation-aligned reference surface, but not as permission to relax the public-doc rule.

### Day 3 Interpretation

- The Day 3 result is concrete enough to drive the rest of the sprint:
  - Day 4 and Day 5 can focus on targeted header truthfulness and wording cleanup under a stable example rule
  - Day 6 through Day 8 can rewrite README/tutorial material by applying one explicit contract instead of improvising per file
- The strongest near-term implementation consequences are now clear:
  - `include/sparse_svd.h` should be fixed for behavior truthfulness
  - `docs/tutorial.md` should be rewritten around current type names and the designated-init/`NULL` split

### Day 3 Outputs

- `artifacts/day3-public-initialization-standard.md`

## Day 4

**Objective:** Apply the Day 3 public example contract to the first installed-header cleanup batch, starting with the highest-signal header truthfulness issue identified on Day 2 and keeping the scope narrow enough to stay documentation-facing and semantically neutral.

### Commands Run

1. Re-read the Sprint 35 Day 4 scope and prior audit/design decisions:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '80,170p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Re-open the target header and cross-check against current implementation:
   - `sed -n '1,260p' include/sparse_svd.h`
   - `rg -n "full SVD not implemented|singular vectors are not computed|economy|compute_uv" include/sparse_svd.h src examples docs -g '!build/**'`
   - `sed -n '720,930p' src/sparse_svd.c`
   - `sed -n '980,1385p' src/sparse_svd.c`
3. Validation after the header edit:
   - `make format`
   - `make lint`
   - `make test`

### Day 4 Implementation Findings

#### 1. The first installed-header batch stayed intentionally narrow

Day 2 and Day 3 were accurate: the installed-header queue is not a large conversion pass. The only Day 4 header that justified direct editing was:

- `include/sparse_svd.h`

Interpretation:

- this is still a real “header batch,” but the batch is driven by truthfulness and current behavior, not by a mechanical syntax sweep

#### 2. `include/sparse_svd.h` now teaches the current public SVD contract

The Day 4 header changes closed the two contradictions identified on Day 2:

- `sparse_svd_compute()` no longer claims that full SVD with `compute_uv && !economy` is unimplemented
- `sparse_svd_partial()` no longer contradicts itself about whether singular vectors can be recovered

The header now states the actual split implemented in `src/sparse_svd.c`:

- `sparse_svd_compute()`:
  - `opts == NULL` => singular values only
  - `compute_uv = 1, economy = 1` => thin/economy `U` and `V^T`
  - `compute_uv = 1, economy = 0` => full `U` and `V^T`
- `sparse_svd_partial()`:
  - approximate singular vectors are supported only when
    `compute_uv = 1, economy = 1`
  - `compute_uv = 1, economy = 0` is rejected

#### 3. The Day 3 public example contract is now applied in a touched installed header

The top-level `sparse_svd.h` usage snippet was also refreshed to match the Day 3 rule more explicitly:

- designated initializer for the non-default path
- minimal fields only
- explicit note that `economy = 0` requests full output

Interpretation:

- Day 4 is not just a wording correction; it is the first concrete application of the Sprint 35 public example standard at the installed-header layer

### Day 4 Interpretation

- The header queue remains small and high-signal.
- Day 5 should now be a consistency/re-review pass across the remaining installed headers rather than a broad rewrite batch.
- The larger Sprint 35 queue still lives outside the headers:
  - `docs/tutorial.md`
  - `README.md`
  - selected shipped examples and example-facing docs
- Validation for the touched header was complete:
  - `make format`
  - `make lint`
  - `make test`
  - all passed

### Day 4 Outputs

- `artifacts/day4-header-batch1.md`
