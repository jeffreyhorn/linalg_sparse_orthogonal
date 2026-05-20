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

## Day 5

**Objective:** Close the installed-header cleanup pass by reconciling the smaller style and wording inconsistencies left after Day 4, and verify that the remaining Sprint 35 queue has genuinely shifted out of headers and into README/tutorial/example-facing docs.

### Commands Run

1. Re-read the Day 5 plan scope and current Sprint 35 state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '95,185p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '1,560p' docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Re-review the remaining installed headers for residual public-surface drift:
   - `sed -n '1,220p' include/sparse_iterative.h`
   - `sed -n '1,220p' include/sparse_reorder.h`
   - `sed -n '1,220p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,220p' include/sparse_ldlt.h`
   - `sed -n '1,220p' include/sparse_ilu.h`
3. Targeted searches for lingering stale type-name / wording signals:
   - `rg -n "sparse_ilu_factor\\(|sparse_ilut_factor\\(|NULL for defaults|default:|reorder = SPARSE_REORDER_COLAMD|SPARSE_REORDER_COLAMD|economy = 0|compute_uv" include/sparse_ilu.h include/sparse_iterative.h include/sparse_reorder.h include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h README.md docs/tutorial.md examples -g '!build/**'`
   - `rg -n "typedef struct .*sparse_cg_opts_t|sparse_cg_opts_t|sparse_ilu_opts_t|sparse_iter_opts_t|sparse_gmres_opts_t" include README.md docs/tutorial.md examples -g '!build/**'`
4. Validation after the header edits:
   - `make format`
   - `make lint`
   - `make test`

### Day 5 Implementation Findings

#### 1. The residual header queue was real, but small

After Day 4, there were no remaining behavior contradictions on the scale of
`include/sparse_svd.h`. The remaining header work was narrower:

- a few high-traffic examples still used inconsistent presentation styles
- one analysis comment still described the reorder set too broadly for the
  normal symmetric-analysis path

The Day 5 touched set was:

- `include/sparse_iterative.h`
- `include/sparse_reorder.h`
- `include/sparse_analysis.h`

#### 2. The public example contract is now more uniform across installed headers

Day 5 brought the remaining high-traffic examples into clearer alignment with
the Day 3 rule:

- non-default examples use designated initializers
- examples are self-contained instead of relying on surrounding snippet state
- the COLAMD / QR snippet now matches the same designated-init presentation
  used elsewhere

This did not change API behavior. It changed how consistently the headers
teach that behavior.

#### 3. The analysis-layer wording is now more explicit about COLAMD's role

`sparse_analysis_opts_t.reorder` still accepts `SPARSE_REORDER_COLAMD`, but
the Day 5 wording now makes the intended split clearer:

- `NONE`, `RCM`, `AMD`, and `ND` are the normal symmetric-analysis choices
- `COLAMD` is accepted, but `sparse_analyze()` applies it symmetrically
- the column-only COLAMD path belongs to QR-specific APIs

Interpretation:

- this reduces the chance that downstream readers infer that COLAMD is the
  normal recommendation for symmetric factor-analysis workflows

#### 4. The remaining Sprint 35 drift has now moved out of the headers

The Day 5 re-review confirms that the larger residual truthfulness queue is no
longer the installed-header layer. It is now clearly:

- `docs/tutorial.md`
- `README.md`
- selected public examples and example-facing docs

### Day 5 Interpretation

- The installed-header cleanup is now effectively complete.
- Sprint 35 can move into README/tutorial/example-facing work without carrying
  a hidden header backlog.
- Day 6 should therefore be a real cross-surface audit of user-facing docs,
  not a continuation of header cleanup.
- Validation for the touched headers was complete:
  - `make format`
  - `make lint`
  - `make test`
  - all passed

### Day 5 Outputs

- `artifacts/day5-header-batch2.md`

## Day 6

**Objective:** Audit the README, tutorial, and example-facing documentation
surface as one public guidance layer, map stale API usage and duplicated
workflow explanations against the current shipped codebase, and turn the
remaining Sprint 35 doc cleanup into named rewrite batches before broad prose
edits begin.

### Commands Run

1. Re-read the Day 6 scope and current Sprint 35 state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '139,170p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '1,760p' docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Audit the highest-value public docs and example-facing files:
   - `sed -n '90,240p' README.md`
   - `sed -n '160,360p' docs/tutorial.md`
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '1,240p' INSTALL.md`
3. Cross-check public docs against current public headers and current names:
   - `rg -n "sparse_cg_opts_t|sparse_ilu_opts_t|sparse_iter_opts_t|sparse_gmres_opts_t|sparse_ilut_opts_t|quality-review|deadcode-check|quality-review-cmake|compute_uv|economy = 0|economy=0|SPARSE_REORDER_COLAMD|SPARSE_REORDER_ND" README.md docs/tutorial.md docs/algorithm.md examples/README.md benchmarks/README.md INSTALL.md examples/*.c -g '!build/**'`
   - `rg -n "typedef struct .*sparse_cg_opts_t|sparse_cg_opts_t|sparse_ilu_opts_t|sparse_iter_opts_t|sparse_gmres_opts_t" include README.md docs/tutorial.md examples -g '!build/**'`
   - `sed -n '1,220p' include/sparse_ilu.h`
   - `sed -n '1,220p' include/sparse_iterative.h`
   - `sed -n '1,220p' include/sparse_analysis.h`

### Day 6 Audit Findings

#### 1. `docs/tutorial.md` is the dominant remaining public truthfulness queue

The README/tutorial/example layer is not evenly stale. The tutorial is
materially behind the shipped public API:

- it still names `sparse_cg_opts_t` in the CG and matrix-free CG examples
- it still names `sparse_ilu_opts_t` for ILUT configuration
- the current public surface instead exposes:
  - `sparse_iter_opts_t`
  - `sparse_gmres_opts_t`
  - `sparse_ilut_opts_t`

Interpretation:

- this is not cosmetic wording debt
- it is the strongest remaining public API truthfulness issue in Sprint 35
- Day 8 should treat `docs/tutorial.md` as the first rewrite target, not as a
  trailing cleanup file

#### 2. `README.md` is mostly current on behavior and workflow names

The README already reflects the Sprint 34 operator workflow accurately:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `make deadcode-check`

Its SVD top-level feature note is also current about `economy = 0` with
`compute_uv = 1`.

The README's residual queue is therefore narrower:

- some public snippets still use compact one-line designated initializers
  instead of the stronger multi-line Day 3 teaching style
- the file now carries a large amount of operator/workflow explanation spread
  across multiple sections, which risks duplication rather than direct API
  falsehood

Interpretation:

- README is a real Sprint 35 target
- but it is primarily a consistency and structure target, not the main
  truthfulness risk

#### 3. Example-facing support docs are secondary, not primary, rewrite debt

`examples/README.md` and `INSTALL.md` are largely aligned with the shipped
surface:

- `examples/README.md` is short, current, and mainly descriptive
- `INSTALL.md` still points to the maintained Make/CMake flows and does not
  carry the stale iterative/ILU type-name drift found in the tutorial

This does **not** mean they are permanently finished. It means they should
follow the README/tutorial rewrite, not lead it.

#### 4. The conflict map is now explicit

The same user-facing topics are currently split across multiple public files:

- iterative solver setup:
  - `README.md`
  - `docs/tutorial.md`
  - `include/sparse_iterative.h`
  - `include/sparse_ilu.h`
  - shipped iterative examples
- SVD usage:
  - `README.md`
  - `docs/tutorial.md`
  - `include/sparse_svd.h`
- user quality/build commands:
  - `README.md`
  - `INSTALL.md`

Current risk by topic:

- iterative setup is the highest conflict area because the headers are current
  while the tutorial still teaches stale types
- SVD is mostly stabilized after Day 4, but the tutorial still uses compact
  snippet style that should be reconciled with the Day 3 public example rule
- build/operator command guidance is currently truthful, but spread across
  enough README sections that Day 7 should choose a clearer division of
  responsibilities between README and `INSTALL.md`

### Day 6 Named Cleanup Queue

1. **Day 8 primary rewrite batch**
   - `docs/tutorial.md`
   - fix stale iterative / ILUT type names
   - apply the Day 3 designated-initializer vs `NULL` teaching split
   - keep SVD wording aligned with the Day 4 installed-header contract
2. **Day 8 secondary rewrite batch**
   - `README.md`
   - normalize public snippet style
   - reduce duplicated operator guidance where one section can be canonical
3. **Follow-on support-doc pass**
   - `examples/README.md`
   - `INSTALL.md`
   - only adjust if the README/tutorial rewrite changes the public wording
     baseline or reveals duplication worth trimming

### Day 6 Interpretation

- The installed-header layer is genuinely closed enough that the remaining
  Sprint 35 risk now lives in user-facing prose.
- The tutorial is the primary truthfulness fix surface.
- The README is the primary consistency/structure surface.
- Example-facing support docs should follow those rewrites rather than compete
  with them for ownership of the public API story.
- Day 7 should therefore choose one canonical division of responsibilities:
  - README = concise entrypoint + maintained workflow map
  - tutorial = fuller API-usage teaching surface
  - headers = authoritative parameter/behavior contract

### Day 6 Outputs

- `artifacts/day6-readme-tutorial-audit.md`

## Day 7

**Objective:** Convert the Day 6 audit into a concrete rewrite plan by
assigning stable responsibilities across README/tutorial/header/example-facing
docs, choosing canonical public wording for the highest-conflict topics, and
defining the implementation order for the remaining Sprint 35 documentation
batches.

### Commands Run

1. Re-read the Day 7 scope and current Sprint 35 state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '170,240p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Re-open the highest-conflict public surfaces from Day 6:
   - `sed -n '200,260p' README.md`
   - `sed -n '620,740p' README.md`
   - `sed -n '1,220p' examples/example_iterative.c`
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '1,240p' INSTALL.md`
3. Cross-check the later sprint-day sequencing against the current plan:
   - `sed -n '240,340p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`

### Day 7 Design Findings

#### 1. The public-doc ownership split should be explicit, not inferred

Day 6 showed that the remaining drift is caused partly by stale content and
partly by multiple files trying to explain the same thing at different levels.

The Sprint 35 ownership split should therefore be:

- `README.md`
  - concise public entrypoint
  - current command map
  - short, stable usage snippets
  - high-level capability summary
- `docs/tutorial.md`
  - fuller user-teaching surface
  - multi-step API workflows
  - iterative/precondition guidance
  - matrix-free and SVD usage walkthroughs
- installed headers in `include/`
  - authoritative parameter and behavior contract
  - current type names
  - accepted option values
  - routine-level preconditions
- `INSTALL.md`
  - installation/platform/build guidance only
  - not a second operator-workflow explainer
- `examples/README.md`
  - short catalog of shipped examples
  - not the canonical API-usage teaching layer

Interpretation:

- README and tutorial can stop competing for the same explanatory depth
- headers remain the contract surface that the prose must follow
- support docs can be trimmed instead of expanded

#### 2. Canonical wording should be fixed once for the highest-conflict topics

The Day 7 rewrite plan needs stable wording rules for the recurring drift
areas.

**Initialization pattern**

- use designated initializers for non-default public examples
- use `NULL` only when the example is intentionally teaching the pure-default
  path
- avoid zero-init sentinel style in public docs

**Iterative / precondition type names**

- CG and matrix-free CG examples should use `sparse_iter_opts_t`
- GMRES examples should use `sparse_gmres_opts_t`
- ILUT configuration should use `sparse_ilut_opts_t`
- ILU(0) should be described as the no-options/default incomplete
  factorization path unless the example is specifically teaching ILUT

**Reorder wording**

- symmetric analysis/factor examples should name `NONE`, `RCM`, `AMD`, and
  `ND` as the normal reorder set
- `COLAMD` should be described as accepted in analysis but not as the normal
  symmetric-analysis recommendation
- QR-specific examples can teach the column-oriented `COLAMD` path directly

**Quality-command wording**

- README owns the concise public command map
- `INSTALL.md` can reference the build/test flows, but should not duplicate the
  full reviewed-quality operator explanation
- tutorial should mention quality commands only when they are directly relevant
  to example/build usage, not as a second command catalog

#### 3. Public precondition guidance needs one home per level

Precondition wording is a real later-sprint topic, so the rewrite plan needs
the location split now rather than during Day 9.

Chosen structure:

- headers:
  - authoritative routine-specific preconditions and accepted modes
- tutorial:
  - user-facing operational guidance
  - which solver/preconditioner path to choose and what assumptions matter
- README:
  - brief signposts only
  - enough context to avoid obviously misleading examples, but not the full
    safety narrative
- `INSTALL.md` / `examples/README.md`:
  - no independent precondition explanation beyond minimal contextual wording

Interpretation:

- Day 9 can audit against a stable destination model instead of deciding
  ownership and wording at the same time

#### 4. The implementation order should follow dependency, not file prestige

The remaining rewrite order should be:

1. **Day 8 core public rewrite**
   - `docs/tutorial.md`
   - `README.md`
2. **Day 9 audit**
   - headers and rewritten docs for residual precondition-language debt
3. **Day 10 implementation**
   - whichever of `README.md`, `docs/tutorial.md`, installed headers, or
     example comments need wording tightening from Day 9
4. **Day 11 support-doc polish**
   - `INSTALL.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - any remaining quality-facing README sections if duplication still survives
5. **Day 12 validation-linked cleanup**
   - shipped example sources only if snippet-to-code validation exposes new
     drift

Why this order is correct:

- tutorial must move first because it contains the strongest current falsehoods
- README should be reconciled immediately after, while the canonical wording is
  fresh
- support docs should follow the main rewrite so they inherit the final public
  wording instead of becoming a third competing source

### Day 7 Interpretation

- Day 7 closes the planning gap that Day 6 intentionally left open.
- Sprint 35 now has one concrete doc architecture:
  - README = entrypoint
  - tutorial = teaching layer
  - headers = contract layer
  - install/examples docs = support layer
- That gives Day 8 a narrow job:
  - rewrite tutorial first
  - reconcile README second
  - avoid reopening support-doc ownership questions until Day 11

### Day 7 Outputs

- `artifacts/day7-readme-tutorial-rewrite-design.md`

## Day 8

**Objective:** Rewrite the main user-facing docs so the tutorial and README
teach the current public API accurately, use the Sprint 35 public example
style consistently, and narrow the remaining public-doc debt to precondition
language rather than stale type names or stale option examples.

### Commands Run

1. Re-read the Day 8 scope and current Sprint 35 rewrite plan:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '200,320p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '150,360p' docs/tutorial.md`
   - `sed -n '180,260p' README.md`
2. Cross-check the rewritten public snippets against the current public
   headers:
   - `sed -n '1,220p' include/sparse_iterative.h`
   - `sed -n '1,220p' include/sparse_ilu.h`
   - `sed -n '1,220p' include/sparse_svd.h`
3. Post-edit documentation sanity checks:
   - `rg -n "sparse_cg_opts_t|sparse_ilu_opts_t|sparse_gmres_opts_t opts = \\{|sparse_svd_opts_t opts = \\{\\.compute_uv|quality-review|deadcode-check" README.md docs/tutorial.md`
   - `git diff -- README.md docs/tutorial.md`

### Day 8 Implementation Findings

#### 1. The tutorial now teaches the current iterative and ILUT public types

The highest-signal Day 6 truthfulness issue is now closed in
`docs/tutorial.md`:

- `sparse_cg_opts_t` examples were replaced with `sparse_iter_opts_t`
- the ILUT example now uses `sparse_ilut_opts_t`
- the CG, GMRES, ILUT, and matrix-free examples now use the same multi-line
  designated-initializer style already established in the installed headers

Interpretation:

- Day 8 removed the main stale public type-name drift instead of just
  restyling the old snippets

#### 2. The tutorial's SVD examples now match the Day 4 header contract

The SVD section in `docs/tutorial.md` was reconciled with
`include/sparse_svd.h`:

- singular-values-only still uses `opts == NULL`
- economy/thin vector recovery uses a designated initializer
- the full-output path is now stated explicitly via `economy = 0`
- partial SVD now says directly that singular vectors are supported only in
  the economy/thin path

Interpretation:

- the tutorial no longer risks teaching an older or looser SVD contract than
  the installed header documents

#### 3. The README rewrite stayed intentionally smaller

Day 7's ownership split was correct: `README.md` needed reconciliation, not a
full second tutorial rewrite.

The touched README surface was the public iterative example:

- the GMRES options snippet now uses the same multi-line designated-init style
  as the tutorial and headers

The reviewed-quality command names in README were already current, so they did
not need a Day 8 semantic rewrite.

#### 4. The remaining queue is now genuinely precondition-language debt

After the Day 8 rewrite, the residual Sprint 35 doc drift is no longer led by
stale type names or stale option examples.

The remaining higher-value queue is now about wording such as:

- where user-facing docs should state that some routines require identity
  permutations or fresh copies
- how explicitly iterative/precondition examples should state SPD vs general
  matrix assumptions
- how much of the ILU / ILUT safety story belongs in tutorial prose versus
  headers

That is the right Day 9 queue, and it is materially narrower than the Day 6
starting state.

### Day 8 Interpretation

- Day 8 completed the main public-facing rewrite batch correctly:
  - tutorial first for truthfulness
  - README second for consistency
- The remaining Sprint 35 doc work is now about tightening public usage
  assumptions, not about replacing stale public API names.
- Day 9 can therefore audit precondition language directly without having to
  finish unfinished Day 6/Day 8 truthfulness cleanup first.

### Day 8 Outputs

- `artifacts/day8-readme-tutorial-implementation.md`

## Day 9

**Objective:** Audit the rewritten public docs and the installed headers for
residual precondition-language debt, identify which usage assumptions are still
implicit or underspecified in user-facing prose, and map each item to the
right public surface before Day 10 wording edits begin.

### Commands Run

1. Re-read the Day 9 scope and the Day 8 end state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '220,290p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Re-open the rewritten user-facing docs:
   - `sed -n '150,360p' docs/tutorial.md`
   - `sed -n '180,260p' README.md`
   - `sed -n '330,365p' README.md`
3. Cross-check user prose against the authoritative public headers:
   - `rg -n "@pre|identity permutations|fresh matrix|sparse_copy\\(|SPD|symmetric positive-definite|not modified|in-place|economy = 0|compute_uv = 1|ILU\\(0\\)|ILUT|precondition" include README.md docs/tutorial.md docs/algorithm.md examples/*.c examples/README.md INSTALL.md -g '!build/**'`
   - `sed -n '1,180p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_qr.h`
   - `sed -n '1,220p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,220p' include/sparse_ilu.h`
   - `sed -n '1,220p' include/sparse_ic.h`
   - `sed -n '1,220p' include/sparse_ldlt.h`
4. Check QR least-squares / minimum-norm wording specifically:
   - `rg -n "minnorm|minimum-norm|underdetermined|least-squares|min \\|\\|Ax - b\\|\\|" README.md docs/tutorial.md examples/README.md include/sparse_qr.h -g '!build/**'`

### Day 9 Audit Findings

#### 1. The main residual queue is now "headers precise, prose implicit"

After Day 8, the public docs no longer teach stale type names. The remaining
debt is narrower: the installed headers often state the important usage
preconditions precisely, while README/tutorial prose still leaves them
implicit.

That pattern shows up in three recurring classes:

- matrix-state assumptions
- matrix-class assumptions
- routine-selection assumptions

Interpretation:

- Day 10 should tighten wording, not expand the docs into header duplicates

#### 2. Matrix-state assumptions are the highest-signal tutorial gap

Several public headers now state clearly that some routines require an
original/unfactored matrix or identity permutations:

- `include/sparse_ilu.h`
- `include/sparse_ic.h`
- `include/sparse_qr.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`

The tutorial and README only partially surface that story:

- LU and Cholesky already explain the in-place overwrite and show `sparse_copy`
- the iterative/precondition sections do **not** explain as clearly that ILU,
  ILUT, IC, QR, LDL^T, SVD, and analysis routines expect an unfactored /
  unreordered matrix view
- the tutorial's ILUT example uses `sparse_copy(A)`, but it does not explain
  why a fresh/original matrix matters

This is the strongest Day 9 wording queue because it can affect whether a user
feeds a previously factored or reordered matrix into the wrong API.

#### 3. Matrix-class assumptions are partly stated, but unevenly

Current state:

- CG is labeled SPD in README/tutorial
- Cholesky is labeled SPD in README/tutorial
- GMRES is labeled general/unsymmetric in README/tutorial
- IC(0) is documented in headers as the SPD-side analogue of ILU(0)

Remaining gap:

- the user-facing iterative/precondition prose does not yet say as directly
  which preconditioner families align naturally with which matrix classes:
  - IC(0) for SPD workflows
  - ILU(0) / ILUT for general or indefinite workflows
- the tutorial also does not yet explain that CG's preconditioned use still
  assumes an SPD problem/operator path

Interpretation:

- this belongs mainly in tutorial prose, with only brief signposts in README

#### 4. QR routine-selection wording still leaves one important distinction implicit

`include/sparse_qr.h` is explicit:

- `sparse_qr_solve()` gives least-squares for overdetermined systems
- for underdetermined systems it gives a basic solution, not the minimum-norm
  solution
- the minimum-norm path is `sparse_qr_solve_minnorm()`

The user-facing docs are looser:

- tutorial QR wording says "rectangular or rank-deficient systems" and then
  shows `sparse_qr_solve()` generically
- README lists both QR least-squares and QR minimum-norm support, but the main
  QR usage example does not help the reader choose between them

This is a real public-usage distinction, not cosmetic polish, and it belongs
in Day 10.

#### 5. SVD and quality-command wording are no longer the main risk

After Day 8:

- SVD examples are aligned enough with `include/sparse_svd.h`
- reviewed-quality command names in README remain current

Residual work there is secondary. The higher-value Day 10 queue is still:

- matrix-state assumptions
- solver/precondition matrix-class assumptions
- QR least-squares vs minimum-norm routine selection

### Day 9 Named Cleanup Queue

1. **Tutorial precondition-language pass**
   - explain fresh/original matrix expectations where users are likely to copy
     factorized or reordered matrices into ILU / ILUT / IC / QR / SVD /
     analysis workflows
   - state matrix-class guidance more directly:
     - CG / IC(0) for SPD paths
     - GMRES / ILU / ILUT for general paths
2. **README signpost pass**
   - add only short clarifying notes where the entrypoint examples currently
     hide a meaningful usage assumption
   - avoid turning README into a second contract surface
3. **QR routine-selection pass**
   - clarify in user-facing docs that `sparse_qr_solve()` is not the
     minimum-norm underdetermined path
   - point underdetermined readers to `sparse_qr_solve_minnorm()`

### Day 9 Surface Mapping

- installed headers:
  - keep the precise contract language they already have
  - only touch on Day 10 if one user-facing clarification reveals a genuine
    header-level ambiguity
- `docs/tutorial.md`:
  - primary Day 10 target
  - best place for matrix-state and matrix-class guidance
- `README.md`:
  - secondary Day 10 target
  - brief signposts only
- `examples/README.md` and support docs:
  - not the right place for the main safety narrative
  - any follow-on cleanup can wait for Day 11

### Day 9 Interpretation

- The remaining Sprint 35 public-doc debt is now sharply bounded.
- Day 10 should be a concise wording pass, not another broad rewrite batch.
- The highest-value fixes are the ones that prevent users from choosing the
  wrong routine or feeding the wrong matrix state into a valid routine.

### Day 9 Outputs

- `artifacts/day9-api-precondition-audit.md`

## Day 10

**Objective:** Tighten the public safety and usage language identified on Day
9, keeping the edits concise and user-facing while aligning tutorial/README
prose more closely with the already-precise installed-header contracts.

### Commands Run

1. Re-read the Day 10 scope and the Day 9 audit queue:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '240,320p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `sed -n '120,320p' docs/tutorial.md`
   - `sed -n '150,260p' README.md`
   - `sed -n '145,185p' include/sparse_qr.h`
2. Re-check the exact Day 9 wording targets:
   - `rg -n "sparse_qr_solve_minnorm|minimum-norm|underdetermined|identity permutations|fresh copy|sparse_copy\\(A\\)" README.md docs/tutorial.md`
   - `sed -n '312,326p' README.md`
   - `sed -n '200,235p' docs/tutorial.md`
3. Post-edit documentation sanity checks:
   - `git diff -- README.md docs/tutorial.md`
   - `rg -n "minimum-norm|identity permutations|fresh `sparse_copy\\(\\)`|IC\\(0\\)|ILUT|underdetermined" README.md docs/tutorial.md`

### Day 10 Implementation Findings

#### 1. The tutorial now surfaces the main matrix-state assumptions directly

The highest-value Day 9 queue was the gap between precise header preconditions
and more implicit tutorial prose. Day 10 closes the most important part of
that gap:

- the QR section now says directly that QR expects an unfactored, unreordered
  matrix with identity permutations
- the SVD section now says the same about using the original matrix view
- the preconditioning section now explains why users may want a fresh
  `sparse_copy()` before ILU(0), ILUT, or IC(0) setup when matrix state is
  uncertain

Interpretation:

- the user-facing docs now expose the main "original matrix vs post-factor /
  post-reorder matrix" distinction instead of leaving it buried in headers

#### 2. Matrix-class / preconditioner guidance is now clearer in user prose

The tutorial now states the practical pairing more directly:

- IC(0) with SPD operators and CG/MINRES workflows
- ILU(0) / ILUT with GMRES and other general or indefinite-system workflows

This is still concise, but it is a materially better operational guide than
the pre-Day-10 version, which implied the families without saying the pairing
plainly.

#### 3. QR routine selection is now clearer across tutorial and README

The Day 9 QR distinction is now surfaced in user-facing docs:

- tutorial QR wording now distinguishes `sparse_qr_solve()` from
  `sparse_qr_solve_minnorm()`
- README's QR API summary now says `sparse_qr_solve()` gives a basic solution
  for underdetermined systems rather than the minimum-norm one

That closes a real public-usage ambiguity without turning either file into a
full duplicate of `include/sparse_qr.h`.

#### 4. No header edits were needed

Day 9 left open the possibility that a user-facing clarification might reveal
an actual header-level ambiguity. That did not happen.

The current installed headers were already the precise contract surface. The
missing work was in the prose layer, so Day 10 stayed in:

- `docs/tutorial.md`
- `README.md`

This is the right outcome. It keeps the sprint focused on public guidance
rather than re-editing already-correct API contracts.

### Day 10 Interpretation

- The remaining Sprint 35 precondition-language queue is now substantially
  smaller.
- Day 11 can move into support-doc polish and duplication cleanup rather than
  still carrying core safety-language debt from README/tutorial.
- The likely Day 11 surfaces are now the intended ones:
  - `INSTALL.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - any remaining quality-facing README duplication

### Day 10 Outputs

- `artifacts/day10-api-precondition-implementation.md`

## Day 11

**Objective:** Align the support-doc and quality-facing surfaces with the
Sprint 35 public-doc rewrite, remove the most obvious remaining cross-doc
mismatches, and define the exact validation scope for Days 12 and 13.

### Commands Run

1. Re-read the Day 11 scope and current Sprint 35 state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '260,340p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Re-open the support-doc targets:
   - `sed -n '1,260p' INSTALL.md`
   - `sed -n '1,240p' examples/README.md`
   - `sed -n '1,260p' benchmarks/README.md`
3. Cross-check those files against the current quality/public-usage wording:
   - `rg -n "quality-review|deadcode-check|tooling-build|examples-build|minimum-norm|underdetermined|ILU\\(0\\)|ILUT|SPD|sparse_qr_solve_minnorm" README.md INSTALL.md examples/README.md benchmarks/README.md -g '!build/**'`
   - `sed -n '620,740p' README.md`
   - `sed -n '100,125p' README.md`

### Day 11 Implementation Findings

#### 1. `INSTALL.md` now points to the maintained reviewed-quality entry points

The install guide previously showed only the basic `make` / `make test`
workflow. That was still valid, but it no longer reflected the maintained
Sprint 34 operator path.

Day 11 now adds the reviewed local wrappers to the Makefile quick start:

- `make tooling-build`
- `make quality-review-compile`
- `make quality-review`

It also points CMake/install readers back to the reviewed local CMake parity
wrappers without turning `INSTALL.md` into a second full command map.

Interpretation:

- `INSTALL.md` now reflects the current repo state
- README remains the canonical operator reference, which preserves the Day 7
  ownership split

#### 2. `examples/README.md` now matches the Day 10 public-usage story

The examples index now surfaces the most important Sprint 35 assumptions
without duplicating the full tutorial:

- in-place factorization examples are described as copying before mutation
- the QR least-squares example is identified explicitly as the overdetermined
  path, with minimum-norm underdetermined solves routed to
  `sparse_qr_solve_minnorm()`
- the iterative example now says the ILU(0) preconditioner is built from a
  fresh matrix copy

This is the right level for the examples catalog: short, accurate signposts
instead of a second tutorial layer.

#### 3. `benchmarks/README.md` now reflects the reviewed compile-quality flow

The benchmark docs already described the compile-only gate correctly through
`make tooling-build` and `make lint`. Day 11 adds the missing reviewed wrapper
surface:

- `make quality-review-compile`

That keeps the benchmark docs aligned with the current local quality path
without pretending that benchmark execution is part of the routine reviewed
wrapper flow.

#### 4. The remaining Sprint 35 queue is now validation, not more rewrite debt

After the Day 11 pass:

- core public API naming is already current
- tutorial/README safety wording is already tightened
- support docs now point at the same maintained workflow and public-usage
  expectations

The remaining work is now the intended last-sprint sequence:

- Day 12: example/snippet/build validation
- Day 13: full validation sweep

### Day 11 Validation Scope Note

Day 12 should validate the public-doc changes against the real shipped example
and tooling surface:

- `make examples`
- targeted example binaries referenced by the rewritten docs
- `make tooling-build`

Day 13 should then re-run the maintained reviewed-quality baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

### Day 11 Outputs

- `artifacts/day11-install-quality-docs-polish.md`

## Day 12

**Objective:** Validate the rewritten public-doc surface against the real
shipped example and tooling targets, confirm that the examples referenced by
the Sprint 35 docs still build and run cleanly, and close any last
snippet-to-code mismatch before the final full validation sweep.

### Commands Run

1. Re-read the Day 12 scope and the Day 11 validation plan:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '300,360p' docs/planning/EPIC_3/SPRINT_35/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`
2. Run the example/tooling build surfaces named on Day 11:
   - `make examples`
   - `make tooling-build`
3. Run the example binaries referenced most directly by the rewritten docs:
   - `./build/example_basic_solve`
   - `./build/example_least_squares`
   - `./build/example_iterative`
   - `./build/example_svd_lowrank`
   - `./build/example_eigs`

### Day 12 Validation Findings

#### 1. The shipped example build surface is clean

The Day 11 validation-scope commands passed:

- `make examples`
  - built all `12` example binaries
- `make tooling-build`
  - built all `14` benchmark binaries
  - built all `12` example binaries

Interpretation:

- the public-doc rewrite did not leave the example-facing or compile-only
  tooling surface in a stale or broken state

#### 2. The rewritten docs still match the high-traffic example binaries

The specific binaries most closely tied to the rewritten public guidance all
ran successfully:

- `example_basic_solve`
  - LU copy-before-factor story still matches the program behavior
- `example_least_squares`
  - QR least-squares path is still the overdetermined solve the docs describe
- `example_iterative`
  - GMRES + ILU(0) copy-before-precondition story still matches the program
- `example_svd_lowrank`
  - the SVD feature surface referenced in README/tutorial is still live
- `example_eigs`
  - the example fixture-based eigensolver workflow still runs from project root
    as `examples/README.md` describes

No doc/code mismatch surfaced that required another rewrite pass.

#### 3. The Sprint 35 residual queue is now fully in final validation

Day 12 did not uncover any new drift in:

- public type names
- public option-struct examples
- the reviewed quality/workflow command names
- the support-doc signposts added on Day 11

That means Day 13 can be the intended full validation sweep, not a mixed
debug-and-validation day.

### Day 12 Interpretation

- Sprint 35's public-doc changes now have both compile-surface and
  runtime-smoke confirmation against the examples users are most likely to use
  as references.
- No additional doc edits were needed on Day 12.
- Day 13 should therefore run the full maintained validation set exactly as
  planned.

### Day 12 Outputs

- `artifacts/day12-example-build-validation.md`
