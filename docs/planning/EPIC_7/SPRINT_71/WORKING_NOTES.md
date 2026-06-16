# Sprint 71 Working Notes

## Day 1 - Scope Audit & Baseline Setup

### Goal

Freeze the Sprint 71 starting point before cleanup work begins by
reconfirming the inherited Sprint 70 architecture contract, the preserved
reviewed baseline, the strongest live public/reference contradiction centers,
and the most important support, proof, and planning surfaces that Sprint 71
may touch next.

### Actions

1. Re-read the Sprint 71 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 retrospective, and the Sprint 70 closeout artifact.
2. Re-read the landed Sprint 71 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - public-surface history audit
   - front-door and install cleanup
   - public header narrative cleanup
   - tutorial/example/benchmark support-surface reconciliation
   - maintainer-guide re-centering
   - truth-surface review and closeout
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 71 touch surfaces directly from the
   live tree across:
   - maintained public/reference surfaces
   - support and policy surfaces
   - strongest proof-owner surfaces most sensitive to truth-surface drift
   - project-level Sprint 70 / Epic 7 planning surfaces

### Findings

#### 1. Sprint 71 starts from the Sprint 70 architecture contract, not from another broad Epic 7 review pass

Sprint 70 already fixed the highest-value starting contract for Epic 7:

- the strongest local reviewed baseline remains unchanged
- the state-of-the-art target, bounded qualities, and deferred ambitions are
  separated explicitly
- the product-model, capability, benchmark, packaging/platform, and non-goal
  fences are all already written down
- the ranked Sprint 71-79 carry-forward queue is already explicit

That means Sprint 71 is not rediscovering Epic 7. It is taking the Sprint 70
contract and cleaning the strongest user-facing and reference-facing surfaces
so the repo reads more like a mature library and less like a sprint archive.

The opening Sprint 71 work should therefore be treated as:

- public-surface history audit
- front-door and install cleanup
- public header narrative cleanup
- support-surface reconciliation
- maintainer-guide re-centering
- truth-surface review and closeout

Interpretation:

- Sprint 71 is a bounded public/reference cleanup sprint, not a disguised
  product-model or capability sprint
- the Sprint 70 contract is already real input, so Day 1 should sharpen the
  exact cleanup map against the live tree rather than reopen architecture
  questions abstractly

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 71 truth surface

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The Day 1 pass also rechecked `make -n quality-review-full`, so Sprint 71
starts from the same reviewed local truth surface and rerun guidance that
Sprint 70 preserved.

Interpretation:

- Sprint 71 inherits the exact same reviewed baseline story as the Sprint 70
  close
- even though Day 1 is docs-only planning work, it still starts from the
  strongest reviewed truth surface rather than from a lighter planning-only
  proxy

#### 3. The highest-value Sprint 71 pressure is now public/reference drift, not capability or implementation drift

The live repo still has the Sprint 70 structural ceilings, but Sprint 71 is
not the sprint that should move them. The strongest current Day 1 Sprint 71
pressure reduces to:

1. top-level public front-door cleanup
2. install/release surface cleanup
3. public header narrative cleanup
4. support-surface teaching/proof reconciliation
5. policy-authority recentering
6. truth-surface review before handoff to Sprint 72

Interpretation:

- Sprint 71 should not widen into product-model or capability implementation
  work
- the real Day 1 task is freezing the exact cleanup and non-goal fence around
  the public/reference package before later days start landing edits

#### 4. The strongest live Sprint 71 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- all counts below are raw Day 1 `wc -l` newline counts captured from the live
  tree
- editor or GitHub line numbers may read one higher when they count displayed
  lines rather than trailing newlines

- maintained public/reference surfaces:
  - `README.md` = `1034`
  - `INSTALL.md` = `237`
  - `include/sparse_cholesky.h` = `232`
- likely support and policy surfaces:
  - `docs/tutorial.md` = `479`
  - `examples/README.md` = `166`
  - `benchmarks/README.md` = `370`
  - `docs/maintainer_guide.md` = `578`
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest proof-owner surfaces most sensitive to wording drift:
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_integration.c` = `2411`
  - `tests/test_fuzz.c` = `651`
- project-level planning and handoff surfaces:
  - `docs/planning/EPIC_7/PROJECT_PLAN.md` = `356`
  - `docs/planning/EPIC_7/SPRINT_70/RETROSPECTIVE.md` = `252`

These are not all immediate edit targets, but they are the real Day 1 map for
where Sprint 71 cleanup pressure and truth-surface sensitivity now live.

#### 5. The current tree still confirms the strongest Sprint 71 contradiction centers from the Sprint 70 queue

The Day 1 pass directly re-confirmed the strongest Sprint 70 carry-forward
centers:

- front-door and install contradiction centers:
  - `README.md`
  - `INSTALL.md`
- strongest header/reference contradiction center:
  - `include/sparse_cholesky.h`
- strongest support-only surfaces unless first-batch cleanup forces them:
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- strongest proof-owner surfaces that Sprint 71 must not let public wording
  steal authority from:
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`

So the Sprint 70 queue remains directionally correct against the live repo.
Day 1 did not uncover a different first-order Sprint 71 problem than the
Sprint 70 contract already named.

## Preserved Day 1 Non-Goal Fence

Sprint 71 Day 1 confirms the following non-goals before deeper work begins:

- no reopening of the Sprint 70 architecture contract
- no implementation or behavior work disguised as public cleanup
- no widening of platform, packaging, benchmark, or capability claims
- no repo-wide chronology cleanup campaign detached from ranked contradiction
  centers
- no public-header cleanup wave wider than the strongest ranked center
- no benchmark, example, or install wording that steals test- or policy-owned
  guarantees

## Day 1 Exit State

Sprint 71 now starts from one explicit cleanup baseline:

- the Sprint 70 architecture contract and carry-forward queue are both active
  and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor remains explicit at `53`
- the broad Sprint 71 goal has already narrowed to front-door/install cleanup,
  public-header cleanup, support-surface reconciliation, maintainer-guide
  recentering, and truth-surface review
- the next step is to recheck the exact docs-only validation and truth-surface
  contract before the deeper public/reference audit begins

## Day 2 - Validation Baseline & Truth-Surface Recheck

### Goal

Reconfirm the docs-only validation contract and the exact truth surfaces that
Sprint 71 cleanup must preserve before the sprint starts rewriting public or
reference-facing wording.

### Actions

1. Reconfirmed the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the strongest current truth-authority surfaces:
   - `README.md`
   - `INSTALL.md`
   - `docs/maintainer_guide.md`
   - `benchmarks/README.md`
   - `examples/README.md`
3. Reconfirmed the Sprint 71 authority split for:
   - docs-only days
   - future `*.c` / `*.h` days
   - substantial architecture or implementation days in later sprints
4. Fixed the targeted Sprint 71 docs-only sanity set explicitly:
   - diff review
   - terminology/alignment scans
   - touched-surface `wc -l`
   - branch-state rechecks
5. Recorded the preserved truth-surface checklist that later Sprint 71 cleanup
   batches must not distort.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 71 start

Sprint 71 still starts from:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 71 inherits the exact same reviewed-baseline authority split as the
  Sprint 70 close
- even though this sprint is public/reference cleanup, it should still read
  from the same reviewed truth surface rather than inventing a lighter local
  rule set

#### 2. The docs-only versus code-touching validation split is now explicit before cleanup starts

Sprint 71 now fixes the following validation split:

- docs-only days:
  - targeted sanity checks only
- bounded `*.c` / `*.h` days, if they appear later:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial architecture, capability, backend, or
  platform work in later sprints:
  - `make quality-review-full`

This is the most useful Day 2 clarification:

- Sprint 71 is intentionally docs-only
- but the sprint still preserves the same code-day and substantial-day
  validation contract the repo already uses

#### 3. The preserved truth surfaces Sprint 71 must not distort are now explicit

The Day 2 pass fixes the following truth-surface checklist for later Sprint 71
cleanup:

- `README.md` must preserve:
  - the orthogonal linked-list public center as the shipped current product
    reading
  - examples vs benchmarks vs tests ownership
  - the threshold-free reading of `make bench-canonical-report`
  - the current platform-confidence summary
- `INSTALL.md` must preserve:
  - static-first install/release shape
  - reviewed Linux/macOS/Windows lane asymmetry
  - local install/package regression ownership without promoting it to a broad
    reviewed install-validation claim
- `docs/maintainer_guide.md` must remain:
  - the main policy authority
  - the home for deeper rationale and deferred-queue reading that should not
    stay duplicated in user-facing docs
- `benchmarks/README.md` must preserve:
  - benchmarks as workflow/performance proof surfaces
  - tests as regression/oracle/property owners
- `examples/README.md` must preserve:
  - examples as adoption and workflow-teaching surfaces
  - no benchmark- or test-owned guarantee widening

Interpretation:

- Sprint 71 can simplify wording
- but it should not simplify by blurring ownership or broadening claims

#### 4. The targeted docs-only sanity set is now fixed for the whole sprint

The maintained Sprint 71 docs-only sanity set is now:

1. diff review on touched public/reference/support surfaces
2. terminology/alignment scans on:
   - workflow ownership
   - benchmark/test/example authority
   - static-first packaging and reviewed-platform wording
3. touched-surface `wc -l` checks where snapshot measurements are recorded
4. branch-state rechecks after each landing batch

This is the main Day 2 operational output:

- later Sprint 71 days now have one exact sanity routine
- not a vague “docs look fine” standard

#### 5. The strongest Day 3 audit targets are now confirmed against the live truth surfaces

The Day 2 reread confirms the strongest Day 3 public-audit targets remain:

- top public contradiction centers:
  - `README.md`
  - `INSTALL.md`
- strongest support surfaces to re-rank:
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
- strongest policy surface that should remain support-first unless later
  cleanup truly forces it:
  - `docs/maintainer_guide.md`

So the Day 1 baseline and Sprint 70 queue remain directionally correct:

- Sprint 71 should audit the public/docs contradiction map next
- not jump early into headers, implementation, or broader policy churn

## Day 2 Exit State

Sprint 71 now has one explicit docs-only validation and truth-surface
contract:

- strongest local reviewed baseline remains unchanged
- docs-only, code-day, and substantial-day validation expectations are all
  explicit
- the preserved product/install/benchmark/example/policy truth surfaces are
  fixed in writing
- the targeted Sprint 71 sanity set is now defined before deeper audit begins

That gives Day 3 one exact job:

- re-rank the strongest remaining chronology and policy-density seams across
  the live public doc surfaces before cleanup design begins

## Day 3 - Public-Surface History Audit I

### Goal

Reduce Sprint 71's broad public/reference cleanup concern to a ranked live
contradiction map across the current user-facing docs before the sprint fixes
its first landing boundary.

### Actions

1. Re-read the Day 3 scope in `docs/planning/EPIC_7/SPRINT_71/PLAN.md`.
2. Re-read the current public-facing Sprint 71 audit targets:
   - `README.md`
   - `INSTALL.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
3. Re-read `docs/maintainer_guide.md` as the support-first policy authority
   rather than as a first cleanup center.
4. Ran targeted terminology scans across those surfaces for:
   - sprint-history spill
   - workflow ownership
   - benchmark/test/example authority
   - static-first packaging and platform wording
5. Rechecked the raw Day 1 `wc -l` hotspot counts for the current public and
   support surfaces.

### Findings

#### 1. `README.md` is still the strongest first public contradiction center

The Day 3 reread confirms that `README.md` carries the densest remaining mix
of:

- top-level product story
- workflow-choice guidance
- examples versus benchmarks versus tests ownership
- benchmark-reporting interpretation
- platform-confidence summary
- install/package summary
- capability and limitation framing

Interpretation:

- `README.md` is not weak because it is missing material
- it is the strongest first cleanup target because it still carries too many
  product, workflow, proof, and platform responsibilities in one front-door
  surface

#### 2. `INSTALL.md` is the strongest second target because operator guidance and contract interpretation are still tightly layered together

`INSTALL.md` is smaller than `README.md`, but it still carries a dense mix of:

- operator install steps
- maintained static-first package-shape wording
- reviewed versus supplemental platform-lane interpretation
- local install/package proof ownership
- packaging-consumer guidance

Interpretation:

- `INSTALL.md` is the strongest second target because it still reads partly
  like a product contract explainer and partly like an operator runbook
- that mix is real and valuable, but it should be tightened rather than left
  as layered sprint-history accumulation

#### 3. `docs/tutorial.md` remains the strongest third target, but as a teaching-flow cleanup rather than a front-door contradiction center

The tutorial still carries some repeated framing around:

- repeated-run direct lifecycle guidance
- handoff to `example_analysis`
- handoff to retained benchmark surfaces
- clarification that tests own regression, oracle, and property guarantees

Interpretation:

- `docs/tutorial.md` is still important
- but it is not the best first cleanup center because its main burden is
  repeated ownership framing inside a teaching document rather than a broad
  front-door contradiction

#### 4. `benchmarks/README.md` is the strongest support-surface contradiction center

Among the support surfaces, `benchmarks/README.md` still carries the densest
mix of:

- benchmark-governance interpretation
- canonical-report framing
- benchmark versus test ownership clarification
- retained benchmark-lane history

Interpretation:

- it is the strongest support-surface contradiction center
- but it should still stay behind the first public landing because the
  front-door and install surfaces are higher-leverage user-facing cleanup
  centers

#### 5. `examples/README.md` is lower-risk support context, not a first cleanup center

The examples surface already reads more narrowly as:

- adoption and workflow-teaching guidance
- explicit non-ownership of regression/oracle/property guarantees
- benchmark handoff after workflow adoption

Interpretation:

- `examples/README.md` still matters for later reconciliation
- but it is weaker than `benchmarks/README.md` as a contradiction center and
  should stay support-only unless the first landing forces it to move

#### 6. `docs/maintainer_guide.md` remains support-first because it is already the right policy authority

The maintainer guide still carries deep rationale, deferred-queue reading, and
platform/package/proof interpretation, but that is intentional policy-density
rather than accidental front-door clutter.

Interpretation:

- `docs/maintainer_guide.md` should remain the main policy home
- Sprint 71 should simplify public surfaces by recentering them around this
  authority, not by making the maintainer guide the first cleanup center

#### 7. The Day 3 ranking is now explicit enough to drive the Day 4 boundary

The current live contradiction ranking is:

1. `README.md`
2. `INSTALL.md`
3. `docs/tutorial.md`
4. `benchmarks/README.md`
5. `examples/README.md`
6. `docs/maintainer_guide.md` as support-first policy authority

Interpretation:

- the broad Sprint 71 cleanup problem is now reduced to a real file ranking
- Day 4 should freeze the first landing boundary around the top user-facing
  centers, with support surfaces moving only if that landing truly forces them

## Day 3 Exit State

Sprint 71 Day 3 closes with one concrete public-surface contradiction map:

1. `README.md` is the strongest first cleanup center
2. `INSTALL.md` is the strongest second cleanup center
3. `docs/tutorial.md` is a real later teaching-flow cleanup target
4. `benchmarks/README.md` is the strongest support-surface contradiction
   center
5. `examples/README.md` and `docs/maintainer_guide.md` remain support-first
   rather than first-batch centers

That gives Day 4 one exact job:

- refine this ranking into a first public/reference cleanup fence for Sprint 71
