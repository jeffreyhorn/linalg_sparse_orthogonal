# Sprint 40 Day 14: Architecture Contract Synthesis

## Objective

Consolidate Sprint 40 into one stable architectural starting point for the
implementation-heavy Epic 4 sprints. This synthesis turns the baseline,
lifecycle inventories, taxonomy, handle-model design, ownership map,
migration-risk audit, and validation anchor into a coherent handoff rather
than leaving them as disconnected day-by-day notes.

## Sprint 40 End-State Summary

Sprint 40 ends with six stable contract layers:

1. measured quality baseline
2. lifecycle/state inventory
3. future handle-model and migration contract
4. quality-truth ownership model
5. public migration-risk map
6. validation anchor for later refactor sprints

Together these now define the architectural starting point for Sprint 41+.

## 1. Preserved Quality Baseline

Sprint 40 explicitly preserves the validated Epic 3 baseline:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake active-suite truth:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- dead-code compile-db benchmark/example coverage gap:
  - closed
- dead-code contract:
  - completeness/reporting workflow
  - authoritative execution remains serialized
- cross-platform contract:
  - Linux = strongest enforced reviewed baseline
  - macOS dead-code = staged
  - Windows local Makefile reviewed-wrapper parity = staged
  - Windows dead-code = staged for future work and excluded from the current
    enforced baseline

Interpretation:

- Sprint 40 is an architecture-definition sprint, not a baseline-reopening
  sprint
- all later Epic 4 refactors inherit this preserve-not-reopen contract

## 2. Lifecycle/State Contract

Sprint 40 reduces the current API surface to a stable lifecycle model.

### Main taxonomy classes

1. matrix-mutating factor builders
2. original-matrix consumers with separate result handles
3. analysis / factor handle pipelines
4. read-only operator consumers
5. bridge / mixed-boundary surfaces

### Main caller-visible contract axes

1. eligibility preconditions
2. mutation boundary
3. cancellation artifact
4. ownership visibility

### Main architectural pressure points

- LU and Cholesky remain the strongest matrix-as-factor-handle outliers
- QR, SVD, analysis, and preconditioner builders are closer to the future
  handle-oriented direction already
- iterative solvers and eigensolvers are operator consumers and should remain
  so
- `sparse_factors_t` is the strongest bridge object between the current and
  future models

Interpretation:

- later Epic 4 work now has a stable classification model for deciding which
  surfaces need ownership refactors and which need compatibility wording only

## 3. Future Handle-Model Contract

Sprint 40 now defines both an end-state object model and a migration path.

### Target object families

1. coefficient matrix objects
2. symbolic analysis objects
3. numeric factor / decomposition objects
4. solve / operation context objects

### Stable keep/move split

Keep on `SparseMatrix`:

- coefficient/value semantics
- structural editing
- matrix arithmetic / query semantics
- read-only utility behavior
- possibly purely value-derived caches

Move behind explicit handles/contexts:

- factor-specific permutations
- solve-readiness state
- factor-specific norm/tolerance telemetry
- cancellation-sensitive partial factor state
- reusable iterative/eigensolver workspaces

### Staged migration contract

1. internal payload separation
2. bridge-object normalization
3. public explicit-handle enrichment
4. compatibility narrowing

### Earliest safe insertion points

- LU internal builders
- Cholesky internal builders
- `sparse_factors_t` payload normalization
- iterative/eigensolver internal workspace plumbing

### High-risk migration edges

- factorization entry points
- analysis/reorder bridge surfaces
- cancellation semantics
- copy-before-reuse/original-state inference burden
- field-level lifecycle escape hatches

Interpretation:

- Sprint 42 and later no longer need to invent the migration shape during code
  churn
- the architecture contract already says where to start and what to protect

## 4. Quality-Truth Ownership Contract

Sprint 40 now makes the post-Epic-3 ownership model explicit:

- commands -> `Makefile`
- machine behavior -> scripts / execution logic
- CI matrix truth -> workflow YAML
- maintainer authority/procedure -> Sprint 30 docs and executable policy
- user/operator summaries -> `README.md`
- API/lifecycle semantics -> headers/tutorial/examples

### Strongest duplication hotspots

- README command/contract density
- README vs Makefile wrapper messaging
- README vs workflow comments
- README vs Sprint 30 warning-authority docs
- dead-code semantics split across scripts, Makefile output, and README

Interpretation:

- later Epic 4 cleanup work now has a stable “where should this truth live?”
  answer before removing or compressing text

## 5. Public Migration-Risk Contract

Sprint 40 now separates public migration risk from internal-only opportunity.

### Highest-risk public surfaces

1. LU and Cholesky direct factorization entry points
2. `README.md`
3. `docs/tutorial.md`

### Significant but bounded public-risk surfaces

- installed headers for lifecycle-sensitive families
- analyze-once bridge surfaces
- maintained examples that encode lifecycle-sensitive workflows

### Mostly internal-only refactor zones

- helper/allocation consolidation
- graph decomposition
- internal LU/Cholesky payload insertion
- internal `sparse_factors_t` payload normalization
- internal iterative/eigensolver workspace plumbing
- support-surface ownership cleanup

Interpretation:

- later sprints can now keep early structural work internal-first
- while planning compatibility help only where the public surface truly needs it

## 6. Validation Contract for Sprint 41+

Sprint 40 now leaves behind a stable validation reference.

### Mandatory floor for `*.c` / `*.h` changes

```bash
make format
make lint
make test
```

### Strong reviewed baseline for substantial refactors

```bash
make quality-review-full
```

### Serial dead-code sibling path

```bash
make deadcode-report
make deadcode-check
```

### Main preserved validation truths

- reviewed CMake active-suite count remains `53`
- Makefile/CMake parity remains explicit
- dead-code execution remains serialized
- `deadcode-check` remains a completeness gate, not a zero-findings claim
- cross-platform enforced/staged/excluded wording remains honest
- after `sanitize` / `asan` / `sanitize-all` / `tsan` / `omp` / `coverage*`,
  return to the normal reviewed path with `make clean`

Interpretation:

- later sprints now have both the command matrix and the truthfulness checklist
  they must preserve

## Sprint 41 Prerequisites

Sprint 41 should inherit these exact prerequisites from Sprint 40:

1. preserve the current quality baseline rather than reopening it
2. use the lifecycle taxonomy and contract map as the source of truth for
   helper/allocation and lifecycle-oriented preparatory refactors
3. keep early structural work internal-first where possible
4. respect the quality-truth ownership map when editing Makefile/scripts/docs
5. use the Day 13 validation anchor as the default validation contract

Interpretation:

- Sprint 41 should begin with bounded internal groundwork, not public API churn

## Sprint 42 Prerequisites

Sprint 42 should inherit these exact prerequisites from Sprint 40:

1. LU and Cholesky are the first major lifecycle-handle landing targets
2. `sparse_factors_t` remains the main bridge normalization seam
3. wrapper preservation is load-bearing for direct factorization families
4. cancellation and copy-before-reuse remain first-class migration risks
5. README/tutorial/examples/headers are the main public migration-sensitive
   surfaces
6. any new workspace API should be additive and wrapper-compatible

Interpretation:

- Sprint 42 can start architecture-heavy lifecycle work without re-auditing the
  problem space

## Scope Coverage Check

Sprint 40 promised:

- measured baseline
- lifecycle inventory
- state-model taxonomy
- future handle-model design
- quality-contract ownership map
- public migration-risk audit
- validation anchor

Sprint 40 now delivers all of them through:

- Days 1-3 baseline/support artifacts
- Days 5-8 lifecycle/taxonomy/contract artifacts
- Days 9-10 handle-model and migration-strategy artifacts
- Day 11 ownership map
- Day 12 migration-risk audit
- Day 13 validation anchor
- this Day 14 synthesis

Interpretation:

- the Sprint 40 scope from `PROJECT_PLAN.md` is fully covered
- no promised workstream remained implicit or only partially defined

## Final Sprint 40 Contract

Sprint 40 closes with one coherent starting point for Epic 4 implementation:

- preserve the validated Epic 3 quality baseline
- treat `SparseMatrix` as a long-term coefficient/value object, not the
  universal factor-state carrier
- land lifecycle ownership changes internally before public API churn
- use `sparse_factors_t` and the analysis pipeline as preserve-and-evolve
  bridge surfaces
- keep public migration work focused on the small set of truly sensitive
  surfaces
- validate later refactors with the Day 13 command matrix and truthfulness
  checklist

That is the stable architecture contract Sprint 41 and Sprint 42 should now
inherit.
