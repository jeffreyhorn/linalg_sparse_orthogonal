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
