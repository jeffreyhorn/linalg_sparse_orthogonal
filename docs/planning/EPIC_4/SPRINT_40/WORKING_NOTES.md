# Sprint 40 Working Notes

## Day 1

**Objective:** Turn the Sprint 40 project-plan scope plus the Epic 4 review
and remediation plan into a concrete baseline/setup package by confirming the
validated starting contract inherited from Epic 3, naming the Sprint 40
workstreams explicitly, and defining the authoritative input surfaces before
deeper lifecycle and architecture auditing begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 40 plan and Epic 4 source artifacts:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
3. Inspect current Epic 4 sprint-folder state:
   - `ls -R docs/planning/EPIC_4`
4. Re-read a representative prior sprint Day 1 structure:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_39/WORKING_NOTES.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_39/artifacts/day1-final-audit-baseline.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`

### Day 1 Findings

#### 1. Sprint 40 starts from a validated quality baseline, not from a fresh cleanup queue

The inherited Epic 3 baseline remains stable and usable as the starting point
for Epic 4 architecture work:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- dead-code reporting/checking is already part of the maintained surface:
  - `make deadcode-report`
  - `make deadcode-check`
- the Epic 4 plan is therefore not opening with “rebuild the baseline”
  work; it is opening with inventory and contract-definition work layered on a
  stable reviewed/dead-code/readiness floor

Interpretation:

- Sprint 40 is an audit-and-architecture sprint, not a quality-repair sprint
- the main preserve-not-reopen contract is the post-Epic-3 validated local
  reviewed baseline

#### 2. The Sprint 40 workstreams are already well-bounded by the Epic 4 review and remediation plan

Day 1 did not surface a hidden extra scope beyond what Sprint 40 already says
it will do. The workstreams remain:

- baseline capture
- lifecycle inventory
- state-model taxonomy
- future handle-model design
- quality-contract ownership map
- public migration-risk audit
- validation anchor

Interpretation:

- Sprint 40 should stay audit-first and architecture-first
- it should not drift into early implementation batches that belong to Sprints
  41-42 and later

#### 3. The authoritative input surfaces for Sprint 40 are explicit

Highest-value authoritative inputs for the sprint are:

- Sprint 40 plan:
  - `docs/planning/EPIC_4/SPRINT_40/PLAN.md`
- Epic 4 review:
  - `docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
- Epic 4 remediation plan:
  - `docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
- Epic 4 project-plan section:
  - `docs/planning/EPIC_4/PROJECT_PLAN.md`
- inherited baseline contracts:
  - `README.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `build/deadcode/report.md`
  - `build/deadcode/report.tsv`
  - `build/deadcode/coverage-notes.txt`

Interpretation:

- Sprint 40 architecture decisions should trace back to these sources rather
  than ad hoc assumptions
- the sprint has enough documented input to proceed without inventing a new
  review surface

#### 4. The current maintainability/correctness risks are structural, which matches Sprint 40 exactly

The Epic 4 review still points to the same high-value structural themes:

- overloaded mutable `SparseMatrix` lifecycle/state responsibilities
- monolithic subsystem hotspots like `src/sparse_graph.c`
- fragmented allocation/overflow helper patterns
- duplicated quality-contract truth across commands/scripts/docs
- benchmark/tooling surfaces that lag the library’s stronger contracts

Interpretation:

- Sprint 40 should focus on naming architecture boundaries and migration risks
- it should not try to prematurely solve these issues before the inventory and
  taxonomy are complete

#### 5. The first concrete audit order is now clear

The strongest Day 1 to Day 4 sequence is:

1. preserve and restate the reviewed/dead-code/CMake/platform baseline
2. inventory hotspot and allocation-density surfaces
3. begin lifecycle inventory in LU/Cholesky/LDLT, then extend across QR/SVD /
   analysis / iterative / eigensolver surfaces
4. delay handle-model design until after the lifecycle inventory and taxonomy
   have concrete inputs

Interpretation:

- Sprint 40 now has a stable front half:
  - baseline
  - scope
  - hotspot inventory
  - lifecycle inventory kickoff
- that order reduces the risk of architecture design drifting away from the
  current code/documentation truth

## Day 2

**Objective:** Capture the current maintained local reviewed-quality baseline
as a concrete Epic 4 reference point by separating direct commands, reviewed
Makefile wrappers, reviewed CMake parity wrappers, and adjacent dead-code / CI
surfaces without conflating their roles.

### Commands Run

1. Re-read the Sprint 40 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working baseline:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Re-read the top-level build command map in `README.md`:
   - `sed -n '80,220p' README.md`
4. Re-read the maintained quality wrapper and dead-code sections in
   `Makefile`:
   - `sed -n '430,760p' Makefile`
5. Sweep current reviewed/dead-code/CI contract wording:
   - `rg -n "quality-review|quality-review-cmake|quality-review-full|deadcode|Cross-Platform CI Contract|Quality Readiness Checklist|warning-workflow" README.md Makefile .github/workflows/*.yml`
6. Re-read the detailed reviewed/dead-code/CI contract sections in
   `README.md`:
   - `sed -n '620,860p' README.md`
7. Re-read the Linux CI workflow as the strongest enforced reviewed baseline:
   - `sed -n '1,220p' .github/workflows/ci.yml`

### Day 2 Findings

#### 1. The maintained local reviewed baseline is a three-layer model, not one command

The current local quality surface splits into three related but distinct
layers:

- direct commands:
  - `make format`
  - `make format-check`
  - `make lint`
  - `make test`
  - `make check`
- reviewed Makefile wrappers:
  - `make quality-review-compile`
  - `make quality-review`
- reviewed CMake parity wrappers:
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`

Then one top-level aggregate sits above both wrapper families:

- strongest local reviewed baseline:
  - `make quality-review-full`

Interpretation:

- Day 2 confirms that the reviewed-quality baseline is not a single flat
  command list
- later Epic 4 ownership work must preserve the distinction between:
  - direct commands
  - reviewed Makefile path
  - reviewed CMake parity path
  - strongest combined local reviewed baseline

#### 2. The reviewed Makefile path is the local authoritative command layer for formatting, static analysis, runtime tests, and dead-code completeness

The current reviewed Makefile path is:

- `quality-review-compile`
  - `format-check`
  - `lint`
- `quality-review`
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

This makes the Makefile reviewed path the authoritative local owner of:

- formatting truth
- static-analysis truth
- direct runtime test execution
- local dead-code completeness invocation

Interpretation:

- `quality-review` is the local “reviewed Makefile path”, not merely a synonym
  for `check`
- the Makefile reviewed path already owns more than compile cleanliness; it
  includes runtime and dead-code completeness in the maintained local contract

#### 3. The reviewed CMake path is parity-oriented and intentionally narrower in responsibility

The reviewed CMake wrappers own:

- configure / clean rebuild of `build/quality-review-cmake`
- `ctest -N`
- Makefile/CMake test-count parity check
- full CTest execution through `quality-review-cmake`

But they explicitly do **not** replace:

- Makefile-authoritative formatting checks
- Makefile-authoritative static analysis
- Makefile dead-code/reporting commands

Interpretation:

- the reviewed CMake path is parity proof, not the full local quality-policy
  owner
- later Epic 4 ownership mapping should preserve this narrower role instead of
  flattening everything into “CMake and Makefile do the same thing”

#### 4. `quality-review-full` is the strongest local reviewed baseline because it composes the two reviewed paths, not because it supersedes them conceptually

`quality-review-full` is currently defined as:

- `quality-review`
- then `quality-review-cmake`

It is the strongest local reviewed baseline because it composes:

- Makefile reviewed correctness/static-analysis/runtime/dead-code completeness
- CMake rebuild and active-suite parity truth

Interpretation:

- Day 2 confirms the precise meaning of “strongest local reviewed baseline”
- later Epic 4 docs/tooling work should keep this as a composition of two
  reviewed paths rather than collapsing it into a monolithic opaque command

#### 5. The main operational caveats are stable and load-bearing

Two caveat classes remain part of the maintained baseline:

- tree-mutating alternate modes:
  - `sanitize`
  - `asan`
  - `sanitize-all`
  - `tsan`
  - `omp`
  - `coverage*`
  - returning to the normal direct/reviewed path still requires:
    - `make clean`
- serialized dead-code execution:
  - `deadcode*` targets still share:
    - `build/deadcode-cmake`
    - `build/deadcode/`
  - concurrent invocation remains a known race

Interpretation:

- these are not incidental footnotes; they are part of the true local
  operator contract
- Sprint 40 should carry them forward into the future ownership map as
  load-bearing caveats

#### 6. README, Makefile, and CI already have a mostly coherent division of labor, but the ownership boundary is still descriptive rather than explicitly designed

Current practical ownership shape is:

- `Makefile`
  - concrete local commands
  - rerun guidance
  - dead-code invocation and wrapper semantics
- `README.md`
  - operator-facing command map
  - reviewed-quality explanations
  - cross-platform CI contract
  - readiness checklist
- `.github/workflows/*.yml`
  - enforced/staged CI realization of the documented contract

Interpretation:

- the current system is coherent enough to use as a baseline
- but it is still exactly the kind of distributed ownership model the Epic 4
  review said should be tightened and simplified later
