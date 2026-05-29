# Sprint 48 Working Notes

## Day 1

**Objective:** Turn the Sprint 48 project-plan scope plus the Sprint 40/42/47
execution rules into a concrete documentation-ownership and quality-contract
starting point by confirming the preserved reviewed contracts, naming the
Sprint 48 workstreams explicitly, and defining the authoritative README,
quality-command, header, tutorial, and CI inputs before simplification begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 48 project-plan source and the new sprint plan:
   - `sed -n '292,323p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,120p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
3. Re-read the immediate prerequisite closeouts:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_47/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
6. Measure the live documentation / quality-contract hotspot sizes:
   - `wc -l README.md benchmarks/README.md examples/README.md docs/tutorial.md Makefile .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml`
   - `wc -l README.md docs/tutorial.md include/sparse_matrix.h include/sparse_lu.h include/sparse_cholesky.h Makefile scripts/deadcode_report.py scripts/deadcode_workflow.sh`
7. Refresh the live duplication seam markers:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode-check|tooling-build" Makefile README.md docs/tutorial.md scripts/deadcode_report.py scripts/deadcode_workflow.sh .github -g '!build'`
   - `rg -n "lifecycle|factored|dead-code|quality-review|maintainer|reviewed baseline|designated initializer|designated-initializer|README|tutorial" README.md docs/tutorial.md include scripts Makefile .github -g '!build'`
   - `sed -n '640,845p' README.md`
8. Re-read one recent Day 1 artifact/notes pattern for format calibration:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day1-scope-and-cli-auxiliary-baseline.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_47/WORKING_NOTES.md`

### Day 1 Findings

#### 1. Sprint 48 starts from a preserved Sprint 40/42/47 baseline, not from quality-baseline repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode`
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized
- Sprint 42 already left behind the lifecycle/cancellation guidance boundary
- Sprint 47 already tightened the benchmark/example/tooling auxiliary surface

Interpretation:

- Sprint 48 is not a solver-correctness or reviewed-baseline repair sprint
- Sprint 48 is a documentation-ownership and quality-contract simplification
  sprint on top of an already-validated Epic 4 baseline

#### 2. README and the quality-contract prose are the main duplication hotspot

The live documentation sizes make the primary hotspot obvious:

- `README.md` = `923`
- `Makefile` = `872`
- `docs/tutorial.md` = `413`
- `scripts/deadcode_report.py` = `550`
- `scripts/deadcode_workflow.sh` = `219`

The highest-density duplication seam is inside `README.md` around the reviewed
quality and dead-code sections:

- reviewed local quality path
- cross-platform CI contract
- quality readiness checklist
- maintainer standards

Interpretation:

- Sprint 48 should treat README reduction and maintainer-policy relocation as
  the main direct landing zone
- it should not treat all docs as equally duplicated

#### 3. The quality-contract is effective but spread across too many ownership surfaces

The live quality-command contract is currently distributed across:

- `Makefile`
- `README.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

The dry-run command surface confirms that `quality-review-full` composes:

- reviewed Makefile path
- `deadcode-check`
- reviewed CMake parity path

Interpretation:

- Sprint 48 should preserve the command semantics
- the main simplification target is explanatory ownership and duplication
- this is not the right sprint for broad command redesign

#### 4. Lifecycle and behavior caveats are still repeated across README, tutorial, and headers

The live repo still carries behavior-policy overlap across:

- README lifecycle / cancellation / quality sections
- `docs/tutorial.md` matrix-state caveats
- public headers such as:
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_analysis.h`

Interpretation:

- Sprint 48 needs a real cross-reference strategy, not just README trimming
- the correct goal is to keep API-relevant caveats local while moving
  maintainer-policy duplication into a clearer home

#### 5. The maintained README has drifted into both user and maintainer roles

Day 1 evidence shows README currently carries both:

- strong user/operator entry content
- detailed maintainer policy about:
  - reviewed wrapper ownership
  - dead-code interpretation
  - cross-platform CI contract
  - maintainer standards
  - release/readiness checklist semantics

Interpretation:

- Sprint 48 must keep README strong as a user-facing entry point
- the main risk is not missing information, but putting too much maintainer
  policy in the wrong home

#### 6. The Sprint 48 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's seven bounded workstreams directly from the plan:

- maintainer-policy home design
- README reduction
- maintainer-guide implementation
- tutorial/header cross-reference reconciliation
- quality-contract ownership simplification
- docs sanity sweep
- validation closeout

Interpretation:

- the front half of the sprint should stay policy-home and README-ownership
  first
- the back half should then pivot into cross-reference cleanup, quality-contract
  simplification, and validation

#### 7. Sprint 48 inherits a clear preserve-not-reopen boundary

Sprint 48 should not reopen:

- core solver architecture work
- broad CI redesign
- dead-code workflow redesign
- benchmark framework redesign
- broad tutorial rewrite
- public API behavioral changes disguised as docs work

Interpretation:

- the correct Sprint 48 shape is:
  - define a maintainer-policy home
  - reduce README to a clearer user-facing entry point
  - reconcile touched docs and headers through references instead of repeated
    policy blocks
  - simplify ownership of the quality contract without changing its substance

#### 8. The Day 1 landing order is fixed before implementation starts

The correct early sprint order is:

1. baseline and seam inventory
2. maintainer-guide design
3. README reduction
4. maintainer-guide implementation
5. tutorial/header cross-reference cleanup
6. quality-contract ownership simplification
7. docs sanity sweep and validation closeout

Interpretation:

- Sprint 48 should preserve Sprint 40's core rule: maintainability cleanup
  should be guided by measured seams and an explicit validation anchor before
  broader documentation redistribution lands

## Day 2

**Objective:** Refresh the documentation and quality-contract seam inventory so
Sprint 48's maintainer-guide design, README reduction, tutorial/header
cross-reference pass, and later quality-contract simplification are sequenced
from the live post-Sprint-47 repo state rather than only from the project-plan
labels.

### Commands Run

1. Re-read the Sprint 48 Day 2 plan section:
   - `sed -n '55,108p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day1-scope-and-quality-contract-baseline.md`
3. Re-read the current high-density README quality/maintainer block:
   - `sed -n '640,845p' README.md`
4. Refresh the live quality-command ownership markers:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode-check|tooling-build" Makefile README.md docs/tutorial.md scripts/deadcode_report.py scripts/deadcode_workflow.sh .github -g '!build'`
5. Refresh the broader duplication markers across docs, headers, scripts, and
   workflows:
   - `rg -n "lifecycle|factored|dead-code|quality-review|maintainer|reviewed baseline|designated initializer|designated-initializer|README|tutorial" README.md docs/tutorial.md include scripts Makefile .github -g '!build'`
6. Reconfirm the current hotspot sizes for the bounded Sprint 48 targets:
   - `wc -l README.md benchmarks/README.md examples/README.md docs/tutorial.md Makefile .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml`
   - `wc -l README.md docs/tutorial.md include/sparse_matrix.h include/sparse_lu.h include/sparse_cholesky.h Makefile scripts/deadcode_report.py scripts/deadcode_workflow.sh`
7. Write the Day 2 seam-inventory artifact:
   - `docs/planning/EPIC_4/SPRINT_48/artifacts/day2-docs-and-quality-contract-surface-inventory.md`

### Day 2 Findings

#### 1. Sprint 48’s live duplication surface reduces cleanly to five seam classes

The current problem is no longer a generic docs backlog. It reduces to:

- quality-command ownership drift
- README user-vs-maintainer scope drift
- tutorial/header behavioral-caveat duplication
- lifecycle/cancellation caveat duplication
- maintainer norms with no stable policy home

Interpretation:

- Sprint 48 should continue from explicit ownership seams
- it should not treat README, headers, tutorial, workflows, and helper scripts
  as one undifferentiated rewrite surface

#### 2. The strongest direct implementation target is still the README quality-policy block

Day 2 confirms that the densest concentration of duplicated maintainer-policy
content remains in `README.md`, especially around:

- dead-code workflow
- reviewed local quality path
- cross-platform CI contract
- quality readiness checklist
- maintainer standards

Interpretation:

- the README reduction pass should be the first real redistribution target
- Sprint 48 should not wait for every other doc to move before reducing that
  block

#### 3. The quality contract currently has three separate authority shapes

The live ownership map is now clearer:

- executable authority:
  - `Makefile`
  - `scripts/deadcode_workflow.sh`
  - `scripts/deadcode_report.py`
- enforced/staged CI authority:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- prose authority:
  - `README.md`

Interpretation:

- Sprint 48 does not need to redesign command semantics first
- it needs a clearer prose home for maintainer-facing policy and contract
  interpretation

#### 4. Tutorial and public headers should keep local behavioral truth, but not own full maintainer policy

The live Day 2 inventory confirms recurring caveats in:

- `docs/tutorial.md`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_analysis.h`

These caveats are still useful locally because they are API-relevant:

- original/unfactored matrix requirements
- factored-state restrictions
- lifecycle/cancellation semantics

Interpretation:

- headers and tutorial should retain concise behavioral truth where users need
  it locally
- repeated maintainer-policy explanation should move elsewhere

#### 5. The strongest “move to maintainer guide” candidates are now explicit

Day 2 confirms the main maintainer-facing policy content currently living in
README:

- reviewed baseline use
- dead-code interpretation
- cross-platform contract reading
- designated-initializer norms
- dormant historical evidence versus live suite truth

Interpretation:

- Sprint 48 does not need to invent new policy from scratch
- it needs a stable maintainer-facing home for policy that already exists

#### 6. The first implementation order is now fixed from live ownership

The correct order after Day 2 is:

1. maintainer-guide design
2. README reduction
3. maintainer-guide implementation
4. tutorial/header cross-reference reconciliation
5. quality-contract ownership simplification

Interpretation:

- policy-home design must come first
- README reduction should happen before broader reconciliation
- quality-contract simplification belongs later, after the prose homes are
  clearer

#### 7. The strongest “leave local” content is also explicit

Sprint 48 should avoid over-centralizing:

- direct command behavior owned by executable surfaces
- concise API-local caveats in headers
- tutorial flow that still teaches user-facing matrix-state expectations
- local benchmark/example usage details already owned by their own READMEs

Interpretation:

- Sprint 48 should simplify ownership, not centralize everything into one
  giant maintainer document
