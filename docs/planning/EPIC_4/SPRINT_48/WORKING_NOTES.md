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

## Day 3

**Objective:** Define the maintainer-facing policy home Sprint 48 will use for
README reduction, tutorial/header reconciliation, and later quality-contract
simplification so the sprint moves toward one stable ownership target rather
than continuing to distribute maintainer policy across user-facing docs.

### Commands Run

1. Re-read the Sprint 48 Day 3 plan section:
   - `sed -n '85,138p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 2 seam inventory:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_48/artifacts/day2-docs-and-quality-contract-surface-inventory.md`
3. Re-read the current Sprint 48 working-notes context:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/WORKING_NOTES.md`
4. Refresh reference patterns for maintainer-facing policy docs already used in
   planning artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
5. Reconfirm the top-level docs surface that would own a stable maintainer
   guide:
   - `find docs -maxdepth 2 -type f | rg 'maintainer|playbook|workflow|guide'`
6. Write the Day 3 design artifact:
   - `docs/planning/EPIC_4/SPRINT_48/artifacts/day3-maintainer-guide-design.md`

### Day 3 Findings

#### 1. Sprint 48 needs one stable maintainer-policy home under `docs/`, not more sprint-only policy capture

The repo already has useful maintainer-policy precedent in sprint artifacts:

- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`

Those files prove the repo can carry rigorous maintainer-facing policy prose,
but they are not the right permanent home for active repo-wide policy.

Interpretation:

- Sprint 48 should create one stable maintainer-policy home under top-level
  `docs/`
- the correct first target is one main guide, not another planning-only policy
  artifact

#### 2. The right target file is one bounded main document: `docs/maintainer_guide.md`

The current top-level docs layout is still small and flat:

- `docs/tutorial.md`
- `docs/algorithm.md`
- `docs/matrix_market.md`

There is no stable maintainer guide there yet.

Interpretation:

- Day 6 should create:
  - `docs/maintainer_guide.md`
- Sprint 48 does not need a new doc subtree or multi-file maintainer handbook
- one bounded main guide is the clearest first landing

#### 3. The guide’s audience is maintainers and high-context contributors, not end users

Day 2 already showed the main ownership problem: README currently serves both
users and maintainers at once.

The new guide should instead serve:

- maintainers
- high-context contributors
- reviewers evaluating quality-contract and documentation-ownership claims

Interpretation:

- README remains the user/operator entry point
- the maintainer guide becomes the policy and interpretation surface

#### 4. Six policy classes should move into the guide as first-class sections

Day 2's strongest “move to maintainer guide” candidates now define the guide’s
core scope:

- reviewed baseline use
- warning authority
- dead-code meaning
- lifecycle/cancellation expectations as maintainer policy
- documentation ownership rules
- designated-initializer / evolving-option-struct norms where still relevant

Interpretation:

- Sprint 48 is not inventing a new policy stack
- it is relocating and tightening the maintainer-policy stack that already
  exists in diluted form across README and nearby docs

#### 5. Executable truth should stay local even after the guide lands

The guide should not try to replace executable or enforced truth owned by:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- CI workflows under `.github/workflows/`

Interpretation:

- the guide should explain ownership, interpretation, and intended use
- it should not become a second command reference or duplicate workflow
  implementations in prose

#### 6. The content that must stay outside the guide is now explicit

Sprint 48 should avoid over-centralizing these content classes:

- end-user quick-start material:
  - stays in `README.md`
- concise API-local caveats:
  - stay in public headers
- tutorial teaching flow and user-facing matrix-state guidance:
  - stays in `docs/tutorial.md`
- benchmark/example usage syntax:
  - stays in `benchmarks/README.md`
  - stays in `examples/README.md`

Interpretation:

- the maintainer guide should centralize policy
- it should not centralize all explanation

#### 7. Cross-reference rules are now fixed before implementation begins

The correct cross-reference shape is:

- `README.md`
  - user-facing entry point
  - links to the maintainer guide for policy
- `docs/maintainer_guide.md`
  - repository-wide policy and ownership interpretation
  - links outward to executable or API-local truth
- `docs/tutorial.md`
  - user-facing behavior guidance
  - only links to the guide when policy interpretation matters
- public headers
  - concise call-site caveats
  - not long maintainer-policy blocks
- local benchmark/example READMEs
  - local usage details
  - not repo-wide policy duplication

Interpretation:

- Sprint 48 now has a concrete “one policy home, local truth where needed”
  redistribution rule

#### 8. The next landing order is now cleaner and more bounded

With the policy-home target fixed, the next order becomes:

1. README reduction
2. maintainer-guide implementation
3. tutorial/header cross-reference reconciliation
4. quality-contract ownership simplification

Interpretation:

- Day 3 was the prerequisite for honest redistribution work
- later days can now move content toward a concrete target instead of a vague
  future guide

## Day 4

**Objective:** Bound the Sprint 48 documentation redistribution batches and
define the focused validation contract before README reduction,
maintainer-guide implementation, and later quality-contract simplification
begin.

### Commands Run

1. Re-read the Sprint 48 Day 4 plan section:
   - `sed -n '120,190p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 3 maintainer-guide design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day3-maintainer-guide-design.md`
3. Re-read the current Sprint 48 working-notes tail:
   - `tail -n 220 docs/planning/EPIC_4/SPRINT_48/WORKING_NOTES.md`
4. Reconfirm the live maintained target names in `Makefile`:
   - `rg -n "^(quality-review-full|tooling-build|deadcode|deadcode-report|deadcode-check|format|lint|test):" Makefile`
   - `sed -n '120,230p' Makefile`
5. Write the Day 4 design artifact:
   - `docs/planning/EPIC_4/SPRINT_48/artifacts/day4-landing-and-validation-design.md`

### Day 4 Findings

#### 1. Sprint 48 needs separate validation rules for docs, scripts, and compiled code

The live sprint scope spans three different edit classes:

- docs-only redistribution
- script and command-surface clarification
- possible compiled-surface touches if later reconciliation reaches `*.c` or
  `*.h`

Interpretation:

- one generic “docs sprint” validation rule would be too loose
- Sprint 48 needs proportionate validation based on the touched surface

#### 2. Docs-only days should use targeted sanity checks rather than the full code gate

The main docs-only redistribution targets are:

- `README.md`
- `docs/maintainer_guide.md`
- `docs/tutorial.md`
- benchmark/example READMEs
- sprint artifacts and notes

Interpretation:

- docs-only days should validate:
  - link and reference correctness
  - local path accuracy
  - command-name accuracy against the live `Makefile`
  - any direct spot-check command needed for truthfulness
- they should not automatically rerun the full `make format` / `make lint` /
  `make test` gate

#### 3. Script and command-surface days should validate directly against the touched executable truth

Sprint 48 may still touch:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`
- README/help text describing maintained quality commands

Interpretation:

- these days should use focused validation such as:
  - `python3 -m py_compile scripts/deadcode_report.py`
  - `bash -n scripts/deadcode_workflow.sh`
  - synthetic malformed/valid input checks where relevant
  - direct `make -n` spot checks for maintained quality targets

#### 4. Any `*.c` or `*.h` change still triggers the full required gate

The sprint may be docs-heavy, but the compiled-surface rule does not change.

Interpretation:

- any Sprint 48 day touching `*.c` or `*.h` must still run:
  - `make format`
  - `make lint`
  - `make test`

#### 5. The stronger reviewed baseline should be reserved for high-signal quality-contract days

The expensive reviewed baseline remains:

- `make quality-review-full`

Interpretation:

- rerun it on:
  - quality-contract simplification days
  - the final validation sweep
- do not pay that cost on every prose-only redistribution day

#### 6. `make tooling-build` is the right maintained compile-only follow-on for touched public auxiliary surfaces

The live auxiliary compile-only target remains:

- `make tooling-build`

Interpretation:

- use it when Sprint 48 touches:
  - example source or example docs coupled to built binaries
  - benchmark docs or compile-only public auxiliary surfaces
- do not invent ad hoc compile-only command matrices when a maintained target
  already exists

#### 7. The implementation order is now fixed as five bounded landing batches

With Day 3's policy-home target fixed, the intended order becomes:

1. README reduction
2. maintainer-guide implementation
3. tutorial/header cross-reference reconciliation
4. quality-contract ownership simplification
5. docs sanity sweep

Interpretation:

- the user-facing scope should become clearer before broader reconciliation
- the maintainer guide should land before local references are retuned toward
  it
- quality-contract simplification belongs later, after the prose homes are
  already stable

#### 8. Out-of-scope items need to stay explicit before redistribution begins

Sprint 48 should continue to exclude:

- broad CI redesign
- dead-code workflow redesign
- broad tutorial rewrite
- large benchmark/example content expansion
- public API behavior changes via docs cleanup
- replacing local executable truth with prose summaries

Interpretation:

- Day 4 locks the sprint into ownership simplification and documentation
  cleanup, not broader workflow redesign

## Day 5

**Objective:** Land the first bounded README reduction pass so the project
entry point remains strong for users and operators while materially shrinking
the embedded maintainer-policy duplication around quality, dead-code, and
readiness guidance.

### Commands Run

1. Re-read the Sprint 48 Day 5 plan section:
   - `sed -n '185,255p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 4 landing/validation design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day4-landing-and-validation-design.md`
3. Re-read the live README hotspot and heading layout:
   - `sed -n '600,930p' README.md`
   - `sed -n '1,180p' README.md`
   - `rg -n "^### |^## " README.md`
4. Refresh the quality/maintainer markers inside `README.md`:
   - `rg -n "quality-review-full|deadcode-check|deadcode-report|Maintainer|maintainer|quality readiness|cross-platform|CI|lifecycle|cancellation" README.md`
5. Land the first README reduction pass:
   - `README.md`
6. Run targeted Day 5 sanity checks:
   - `rg -n "Compile Hygiene Playbook|Rebuild Workflow|Cross-Platform CI Contract|Maintainer References|Quality Readiness Checklist" README.md`
   - `test -f docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
   - `test -f docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`

### Day 5 Findings

#### 1. The README can shrink materially without losing the operator command map

The Day 5 pass kept the high-signal operator surfaces visible:

- direct build/test commands
- reviewed local wrappers
- dead-code commands
- cross-platform CI truth table
- concise readiness checks

Interpretation:

- README can become smaller and more user-facing without becoming shallow

#### 2. The dead-code section was carrying too much maintainer interpretation for a user entry point

The Day 5 reduction keeps the user-relevant contract:

- what the three dead-code commands do
- that `deadcode-check` is a completeness gate rather than a zero-findings
  claim
- that the dead-code path remains serialized

Interpretation:

- the operator truth remains
- the oversized maintainer-policy density is lower

#### 3. The reviewed-quality wrapper section was the clearest place to reduce repeated prose

The README now still identifies:

- reviewed Makefile path
- strongest local reviewed baseline
- reviewed CMake parity path
- additive relationship between wrappers and direct commands

Interpretation:

- the command map is still visible
- the long wrapper-expansion prose no longer dominates the entry document

#### 4. The CI truth table remains worth keeping in README, but the surrounding restatement was not

The cross-platform CI contract table is still useful because it tells users and
operators what is enforced, staged, and supplemental.

Interpretation:

- the table should stay
- the large amount of adjacent policy restatement did not need to stay

#### 5. Day 5 reduced maintainer-policy density before the guide exists, but did not pretend relocation is finished

The README now keeps only concise maintainer references:

- Sprint 30 compile-hygiene and rebuild docs
- designated-initializer reminder for non-default examples
- historical-evidence / test-truth reminders

Interpretation:

- this is a first reduction pass, not the final policy-home move
- Day 6 still needs to create the real maintainer guide
