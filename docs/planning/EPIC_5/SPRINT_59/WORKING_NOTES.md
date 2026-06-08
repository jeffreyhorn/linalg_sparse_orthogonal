# Sprint 59 Working Notes

## Day 1

**Objective:** Turn the Sprint 59 project-plan scope plus the Sprint 58
validated close state into a concrete final Epic 5 quality/platform and
closeout starting point by confirming the preserved reviewed baseline, naming
the Sprint 59 follow-through workstreams explicitly, and defining the
authoritative quality-contract, platform, and closeout hotspots before any
follow-through edits begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 59 project-plan source and the new sprint plan:
   - `sed -n '312,341p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
3. Re-read the strongest inherited Sprint 58 closeout sources:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md`
4. Re-read the Epic 5 review/todo guidance for the remaining quality/platform
   queue:
   - `sed -n '278,320p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '196,210p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
   - `rg -n "Quality-Platform|dead-code|coverage|Windows|macOS staging|reviewed-wrapper parity|deadcode|platform" docs/planning/EPIC_5 docs/maintainer_guide.md README.md Makefile .github/workflows -g '!build'`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Re-read the strongest live quality/platform contract surfaces:
   - `sed -n '840,872p' README.md`
   - `sed -n '498,650p' Makefile`
8. Measure the main Sprint 59 quality/platform and closeout hotspot surfaces:
   - `wc -l scripts/deadcode_workflow.sh scripts/deadcode_report.py .github/workflows/ci.yml README.md Makefile docs/maintainer_guide.md docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md docs/planning/EPIC_5/SPRINT_58/artifacts/day13-full-validation-sweep.md docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md`

### Day 1 Findings

#### 1. Sprint 59 starts from a validated final-sprint baseline, not from renewed feature, lifecycle, or API design work

The inherited starting state is already explicit and stable:

- Sprint 58 closed with:
  - public-surface simplification complete across README/tutorial/headers/
    examples/benchmarks
  - no public API redesign
  - no workflow-boundary drift
  - no implementation recovery queue hidden behind the docs work
- Sprint 58 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited product contract remains unchanged:
  - one-shot APIs remain first-class entry points
  - repeated direct-run lifecycle remains the validated Sprint 50-53 shape
  - repeated-run iterative/eigensolver handles remain the validated Sprint 54
    support boundary

Interpretation:

- Sprint 59 is not a feature sprint
- Sprint 59 is not a lifecycle-recovery sprint
- Sprint 59 is a bounded quality/platform/productization and closeout sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the final Epic 5 follow-through

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`
- `quality-review` still includes:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

Interpretation:

- Sprint 59 should keep using the exact `strongest local reviewed baseline`
  phrasing
- later code-touching days should keep the reviewed CMake count and
  Makefile/CMake parity contract as the main truthfulness anchors
- the final Epic 5 closeout should still treat `deadcode-check` as a report-
  completeness gate rather than a zero-findings gate

#### 3. The Epic 5 review queue is now concentrated in staged quality/platform follow-through rather than implementation ownership

The project plan and Epic 5 review/todo notes already fix the remaining
quality/platform queue:

- dead-code execution remains serialized
- macOS dead-code remains staged
- Windows local reviewed-wrapper parity remains staged
- Windows dead-code remains excluded
- coverage remains calibrated to the current enforced 80% reality

The inherited review guidance is explicit:

- the quality contract is already honest and strong
- the remaining gaps should be treated as a bounded productization pass
- the goal is to refresh each staged/excluded quality surface with a current
  disposition:
  - fixed
  - still intentionally staged
  - or explicitly deferred again with current rationale

Interpretation:

- Sprint 59 should treat the remaining quality/platform queue as real but
  bounded
- the strongest remaining maintainability pressure is now in platform-story
  truthfulness and residual disposition, not solver behavior

#### 4. Sprint 59 reduces cleanly to six bounded work classes

The Sprint 59 project-plan items reduce to six bounded work classes:

1. quality/platform residual audit
2. bounded quality follow-through batch
3. final cross-surface compatibility sweep
4. full validation sweep
5. Epic 5 summary and handoff
6. project-plan / residual-journal finalization

The strongest architectural narrowing is:

- keep the work centered on final truthfulness and bounded convergence
- prefer explicit residual disposition over broad platform ambition
- preserve the Sprint 50-58 public and validation fence exactly
- do not broaden into feature, API, benchmark-framework, or CI-redesign work

Interpretation:

- Sprint 59 is about closing the remaining stated gaps honestly
- the right output shape is a smaller, measured residual queue and a final
  integrated closeout package

#### 5. The authoritative Sprint 59 quality/platform hotspots are now fixed directly from the live repo

The strongest Sprint 59 touched surfaces are now explicit:

- quality-contract and platform-story surfaces:
  - `README.md` = `973`
  - `Makefile` = `878`
  - `docs/maintainer_guide.md` = `294`
  - `.github/workflows/ci.yml` = `221`
- dead-code and classification workflow surfaces:
  - `scripts/deadcode_workflow.sh` = `219`
  - `scripts/deadcode_report.py` = `550`
- project-level planning and closeout surfaces:
  - `docs/planning/EPIC_5/PROJECT_PLAN.md` = `340`
  - `docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md` = `226`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day13-full-validation-sweep.md` = `107`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md` = `125`

Interpretation:

- the strongest implementation-adjacent follow-through pressure is in
  `Makefile`, dead-code workflow scripts, and the platform contract wording
- the strongest closeout-writing pressure is now in Epic-level summary and
  residual disposition surfaces rather than sprint-local implementation notes

#### 6. The inherited public and quality compatibility fence gives Sprint 59 a clean non-goal boundary

The inherited fence remains:

- no public API redesign
- no reopening the direct lifecycle contract
- no reopening the repeated-run iterative/eigensolver support boundary
- no broad CI/platform redesign disguised as “final polish”
- no fake closure claims on staged quality surfaces without fresh evidence

Interpretation:

- Sprint 59 should improve residual disposition and final truthfulness under
  the already-validated product surface
- bounded convergence and honest closeout, not maximal platform expansion, are
  the success criteria

#### 7. The final Epic 5 closeout should now be treated as a measured integration package rather than a generic retrospective pass

The inherited Sprint 50-58 work already established:

- direct lifecycle, CSC, and factor-many behavior
- repeated-run iterative/eigensolver support boundaries
- large-source and giant-test maintainability improvements
- public-surface simplification

Interpretation:

- Sprint 59 should focus on:
  - final quality/platform disposition
  - final cross-surface caller-story agreement
  - final validation baseline
  - final project-level residual and handoff clarity
- the right closeout shape is a measured Epic 5 finish, not another broad
  storytelling sweep

## Day 1 Close

Sprint 59 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated public-contract and workflow fence from Sprint 58
- named quality/platform/productization hotspots
- clear final-sprint workstreams
- explicit non-goal fence against feature, API, or broad platform redesign

That is enough to move to the Day 2 validation and truthfulness-anchor recheck
without reopening Sprint 50-58 product-surface decisions.

## Day 2

**Objective:** Reconfirm the maintained reviewed validation baseline, the exact
truthfulness anchors, and the final-sprint rerun set Sprint 59 must preserve
before any quality/platform follow-through or Epic 5 closeout edits begin.

### Commands Run

1. Re-read the Sprint 59 Day 2 plan scope:
   - `sed -n '80,120p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the current Sprint 59 working-notes starting state:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/WORKING_NOTES.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
5. Re-read the live quality-contract wording across the main maintained
   authority surfaces:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check|dead-code|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
6. Reconfirm the targeted final-sprint rerun set from the live build tree:
   - `ls build/test_integration build/test_iterative build/test_eigs build/test_eigs_lobpcg build/test_chol_csc build/test_ldlt_csc build/example_analysis build/example_iterative build/example_ic_minres build/example_eigs build/example_svd_lowrank build/bench_refactor build/bench_refactor_csc build/bench_iterative_reuse build/bench_eigs_reuse`

### Day 2 Findings

#### 1. The maintained reviewed baseline remains unchanged and should stay the explicit final Epic 5 truthfulness anchor

The maintained Sprint 59 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The wrapper wording also remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 59 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the main
  truthfulness anchors for the final sprint

#### 2. The authority split remains explicit and stable enough to carry unchanged through the final closeout

The maintained authority split remains:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 59 does not need to reopen the quality-contract hierarchy itself
- the final Epic 5 closeout can keep treating `deadcode-check` as a
  completeness and interpretation gate, not as a claim that dead-code findings
  must be zero

#### 3. The code-day validation boundary remains the same as the live repo contract

The mandatory gate for later `*.c` / `*.h` days remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial quality/platform follow-through
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- any Sprint 59 code-touching day still does
- substantial quality/platform changes should continue to use the stronger
  reviewed wrapper path too

#### 4. The maintained quality/platform wording is still aligned across the repo's authoritative surfaces

The quality-contract wording remains aligned across:

- `README.md`
  - strongest local reviewed baseline command map
  - dead-code completeness-gate meaning
  - reviewed platform matrix and staged-platform caveats
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - rerun guidance
  - test-count parity checks
- GitHub workflows
  - reviewed CMake execution path
  - dead-code report/check execution path
  - coverage-contract execution path

Interpretation:

- Sprint 59 does not need to reopen baseline wording just to restate the
  existing contract
- the real remaining work is bounded residual disposition on staged/excluded
  platform and dead-code surfaces, not rediscovering the quality hierarchy

#### 5. The authoritative Sprint 59 rerun set is now fixed from the live build tree

The main Sprint 59 follow-on binaries already present in `build/` are:

- `./build/test_integration`
- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Interpretation:

- Sprint 59 can keep its rerun focus on:
  - direct/public lifecycle proof
  - representative maintained examples
  - representative maintained benchmark drivers
  - the reviewed CMake parity anchor
- no broader default rerun set is required on Day 2

#### 6. Day 2 removes validation ambiguity before the final quality/platform audit begins

Day 2 now leaves Sprint 59 with:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative final-sprint rerun set from the live build tree

Interpretation:

- the sprint can move to the Day 3 residual audit without uncertainty about
  what the maintained validation contract actually is
- any later change to the platform/quality story can now be judged against a
  fixed, explicit baseline

## Day 2 Close

Sprint 59 now has an explicit validation and truthfulness boundary:

- preserved reviewed baseline wording
- explicit authority split
- exact code-day validation gate
- stronger reviewed-wrapper default for substantial follow-through
- authoritative tests/examples/benchmarks rerun set from the live tree

That is enough to move to the Day 3 quality/platform residual audit without
validation ambiguity.
