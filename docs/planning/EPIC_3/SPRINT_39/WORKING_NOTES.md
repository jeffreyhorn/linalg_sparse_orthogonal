# Sprint 39 Working Notes

## Day 1

**Objective:** Turn the Sprint 38 closeout state plus the Sprint 39
project-plan scope into a concrete final-audit baseline by confirming the
validated reviewed/dead-code contract, inventorying the current residual
queues, and naming the first final-audit targets before any implementation
begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 39 scope and Sprint 38 closeout:
   - `sed -n '336,367p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/RETROSPECTIVE.md`
3. Reconfirm the inherited reviewed CMake suite baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Recheck local prerequisite tool availability:
   - `command -v cppcheck`
   - `command -v clang-tidy`
   - `command -v xunused`
   - `command -v ctest`
5. Inventory the current maintained reviewed/dead-code surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
6. Reconfirm the current dead-code end-state artifacts:
   - `python3` bucket-count read of `build/deadcode/report.tsv`
   - `sed -n '1,220p' build/deadcode/coverage-notes.txt`

### Day 1 Findings

#### 1. Sprint 39 starts from a validated final-audit baseline, not unresolved Sprint 38 cleanup debt

Sprint 39 inherits the Sprint 38 close state exactly as intended:

- strongest local reviewed baseline is already named and maintained:
  - `make quality-review-full`
- current reviewed CMake parity baseline remains:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- dead-code compile-db benchmark/example coverage gap is already closed:
  - `benchmarks 14`
  - `examples 12`
  - empty `missing_benchmarks`
  - empty `missing_examples`

Interpretation:

- Sprint 39 is not reopening Sprint 38 compile-db coverage work
- Sprint 39 is not a baseline-building sprint
- Sprint 39 is a final audit and closeout sprint layered on top of a stable
  reviewed/dead-code/readiness contract

#### 2. The strongest maintained quality surface is already broad; the main Day 1 job is ordering the final audit correctly

The maintained local reviewed/dead-code surface is already explicit:

- strongest local reviewed baseline:
  - `quality-review-full`
- reviewed Makefile path:
  - `quality-review`
- reviewed CMake parity path:
  - `quality-review-cmake`
- dead-code/reporting path:
  - `deadcode-report`
  - `deadcode-check`

Interpretation:

- Sprint 39 does not start by inventing new top-level gates
- it starts by verifying the remaining residual claims around warnings,
  dead-code findings, cross-platform limits, and maintainer standards

#### 3. The highest-value open dead-code work is now content-level disposition plus the known serialized-execution limit

Current dead-code residual buckets are:

- `public-surface-review = 4`
- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

Already closed and therefore not a Day 1 open queue:

- `coverage-gap = 0`
- `definitely-unused-internal-candidate = 0`

Still open as workflow topology, not content-level debt:

- authoritative dead-code execution remains serialized
- the shared-path model still runs through:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Interpretation:

- Sprint 39 dead-code work should focus first on final disposition and
  justification of the residual buckets
- it should keep the serialized-execution limit explicit rather than pretending
  dead-code has become concurrent-safe

#### 4. The cross-platform residual queue is bounded and already known

Sprint 38 already narrowed the carried-forward cross-platform closeout queue to:

- macOS dead-code remains staged
- Windows local Makefile reviewed-wrapper parity remains staged
- Windows dead-code remains excluded

Interpretation:

- Sprint 39 cross-platform work should reconcile and record the final enforced /
  staged / excluded contract
- it should not broaden into fake all-platform symmetry work

#### 5. The first final-audit queues are already explicit

Highest-value Sprint 39 surfaces at Day 1:

- final warning audit:
  - Sprint 30 Apple Clang CMake full-tree model remains authoritative
  - Makefile `all` remains the narrower library-only cross-check
- final dead-code audit:
  - residual public/supporting/noise buckets
  - serialized-execution limitation remains explicit
- final cross-platform audit:
  - Linux enforced baseline
  - macOS staged dead-code
  - Windows staged/excluded reviewed/dead-code surfaces
- standards/documentation closeout:
  - maintainer guidance for warning cleanliness, designated initializers,
    dormant-test truthfulness, and dead-code workflow
- temporary-scaffolding cleanup:
  - remove only what is clearly transitional and no longer load-bearing

Interpretation:

- Sprint 39 already has a bounded final-audit sequence
- the initial days should stay audit-first so the later closeout edits remain
  honest, attributable, and final

## Day 2

**Objective:** Reconfirm the final warning-clean contract against the Sprint 30
compile-hygiene playbook so Sprint 39 warning closeout starts from the right
authoritative surfaces and does not confuse narrower maintained quality paths
with full-tree warning proof.

### Commands Run

1. Re-read the Sprint 39 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Sprint 30 warning policy and rebuild workflow:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
3. Sweep current warning-facing contract surfaces:
   - `rg -n "warning|wall-check|Apple Clang|quality-review|lint|Werror|full-tree|authoritative" README.md Makefile CMakeLists.txt docs/planning/EPIC_3 -g '!docs/planning/EPIC_3/SPRINT_39/**'`
4. Re-read the main current user/operator warning and reviewed-path sections:
   - `sed -n '80,150p' README.md`
   - `sed -n '640,760p' README.md`
   - `sed -n '560,620p' Makefile`
5. Re-read the latest full measured baseline that Sprint 39 inherits:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/artifacts/day13-full-validation-sweep.md`

### Day 2 Findings

#### 1. The authoritative final warning claim is still the Sprint 30 Apple Clang CMake full-tree model

The Sprint 30 compile-hygiene playbook remains unambiguous:

- Apple Clang CMake full-tree build is the authoritative warning inventory
- Makefile `all` remains a narrower library-only cross-check
- supported build surfaces define the quality bar, not just the easiest local
  command

The Sprint 30 rebuild workflow also remains the intended reproducible entry
point for warning inventory work:

- `make warning-workflow WARNING_WORKFLOW_LABEL=<label>`

Interpretation:

- Sprint 39 warning closeout must still ground any repository-wide warning
  claim in the warning-workflow / Apple Clang CMake path
- `make quality-review-full` is the strongest local reviewed baseline, but it
  is not itself a replacement for the authoritative full-tree warning
  inventory

#### 2. The current maintained local quality contract is strong, but it is not the same thing as a full-tree warning audit

Current maintained reviewed/local surfaces do provide strong supporting
evidence:

- `make lint`
  - strict `src/*.c` compile with `-Werror`
  - `clang-tidy`
  - `cppcheck`
  - `tooling-build` for benchmarks/examples compile-only coverage
- `make quality-review-full`
  - reviewed Makefile path
  - reviewed CMake parity path

But they do not restate the Sprint 30 authoritative warning claim directly in
their top-level semantics:

- `quality-review-*` is a reviewed baseline command family
- `warning-workflow` remains the dedicated full-tree warning inventory workflow

Interpretation:

- there is no Day 2 evidence of an actual warning regression
- there is a real closeout distinction to preserve between:
  - strongest routine reviewed baseline
  - final authoritative repository-wide warning proof

#### 3. Current warning-facing wording drift is smaller than the Sprint 38 coverage drift was

Compared with Sprint 38 Day 2, the warning-facing contract is already fairly
disciplined:

- `README.md` accurately presents:
  - `make lint`
  - reviewed wrapper commands
  - cross-platform CI contract
- `Makefile` still labels `warning-workflow` explicitly as the Sprint 30
  reproducible Epic 3 warning-capture workflow
- Sprint 38 inherited validation already records the strongest local reviewed
  baseline and separates it from dead-code/reporting semantics

The main remaining risk is not stale counts or fake warning-clean claims in the
user-facing command map. It is more subtle:

- later closeout language could accidentally treat `quality-review-full` as if
  it were the authoritative proof of repository-wide warning cleanliness

Interpretation:

- Day 5 likely does not need a broad warning-doc rewrite
- it needs a narrow closeout batch that preserves the Sprint 30 authority model
  while aligning final Epic 3 summary language with the current maintained
  reviewed baseline

#### 4. The final warning queue is currently truthfulness- and evidence-oriented, not clearly code-oriented

What Day 2 did **not** find:

- a named open warning class in `src/`
- a known post-Sprint-38 regression in reviewed paths
- a current documented mismatch like the earlier Sprint 38 coverage numbers

What Day 2 **did** confirm as the remaining warning-closeout queue:

- preserve the distinction between:
  - warning-workflow / Apple Clang CMake full-tree inventory
  - Makefile `all` narrower library-only cross-check
  - reviewed local wrappers used for routine local protection
- ensure final Epic 3 summary/standards language does not collapse those
  separate evidence tiers into one overclaim

Interpretation:

- Day 5 should begin with the smallest possible warning-closeout batch
- if no real warning regression is found during the stronger Day 5 rerun, the
  likely implementation work is maintainer-standard / closeout wording rather
  than source cleanup

## Day 3

**Objective:** Reassess the residual dead-code buckets after Sprint 38 closed
the compile-db coverage gap so Sprint 39 can separate final content-level
disposition work from the still-intentional serialized-execution workflow
limit.

### Commands Run

1. Re-read the Sprint 39 Day 3 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Sprint 33 dead-code workflow lineage:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/RETROSPECTIVE.md`
3. Re-read the Sprint 38 dead-code closeout state:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/RETROSPECTIVE.md`
4. Re-read the current dead-code report outputs:
   - `sed -n '1,260p' build/deadcode/report.md`
   - `sed -n '1,220p' build/deadcode/report.tsv`
   - `sed -n '1,220p' build/deadcode/coverage-notes.txt`

### Day 3 Findings

#### 1. Sprint 39 no longer inherits a dead-code discovery problem; it inherits a final disposition problem

Compared with the original Sprint 33 workflow closeout, two earlier categories
are already fully closed:

- `coverage-gap = 0`
- `definitely-unused-internal-candidate = 0`

Current residual buckets are:

- `public-surface-review = 4`
- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

Interpretation:

- Sprint 39 is not reopening compile-db breadth work
- Sprint 39 is not reopening a known internal cleanup-ready code-removal queue
- the remaining dead-code work is now primarily about final classification and
  justification

#### 2. The public bucket is effectively already an audited keep set, not an active review queue

All four current `public-surface-review` rows already carry audited keep
dispositions:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Current report language already says:

- these remain exported through installed headers
- current audited outcome for all listed rows is `keep`, not cleanup

Interpretation:

- the public bucket name survives for reporting continuity
- but from a Sprint 39 closeout perspective, this is already a justified keep
  context list rather than a live unresolved review backlog

#### 3. The residual `cppcheck` bucket is still supporting evidence, not a cleanup-ready queue

The current `secondary-candidate-signal` rows remain summarized, not escalated:

- strongest concentrations include:
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_matrix.c`
  - `src/sparse_qr.c`
  - `src/sparse_graph.c`

But the current report contract still frames them as:

- supporting signals only
- not direct removal instructions
- not pass/fail criteria in the staged workflow

Interpretation:

- Day 3 found no evidence that these rows have crossed the threshold into a new
  cleanup-ready deletion batch
- the likely Sprint 39 closeout work is to preserve and clarify this boundary,
  not to manufacture a larger removal queue from noisy scanner output

#### 4. The `non-deadcode-static-analysis-noise` bucket is a documentation/appendix issue, not a cleanup issue

The remaining noise summary is still:

- `constVariablePointer = 106`
- `normalCheckLevelMaxBranches = 23`
- `variableScope = 4`
- `constParameterPointer = 1`
- `constVariable = 1`
- `unreadVariable = 1`

These rows are already classified as:

- appendix-only
- not cleanup candidates

Interpretation:

- Sprint 39 should not spend dead-code closeout time pretending this is
  removal-ready engineering debt
- the correct final treatment is honest explanation and bounded retention

#### 5. The serialized dead-code execution limit remains a workflow-topology constraint, not a content finding

The current coverage notes and Sprint 38 closeout still imply the same workflow
topology:

- shared compile-db path:
  - `build/deadcode-cmake`
- shared artifact path:
  - `build/deadcode/`

The current report/check contract remains truthful only under serialized
execution.

Interpretation:

- Sprint 39 dead-code closeout must keep two separate categories visible:
  - content-level bucket disposition
  - workflow-topology limitation
- a clean final report does **not** imply concurrent-safe execution

#### 6. The likely Day 6 batch is narrow and explanation-heavy

Day 3 narrows the expected Sprint 39 Day 6 dead-code batch to:

- final disposition wording for the public audited keeps
- final disposition wording for the `cppcheck` supporting-signal bucket
- final disposition wording for the static-analysis-noise appendix bucket
- preservation of the serialized-execution limitation as a still-open workflow
  boundary

What Day 3 did **not** justify:

- a new code-removal batch
- a new compile-db expansion batch
- stronger content-based `deadcode-check` failure logic

## Day 4

**Objective:** Reconfirm the final enforced/staged/excluded cross-platform
contract against the current workflows and README so Sprint 39 can separate
real platform drift from the intentionally preserved staged or excluded
surfaces.

### Commands Run

1. Re-read the Sprint 39 Day 4 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Sprint 36 cross-platform parity closeout:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/RETROSPECTIVE.md`
3. Re-read the Sprint 38 carried-forward residual queue:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_38/HANDOFF.md`
4. Re-read the current cross-platform contract surfaces:
   - `sed -n '720,760p' README.md`
   - `sed -n '1,260p' .github/workflows/ci.yml`
   - `sed -n '1,260p' .github/workflows/macos-ci.yml`
   - `sed -n '1,260p' .github/workflows/windows-ci.yml`

### Day 4 Findings

#### 1. The current cross-platform contract still matches the Sprint 36 model cleanly

The README and workflow files still describe the same top-level platform model:

- Linux:
  - enforced reviewed Makefile compile-quality path
  - enforced reviewed CMake parity path
  - enforced dead-code report/check path
- macOS:
  - enforced Apple Clang reviewed path
  - enforced `wall-check`
  - enforced `sanitize`
  - supplemental Homebrew GCC leg
  - staged dead-code
- Windows:
  - enforced reviewed CMake subset
  - staged local Makefile reviewed-wrapper parity
  - staged/excluded dead-code

Interpretation:

- Day 4 found no evidence that the Sprint 36 parity model has drifted out of
  sync between README and CI workflows
- Sprint 39 does not start from a broken platform-contract surface

#### 2. The real remaining queue is exactly the intentionally staged/excluded queue

Sprint 38 already narrowed the carried-forward platform queue to:

- macOS dead-code remains staged
- Windows local Makefile reviewed-wrapper parity remains staged
- Windows dead-code remains excluded

The current workflow/docs surfaces still present those limits honestly:

- macOS workflow does **not** claim enforced dead-code
- Windows workflow does **not** claim local Makefile reviewed-wrapper parity
- Windows still names the enforced reviewed subset through CMake configure /
  build / `ctest -N` / full `ctest`

Interpretation:

- the residual platform queue is not hidden or contradictory
- the remaining Sprint 39 closeout work is final reconciliation and summary,
  not discovery of a new parity gap

#### 3. Reviewed CMake parity remains the only fully honest shared cross-platform reviewed baseline

The current contract still clearly implies:

- Linux has the strongest full reviewed baseline overall
- reviewed CMake parity is the strongest shared cross-platform reviewed subset
- local Makefile reviewed wrappers are not yet truthfully universal

That remains consistent with:

- Sprint 36 handoff
- current README cross-platform contract section
- current Windows workflow enforcement model

Interpretation:

- Sprint 39 should preserve this distinction in final summary language
- Day 7 should not broaden into fake “all platforms now support the same local
  reviewed command surface” claims

#### 4. No new platform-specific regression surfaced at the workflow/docs level

Day 4 did **not** find:

- a mismatch between README platform claims and workflow enforcement names
- a hidden new Windows exclusion beyond the already-named test set
- a new macOS dead-code enforcement claim
- a reopened Linux dead-code or reviewed-wrapper ambiguity

Interpretation:

- the likely Day 7 batch is narrow
- it should focus on final wording reconciliation and closeout clarity rather
  than new workflow topology or platform-feature changes

#### 5. The likely Day 7 batch is contract-focused, not implementation-heavy

Day 4 narrows the expected Sprint 39 Day 7 work to:

- preserve the Linux / macOS / Windows enforced-staged-excluded model in final
  closeout language
- make sure final Epic 3 standards/summary docs do not overclaim Windows local
  reviewed-wrapper parity or universal dead-code parity
- keep reviewed CMake parity framed as the strongest shared reviewed baseline

What Day 4 did **not** justify:

- a new workflow-matrix expansion
- new Windows Makefile parity work
- new macOS dead-code enforcement work
- any attempt to erase the staged/excluded distinctions for simplicity

## Day 5

**Objective:** Land the smallest warning-closeout batch that preserves the
Sprint 30 authority model explicitly in the current operator-facing contract
without inventing a broader warning rewrite or new gate semantics.

### Commands Run

1. Re-read the Sprint 39 Day 5 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Day 2 warning audit:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/artifacts/day2-final-warning-audit.md`
3. Re-read the current README warning/reviewed-quality sections:
   - `sed -n '96,140p' README.md`
   - `sed -n '680,780p' README.md`
4. Re-read the Sprint 30 authoritative warning references:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
5. Validate the touched doc surface directly:
   - `rg -n "warning-workflow|authoritative repository-wide warning inventory|authoritative warning proof|quality-review-full" README.md`
   - `sed -n '104,120p' README.md`
   - `sed -n '758,776p' README.md`

### Day 5 Findings

#### 1. The smallest useful warning-closeout batch was README contract clarification, not new tooling or code cleanup

Day 2 already showed there was no known warning regression queue. The highest
value remaining risk was contract drift between:

- strongest routine local reviewed baseline:
  - `make quality-review-full`
- authoritative repository-wide warning proof:
  - `make warning-workflow WARNING_WORKFLOW_LABEL=<label>`

Day 5 therefore stayed intentionally small and operator-facing.

#### 2. The README now teaches the warning authority model directly

Two concrete clarifications landed:

- the top-level Make command list now includes:
  - `make warning-workflow WARNING_WORKFLOW_LABEL=<label>`
- the Quality Readiness Checklist now states explicitly:
  - repository-wide warning claims still use the Sprint 30 authoritative path
  - the Apple Clang CMake full-tree inventory is the authoritative warning
    proof
  - Makefile `all` remains the narrower library-only cross-check

Interpretation:

- final Epic 3 closeout language now has a current user/operator-facing anchor
  for the warning authority model
- later Sprint 39 summary/standards work no longer has to infer this only from
  Sprint 30 artifacts or Makefile comments

#### 3. The reviewed-baseline contract stays separate and unchanged

Day 5 intentionally did **not** collapse the existing reviewed-quality
language:

- `make quality-review-full` still means strongest routine local reviewed
  baseline
- it still does **not** mean repository-wide warning inventory proof

Interpretation:

- the warning-closeout batch improved truthfulness without changing gate
  semantics
- the current local reviewed baseline remains strong, but its evidence tier is
  now stated more precisely in the README

## Day 6

**Objective:** Land the narrow dead-code closeout batch identified on Day 3 by
turning the residual report buckets into explicit final-state explanations,
while keeping the serialized-execution limit visible as workflow topology
rather than implying a new cleanup queue.

### Commands Run

1. Re-read the Sprint 39 Day 6 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Day 3 dead-code audit:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/artifacts/day3-final-deadcode-audit.md`
3. Re-read the current dead-code workflow surfaces before editing:
   - `sed -n '1,320p' scripts/deadcode_report.py`
   - `sed -n '320,520p' scripts/deadcode_report.py`
   - `sed -n '640,720p' README.md`
   - `sed -n '600,650p' Makefile`
   - `sed -n '1,220p' build/deadcode/report.md`
4. Validate the touched support surfaces:
   - `python3 -m py_compile scripts/deadcode_report.py`
   - `make deadcode-report`
   - `make deadcode-check`

### Day 6 Findings

#### 1. The shipped batch is pure closeout clarification, not new dead-code policy

Day 3 already showed the remaining dead-code work was explanation-heavy:

- no current benchmark/example compile-db gap
- no current definitely-unused internal cleanup batch
- public rows already audited as keeps
- `cppcheck` density still supporting-only
- static-analysis noise still appendix-only

Day 6 therefore kept the batch narrow and operator-facing.

#### 2. The generated report now reads like a closeout-state report instead of a lingering active cleanup queue

The report wording was tightened in these ways:

- `Public-Surface Reviewed Keeps` became:
  - `Public-Surface Justified Keeps`
- the public section now states explicitly that it is closeout context, not an
  active removal queue
- `Secondary cppcheck Candidate Signals` became:
  - `Secondary cppcheck Supporting Signals`
- the secondary section now states directly that these rows do not currently
  justify a new Sprint 39 removal batch
- `Deferred Noise Summary` became:
  - `Appendix Noise Summary`
- the next-action queue now says explicitly:
  - public justified keeps are closeout context
  - static-analysis noise is appendix-only
  - serialized execution is a separate workflow-topology limit

Interpretation:

- the dead-code report now matches the real post-Sprint-38 / post-Day-3 state
  more closely
- final Epic 3 closeout language now has a clearer generated artifact to point
  at

#### 3. The README and `deadcode-check` output now teach the same final contract

README dead-code workflow wording now states:

- current benchmark/example compile-db coverage gap is closed
- there is no current definitely-unused internal cleanup batch
- the public bucket is currently an audited keep list
- `cppcheck` secondary rows are supporting-only
- static-analysis noise is appendix-only

`deadcode-check` output now states more precisely:

- passing is not a zero-findings or removal-ready gate
- residual buckets are closeout/supporting context only
- authoritative execution remains serialized

Interpretation:

- the generated report, the Makefile operator message, and the README dead-code
  docs now all describe the same final-state contract

## Day 7

**Objective:** Land the narrow cross-platform reconciliation batch identified
on Day 4 by making the README state the final staged/excluded platform queue
and the shared reviewed-baseline distinction directly, without changing any CI
workflow behavior.

### Commands Run

1. Re-read the Sprint 39 Day 7 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Day 4 cross-platform audit:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/artifacts/day4-final-cross-platform-audit.md`
3. Re-read the current cross-platform contract surfaces:
   - `sed -n '720,780p' README.md`
   - `sed -n '1,120p' .github/workflows/ci.yml`
   - `sed -n '1,120p' .github/workflows/macos-ci.yml`
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`
4. Validate the touched doc surface directly:
   - `rg -n "strongest shared reviewed baseline|macOS dead-code = staged|Windows local Makefile reviewed-wrapper parity = staged|Windows dead-code = excluded" README.md`
   - `sed -n '736,748p' README.md`
   - `sed -n '788,798p' README.md`

### Day 7 Findings

#### 1. The shipped batch is contract clarification only, not workflow expansion

Day 4 already showed the current workflows and README were broadly consistent.
The remaining closeout gap was subtle:

- the README implied, but did not state directly enough, that reviewed CMake
  parity is the strongest **shared** reviewed baseline
- the README warned against overclaiming staged/supplemental paths, but did not
  restate the exact remaining intentionally non-universal queue in the
  readiness checklist

Day 7 therefore stayed purely documentation-focused.

#### 2. The README now states the shared-baseline distinction directly

The `Cross-Platform CI Contract` interpretation section now says explicitly:

- reviewed CMake parity remains the strongest shared reviewed baseline across
  platforms

Interpretation:

- this preserves the Sprint 36 / Sprint 39 audit distinction between:
  - Linux as the strongest overall enforced baseline
  - reviewed CMake parity as the strongest shared reviewed baseline

#### 3. The readiness checklist now names the exact staged/excluded queue directly

The `Quality Readiness Checklist` now restates the current intentionally
non-universal surfaces explicitly:

- macOS dead-code = staged
- Windows local Makefile reviewed-wrapper parity = staged
- Windows dead-code = excluded

Interpretation:

- final Epic 3 closeout language now has a concise operator-facing anchor for
  the remaining platform-specific limits
- later summary/standards work does not have to infer them only from the table
  or workflow comments

## Day 8

**Objective:** Audit the current maintainer-facing standards and ownership
surfaces so Sprint 39 can consolidate only the guidance that should actually
survive Epic 3, while leaving sprint-narrative material in artifacts rather
than promoting it into permanent repo-level docs.

### Commands Run

1. Re-read the Sprint 39 Day 8 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_39/PLAN.md`
2. Re-read the Sprint 30 warning-standard sources:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
3. Re-read the current repo-root maintainer/operator contract surfaces:
   - `sed -n '560,820p' README.md`
   - `sed -n '1,260p' tests/test_framework.h`
4. Sweep the repo for current standard-like language tied to the Epic 3 themes:
   - `rg -n "designated initializer|designated-initializer|warning-workflow|deadcode|dormant|historical evidence|RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|SPARSE_TEST_LARGE|Cross-Platform CI Contract|quality-review-full|warning-clean|compile-hygiene" README.md INSTALL.md docs include tests Makefile -g '!docs/planning/EPIC_3/SPRINT_39/**'`

### Day 8 Findings

#### 1. Maintainer-standard ownership is already narrower than the total amount of Epic 3 narrative

The current repo has multiple kinds of “documentation,” but not all of them
should survive as equal long-term standard sources.

The stable top-level standards are already concentrated in:

- `README.md`
  - reviewed-quality command map
  - dead-code workflow contract
  - cross-platform CI contract
  - readiness checklist
  - test-category policy
- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
  - authoritative warning-clean evidence model
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
  - authoritative warning-workflow reproduction method
- `tests/test_framework.h`
  - executable truth for skip/slow/experimental test semantics

Interpretation:

- the closeout batch should consolidate references and trim duplication
- it should not try to collapse all Epic 3 policy into one monolithic document

#### 2. The strongest long-term standards already have clear natural homes

Current best ownership by topic:

- repository-wide warning authority model:
  - `COMPILE_HYGIENE_PLAYBOOK.md`
  - `REBUILD_WORKFLOW.md`
- routine operator command map and reviewed/dead-code/platform contract:
  - `README.md`
- executable dormant/slow/experimental truthfulness policy:
  - `tests/test_framework.h`
- public non-default example style / designated-initializer teaching:
  - `README.md` and public headers/tutorial surface from Sprint 35

Interpretation:

- Day 9 should primarily make these ownership boundaries more explicit
- it should avoid creating a new permanent standards document unless a real gap
  exists, because the repo already has authoritative homes for the major
  categories

#### 3. The main residual standards problem is duplication and implicitness, not missing topics

Day 8 did **not** find a missing repo-level standard for the main Epic 3 areas:

- warning-clean authority model exists
- reviewed local quality path exists
- dead-code workflow contract exists
- cross-platform enforced/staged/excluded contract exists
- test truthfulness contract exists
- public designated-initializer guidance exists in the Sprint 35 closeout
  material and public-facing docs

The main residual issue is that some of this remains implicit across multiple
surfaces:

- `README.md` teaches the operator contract but does not yet point back to the
  Sprint 30 warning playbook/workflow as explicitly as it could in the
  readiness/quality sections
- designated-initializer guidance is still distributed across Sprint 35
  artifacts and public-facing docs rather than summarized as one concise
  maintainer expectation
- dormant/historical-test guidance is currently clearest in the README test
  category policy plus `test_framework.h`, but that link is still partly
  inferential

Interpretation:

- Day 9 should be a consolidation/compression pass
- it should prefer short cross-references and crisp policy statements over new
  long-form narrative

#### 4. Sprint-only narrative should stay in artifacts, not be promoted into permanent top-level docs

Large amounts of Epic 3 explanation are useful historically, but they are not
all stable standards:

- Sprint working notes
- day-by-day design artifacts
- review documents that describe pre-cleanup debt in detail

Interpretation:

- the closeout batch should keep using sprint artifacts for historical rationale
- permanent repo-level docs should keep only the stable contract and reference
  the artifacts where needed

#### 5. The likely Day 9 batch is small and reference-oriented

Day 8 narrows the expected standards/documentation closeout batch to:

- tighten README cross-references to the Sprint 30 warning authority docs
- add one concise maintainer-facing statement about designated initializers as
  the default public non-default-options style
- add one concise maintainer-facing statement about dormant/historical test
  evidence living in artifacts rather than the active suite
- avoid broad README restructuring or new standalone standards files
