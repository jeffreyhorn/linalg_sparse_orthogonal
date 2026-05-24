# Sprint 40 Day 11: Quality-Contract Ownership Map

## Objective

Define where quality, policy, and workflow truth should live after Epic 4.
This is not yet a simplification patch. It is the ownership model later Epic 4
cleanup work should implement when reducing duplication across commands,
scripts, workflows, and docs.

## Audit Inputs

This audit is grounded in the current live surfaces:

- `Makefile`
- workflow YAML:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- scripts:
  - `scripts/deadcode_workflow.sh`
  - `scripts/deadcode_report.py`
  - `scripts/epic3_warning_workflow.sh`
- operator docs:
  - `README.md`
- maintainer authority docs:
  - `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
  - `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
- executable policy truth:
  - `tests/test_framework.h`

## Current Quality Truth Surfaces

The repo currently exposes five distinct kinds of “truth”:

1. command entry points
2. machine behavior
3. CI matrix enforcement truth
4. maintainer policy / authority model
5. user/operator-facing summaries

The problem is not that these surfaces are missing. The problem is that some of
them currently overlap more than they should.

## Target Ownership Model

### 1. Commands

#### Stable owner

`Makefile`

#### What it should own

- top-level command names
- command composition / sequencing
- local rerun guidance printed during wrappers
- default build-tree paths and helper target wiring

#### What it should not own

- broad policy prose
- CI matrix interpretation
- long explanatory contract summaries already documented elsewhere

#### Examples

- `quality-review-compile`
- `quality-review`
- `quality-review-cmake`
- `quality-review-full`
- `deadcode-report`
- `deadcode-check`
- `warning-workflow`

### 2. Machine Behavior

#### Stable owners

- scripts
- lower-level Makefile rules only where they are the execution implementation

#### What they should own

- actual dead-code workflow behavior
- report generation / invariant checks
- warning-workflow reproduction behavior
- parser/formatter/report logic

#### What they should not own

- operator-facing summary policy beyond concise runtime messages
- broad CI interpretation
- user-facing tutorial material

#### Examples

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- `scripts/epic3_warning_workflow.sh`

### 3. CI Matrix Truth

#### Stable owners

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

#### What they should own

- which commands each platform/job actually runs
- which paths are enforced vs supplemental inside that workflow
- job-local prerequisites and setup
- platform-specific exclusions/staged status when the workflow itself is the
  authoritative executor

#### What they should not own

- long duplicated operator guidance
- lifecycle architecture policy
- README-scale explanation of why the model exists

### 4. Maintainer Policy / Authority Model

#### Stable owners

- Sprint 30 authority docs for warning-clean proof
- dedicated maintainer-policy docs under `docs/planning/`
- executable policy headers where behavior is part of the code contract

#### What they should own

- repository-wide warning authority model
- rebuild/reproduction workflow for warning audits
- dormant/historical-test policy
- executable opt-in test semantics
- deeper rationale that is too detailed for operator summaries

#### What they should not own

- routine command-map duplication
- workflow-specific enforcement listings already present in CI or README

#### Examples

- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
- `tests/test_framework.h`

### 5. User/Operator Summaries

#### Stable owner

`README.md`

#### What it should own

- concise top-level command map
- high-level interpretation of reviewed local baseline
- high-level dead-code meaning
- concise cross-platform enforced/staged/excluded summary
- compact readiness checklist
- short maintainer signposts to deeper authority docs

#### What it should not own

- full machine-behavior detail already encoded in scripts
- the entire warning-authority procedure
- duplicated CI-job implementation detail
- long repeated rerun guidance better printed by the commands themselves

### 6. API / Lifecycle Semantics

#### Stable owners

- installed headers
- focused API docs/tutorial/example docs

#### What they should own

- matrix/factor eligibility requirements
- mutation semantics
- cancellation semantics where part of public API behavior
- option-field and object-lifecycle meaning

#### What they should not own

- quality-policy prose
- CI matrix summaries
- general maintainer workflow policy

## Ownership Map by Truth Type

| Truth Type | Primary Owner | Secondary / Reference Owners | Later Epic 4 Cleanup Direction |
|---|---|---|---|
| Command names and wrapper composition | `Makefile` | `README.md` command map | Keep execution truth in `Makefile`; keep README concise |
| Dead-code execution behavior | scripts + low-level `Makefile` rules | README dead-code section | Reduce repeated explanation; keep runtime text concise |
| CI platform enforcement truth | workflow YAML | README cross-platform contract | README should summarize, not restate every workflow step |
| Repository-wide warning authority | Sprint 30 docs | README maintainer standards | Keep deep procedure in Sprint 30 docs; README only signposts |
| Opt-in test semantics | `tests/test_framework.h` | README maintainer standards | Preserve header as executable truth |
| Lifecycle/API semantics | headers/tutorial/examples | README high-level references | Keep policy out of these surfaces |
| Readiness interpretation | README checklist | `Makefile` wrapper output | Keep README criterion-based, wrappers action-based |

## Strongest Current Duplication Hotspots

### 1. README command/contract density

`README.md` currently carries several overlapping layers:

- top-level command map
- dead-code explanation
- reviewed-wrapper explanation
- cross-platform CI contract
- readiness checklist
- maintainer standards
- rerun guidance

Interpretation:

- this is not wrong, but it is the strongest concentration hotspot
- later Epic 4 simplification should keep README as the operator-facing summary
  layer and avoid expanding it further

### 2. README vs Makefile wrapper messaging

The wrapper targets in `Makefile` already print:

- phase banners
- rerun commands
- reset guidance

The README also documents much of the same reviewed-path interpretation.

Interpretation:

- `Makefile` should own action-oriented rerun behavior at runtime
- README should summarize the command meaning, not restate every rerun path

### 3. README vs workflow comments

Workflow file headers and step names still repeat part of the cross-platform
contract that README also summarizes.

Interpretation:

- workflows should keep concise job-local framing
- README should remain the human summary for platform enforcement/staged status
- neither should become the other’s full duplicate

### 4. README vs Sprint 30 authority docs

README now correctly points to the warning authority docs, but some of the
warning-authority explanation still appears in both places.

Interpretation:

- Sprint 30 docs should remain the authority for warning-proof procedure
- README should only keep the short pointer plus the minimum readiness meaning

### 5. Dead-code semantics across Makefile, script output, and README

The current dead-code closeout state is described in:

- `scripts/deadcode_report.py`
- `Makefile` runtime output
- README dead-code section

Interpretation:

- this is necessary to some extent because different audiences need the
  information at different times
- but later cleanup should keep the exact invariant meaning short and
  consistent across all three surfaces

## Stable Ownership Decisions for Later Sprints

1. `Makefile` is the authoritative home for command existence, wrapper
   sequencing, and rerun entry points.
2. Scripts are the authoritative home for actual dead-code and warning-workflow
   behavior.
3. Workflow YAML is the authoritative home for what each platform CI path
   actually enforces.
4. Sprint 30 authority docs remain the authoritative home for repository-wide
   warning-proof procedure.
5. `tests/test_framework.h` remains the executable truth for skip/slow/
   experimental test semantics.
6. `README.md` should remain concise and summary-oriented:
   - command map
   - contract summary
   - signposts
   - readiness checklist
7. Headers/tutorial/examples remain the home of API/lifecycle semantics, not
   quality-policy truth.

## Day 11 Output for Later Sprints

Later Epic 4 cleanup work should use this ownership model when deciding whether
to delete, compress, move, or preserve text:

- move deep procedure to authority docs, not to README
- keep execution behavior in code/scripts, not duplicated as long prose
- keep CI truth in workflow YAML, with README only summarizing platform status
- keep API/lifecycle semantics in headers/docs/examples, not mixed with quality
  policy

That gives later simplification work a stable “where should this truth live?”
answer before any text is removed.
