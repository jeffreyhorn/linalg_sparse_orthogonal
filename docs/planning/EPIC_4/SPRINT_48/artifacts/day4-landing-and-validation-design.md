# Sprint 48 Day 4: Landing and Validation Design

## Objective

Bound the Sprint 48 documentation redistribution batches and define the focused
validation contract before README reduction, maintainer-guide implementation,
and later quality-contract simplification begin.

## Commands Run

1. Re-read the Sprint 48 Day 4 plan section:
   - `sed -n '120,190p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 3 maintainer-guide design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day3-maintainer-guide-design.md`
3. Re-read the current Sprint 48 working-notes tail:
   - `tail -n 220 docs/planning/EPIC_4/SPRINT_48/WORKING_NOTES.md`
4. Reconfirm the live maintained target names in `Makefile`:
   - `rg -n "^(quality-review-full|tooling-build|deadcode|deadcode-report|deadcode-check|format|lint|test):" Makefile`
   - `sed -n '120,230p' Makefile`

## Design

#### 1. Sprint 48 needs three different validation shapes, not one generic docs rule

The live Sprint 48 scope spans three distinct edit classes:

- docs-only redistribution
- script and command-surface clarification
- possible C or header touch points if a later reconciliation unexpectedly
  reaches a compiled surface

Design decision:

- Day 4 should define separate validation expectations for each class instead
  of treating all non-core work the same

Interpretation:

- this keeps docs-only days lightweight
- it still preserves the full required gate whenever compiled source changes

#### 2. Docs-only days should use targeted sanity checks, not the full code gate by default

Docs-only work in Sprint 48 is expected on:

- `README.md`
- `docs/maintainer_guide.md`
- `docs/tutorial.md`
- benchmark/example READMEs
- sprint artifacts and notes

Design decision:

- docs-only days should validate with targeted sanity checks:
  - link/reference sanity for touched docs
  - local path accuracy
  - command-name accuracy against the live `Makefile`
  - any direct spot-check command needed to confirm described behavior

Interpretation:

- Sprint 48 should not rerun `make format`, `make lint`, and `make test` after
  every prose-only move
- the goal is still truthfulness, just with proportionate validation

#### 3. Script and command-surface days should use focused validation against the touched executable truth

Sprint 48 may touch:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`
- README/help text describing maintained quality commands

Design decision:

- touched script/command-surface days should use focused validation tied to the
  edited surface, such as:
  - `python3 -m py_compile scripts/deadcode_report.py`
  - `bash -n scripts/deadcode_workflow.sh`
  - targeted synthetic input/output checks for the touched failure or reporting
    path
  - direct command spot checks like:
    - `make -n quality-review-full`
    - `make -n deadcode-report`
    - `make -n deadcode-check`

Interpretation:

- the executable truth for these surfaces lives in scripts and the `Makefile`
- Sprint 48 should validate those surfaces directly when it changes the prose
  or behavior around them

#### 4. Any `*.c` or `*.h` change still triggers the full required gate

This sprint is expected to be mostly docs and scripts, but the validation rule
must still stay explicit.

Design decision:

- any Sprint 48 change touching `*.c` or `*.h` must still run:
  - `make format`
  - `make lint`
  - `make test`

Interpretation:

- Sprint 48 does not weaken the standing compiled-surface contract
- docs cleanup cannot be used as cover for softer validation on code changes

#### 5. The stronger reviewed baseline should be reserved for high-signal contract days

The repo already has a stronger reviewed baseline:

- `make quality-review-full`

That command is more expensive and should stay tied to days where Sprint 48 is
really testing the quality-contract story itself.

Design decision:

- rerun `make quality-review-full` on:
  - quality-contract simplification days
  - the final validation sweep
- do not require it for every docs-only redistribution day

Interpretation:

- Sprint 48 should preserve Sprint 40 truthfulness without paying the full
  reviewed-baseline cost on every prose move

#### 6. `make tooling-build` is the right compile-only follow-on for touched public auxiliary surfaces

The maintained auxiliary compile-only target is:

- `make tooling-build`

That target compiles:

- benchmarks
- examples

without turning the day into a full end-to-end runtime sweep.

Design decision:

- use `make tooling-build` when Sprint 48 touches:
  - benchmark docs tightly coupled to buildable benchmark entry points
  - example docs or example source files
  - auxiliary compile-only public surfaces

Interpretation:

- Sprint 48 already has a maintained auxiliary build sanity target
- it should use that instead of inventing ad hoc compile-only coverage

#### 7. The intended landing order is now fixed as five bounded batches

With Day 3's policy-home target fixed, the implementation order should be:

1. README reduction
2. maintainer-guide implementation
3. tutorial/header cross-reference reconciliation
4. quality-contract ownership simplification
5. docs sanity sweep

Interpretation:

- README reduction should happen before broader reconciliation so the
  user-facing scope is visible early
- the guide should land before tutorial/header cleanup so references have a
  stable target
- quality-contract simplification should follow once the prose homes are
  already clearer

#### 8. The quality-contract simplification batch should stay about ownership, not command redesign

Sprint 48 Day 2 already established that the quality contract has three
authority shapes:

- executable authority
- CI/staged authority
- prose authority

Design decision:

- later quality-contract simplification should focus on:
  - who explains what
  - where maintainers should read policy
  - which command surface is authoritative for which claim
- it should not focus on:
  - redefining `quality-review-full`
  - redesigning `deadcode` workflow semantics
  - broad CI YAML changes

Interpretation:

- Sprint 48 should simplify ownership first
- behavior changes are out of scope unless a narrow safety issue forces one

#### 9. Explicit out-of-scope items remain necessary to keep later days honest

Day 4 should freeze these out-of-scope items before redistribution begins:

- broad CI redesign
- dead-code workflow redesign
- broad tutorial rewrite
- large benchmark/example content expansion
- public API behavior changes via docs cleanup
- replacing local executable truth with prose summaries

Interpretation:

- later days should stay bounded even if touching README, scripts, and guide
  surfaces reveals more possible cleanup

## Bottom Line

Sprint 48 Day 4 fixes the execution contract for the rest of the sprint:

- docs-only days:
  - targeted sanity checks only
- script/command-surface days:
  - focused validation against the touched executable truth
- any `*.c` / `*.h` day:
  - `make format`
  - `make lint`
  - `make test`
- stronger reviewed baseline:
  - `make quality-review-full` on quality-contract days and final sweep
- auxiliary compile-only follow-on when justified:
  - `make tooling-build`

It also fixes the implementation order:

1. README reduction
2. maintainer-guide implementation
3. tutorial/header reconciliation
4. quality-contract ownership simplification
5. docs sanity sweep

That is the right Day 4 state before README reduction begins.
