# Sprint 48 Day 9: Quality-Contract Ownership Audit

## Objective

Audit the live Makefile, script, README, guide, and workflow wording around
the quality-command surface so Sprint 48 can separate the remaining ownership
simplification work from command behavior that should stay local and unchanged.

## Commands Run

1. Re-read the Sprint 48 Day 9 plan section:
   - `sed -n '330,410p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 8 cross-reference artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day8-tutorial-and-header-cross-reference-batch.md`
3. Refresh the quality-command ownership markers across live surfaces:
   - `rg -n "quality-review-full|quality-review-cmake|quality-review-compile|deadcode-report|deadcode-check|warning-workflow|tooling-build" README.md docs/maintainer_guide.md docs/tutorial.md Makefile scripts/deadcode_report.py scripts/deadcode_workflow.sh .github/workflows -g '!build'`
4. Re-read the maintained quality target definitions in `Makefile`:
   - `sed -n '540,670p' Makefile`
5. Re-read the current README quality-contract tail:
   - `sed -n '700,790p' README.md`
6. Re-read the maintainer-guide quality-contract sections:
   - `sed -n '40,150p' docs/maintainer_guide.md`

## Findings

#### 1. The quality-contract now has clearer document homes, but the wording still repeats the same three ownership claims

After Day 6 and Day 8, the live quality contract already has better homes:

- command truth:
  - `Makefile`
- maintainer-policy interpretation:
  - `docs/maintainer_guide.md`
- concise operator-facing reminders:
  - `README.md`

What still repeats across surfaces is mostly three claims:

- which local command is the strongest reviewed baseline
- what `deadcode-check` actually means
- where cross-platform enforced/staged truth should be read

Interpretation:

- Day 10 does not need another broad redistribution pass
- it needs a small ownership-tightening pass around those repeated claims

#### 2. `Makefile` should remain the authoritative home for rerun guidance and wrapper expansion details

The maintained `Makefile` target help still carries the richest live command
surface for:

- rerun-failing-phase guidance
- reviewed wrapper composition
- parity-path details
- dead-code completeness wording

Interpretation:

- Day 10 should not copy more of this detail into docs
- instead, docs should point to the `Makefile`-owned command surface when that
  detail is needed

#### 3. `README.md` is still slightly over-describing the reviewed-quality surface for a user/operator entry point

The current README quality tail is already much smaller than Day 1, but it
still repeats several maintainability-oriented interpretations:

- strongest reviewed baseline naming
- reviewed CMake truthfulness framing
- dead-code completeness framing
- staged/enforced boundary framing

These statements are not wrong, but some are closer to maintainer-policy
interpretation than to user/operator quick-reference.

Interpretation:

- Day 10 should keep the concise operator checklist
- it should tighten the remaining phrasing so README points to the guide or the
  table rather than restating policy where not needed

#### 4. The maintainer guide is the right home for meaning, but not for every command variant detail

`docs/maintainer_guide.md` now owns the policy interpretation correctly:

- strongest local reviewed baseline meaning
- warning authority
- dead-code meaning
- documentation ownership

But it should not become a second command reference for:

- every rerun variant
- every wrapper echo message
- every build-tree path detail already emitted by `Makefile`

Interpretation:

- Day 10 should keep the guide interpretive
- it should not broaden the guide into a shadow CLI manual

#### 5. The dead-code scripts themselves are not the right Day 10 edit target unless wording truth forces it

The live dead-code scripts currently appear only as executable/support surfaces
plus a small reporting note in `deadcode_report.py`.

Interpretation:

- the main Day 10 target should be doc ownership, not script behavior
- touching scripts would only be justified if documentation truth currently
  mismatches the emitted script-side wording

#### 6. The cross-platform CI table should stay in README, but its policy interpretation should stay minimal there

The table in `README.md` is still valuable because it gives a compact user and
operator map of:

- enforced
- staged
- supplemental/excluded

What should not keep growing around it is policy interpretation that the
maintainer guide now owns better.

Interpretation:

- Day 10 should keep the table
- it should trim or tighten any remaining prose that restates what the guide
  already explains

#### 7. The bounded Day 10 target set is now explicit

The highest-value Day 10 simplification targets are:

- `README.md`
  - tighten the remaining quality-tail phrasing so README stays operator-facing
- `docs/maintainer_guide.md`
  - clarify that command-semantics detail lives in `Makefile`, while the guide
    owns interpretation
- possibly one small `Makefile` comment/help wording adjustment only if needed
  to keep the documentation ownership story honest

Interpretation:

- Day 10 should be mostly README + guide
- it should not become a workflow or script redesign batch

#### 8. Content that should stay local is now explicit

The following should stay local to their current surfaces:

- rerun-failing-phase details:
  - `Makefile`
- dead-code report generation and check semantics as emitted by the targets:
  - `Makefile`
  - dead-code scripts
- enforced/staged platform execution:
  - CI workflows
  - the compact README table

Interpretation:

- Day 10 should simplify ownership by reducing repeated prose
- not by relocating executable detail into prose again

## Bottom Line

Sprint 48 Day 9 makes the remaining quality-contract queue concrete:

- Day 10 primary targets:
  - `README.md`
  - `docs/maintainer_guide.md`
- possible secondary touch only if truly needed:
  - one small `Makefile` wording adjustment
- keep local:
  - rerun details in `Makefile`
  - dead-code execution semantics in `Makefile` and scripts
  - platform execution truth in workflows plus the compact README table

That is the right audit state before the Day 10 ownership-simplification
batch.
