# Sprint 48 Day 10: Quality-Contract Simplification Batch

## Objective

Land the bounded README/maintainer-guide ownership simplification identified
on Day 9 so `README.md` stays the concise operator map, the maintainer guide
owns repository-wide interpretation, and executable command details remain
local to `Makefile` and the dead-code supporting surfaces.

## Commands Run

1. Re-read the Sprint 48 Day 10 plan section:
   - `sed -n '379,452p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 9 ownership audit:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day9-quality-contract-ownership-audit.md`
3. Re-read the current quality-contract wording in the main doc surfaces:
   - `sed -n '690,790p' README.md`
   - `sed -n '40,170p' docs/maintainer_guide.md`
4. Reconfirm the maintained command-detail home before editing:
   - `sed -n '500,640p' Makefile`
5. Run targeted Day 10 sanity checks after editing:
   - `rg -n "Maintainer Guide|quality-review-full|deadcode-check|Cross-Platform CI Contract|quality-review-cmake" README.md docs/maintainer_guide.md`
   - `wc -l README.md docs/maintainer_guide.md`
   - `make -n quality-review-full deadcode-report deadcode-check`

## Changes

#### 1. Tightened the README quality tail so it behaves more clearly as an operator map

`README.md` now:

- keeps the reviewed wrapper command map
- keeps the cross-platform quality table
- keeps the readiness checklist
- points readers back to `make <target>` for wrapper expansion and rerun detail
- points readers back to `docs/maintainer_guide.md` for repository-wide
  interpretation

Interpretation:

- the README stayed useful for operators
- it now repeats less maintainer-policy prose

#### 2. Made the command-detail boundary explicit in the maintainer guide

`docs/maintainer_guide.md` now states directly that:

- wrapper expansion
- rerun guidance
- build-tree paths
- dead-code execution detail

stay with the executable surfaces:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

while the guide owns repository-wide interpretation of those surfaces.

Interpretation:

- the guide is now a clearer policy layer
- it did not expand into a shadow CLI manual

#### 3. Reduced the remaining repeated quality-contract claims to a cleaner ownership split

The three repeated Day 9 claims were tightened:

- strongest/default local reviewed closeout naming
- `deadcode-check` meaning
- cross-platform enforced/staged interpretation

The resulting ownership split is now:

- command semantics and emitted detail:
  - `Makefile`
- compact operator map:
  - `README.md`
- repository-wide interpretation:
  - `docs/maintainer_guide.md`

Interpretation:

- Day 10 simplified ownership without changing behavior
- fewer future wording edits need to be mirrored across multiple prose homes

## Bottom Line

Sprint 48 Day 10 completed the bounded quality-contract simplification exactly
where Day 9 said it should:

- touched:
  - `README.md`
  - `docs/maintainer_guide.md`
- preserved:
  - `Makefile` and script authority for executable command detail
- tightened:
  - README operator framing
  - maintainer-guide policy framing
- avoided:
  - command redesign
  - workflow/script churn

That is the right ownership state before the Day 11 docs sanity sweep.
