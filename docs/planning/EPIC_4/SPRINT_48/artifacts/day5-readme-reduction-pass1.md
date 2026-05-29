# Sprint 48 Day 5: README Reduction Pass I

## Objective

Land the first bounded README reduction pass so the main project entry point
stays strong for users and operators while materially shrinking the embedded
maintainer-policy duplication around quality, dead-code, and readiness
guidance.

## Commands Run

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

## Changes

#### 1. Kept the user/operator-facing command surfaces visible

The README still keeps the high-signal operator entry points:

- direct build/test commands
- reviewed local wrappers
- dead-code commands
- the cross-platform CI contract table
- concise readiness checks

Interpretation:

- Day 5 did not turn README into a shallow landing page
- it kept the commands a user or operator actually needs close at hand

#### 2. Compressed the dead-code explanation into a user-facing summary

The README still explains:

- what `make deadcode`, `make deadcode-report`, and `make deadcode-check` do
- that `deadcode-check` is a completeness gate rather than a zero-findings
  claim
- that the dead-code path remains serialized

Removed from README:

- oversized interpretation detail better suited to maintainer policy
- low-level bucket commentary that does not help a first operator run the
  commands

Interpretation:

- README now keeps the operator contract without carrying the full maintainer
  interpretation burden

#### 3. Compressed the reviewed-quality wrapper section without hiding the contract

The README still identifies:

- reviewed Makefile path
- strongest local reviewed baseline
- reviewed CMake parity path
- additive relationship between the wrappers and direct commands

Removed from README:

- long step-by-step command expansion for each wrapper
- repeated explanations already evident from the command names and `Makefile`

Interpretation:

- the user-facing quality map remains visible
- the maintainer-policy density is materially lower

#### 4. Kept the CI truth table but removed extra prose drift around it

The cross-platform CI contract table remains because it is a useful user and
operator truth surface.

The surrounding prose is now tighter:

- the table remains the main truth source
- the long restatement of each platform’s meaning is gone

Interpretation:

- README still names the enforced/staged/supplemental boundaries honestly
- it no longer repeats the same policy in multiple paragraphs

#### 5. Reduced the readiness and maintainer sections to concise boundary markers

The README now keeps:

- a short readiness checklist
- a short maintainer-reference section pointing to the Sprint 30 authoritative
  warning/rebuild docs
- a short note on designated initializers, historical evidence, and test
  truth
- the `make clean` return-to-normal-path reminder for tree-mutating modes

Removed from README:

- the large embedded maintainer standards block
- the long per-wrapper rerun command dump

Interpretation:

- Day 5 did not fully relocate maintainer policy yet
- it did reduce README duplication while preserving essential operator clarity

## Bottom Line

Sprint 48 Day 5 makes `README.md` smaller and more user-facing without waiting
for the Day 6 maintainer guide:

- kept:
  - direct operator command map
  - CI truth table
  - readiness checklist
  - concise maintainer references
- reduced:
  - oversized maintainer-policy prose
  - repeated wrapper-expansion details
  - repeated interpretation already better owned by future guide material

That is the right first README reduction pass before the maintainer-guide batch
lands.
