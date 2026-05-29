# Sprint 48 Day 2: Documentation and Quality-Contract Surface Inventory

## Objective

Refresh the live documentation and quality-contract seam inventory so Sprint
48's maintainer-guide design, README reduction, tutorial/header
cross-reference pass, and quality-contract simplification batches are
sequenced from the actual post-Sprint-47 repo state rather than only from the
project-plan labels.

## Commands Run

1. Re-read the Sprint 48 Day 2 plan section:
   - `sed -n '55,108p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day1-scope-and-quality-contract-baseline.md`
3. Re-read the current high-density README quality/maintainer sections:
   - `sed -n '640,845p' README.md`
4. Refresh the live quality-command ownership markers:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode-check|tooling-build" Makefile README.md docs/tutorial.md scripts/deadcode_report.py scripts/deadcode_workflow.sh .github -g '!build'`
5. Refresh the broader duplication markers across docs, headers, scripts, and
   workflows:
   - `rg -n "lifecycle|factored|dead-code|quality-review|maintainer|reviewed baseline|designated initializer|designated-initializer|README|tutorial" README.md docs/tutorial.md include scripts Makefile .github -g '!build'`
6. Reconfirm the main hotspot sizes:
   - `wc -l README.md benchmarks/README.md examples/README.md docs/tutorial.md Makefile .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml`
   - `wc -l README.md docs/tutorial.md include/sparse_matrix.h include/sparse_lu.h include/sparse_cholesky.h Makefile scripts/deadcode_report.py scripts/deadcode_workflow.sh`

## Findings

#### 1. The live Sprint 48 surface reduces cleanly to five seam classes

The current duplication problem is not one generic docs backlog. It now
reduces to five bounded seam classes:

- quality-command ownership drift
- README user-vs-maintainer scope drift
- tutorial/header behavioral-caveat duplication
- lifecycle/cancellation caveat duplication
- maintainer norms with no stable policy home

Interpretation:

- Sprint 48 should work from explicit ownership seams
- it should not treat README, headers, tutorial, workflows, and helper scripts
  as one undifferentiated rewrite surface

#### 2. The strongest direct implementation target is still the README quality-policy block

Day 2 confirms that the densest concentration of maintainer-policy duplication
still sits in `README.md`, especially around:

- dead-code workflow
- reviewed local quality path
- cross-platform CI contract
- quality readiness checklist
- maintainer standards

That region overlaps directly with:

- `Makefile` quality wrapper/help text
- workflow comments under `.github/workflows/`
- dead-code support-script behavior

Interpretation:

- the README reduction pass should be the first real redistribution target
- Sprint 48 should not delay that pass until after every other doc is touched

#### 3. The quality-command contract is effective but has three different “authority” layers

The current quality contract splits into three different authority shapes:

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

- Sprint 48 needs a clearer prose home for maintainer policy
- command semantics themselves are not the main problem
- the main problem is that prose authority is too distributed and too often
  repeats command ownership already visible elsewhere

#### 4. Tutorial and public headers should keep behavioral caveats, but not own full maintainer policy

The live Day 2 inventory shows recurring caveats in:

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
- Sprint 48 should move repeated maintainer-policy explanations out of those
  surfaces, not strip away API-relevant caveats

#### 5. Maintainer norms currently lack a stable policy home

Day 2 confirms that several maintainer-facing expectations currently live
inside README rather than a maintainer-facing guide:

- reviewed baseline use
- dead-code interpretation
- cross-platform contract reading
- designated-initializer norms
- what counts as dormant historical evidence versus live suite truth

Interpretation:

- the strongest “move to maintainer guide” candidates are now explicit
- Sprint 48 does not need to invent new policy so much as relocate and tighten
  the policy that already exists

#### 6. The first implementation order is now fixed from live ownership rather than only from plan labels

The correct order after Day 2 is:

1. maintainer-guide design
2. README reduction
3. maintainer-guide implementation
4. tutorial/header cross-reference reconciliation
5. quality-contract ownership simplification

Interpretation:

- the policy home must be designed before broad redistribution starts
- README reduction should happen before tutorial/header cleanup so the user-vs-
  maintainer boundary is already visible
- quality-contract simplification belongs later, after the prose homes are
  clearer

#### 7. The strongest “leave local” content is also explicit

Sprint 48 should avoid moving or over-centralizing:

- direct command behavior owned by executable surfaces
- concise API-local caveats in headers
- tutorial flow that still teaches user-facing matrix-state expectations
- local benchmark/example usage details already owned by their own READMEs

Interpretation:

- the sprint should simplify ownership, not centralize everything into one huge
  maintainer document

## Bottom Line

Sprint 48 Day 2 reduces the live duplication problem to a bounded ownership map:

- primary direct target:
  - README quality/maintainer-policy reduction
- policy-home target:
  - new maintainer-facing guide
- later cross-reference target:
  - tutorial/header lifecycle and behavior caveat reconciliation
- later command-surface target:
  - quality-contract ownership simplification

That is the right Day 2 state before maintainer-guide design begins.
