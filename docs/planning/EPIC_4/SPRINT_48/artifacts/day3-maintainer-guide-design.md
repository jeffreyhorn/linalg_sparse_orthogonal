# Sprint 48 Day 3: Maintainer-Guide Design

## Objective

Define the new maintainer-facing policy home for Sprint 48 so README
reduction, tutorial/header reconciliation, and later quality-contract
simplification land against one explicit ownership target instead of continuing
to spread maintainer policy across user-facing docs.

## Commands Run

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

## Design

#### 1. Sprint 48 should add one stable maintainer-policy home under `docs/`

The repo already has good maintainer-facing policy examples, but they live in
sprint artifact space:

- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`

Those files are useful precedent, but they are not the right permanent home
for active repo-wide maintainer policy.

Design decision:

- create one stable maintainer-facing policy document at:
  - `docs/maintainer_guide.md`

Rationale:

- it matches the current top-level docs layout better than a new doc subtree
- it gives Sprint 48 one clear policy entry point
- it avoids turning `README.md` back into the maintainer-policy dump
- it keeps active policy out of sprint-history artifacts without erasing the
  value of those artifacts as evidence

#### 2. The guide should be one main document, not a broad guide cluster

Sprint 48 does not need a large docs-site redesign or a multi-document
maintainer handbook. The current duplication is concentrated enough that one
main guide is the right first landing.

Design decision:

- Day 6 should create one main guide, not a broad guide cluster
- sub-guides should only exist later if a stable section genuinely outgrows
  the main guide

Interpretation:

- the first goal is ownership clarity
- not every maintainer topic needs its own file

#### 3. The intended audience is maintainers and high-context contributors, not end users

The new guide should serve people who need to interpret repository policy,
quality expectations, and documentation ownership rules while changing the
tree.

Primary audience:

- maintainers
- high-context contributors working on repo-wide cleanup
- reviewers evaluating quality-contract claims

Explicit non-audience:

- first-time library users looking for quick-start instructions
- API consumers trying to learn one solver entry point
- benchmark/example users looking for command syntax

Interpretation:

- `README.md` remains the user/operator entry point
- the guide becomes the policy and ownership reference point for maintainers

#### 4. The guide should own six policy classes directly

Sprint 48 Day 2 already identified the strongest maintainer-policy candidates.
Those should move into the new guide as first-class sections.

The guide should directly own:

- reviewed baseline use
- warning authority and how to interpret supported build surfaces
- dead-code meaning and how to read `deadcode-report` / `deadcode-check`
- lifecycle/cancellation expectations that matter as maintainer policy rather
  than user API wording
- documentation ownership rules
- designated-initializer / evolving-option-struct norms where still relevant

Interpretation:

- Sprint 48 is not inventing new policy here
- it is giving existing policy one stable home and one stable voice

#### 5. The guide should explain policy, but executable truth stays local

The guide must not replace the authoritative executable surfaces.

Executable truth stays with:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- CI workflow files under `.github/workflows/`

Guide responsibilities:

- explain what those surfaces are for
- explain when a maintainer should use them
- explain how to interpret their outputs
- explain which surface is authoritative for which claim

Guide non-responsibilities:

- duplicating command implementations
- becoming a second CLI reference for every wrapper target
- restating full workflow YAML behavior in prose

Interpretation:

- the guide is a policy and ownership home, not an executable reference dump

#### 6. Four content classes should explicitly remain outside the guide

The guide should not absorb content that already has a better local home.

Content that should stay outside:

- end-user quick-start and high-level project entry material:
  - stays in `README.md`
- concise API-reference and call-site caveats:
  - stay in public headers
- tutorial teaching flow and user-facing matrix-state guidance:
  - stays in `docs/tutorial.md`
- benchmark/example command usage details:
  - stay in `benchmarks/README.md`
  - stay in `examples/README.md`

Interpretation:

- the guide should centralize maintainer policy
- it should not centralize every explanation in the repository

#### 7. Cross-reference rules should be explicit and minimal

Sprint 48 needs stable cross-reference rules so later edits reduce duplication
instead of just moving it around.

Recommended cross-reference rules:

1. `README.md`
   - keeps quick-start, feature map, build/test essentials, and direct links to
     deeper docs
   - links to `docs/maintainer_guide.md` for maintainer policy instead of
     restating it
2. `docs/maintainer_guide.md`
   - owns repository-wide policy interpretation
   - links outward to the local executable or API surfaces when specific truth
     lives there
3. `docs/tutorial.md`
   - keeps user-facing behavioral guidance needed to use the library
   - links to the maintainer guide only when policy interpretation, not end-user
     workflow, is the topic
4. public headers
   - keep concise local caveats needed at call sites
   - avoid long maintainer-policy explanation blocks when the guide can carry
     that burden
5. local benchmark/example READMEs
   - keep local usage syntax and surface-specific notes
   - link to the maintainer guide only for repo-wide policy, not local usage

Interpretation:

- the repo should move toward:
  - local truth where a user needs it
  - one policy home where a maintainer needs it

#### 8. The Day 6 implementation batch should stay intentionally narrow

This design implies a bounded Day 6 implementation shape:

- create `docs/maintainer_guide.md`
- move the highest-density maintainer-policy prose out of `README.md`
- add only the minimum necessary cross-references

It should not:

- rewrite the whole tutorial
- rewrite all touched headers in the same batch
- redesign CI/workflow architecture
- create a large docs hierarchy

Interpretation:

- the guide landing should establish ownership first
- later days can then reconcile the remaining local references cleanly

## Bottom Line

Sprint 48 Day 3 fixes the maintainer-policy target before documentation
movement begins:

- stable target file:
  - `docs/maintainer_guide.md`
- audience:
  - maintainers and high-context contributors
- direct policy ownership:
  - reviewed baseline
  - warning authority
  - dead-code interpretation
  - lifecycle/cancellation maintainer expectations
  - documentation ownership
  - designated-initializer / evolving-option-struct norms
- explicit non-goals:
  - user quick-start duplication
  - API reference migration out of headers
  - benchmark/example usage migration out of local READMEs
  - broad docs-site or CI redesign

That is the right Day 3 state before README reduction and maintainer-guide
implementation begin.
