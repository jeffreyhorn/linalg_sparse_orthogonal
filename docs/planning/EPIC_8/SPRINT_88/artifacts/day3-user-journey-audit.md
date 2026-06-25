# Sprint 88 Day 3: User-Journey Audit

## Purpose

Reduce Sprint 88's broad front-door usability problem to one ranked live
contradiction map so the sprint can choose one bounded adoption-guidance lane
instead of another generic docs or example bucket.

## Main Result

Sprint 88's broad front-door usability problem is now reduced to one ranked
live contradiction map:

- strongest first target:
  - bounded front-door simplification centered on `README.md`, with direct
    follow-through only where the first user path currently leaks too much
    install, benchmark, or maintainer density
- strongest second target:
  - bounded examples / workflow simplification centered on the example
    references in `README.md` plus the maintained downstream example surface
    in `examples/cmake_example/`
- strongest third target:
  - bounded support-surface consolidation centered on `INSTALL.md`,
    `benchmarks/README.md`, and `docs/maintainer_guide.md` after the
    front-door contract is explicit
- strongest fourth target:
  - bounded header / API narrative cleanup centered on the highest-signal
    public headers:
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_matrix.h`
    - `include/sparse_types.h`
- strongest support-only but real target:
  - workflow and proof-surface wording only where a landed usability batch
    truly changes how users should interpret those surfaces

## Strongest Current Contradiction

The strongest current contradiction is now explicit:

- `README.md` already contains a real user entry path through:
  - `Choose a Workflow`
  - `Quick Start`
  - repeated-run workflow guidance
  - installation references
- but the same file still carries advanced benchmark, dead-code, maintainer,
  and support references deep into the front-door reading path
- the result is a truthful but over-dense front door: first-adoption
  decisions, advanced workflow interpretation, and maintainer-facing
  references still coexist too closely

That makes the strongest first Sprint 88 move clear:

- do not jump straight to generic support-doc cleanup
- first define the exact front-door user path the repo wants to teach
- then make README and its direct support references match that contract
  cleanly

## Second-Tier Contradictions

### Example / Workflow Adoption Asymmetry

The strongest second contradiction is examples/workflow asymmetry:

- the example surfaces are real and maintained
- the downstream CMake example is minimal and bounded
- but README still has to do too much work explaining how to move from
  one-shot examples to repeated-run, benchmark, and maintained proof lanes

This makes examples/workflow simplification real Sprint 88 work, but it still
reads as second after the front-door contract is explicit.

### Support-Surface Audience Blur

The strongest third contradiction is support-surface audience blur:

- `INSTALL.md` already says README is the canonical front door
- `benchmarks/README.md` already self-limits toward benchmark ownership
- `docs/maintainer_guide.md` is already maintainer-facing
- but the audience boundaries among these surfaces are still not quite sharp
  enough, so README still carries more support-routing burden than it should

This means support-surface consolidation is real Sprint 88 work, but it
remains bounded and must stay behind a truthful front-door contract.

### Public Narrative Spillover

The strongest fourth contradiction is public-narrative spillover:

- the highest-signal public headers remain large and valuable
- but they still read with more internal workflow/policy context than an
  adoption-focused public narrative ideally needs

This keeps header/API narrative cleanup real Sprint 88 work, but it remains
ordered behind the first front-door and example lanes.

## Deferred Claims

Broad support and product rewriting remains lower-value first work:

- no package/platform contract reopening
- no correctness-ownership redistribution
- no benchmark-policy rewrite detached from adoption guidance
- no internal architectural rewrite disguised as usability cleanup
- no workflow/platform claim broadening beyond the already-maintained proof
  and support surfaces

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 88 move is not generic "improve docs"
- it is one bounded front-door simplification pass on the README-level user
  decision path
- examples/workflow simplification follows next where the README contract
  exposes a real maintained adoption gap
- support-surface consolidation comes after that where audience boundaries
  need sharpening
- public header narrative cleanup remains real, but later than the first
  adoption-flow lanes

The Sprint 87 carry-forward reading is now fixed:

- Sprint 87 already stabilized the static-first package/platform contract
- Sprint 87 already strengthened local consumer proof
- Sprint 88 therefore begins with adoption-friction truthfulness rather than
  another package or workflow-proof lane

## Exit State

- Sprint 88 now has one ranked live front-door usability contradiction map
  grounded in the current tree and maintained support surfaces.
- The first implementation center is fixed to bounded README/front-door
  simplification, not immediate support-surface or header cleanup.
- Later examples/workflow simplification, support-surface consolidation, and
  public narrative cleanup are explicitly ordered behind that first lane.
