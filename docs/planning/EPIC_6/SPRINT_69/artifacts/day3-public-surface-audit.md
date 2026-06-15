# Sprint 69 Day 3: Public Surface Audit

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Reduce Sprint 69's broad "public product surface finalization" scope to a
ranked live seam map before any final simplification or compatibility work
lands.

## Audit Inputs

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- targeted terminology scans for workflow / ownership / benchmark / platform /
  install / proof wording
- current touched-surface measurements from the live tree

## Day 3 Audit Conclusions

### 1. The main residual pressure is duplicated product-story ownership, not absent explanation

The current repo already explains the final Epic 6 state in enough places.
The strongest remaining problem is that the same product-story lane still
appears across too many maintained surfaces:

- workflow choice
- repeated-run adoption
- examples vs benchmarks vs tests ownership
- canonical benchmark/report interpretation
- install/platform confidence limits
- maintainer-policy boundaries

So Sprint 69 should optimize for simplification and sharper authority splits,
not for adding more explanatory mass.

### 2. README is the strongest first target

`README.md` is the strongest live hotspot because it combines the densest mix
of:

- top-level product narrative
- workflow-choice guidance
- examples/benchmarks/tests ownership summary
- canonical benchmark/reporting summary
- platform-quality summary
- install/package summary

This makes it the best first landing: a bounded README simplification can
remove duplicated explanation pressure from several adjacent surfaces at once.

### 3. Tutorial is the strongest second target

`docs/tutorial.md` is the strongest second hotspot because it still repeats
some product-story framing that now overlaps with:

- README workflow-choice guidance
- example-analysis adoption handoff
- examples vs benchmarks interpretation
- repeated-run direct-path explanation

It remains valuable, but its biggest residual risk is overlap rather than
missing usage content.

### 4. The maintainer guide is already the right policy home, so it is support-first rather than first-landing by volume

`docs/maintainer_guide.md` is large and important, but it is already the best
home for:

- documentation ownership interpretation
- benchmark-governance interpretation
- packaging/platform residual interpretation
- proof-ownership policy

That means the first Sprint 69 win is not shrinking the maintainer guide in
isolation. It is reducing what README/tutorial/examples/benchmarks still need
to say because the policy home already exists.

### 5. Public headers are real final product surfaces, but only one is a strong first support candidate

The live header ranking is:

- strongest header-side support candidate:
  - `include/sparse_cholesky.h`
- real but lower-priority support headers:
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

`include/sparse_cholesky.h` stands out because it carries the densest
public-path explanation tied to the final Epic 6 product story:

- transparent CSC dispatch
- benchmark/test reference notes
- backend-contract error semantics
- one-shot vs repeated-run lifecycle interpretation

The other large headers are substantial, but their remaining pressure is more
reference breadth than first-order public-story contradiction.

### 6. Examples and benchmark READMEs are important support surfaces, but weaker first design centers

`examples/README.md` and `benchmarks/README.md` are already relatively sharp:

- examples README stays local to adoption entry points
- benchmarks README stays local to benchmark categories, schemas, and
  ownership limits

Their main remaining risk is support-side drift relative to README/tutorial,
not that they need broad redesign first.

## Ranked Sprint 69 Surface Map

### Strongest first target

- `README.md`

### Strongest second target

- `docs/tutorial.md`

### Strongest policy/support surface

- `docs/maintainer_guide.md`

### Strongest header-side support candidate

- `include/sparse_cholesky.h`

### Important support surfaces, but weaker first design centers

- `examples/README.md`
- `benchmarks/README.md`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

## Day 3 Exit State

Sprint 69’s broad public-surface finalization story is now reduced to a real
ranked seam map. The next step is to turn that map into one explicit first
landing boundary instead of a generic docs/header shortlist.
