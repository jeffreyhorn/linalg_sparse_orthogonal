# Sprint 95 Retrospective

**Sprint:** 95 - Public Narrative, Docs & Workflow Coherence
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 95 started from a live public-surface inventory rather than a
      generic docs-polish list
- [x] the highest-risk narrative problems were ranked by reader impact, truth
      risk, implementation cost, and validation risk
- [x] the permanent audience and ownership model was defined before broad
      rewrite work
- [x] the README was reduced back to a concise public front door
- [x] the tutorial and examples were aligned with the cleaned adoption path
- [x] selected public headers were rewritten to describe stable API behavior
      instead of sprint provenance
- [x] install, benchmark, and maintainer support surfaces were consolidated
      around clear ownership rules
- [x] the highest-value sprint-named proof-owner cluster was renamed to
      product-oriented direct CSC names
- [x] historical planning artifacts remained historical instead of being
      rewritten as if they were product docs
- [x] the residual queue was frozen with explicit future-work boundaries
- [x] the full branch-level validation chain passed before closeout:
  - `make format`
  - `make lint`
  - `make test`
- [x] Sprint 95 closed with a bounded Sprint 96 handoff queue

## What Went Well

1. **The sprint fixed ownership before rewriting prose.**
   Day 3's audience model kept the sprint from spreading the same support,
   benchmark, and proof narrative across every public page. That made later
   cleanup smaller and easier to validate.

2. **The README cleanup removed the largest public-reader distraction.**
   The README now works as the project front door: current capability story,
   workflow choice, quick-start path, compact command map, and links to owner
   surfaces. It no longer tries to carry sprint chronology, long benchmark
   evidence, detailed install policy, and maintainer proof maps inline.

3. **The tutorial and examples now have clearer jobs.**
   The tutorial owns the longer learning path after README. Examples own
   executable usage selection and small local caveats. Benchmark and
   maintainer-policy interpretation now point to their owning surfaces instead
   of being repeated in example prose.

4. **Public header cleanup stayed API-local.**
   The touched headers now describe caller-visible contracts, option behavior,
   matrix ownership, and compatibility boundaries without turning generated API
   documentation into a sprint archive.

5. **The proof-owner rename was bounded to a coherent cluster.**
   Sprint 95 did not rename every `test_sprint*_integration.c` file. It moved
   only the direct CSC cluster:
   - `test_sprint18_integration` -> `test_direct_csc_dispatch`
   - `test_sprint19_integration` -> `test_direct_csc_regression`
   - `test_sprint20_integration` -> `test_ldlt_backend_dispatch`
   That gave the highest-value proof owners product names without forcing a
   broad platform or mixed-capability test split.

6. **Support-surface consolidation left a reusable map.**
   `INSTALL.md`, `benchmarks/README.md`, and `docs/maintainer_guide.md` now
   have a clearer split: install owns operational setup, benchmarks own
   measurement surfaces, and the maintainer guide owns policy interpretation.

7. **The sprint closed from validation rather than intent.**
   Day 13 ran the full quality chain for the final branch state and then added
   focused scans for stale proof-owner names, whitespace, diff hygiene, and
   local Markdown links.

## What Didn't Go Well

1. **`docs/algorithm.md` remains a large historical surface.**
   Sprint 95 reduced the worst public-facing adoption and support surfaces, but
   the algorithm reference still contains substantial chronological sections.
   That needs a separate bounded modernization plan.

2. **Some historical test names remain.**
   Several `tests/test_sprint*_integration.c` files are still present because
   they are mixed-capability bundles, historical regression owners, or
   platform-policy coupled. Renaming them safely requires split-first design.

3. **Active benchmark command names still expose history.**
   `bench-reorder-sprint86` and `--sprint86-slice` remain live compatibility
   surfaces. Sprint 95 clarified their current meaning but correctly avoided a
   rename without aliases or a compatibility decision.

4. **The maintainer guide still carries provenance.**
   That is appropriate where provenance explains current policy, but it means
   the guide should continue to be watched so moved public-doc history does not
   become unbounded maintainer clutter.

5. **The branch touched more than prose.**
   Header edits, example source edits, test file renames, Makefile updates, and
   CMake registrations all increased validation cost. That was justified by
   the proof-owner cleanup, but it made a full quality-chain closeout necessary.

## Final Metrics

### Validation

| Metric | Sprint 95 close state |
|---|---:|
| standard branch-level gate | `make format && make lint && make test` passed |
| final test summary | `All tests passed.` |
| stale selected proof-owner scan | no active `test_sprint18/19/20` references outside planning/build |
| local Markdown link check | passed on touched docs and Sprint 95 artifacts |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched public docs, headers, examples, and Sprint 95 artifacts |

### Sprint 95 artifact package

| Metric | Sprint 95 close state |
|---|---:|
| total artifact files under `SPRINT_95/artifacts/` | `15` |
| inventory/audit/ownership artifacts | `4` |
| rewrite and cleanup artifacts | `6` |
| proof/support/validation/closeout artifacts | `5` |

Notes:

- inventory/audit/ownership artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-public-surface-inventory.md`
  - `day2-ranked-public-surface-audit.md`
  - `day3-audience-ownership-model.md`
- rewrite and cleanup artifacts:
  - `day4-readme-boundary-and-rewrite-outline.md`
  - `day5-readme-cleanup-batch.md`
  - `day6-tutorial-quickstart-cleanup.md`
  - `day7-public-docs-coherence.md`
  - `day8-public-header-cleanup.md`
  - `day9-example-cleanup.md`
- proof/support/validation/closeout artifacts:
  - `day10-proof-owner-naming-design.md`
  - `day11-proof-owner-cleanup.md`
  - `day12-support-surface-consolidation.md`
  - `day13-validation-and-residual-queue.md`
  - `day14-sprint95-closeout.md`

### Landed cleanup package

| Metric | Sprint 95 close state |
|---|---:|
| primary public narrative docs touched | `6` |
| public headers touched | `8` |
| example sources touched | `2` |
| benchmark/support source comments touched | `3` |
| proof-owner test files renamed | `3` |
| build registration surfaces touched | `2` |

Notes:

- primary public narrative docs touched:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `docs/algorithm.md`
  - `examples/README.md`
  - `benchmarks/README.md`
- public headers touched:
  - `include/sparse_eigs.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_lu.h`
  - `include/sparse_lu_csr.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_types.h`
- example sources touched:
  - `examples/example_analysis.c`
  - `examples/example_eigs.c`
- benchmark/support source comments touched:
  - `benchmarks/bench_ldlt_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
- proof-owner test file renames:
  - `tests/test_direct_csc_dispatch.c`
  - `tests/test_direct_csc_regression.c`
  - `tests/test_ldlt_backend_dispatch.c`
- build registration surfaces touched:
  - `Makefile`
  - `CMakeLists.txt`

## Residual Deferred Debt

Sprint 95 deliberately stopped after the highest-value public narrative and
selected proof-owner cleanup package. The main open work it hands forward is:

- bounded `docs/algorithm.md` modernization
- split-first redesign for mixed historical integration owners before any
  further product-oriented renames
- compatibility-aware aliases or migration plan before renaming active
  historical benchmark commands
- continued maintainer-guide history reduction only where current policy
  interpretation remains clear
- generated API refresh only through the established source-comment workflow

Still consciously constrained rather than silently solved:

- no repo-wide removal of all sprint/day text
- no blanket rename of every `test_sprint*_integration.c` file
- no benchmark CLI or Makefile target compatibility break
- no hand-edited generated API HTML
- no claim that planning docs should stop being historical

Not carried forward as unresolved Sprint 95 debt:

- public-surface inventory
- ranked audit and cleanup queue
- audience ownership model
- README boundary and cleanup batch
- tutorial/example coherence pass
- selected public-header narrative cleanup
- direct CSC proof-owner rename batch
- support-surface consolidation
- full branch-level validation sweep
- explicit Sprint 96 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day2-ranked-public-surface-audit.md](./artifacts/day2-ranked-public-surface-audit.md)
- [day3-audience-ownership-model.md](./artifacts/day3-audience-ownership-model.md)
- [day5-readme-cleanup-batch.md](./artifacts/day5-readme-cleanup-batch.md)
- [day6-tutorial-quickstart-cleanup.md](./artifacts/day6-tutorial-quickstart-cleanup.md)
- [day8-public-header-cleanup.md](./artifacts/day8-public-header-cleanup.md)
- [day9-example-cleanup.md](./artifacts/day9-example-cleanup.md)
- [day10-proof-owner-naming-design.md](./artifacts/day10-proof-owner-naming-design.md)
- [day11-proof-owner-cleanup.md](./artifacts/day11-proof-owner-cleanup.md)
- [day12-support-surface-consolidation.md](./artifacts/day12-support-surface-consolidation.md)
- [day13-validation-and-residual-queue.md](./artifacts/day13-validation-and-residual-queue.md)
- [day14-sprint95-closeout.md](./artifacts/day14-sprint95-closeout.md)

## Bottom Line

Sprint 95 achieved its goal:

- permanent public docs now read more like product docs and less like sprint
  archives
- the public workflow narrative is smaller and better routed through owner
  surfaces
- selected public headers and examples no longer carry unnecessary sprint-era
  narrative
- the highest-value direct CSC proof owners now have product-oriented names
- support docs have a clearer install/benchmark/maintainer ownership split
- the branch validates cleanly under the full quality chain
- Sprint 96 receives a bounded handoff queue instead of a broad narrative
  cleanup backlog

Future documentation work can now start from an explicit ownership model and
residual queue instead of rediscovering whether public docs, support docs,
proof owners, or planning history are supposed to own each narrative.
