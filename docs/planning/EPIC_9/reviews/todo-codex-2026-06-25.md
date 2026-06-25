# Epic 9 Gap-Closure Todo - 2026-06-25

This todo converts the 2026-06-25 review findings into a concrete closure
sequence. The order is deliberate: it starts from the frozen Sprint 90 target,
comparison, and non-goal contracts, then attacks the structural product
ceilings before widening into coherence, duplication, and broader comparison
depth.

The sequence assumes the project enters Epic 9 from the validated Epic 8 close
state carried through Sprint 90:

- `make quality-review-full`
- reviewed CMake parity = `53`
- maintained install/export proof passing
- bounded external SPD comparison lane in place
- front-door/public support surfaces materially cleaner than before Epic 8
- explicit Sprint 90 target-state, comparison, and risk-fence package in
  writing

## Stage 1: Hold the Epic 9 target and claim fence

### 1. Reconfirm the strongest current baseline

- rerun the strongest reviewed baseline when implementation work begins
- recapture reviewed runtime concentration
- recapture install/export proof status
- recapture the maintained bounded external comparison outputs

**Done when:** Epic 9 implementation starts from one fresh measured baseline
instead of only from Sprint 90 planning notes.

### 2. Keep the bounded target-state reading intact

- preserve the Sprint 90 target:
  - bounded state-of-the-art sparse linear algebra library
- reject drift toward:
  - generic research sprawl
  - fake industrial-platform claims
  - benchmark-supremacy theater

**Done when:** later sprint work still reads against one stable target-state
contract.

### 3. Keep the anti-sprawl fence explicit

- explicitly reject:
  - blanket full-library compressed rewrite claims
  - fake platform symmetry
  - fake broad complex/mixed-precision claims
  - fake broad shared-library maturity
  - fake benchmark supremacy claims

**Done when:** Epic 9 retains one durable truthfulness fence throughout the
implementation sprints.

## Stage 2: Heal the core product and compute model

### 4. Audit the remaining linked-list-first costs

- measure where public workflows still begin or bounce through the linked-list
  shell unnecessarily
- rank conversion/publication costs by user value and runtime payoff

**Done when:** the highest-value linked-list-first costs are named and ranked.

### 5. Design a compressed-first product model

- define the future role of the linked-list shell
- define the future role of CSC/CSR-backed construction/import
- define how one-shot and repeated-run direct workflows should read publicly

**Done when:** the repo has one explicit product model where linked lists are
bounded rather than conceptual center stage.

### 6. Land compressed-first construction/import/publication batches

- add the highest-value compressed-first entry points
- reduce shell-centric round-trips on major direct and interop workflows
- keep compatibility shims where needed

**Done when:** large workflows can begin in compressed form without the shell
remaining the default mental model.

### 7. Tighten lifecycle ownership across direct families

- make one-shot vs repeated-run rules simpler
- reduce solve-ready ambiguity on mutated shells
- ensure factor/publication contracts are easier to reason about

**Done when:** direct workflow semantics are smaller, clearer, and harder to
misuse.

## Stage 3: Raise the dense/backend and runtime ceiling

### 8. Profile dense-kernel and reorder/runtime hotspots again

- re-measure supernodal direct-family dense consumers
- re-measure reorder/ND runtime concentration
- separate algorithm cost from proof-surface organization cost

**Done when:** backend and runtime work are driven by current measurements.

### 9. Design a portable backend ABI

- preserve the builtin backend as the default self-contained surface
- define a portable BLAS/LAPACK-class path
- keep backend selection observable and bounded

**Done when:** the repo can support more than one serious backend lane without
framework theater.

### 10. Integrate portable dense acceleration on the highest-value paths

- start with the direct-family supernodal lanes
- keep all fallbacks intact
- make results benchmarkable and testable

**Done when:** the repo has a serious portable backend ceiling beyond the
Darwin-only optional path.

### 11. Decide the threading/runtime model

- define where OpenMP is product-level and where it remains family-local
- remove or bound unsafe global/current-thread tuning hooks where possible
- document the supported thread-safety model clearly

**Done when:** concurrency and runtime behavior read as product design rather
than accumulated caveats.

### 12. Reduce the reviewed runtime long pole again

- optimize or reorganize the ND/reorder proof path
- keep correctness stronger than speed-only wins
- preserve fixture-level trust

**Done when:** the reviewed baseline is materially faster or materially easier
to scale.

## Stage 4: Expand the capability envelope

### 13. Re-rank the next capability targets

- compare:
  - complex scalar support
  - mixed precision
  - wider index maturity
  - broader eigensolver breadth
  - broader reusable iterative/workspace surfaces

**Done when:** the next capability sprint works on the highest-value breadth
gap first.

### 14. Land one real scalar-family widening

- introduce the minimum viable abstraction for a second real scalar family or
  mixed-precision lane
- avoid fake genericity across untouched families

**Done when:** at least one widened scalar capability is truly implemented,
tested, and documented.

### 15. Mature the index-width and ABI story

- review remaining 32-bit assumptions
- strengthen 64-bit build/package/test confidence
- keep overflow behavior explicit

**Done when:** larger-index support reads like a supported product lane rather
than just a compile-time seam.

### 16. Broaden one solver-family capability lane

- choose one bounded but high-value target:
  - unsymmetric eigensolver support
  - broader iterative-handle coverage
  - a wider direct-family public surface

**Done when:** the repo ships one real additional capability lane with proof.

## Stage 5: Heal documentation, naming, and product coherence

### 17. Remove sprint-era chronology from public product surfaces

- scrub sprint-history residue from:
  - `README.md`
  - `INSTALL.md`
  - public headers
  - example docs
- keep durable technical rationale only

**Done when:** public docs read like product docs, not historical changelogs.

### 18. Clean up permanent test and example naming

- rename or regroup sprint-named test binaries where practical
- make integration/proof owners easier to discover from names alone

**Done when:** proof surfaces are easier to navigate without planning history.

### 19. Consolidate the user-journey narrative

- keep the support split truthful
- reduce duplication across README, tutorial, examples, and install docs
- make the main user journeys more obvious

**Done when:** a new user can find the right workflow with less surface-hopping.

## Stage 6: Reduce maintainability concentration

### 20. Extract the next source hotspots

- start with `src/sparse_ldlt_csc.c`
- then `src/sparse_iterative.c`
- then the highest-value residual direct/algorithm families

**Done when:** the largest mixed-role owners are structurally smaller and more
reviewable.

### 21. Extract the next giant-test hotspots

- start with `tests/test_chol_csc.c`
- then `tests/test_ldlt_csc.c` or `tests/test_graph.c`
- keep family-local helpers local where possible

**Done when:** proof ownership is clearer and giant registration walls are
smaller.

### 22. Scrub historical internal comments that no longer add product value

- keep algorithm rationale
- remove sprint/date/history noise where it no longer helps maintenance

**Done when:** the code is easier to read as code, not as archive.

## Stage 7: Reduce duplication in build/package/workflow surfaces

### 23. Audit build-topology duplication

- identify source-list duplication and workflow duplication between Make and
  CMake
- decide which duplication is intentional and which should be reduced

**Done when:** the repo has one explicit build-convergence plan instead of
passive duplication acceptance.

### 24. Reduce source-list and workflow duplication where safe

- centralize or generate lists where practical
- keep reviewed parity intact
- avoid breaking install/export proof

**Done when:** long-term maintenance cost is lower without sacrificing proof.

### 25. Decide whether the package surface stays permanently static-first

- keep exact-version truthfulness
- decide whether a bounded shared-library lane is worth adding
- do not broaden claims unless the proof matrix broadens with them

**Done when:** the package surface is both truthful and strategically settled.

## Stage 8: Broaden comparison and assurance evidence

### 26. Broaden maintained external correctness comparison

- expand beyond the current bounded SPD lane where payoff is highest
- keep the comparison matrix reviewable and deterministic

**Done when:** the repo has stronger external evidence across more than one
core family/path.

### 27. Broaden performance-comparison evidence

- add bounded comparison artifacts for reorder/fill/runtime and direct-family
  performance where feasible
- keep machine-class caveats explicit

**Done when:** the repo has a more competitive evidence package, not just
isolated local benchmarks.

### 28. Revisit cross-platform proof only where evidence justifies it

- decide which install/test/proof lanes can become broader on macOS/Windows
- keep Linux as strongest truth unless proof really broadens

**Done when:** platform claims remain truthful but less asymmetric where
practical.

## Stage 9: Final convergence and closeout

### 29. Re-audit the full state against the Epic 9 target

- verify whether the biggest structural ceilings actually moved
- distinguish resolved gaps from deliberate non-claims

**Done when:** the final epic close reads from live evidence, not from intent.

### 30. Close from one validated final baseline

- rerun the strongest reviewed baseline
- rerun install/export proof
- rerun canonical reporting
- rerun the maintained external comparison surfaces

**Done when:** Epic 9 ends from one explicit measured baseline and one explicit
residual queue.
