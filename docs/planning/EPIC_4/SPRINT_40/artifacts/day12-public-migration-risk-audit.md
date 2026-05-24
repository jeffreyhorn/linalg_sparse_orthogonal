# Sprint 40 Day 12: Public Migration-Risk Audit

## Objective

Identify the public-facing surfaces most likely to need compatibility help
during Epic 4 when lifecycle, factor-handle, and workspace refactors begin.
This audit separates internal-only opportunities from externally visible risk
zones so later implementation sprints know where wrapper preservation, doc
updates, example rewrites, or migration notes are most likely to be required.

## Audit Inputs

This audit is grounded in the current public or near-public surfaces:

- installed headers under `include/`
- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- maintained examples under `examples/`
- public type/result/handle names already in use:
  - `SparseMatrix`
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_ldlt_t`
  - `sparse_qr_t`
  - `sparse_svd_t`
  - iterative/preconditioner option structs and builder APIs

## Core Day 12 Conclusion

Epic 4 compatibility risk is concentrated in a fairly small set of public
surfaces:

1. direct factorization entry points that currently imply matrix-state
   transitions
2. documentation that teaches original-matrix / identity-permutation /
   copy-before-reuse rules
3. examples that mirror those workflows
4. bridge-handle surfaces like `sparse_analysis_t` / `sparse_factors_t`
5. any future repeated-run workspace additions for iterative/eigensolver APIs

The rest of Epic 4’s structural work is much more likely to be internal-only.

## Public-Facing Risk Zones

### Tier 1: Highest migration sensitivity

These are the surfaces most likely to need compatibility wrappers, careful
review, and explicit migration notes.

#### 1. LU and Cholesky public entry points

Why they are high-risk:

- they sit on top of the strongest matrix-as-factor-handle burden
- current docs already teach cancellation and in-place mutation semantics
- later explicit handle work could change how callers think about matrix reuse
  even if one-shot wrappers remain

Likely compatibility needs:

- stable one-shot wrapper preservation
- careful docs wording around mutation semantics
- possible later opt-in explicit-handle examples

#### 2. `README.md` lifecycle and solver sections

Why it is high-risk:

- it is the main operator-facing summary for lifecycle-sensitive workflows
- it currently teaches:
  - original-matrix requirements
  - factored-state validation
  - copy-before-reuse guidance
  - cancellation semantics
- it will need to stay truthful if internal ownership changes while public
  semantics remain stable

Likely compatibility needs:

- wording rewrites as internal ownership moves
- explicit distinction between preserved semantics and changed internals
- possible short migration notes if new explicit-handle APIs appear

#### 3. `docs/tutorial.md`

Why it is high-risk:

- it is the fuller teaching surface for QR, SVD, iterative solvers, and
  preconditioners
- it currently explains original matrix view, identity permutations, and fresh
  copies more directly than README does
- it will likely be the first place where users expect concrete examples of any
  new explicit-handle or workspace-backed workflows

Likely compatibility needs:

- example rewrites
- staged side-by-side old/new workflow explanations if public explicit handles
  arrive
- migration notes for copy-before-reuse / original-state expectations

### Tier 2: Significant but more bounded public risk

#### 4. Installed headers for lifecycle-sensitive families

Highest-sensitivity headers:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_ilu.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Why they are bounded:

- headers already own the API semantics and preconditions cleanly
- later refactors can often preserve those semantics while changing internal
  ownership

Why they are still risky:

- any public explicit-handle enrichment or new workspace API will eventually
  need header-level contracts
- any narrowing of field-level lifecycle reasoning must stay consistent with
  the installed-header story

Likely compatibility needs:

- additive declarations before removals
- careful precondition wording updates
- possible new examples that preserve old one-shot entry points

#### 5. Analyze-once bridge surfaces

Primary surfaces:

- `sparse_analysis_t`
- `sparse_factors_t`
- README/tutorial examples that teach analyze-then-factor usage

Why they are risky:

- they are already close to the future architecture conceptually
- changing them carelessly could break one of the cleanest current public seams
- later internal payload normalization may require docs clarification even if
  the public shape remains stable

Likely compatibility needs:

- preserve public conceptual shape
- update docs if ownership stops being matrix-centric internally
- likely little or no visible deprecation at first

#### 6. Examples that encode lifecycle-sensitive expectations

Highest-sensitivity examples:

- `examples/example_lu.c`
- `examples/example_cholesky.c`
- `examples/example_qr.c`
- `examples/example_iterative.c`
- `examples/example_ldlt.c`
- `examples/example_svd_lowrank.c`
- `examples/example_ic_minres.c`
- `examples/example_matrix_free.c`

Why they are risky:

- examples are often the first “real code” users copy
- they reinforce copy-before-reuse, original-matrix, and preconditioner
  composition expectations

Likely compatibility needs:

- example refreshes
- preservation of one-shot sample code even if richer explicit-handle APIs are
  added

### Tier 3: Lower public risk but still externally visible

#### 7. Benchmark and support-tooling references

Why they are lower-risk:

- they are not the main user API surface
- most Epic 4 lifecycle refactors should not require benchmark-facing contract
  changes

Residual risk:

- usage/help text may need small updates if terminology changes
- benchmark/example docs may need consistency sweeps after larger public-doc
  changes

#### 8. Maintainer-facing quality/procedure docs

Why they are lower-risk:

- these are not the main lifecycle-teaching surfaces
- most handle/workspace refactors only affect them indirectly

Residual risk:

- README signposts and maintainer notes may need small updates when new public
  migration notes appear

## Internal-Only vs Public-Facing Refactor Boundary

### Mostly internal-only zones

These later Epic 4 changes should usually be able to land with little or no
direct public compatibility burden if wrapper behavior stays stable:

- internal LU/Cholesky factor payload insertion
- internal `sparse_factors_t` payload normalization
- internal helper/allocation/overflow consolidation
- `src/sparse_graph.c` decomposition
- internal iterative/eigensolver reusable workspace plumbing
- script/tooling ownership cleanup

### Mixed boundary zones

These are likely to start internally but may eventually need public explanation
or additive API surface:

- explicit LU/Cholesky factor-handle enrichment
- public workspace/context APIs for repeated iterative/eigensolver runs
- narrowing public dependence on `factored`-style lifecycle escape hatches

### Strongly public-facing zones

These later changes will require careful compatibility messaging even if the
code impact is small:

- README lifecycle wording
- tutorial workflow examples
- installed-header precondition wording
- examples that users copy into downstream code

## Prioritized Compatibility-Sensitive Surface List

1. `README.md`
2. `docs/tutorial.md`
3. LU / Cholesky public APIs and any future explicit-handle sibling APIs
4. installed headers for lifecycle-sensitive families
5. analyze-once bridge surfaces (`sparse_analysis_t`, `sparse_factors_t`)
6. maintained examples that encode matrix-lifecycle and preconditioner usage
7. iterative/eigensolver public API entry points if opt-in workspace APIs are
   added later

## Likely Compatibility Tools by Surface

### Wrapper preservation

Best fit for:

- LU one-shot APIs
- Cholesky one-shot APIs
- iterative/eigensolver one-shot APIs if workspace-backed internals are added

### Documentation rewrites

Best fit for:

- README
- tutorial
- example comments
- installed-header examples

### Additive APIs before migration pressure

Best fit for:

- explicit handle siblings to existing one-shot factorization paths
- opt-in workspaces/contexts for repeated iterative/eigensolver runs

### Migration notes

Best fit for:

- any future explicit-handle public enrichment
- any future de-emphasis of field-level lifecycle reasoning
- any workflow where “original matrix required” becomes easier to satisfy but
  old assumptions remain documented

## Day 12 Decisions

1. The highest public migration risk is concentrated in lifecycle-teaching docs
   and matrix-mutating direct factorization entry points.
2. `README.md` and `docs/tutorial.md` are the two strongest externally visible
   compatibility surfaces.
3. Most early Epic 4 structural work can and should remain internal-first.
4. Analyze-once surfaces should be treated as preserve-and-evolve bridges, not
   as rewrite-first targets.
5. Any future public workspace API should be additive and wrapper-compatible
   rather than replacing existing one-shot solver entry points immediately.

## Day 12 Output for Later Sprints

Later Epic 4 implementation sprints now have:

- a ranked compatibility-sensitive surface list
- an internal-only vs public-facing refactor boundary map
- a clearer sense of where wrapper preservation is enough
- a clearer sense of where docs/examples/header updates will be required even
  if code behavior remains mostly stable

That should reduce the chance that public migration work is discovered too late
in the implementation cycle.
