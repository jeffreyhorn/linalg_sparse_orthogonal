# Sprint 91 Day 3: Remaining Linked-List-First Cost Audit

## Purpose

Reduce Sprint 91's broad compressed-first product-model problem to one ranked
live contradiction map centered on the remaining linked-list-first
construction, import/export, publication, and lifecycle costs.

## Main Result

Sprint 91's broad compressed-first problem is now reduced to one ranked live
map of the highest-value linked-list-first costs:

- strongest first target:
  - compressed-first construction and import entry points on the public matrix
    shell
- strongest second target:
  - shell-centric publication and export round-trips that still keep the
    linked-list shell as the default conceptual owner even when compute wants
    CSC/CSR-backed state
- strongest third target:
  - one-shot direct-workflow entry paths that still read as shell-first even
    when the explicit repeated-run direct lifecycle already exists
- strongest fourth target:
  - lifecycle ambiguity on mutated vs solve-ready shell state and on where the
    long-lived direct-workflow owner really lives
- strongest support-only but real target:
  - README, maintainer, and public-header wording that still teaches the shell
    as conceptual center instead of as bounded mutable compatibility surface

## Strongest Current Contradiction

The strongest current contradiction is still the public construction and
ownership reading:

- `README.md` still opens by describing the project as an orthogonal linked-
  list sparse matrix library
- `include/sparse_matrix.h` still describes the public API as the orthogonal
  linked-list sparse matrix shell
- the same header explicitly says the shell remains the library's mutable
  sparse construction and one-shot direct-workflow compatibility shell
- `src/sparse_matrix.c` remains a major shell, mutation, and utility owner

That fixes the strongest first Sprint 91 move:

- the sprint no longer most urgently needs another generic proof or support
  tightening pass
- it needs one clearer compressed-first entry reading on the highest-value
  public direct and interop workflows
- the linked-list shell remains a real strength for pedagogy and mutation, but
  it still reads as the first thing the product wants callers to own

## Second-Tier Contradictions

### Construction and Import Still Read Shell-First

The strongest second contradiction is construction/import cost:

- `sparse_create()` and `sparse_copy()` still anchor the public ownership story
  in `include/sparse_matrix.h`
- `sparse_from_csr()` / `sparse_from_csc()` still convert compressed inputs
  into the shell rather than reading like compressed-first public entry points
- `README.md` still presents CSR/CSC conversion as conversion around the shell
  rather than as part of a compressed-first compute model

This is real Sprint 91 work because large workflows can already reason in
compressed storage, but the public product story still routes them through the
shell by default.

### Publication and Export Still Preserve Shell Centrality

The strongest third contradiction is publication/export ownership:

- `sparse_to_csr()` / `sparse_to_csc()` still read like export helpers hanging
  off the shell-centered product model
- `include/sparse_matrix.h` still says internal construction/import and
  publication paths may use bounded compressed-first helpers while public
  ownership stays with the shell
- maintainer wording still treats CSC publish-back on the direct-family lanes
  as family-local completion behavior rather than as a wider product-model
  change

This remains lower than first-center construction/import work, but it is still
real because compressed compute can exist internally while the product still
teaches the shell as the only durable owner.

### One-Shot vs Repeated-Run Ownership Is Still Split Too Sharply

The strongest fourth contradiction is lifecycle reading:

- `include/sparse_matrix.h` still makes the shell the one-shot direct-workflow
  compatibility owner
- `include/sparse_analysis.h` already describes the explicit repeated-run
  direct-solver path with reusable symbolic and factor/workspace state
- `README.md` teaches both stories, but the shell-first one still feels more
  primary than the repeated-run direct owner on major workflows
- the shell still carries factored-state compatibility hooks such as
  `sparse_mark_factored()`, which helps compatibility but also blurs where the
  durable direct owner should live

This is real Sprint 91 work, but it reads after construction/import because it
should tighten the product story around the highest-value workflows, not try
to remove every compatibility seam at once.

## Fix-Now vs Compatibility-Only Split

The current tree now separates cleanly into:

### Contradictions that should drive Sprint 91 implementation

- shell-first construction and import entry points
- shell-centric publication/export reading on major direct and interop paths
- one-shot vs repeated-run ownership ambiguity on the public direct workflow

### Contradictions that should remain compatibility-only for now

- broad shell removal
- family-wide direct-API rewrites
- fully compressed-first ownership on every solver family and every helper path
- package, backend, or capability widening under the guise of product cleanup

### Contradictions already materially reduced before Sprint 91

- the explicit repeated-run direct lifecycle now already has a real public
  owner in `include/sparse_analysis.h`
- CSR/CSC conversion and import/export already exist as public seams
- touched direct-family CSC paths already prove bounded compressed-first
  internal completion routes
- install/export and support surfaces are cleaner and better partitioned than
  they were before Epic 8

## Strongest Owner Surfaces

The highest-value owner surfaces tied to this audit are now explicit:

- product-model owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- repeated-run ownership owner:
  - `include/sparse_analysis.h`
- compressed conversion owners:
  - `include/sparse_csr.h`
  - touched import/export implementation seams behind the matrix shell
- strongest proof and workflow owners likely to matter later:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`

## Deferred Structural Claims

Broad claim widening remains lower-value first work:

- no fake "already compressed-first everywhere" story
- no fake deprecation of the linked-list shell before compatibility policy is
  explicit
- no reopening of backend, runtime, capability, or packaging lanes under Day 3
- no generic lifecycle rewrite detached from the highest-value direct workflows

## Interpretation

The useful Day 3 clarification is now explicit:

- Sprint 91 does not begin with generic direct-family cleanup
- it begins with one ranked shell-cost map
- the best first implementation center is compressed-first construction/import
  on the public matrix-shell story
- publication/export and one-shot vs repeated-run lifecycle tightening follow
  after that
- broader shell removal and family-wide workflow rewriting stay explicitly
  later unless the bounded product-model design proves otherwise

## Exit State

- Sprint 91 now has one ranked live shell-cost contradiction map grounded in
  the current post-Sprint-90 tree.
- The first compressed-first implementation center is fixed to construction and
  import entry points on the public matrix-shell story.
- Publication/export and lifecycle clarification are explicitly ordered behind
  that first center.
