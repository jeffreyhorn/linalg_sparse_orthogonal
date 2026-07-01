# Sprint 101 Day 2 Public Storage Surface Audit

## Purpose

Day 2 audits public construction, import/export, mutation, and publication
surfaces for remaining linked-list-first product costs. This is an audit-only
artifact. It does not recommend implementation before the Day 4 API design and
Day 5 implementation boundary freeze.

## Audited Surfaces

| surface | role |
|---|---|
| `include/sparse_matrix.h` | mutable matrix-shell lifecycle, mutation, Matrix Market I/O, display, permutations |
| `include/sparse_csr.h` | public CSR/CSC structs, compressed export, compressed-first construction wrappers |
| `src/sparse_matrix.c` | matrix-shell implementation, Matrix Market bulk build, mutation, arithmetic publication |
| `src/sparse_csr.c` | CSR/CSC validation, conversion, and compressed input build path |
| `README.md` | front-door workflow chooser and API summary |
| `docs/tutorial.md` | longer user learning path |
| `examples/README.md` | example selection and example-local storage guidance |

Day 3 will audit solver entry paths separately. This artifact only covers
storage-facing surfaces and the user narrative around them.

## Public Construction and Mutation Map

| surface | API or wording | current classification | notes |
|---|---|---|---|
| empty matrix construction | `sparse_create(rows, cols)` | compatibility-shell | correct mutable-shell entry point; still the tutorial's first construction path |
| incremental mutation | `sparse_insert`, `sparse_remove`, `sparse_set` | compatibility-shell | essential for editable matrix construction; not the desired conceptual front door for callers already holding CSR/CSC |
| deep copy | `sparse_copy` | compatibility-shell | preserves matrix-shell solve compatibility state; important but explanation-heavy |
| permutation reset | `sparse_reset_perms` | compatibility-shell | recovery tool after one-shot factorization/reorder state |
| Matrix Market load | `sparse_load_mm` | already partly compressed-first internally | header says imported coordinates are bulk-built into mutable shell while preserving visible API |
| Matrix Market save | `sparse_save_mm` | compatibility-shell publication | writes from `SparseMatrix`; no compressed output file owner |
| CSR export | `sparse_to_csr` | compressed publication | useful bridge to compressed workflows; caller owns returned `SparseCsr` |
| CSC export | `sparse_to_csc` | compressed publication | useful bridge to compressed workflows; caller owns returned `SparseCsc` |
| CSR simple constructor | `sparse_create_from_csr` | compressed-first | strongest current public compressed-first entry point; returns `NULL` on bad input |
| CSC simple constructor | `sparse_create_from_csc` | compressed-first | strongest current public compressed-first entry point; returns `NULL` on bad input |
| CSR status constructor | `sparse_from_csr` | compressed-first compatibility wrapper | same build path with explicit `sparse_err_t`; name still reads older conversion-first |
| CSC status constructor | `sparse_from_csc` | compressed-first compatibility wrapper | same build path with explicit `sparse_err_t`; name still reads older conversion-first |
| CSR/CSC free | `sparse_csr_free`, `sparse_csc_free` | compressed publication cleanup | clear owner for exported compressed structures |

## Current Implementation Signals

| implementation detail | observed behavior | product-model implication |
|---|---|---|
| CSR validation | rejects null structures, negative dimensions, invalid row pointers, out-of-range columns, unsorted or duplicate row entries | compressed input has meaningful validation before shell construction |
| CSC validation | rejects null structures, negative dimensions, invalid column pointers, out-of-range rows, unsorted or duplicate column entries | mirrors CSR validation for CSC callers |
| CSR/CSC build path | validates then calls `sparse_create` and `sparse_insert` for each entry | public entry is compressed-first, but implementation still bulk-builds into the mutable shell |
| simple constructors | `sparse_create_from_csr/csc` collapse errors to `NULL` | ergonomic for simple callers, but not enough for callers that need diagnostics |
| status constructors | `sparse_from_csr/csc` preserve `sparse_err_t` | useful but the `from_*` naming reads like conversion compatibility rather than front-door construction |
| Matrix Market load | bulk-builds into mutable shell | file import already avoids teaching callers incremental insertion, but still publishes only `SparseMatrix` |

## Public Documentation Narrative

| surface | current reading | classification |
|---|---|---|
| README capabilities | names "CSR/CSC export plus compressed-first construction" | already compressed-first aware |
| README workflow chooser | tells CSR/CSC callers to use `sparse_create_from_csr/csc` before one-shot direct APIs | already compressed-first aware |
| README quick start | still starts with `sparse_create` and `sparse_insert` | acceptable compatibility-shell teaching path for first solve |
| README API reference | lists CSR/CSC export and compressed-first construction alongside Matrix Market I/O | already compressed-first aware |
| README known limitations | explains linked-list in-place factorization and imported factor-state caveat | compatibility-shell caveat remains necessary |
| tutorial section 1 | starts with creating and mutating `SparseMatrix` via insertions | linked-list-first learning path |
| examples README | examples lean on one-shot APIs and mutable matrix copies | compatibility-shell learning path; no compressed-first example called out |

## Linked-List-First Cost Table

| cost | where visible | impact | likely owner |
|---|---|---|---|
| tutorial starts with incremental insertion only | `docs/tutorial.md` | first longer learning path still teaches mutable shell as the natural beginning | Day 10-11 docs/example work |
| examples map has no compressed-input example | `examples/README.md`, examples directory | users with CSR/CSC data do not get an executable compressed-first adoption reference | Day 10-12 docs/example/test work |
| status-return compressed constructors have compatibility-style names | `sparse_from_csr/csc` | users may miss them as diagnostic front-door constructors | Day 4 API design |
| simple constructors lose error detail | `sparse_create_from_csr/csc` | bad compressed input returns `NULL` without distinguishing null, shape, ordering, duplicate, or allocation failure | Day 4 API design and Day 12 tests |
| implementation still inserts one entry at a time | `src/sparse_csr.c` | public product model is compressed-first, but construction cost still uses shell mutation internally | Day 4-7 implementation candidate, if worth scope |
| compressed publication is one-way through owned structs | `sparse_to_csr/csc` | no direct compressed-to-solver public workflow beyond shell construction yet | Day 3 solver entry audit |
| Matrix Market import publishes shell only | `sparse_load_mm` | file import is not an independent compressed object story | out of Sprint 101 unless Day 4 promotes it |

## Surface Classification

| classification | surfaces |
|---|---|
| already compressed-first | `sparse_create_from_csr`, `sparse_create_from_csc`, README workflow chooser, `sparse_csr.h` overview |
| compressed publication | `sparse_to_csr`, `sparse_to_csc`, `sparse_csr_free`, `sparse_csc_free` |
| compatibility-shell | `sparse_create`, `sparse_insert`, `sparse_remove`, `sparse_set`, `sparse_copy`, `sparse_reset_perms`, README quick start, tutorial first matrix section, example-local matrix setup |
| unclear or candidate for refinement | `sparse_from_csr`, `sparse_from_csc`, error-reporting model for compressed constructors, lack of compressed-input example, insert-loop build path |

## Ranked Compressed-First Candidates

| rank | candidate | user value | compatibility risk | Day 2 recommendation |
|---:|---|---|---|---|
| 1 | add or promote an explicit diagnostic compressed constructor story | high | low-medium | Day 4 should decide whether naming/docs are enough or API refinement is needed |
| 2 | add a compressed-input example or tutorial subsection | high | low | strong docs/example candidate after implementation/design is settled |
| 3 | strengthen tests around bad CSR/CSC input diagnostics and ownership | high | low | Day 12 regression proof candidate even if API does not change |
| 4 | reduce internal shell-insertion cost for CSR/CSC build | medium-high | medium | implementation candidate only if measurable and locally bounded |
| 5 | clarify `sparse_from_csr/csc` as retained explicit-status front doors | medium | low | docs/header wording candidate |
| 6 | add compressed publication guidance for round-trip ownership | medium | low | docs/header candidate |
| 7 | change Matrix Market import product shape | low-medium | high | defer unless later evidence shows high user value |

## Compatibility-Shell Preservation Notes

The following remain supported and should not be deprecated by Sprint 101:

- constructing small or ad hoc matrices with `sparse_create` and
  `sparse_insert`;
- mutating matrix shells through insert/remove/set where editable storage is
  the point;
- one-shot direct examples that copy before in-place factorization;
- `SparseMatrix` as the public ownership object after CSR/CSC construction;
- compatibility wrappers that return explicit `sparse_err_t` for callers that
  need diagnostics.

Sprint 101 should change the product center, not erase the compatibility
shell.

## Day 2 Conclusion

The project already has real compressed-first public constructors and README
workflow guidance. The remaining linked-list-first cost is concentrated in
the longer tutorial, example set, diagnostic constructor story, and internal
CSR/CSC build mechanics. Day 3 should now audit solver entry paths to decide
whether constructor/import work alone is enough or whether solver-facing
compressed workflows still need clearer ownership.
