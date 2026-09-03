# Sprint 194 Day 10 Header Narrative Audit

## Objective

Audit public headers under `include/` for workflow narrative, duplicated
examples, support claims, and adoption guidance that can move to user-facing
docs while preserving generated API documentation value. This is an audit-only
day: no public headers are edited here.

## Scope

Reviewed headers:

- `include/sparse_matrix.h`
- `include/sparse_csr.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- representative direct, dense, reorder, vector, and type headers found by
  narrative search

Search focus:

- long-form workflow guidance;
- README/tutorial/cookbook/example routing text;
- support and evidence boundaries;
- benchmark or parity caveats;
- usage examples large enough to belong in docs;
- prose that risks becoming stale independently of declarations.

## Preservation Rules

Do not remove or materially change:

- declarations, typedefs, enum values, macro names, struct fields, or public
  numeric constants;
- Doxygen `@file`, `@brief`, `@param`, `@return`, `@note`, `@par`,
  `@warning`, `@see`, and ownership text that explains API use at the point of
  declaration;
- status-code mappings such as `SPARSE_ERR_*`;
- callback lifetime and cancellation semantics;
- input/output ownership, allocation, and cleanup requirements;
- tolerance, rank, residual, and backend field definitions;
- generated API anchors likely derived from declaration adjacency;
- short examples required to understand non-obvious option structs or result
  population.

Header cleanup should be declaration-preserving and documentation-only unless
a separate defect is found. If Day 11 changes any `.h` file, run the full
quality gate required for C/header changes.

## Candidate Move List

| Header | Candidate text | Proposed owner | Rationale | Day 11 action |
| --- | --- | --- | --- | --- |
| `sparse_matrix.h` | File-level "For first-use paths..." routing paragraph naming cookbook, tutorial, solver selection, and examples. | `docs/api_reference.md` and existing README/doc links. | Cross-doc routing is adoption-map prose, not matrix-shell declaration detail. | Replace with one short sentence that this header owns exact matrix-shell declarations and ownership contracts. |
| `sparse_matrix.h` | `SPARSE_CSC_THRESHOLD` paragraph referencing maintained Cholesky CSC benchmark corpus and broad performance/package non-claims. | `docs/solver_selection.md`, `docs/maintainer_guide.md`, benchmark docs. | The macro needs dispatch semantics and override rules, but benchmark-corpus interpretation is support evidence prose. | Keep threshold behavior and override text; shorten evidence/non-claim wording and link to solver-selection/support docs if needed. |
| `sparse_csr.h` | File-level workflow phrase about entering the public matrix-shell workflow without linked-list mutation as conceptual starting point. | `docs/cookbook.md` compressed-first ladder. | Useful adoption framing already belongs in cookbook. | Keep conversion/import purpose and ownership/identity notes; move conceptual starting-point wording to docs if not already present. |
| `sparse_iterative.h` | File-level routing paragraph naming solver-selection, tutorial, cookbook, and examples before using the header. | `docs/api_reference.md`, `docs/solver_selection.md`. | Header should own exact iterative declarations/result contracts; routing is doc-index prose. | Shorten to "Use this header for exact option/result declarations." |
| `sparse_iterative.h` | Reusable-handle narrative explaining one-shot entries remain first-class and repeated-run lifecycle positioning. | `README.md`, `docs/solver_selection.md`, `docs/tutorial.md`. | Much of this is workflow positioning, but handle ownership/reuse semantics are API-critical. | Preserve zero-init, prepare/run/free, ownership, allocation failure, and state-discard semantics; trim broad positioning only. |
| `sparse_qr.h` | File-level routing sentence naming examples and solver selection for runnable workflow guidance/evidence boundaries. | `docs/solver_selection.md`, `examples/README.md`, `docs/api_reference.md`. | Cross-doc routing can stale independently of QR declarations. | Keep QR API-local contract list; shorten routing. |
| `sparse_qr.h` | `sparse_qr_opts_t.reorder` prose saying COLAMD recommended and ND best on PDE meshes. | `docs/solver_selection.md`, `docs/cookbook.md`. | Solver preference and matrix-family guidance is workflow-selection prose. | Preserve accepted enum behavior; move recommendation language to solver-selection docs unless needed for option semantics. |
| `sparse_qr.h` | `sparse_qr_rank_info()` note linking noisy/known-rank tolerance selection to solver-selection evidence boundary. | `docs/solver_selection.md`. | The tolerance formula is API detail; evidence-boundary routing is doc prose. | Keep tolerance formula and QR-local/global-rank warning; trim evidence-boundary link if coverage remains in docs. |
| `sparse_svd.h` | File-level routing paragraph naming examples and solver selection. | `docs/api_reference.md`, `docs/solver_selection.md`, `examples/README.md`. | Routing belongs outside declarations. | Shorten header to output shapes, ownership, convergence errors, and cleanup. |
| `sparse_svd.h` | Partial-SVD corpus evidence paragraph in `sparse_svd_partial(...)`. | `docs/solver_selection.md`, `docs/cookbook.md`, `docs/maintainer_guide.md`. | Fixture evidence and support boundaries are already maintained in docs; the header only needs algorithm/result limitations. | Preserve compute mode, vector approximation, full-U/full-VT exclusion, and non-convergence return; move fixture evidence wording out. |
| `sparse_svd.h` | Low-rank sparse note about `SPARSE_SVD_LOWRANK_OUTER` validation and bit-level equivalence. | `docs/maintainer_guide.md` runtime/backend section. | Environment-variable tuning and validation interpretation are maintainer/runtime prose. | Preserve option existence and output-memory tradeoff only if the env var remains public-facing; otherwise move extended validation wording. |
| `sparse_eigs.h` | Large file-level usage pattern example. | `docs/tutorial.md`, `examples/README.md`, `docs/solver_selection.md`. | The example is useful but long for a header and duplicates workflow docs. | Consider replacing with a minimal allocation/result-field sketch only if Doxygen still needs one; otherwise move to docs. |
| `sparse_eigs.h` | File-level workflow-routing paragraph naming solver selection, tutorial, cookbook, examples, and algorithm docs. | `docs/api_reference.md`. | Routing is index prose. | Shorten to API/result/option ownership and `@see` links. |
| `sparse_eigs.h` | Backend enum prose describing sweet spots, memory examples, bcsstk matrices, and backend superiority non-claims. | `docs/solver_selection.md`, `docs/algorithm.md`, `docs/maintainer_guide.md`, benchmarks docs. | Enum docs need dispatch behavior and field semantics; benchmark examples and workload advice belong in docs. | Preserve exact AUTO routing rules, enum meanings, and `backend_used`; move workload examples and broad evidence caveats out. |
| `sparse_eigs.h` | Reusable eigensolver handle narrative about one-shot remaining first-class and backend coverage. | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`. | Workflow positioning belongs in docs; handle lifecycle/ownership must stay. | Preserve zero-init, prepare/run/free, workspace ownership, and capacity reuse semantics; trim adoption positioning. |
| `sparse_cholesky.h` | File-level one-shot workflow narrative and repeated-run lifecycle routing. | `README.md`, `docs/solver_selection.md`, `docs/tutorial.md`. | It duplicates direct-solver workflow guidance. | Preserve in-place factorization, SPD, copy-before-factor, and return-code semantics; trim repeated-run routing. |
| `sparse_cholesky.h` | Usage pattern block. | `docs/tutorial.md`, `examples/README.md`, `docs/solver_selection.md`. | Example-level material can live with runnable examples. | Keep only if needed for Doxygen comprehension; otherwise move/shorten. |
| `sparse_ldlt.h` | Usage pattern block with Bunch-Kaufman, reordering, inertia, and solve. | `docs/tutorial.md`, `examples/README.md`, `docs/solver_selection.md`. | Useful adoption content, but long-form workflow example. | Preserve factor object, D blocks, pivot, inertia, solve, and cleanup declarations; consider moving full example. |
| `sparse_types.h` | Scalar-width preparation wording and non-claim notes. | Keep mostly in header; mirror summary in docs only. | Type width and scalar contract are declaration-adjacent API facts. | Do not remove unless docs already preserve exact compile-time contract. |

## Required Header Content

The following content should remain in headers because it is API-adjacent and
important for generated Doxygen:

- lifecycle ownership (`caller-owned`, borrowed, freed with exact helper);
- exact output shapes, dense layout, and column-major ordering;
- result-field population rules;
- callback lifetime and cancellation behavior;
- allocation failure and state-preservation behavior;
- tolerance formulas and default values;
- shape, permutation, factored-state, and same-pattern preconditions;
- in-place mutation warnings and copy-before-factor guidance where the API
  mutates caller-visible objects;
- exact cleanup functions such as `sparse_free`, `sparse_qr_free`,
  `sparse_svd_free`, `sparse_iter_handle_free`, and
  `sparse_eigs_handle_free`;
- exact enum behavior, including AUTO dispatch rules when a field directly
  controls dispatch.

## Candidate Relocation Targets

| Destination | Content to receive or already own |
| --- | --- |
| `README.md` | Compact first-use and escalation map; one-shot versus repeated-run positioning. |
| `docs/tutorial.md` | Step-by-step workflows and small code examples. |
| `docs/cookbook.md` | CSR/CSC/Matrix Market first-use and compressed-first workflow framing. |
| `docs/solver_selection.md` | Solver family selection, backend escalation, QR/SVD/eigs evidence boundaries, rank/residual terminology. |
| `docs/api_reference.md` | Header index and "headers own exact declarations" routing. |
| `docs/algorithm.md` | Algorithm notes that are not required at declaration points. |
| `docs/maintainer_guide.md` | Evidence interpretation, claim boundaries, benchmark/report support levels, and maintainer validation ownership. |
| `examples/README.md` | Runnable example selection and example-local output interpretation. |

## Header Non-Goals

Day 11 should not:

- change public declarations;
- rename macros, typedefs, enums, fields, or functions;
- change Doxygen anchors needed for generated API docs;
- remove parameter or return-value documentation;
- remove allocation, ownership, cleanup, or cancellation contracts;
- rewrite numerical tolerances or status-code semantics;
- change backend dispatch behavior or options;
- move all narrative out of headers indiscriminately;
- make style-only churn across every header.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| Removing useful Doxygen context for public users. | Keep declaration-adjacent API facts; move only cross-doc routing, examples, and evidence/support interpretation. |
| Breaking generated API docs or anchors. | Preserve declaration adjacency, Doxygen command structure, and `@see` targets; run API docs freshness checks after header edits. |
| Accidentally changing ABI or public behavior. | Treat Day 11 as comment-only unless a separate defect is found; run full C/header quality gates if headers change. |
| Losing evidence-boundary wording. | Confirm moved text already exists in solver selection, maintainer guide, benchmark docs, or planning artifacts before trimming. |
| Over-cleaning headers that correctly own API semantics. | Keep exact ownership, result-field, tolerance, status-code, and lifecycle documentation in headers. |

## Day 11 Recommended Cut

Keep Day 11 narrow:

1. Trim cross-doc routing paragraphs in `sparse_matrix.h`,
   `sparse_iterative.h`, `sparse_qr.h`, and `sparse_svd.h`.
2. Move or shorten the longest example/evidence paragraphs in
   `sparse_eigs.h`.
3. Preserve all declarations, parameter docs, return docs, option/default
   docs, result-field docs, and cleanup docs.
4. Run the full C/header quality gate because `.h` files will be modified.

Do not attempt every candidate in one pass if preserving Doxygen quality
becomes unclear. Complete a smaller declaration-preserving cleanup rather than
partially rewriting many headers.
