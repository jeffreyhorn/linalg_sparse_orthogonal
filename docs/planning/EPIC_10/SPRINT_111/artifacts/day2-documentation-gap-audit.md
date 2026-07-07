# Day 2 Documentation Gap Audit

## Purpose

Day 2 compares the current user-facing documentation and examples against the
actual public contracts in headers and implementation. The output is a
prioritized, audience-classified gap list for Days 3-14. This audit does not
change product documentation yet; it creates the source of truth for the
solver-selection guide, compressed-first example batch, Matrix Market behavior
docs, benchmark interpretation work, and maintainer/user split cleanup.

## Source Inputs

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `docs/matrix_market.md`
- `docs/algorithm.md`
- `docs/maintainer_guide.md`
- `examples/README.md`
- representative examples:
  - `examples/example_basic_solve.c`
  - `examples/example_compressed_input.c`
  - `examples/example_analysis.c`
  - `examples/example_iterative.c`
  - `examples/example_eigs.c`
  - `examples/example_svd_lowrank.c`
  - `examples/example_colamd.c`
- `benchmarks/README.md`
- public headers under `include/`
- Matrix Market implementation owner:
  - `src/sparse_matrix_io.c`

## Public Contract Baseline

### Compressed Input

The public compressed-first baseline is defined by `include/sparse_csr.h`:

- `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` are the
  simplest public constructors for callers that already own compressed data.
- The simple constructors validate and copy input arrays into an independent
  public `SparseMatrix` shell.
- The simple constructors return `NULL` on invalid input or allocation
  failure.
- `sparse_from_csr(...)` and `sparse_from_csc(...)` provide explicit
  `sparse_err_t` diagnostics.
- Caller-owned CSR/CSC arrays are not adopted or modified.
- The returned matrix is caller-owned and freed with `sparse_free(...)`.
- Exported `SparseCsr` and `SparseCsc` objects are freed with
  `sparse_csr_free(...)` and `sparse_csc_free(...)`.

### Matrix Market

The public Matrix Market baseline is defined by `include/sparse_matrix.h` and
implemented in `src/sparse_matrix_io.c`:

- `sparse_save_mm(...)` writes coordinate real general Matrix Market files.
- Values are written with `%.15g`.
- Only stored nonzeros are written.
- `sparse_load_mm(...)` supports coordinate real, integer, and pattern values
  with general or symmetric symmetry.
- Pattern entries default to value `1.0`.
- Symmetric off-diagonal entries are mirrored during load.
- Coordinates are one-based in files and validated against matrix dimensions.
- Unsupported headers, bad dimensions, invalid coordinates, malformed data, or
  symmetric rectangular inputs return `SPARSE_ERR_PARSE`.
- File open/read/write/close failures return `SPARSE_ERR_IO` and capture the
  system `errno` retrievable through `sparse_errno()`.
- Successful load/save resets `sparse_errno()` to `0`.
- Loaded matrices are caller-owned and freed with `sparse_free(...)`.
- The Matrix Market implementation uses private matrix builder and I/O owners;
  the public docs must not claim a public Matrix I/O module or public builder
  API.

### Solver and Workflow Routing

The public adoption baseline is distributed across README, tutorial, examples,
and headers:

- one-shot direct solve: LU, Cholesky, LDLT, and QR public entry points;
- repeated direct solve: `sparse_analyze(...)` ->
  `sparse_factor_numeric(...)` -> `sparse_factor_solve(...)` ->
  `sparse_refactor_numeric(...)`;
- compressed-first direct path: construct with CSR/CSC then use normal public
  matrix shell and solver APIs;
- one-shot iterative path: CG, GMRES, MINRES, and BiCGSTAB public solve APIs;
- repeated iterative handles: CG, GMRES, and MINRES;
- eigensolver path: `sparse_eigs_sym(...)` and explicit repeated-run handle
  support for stable-dimension symmetric eigensolves;
- SVD path: SVD, rank, condition, pseudoinverse, and low-rank APIs;
- reorder/fill path: RCM, AMD, ND, and COLAMD with symmetric versus column
  ordering distinctions.

## Gap Register

| ID | Surface | Gap | Classification | Later Owner |
|---|---|---|---|---|
| G1 | README/tutorial/examples | Solver-selection guidance is spread across several places; users can follow it, but there is no compact decision guide that starts from matrix format and workload shape. | User-blocking | Days 3-4 |
| G2 | `docs/matrix_market.md` | Matrix Market docs omit duplicate-entry last-write behavior inherited from the builder path. | User-blocking | Day 9 |
| G3 | `docs/matrix_market.md` | Matrix Market docs do not spell out final-zero elision after duplicate resolution and sparse storage filtering. | Confusing | Day 9 |
| G4 | `docs/matrix_market.md` | Ownership is implied by examples but not stated as a clear loaded-matrix lifecycle rule. | User-blocking | Day 9 |
| G5 | `docs/matrix_market.md` | `sparse_errno()` behavior is underdocumented: I/O failures capture errno and successful load/save resets it to `0`. | Confusing | Day 9 |
| G6 | `docs/matrix_market.md` and examples | No dedicated Matrix Market example is visible in `examples/README.md`; the current load snippet appears in docs but not as a shipped workflow example. | User-blocking | Days 5 and 8 |
| G7 | `README.md` | Capability sections contain useful but dense backend and evidence wording before a new user has one compact solver guide. | Confusing | Days 3-4 and 10 |
| G8 | `docs/tutorial.md` | Tutorial is broadly coherent, but it duplicates workflow-selection material that should point to the future concise guide once created. | Confusing | Days 4 and 10 |
| G9 | `examples/README.md` | Examples map is strong, but it needs explicit alignment with the future solver guide and a clearer Matrix Market path. | User-blocking | Days 5-8 |
| G10 | `benchmarks/README.md` | Benchmark docs are accurate and bounded but dense; first-time users need a shorter interpretation route before reading schema and proof-owner detail. | Confusing | Day 11 |
| G11 | `docs/algorithm.md` | Algorithm docs contain extensive sprint history and benchmark evidence; this is useful reference material but too chronology-heavy for adoption. | Maintainer-only | Day 12 |
| G12 | `docs/maintainer_guide.md` | Maintainer guide correctly owns proof policy, but links from adoption docs should avoid making it required reading for normal use. | Maintainer-only | Days 10 and 12 |
| G13 | Public headers | Header comments are generally authoritative; later documentation should quote behavior from headers instead of introducing new unsupported contracts. | Guardrail | Days 3-10 |
| G14 | Future public module wording | Any language that suggests a public Matrix I/O module or public builder API would exceed the current contract. | Future-work / no-go | Days 8-10 |

## Stale or Risky Claim Inventory

| Claim Area | Risk | Required Handling |
|---|---|---|
| Matrix Market duplicate handling | The public docs currently describe supported formats but not duplicate resolution. | Day 9 should document last-write behavior and final-zero elision from the private builder path. |
| Matrix Market symmetric expansion | Existing docs say off-diagonal entries are mirrored, but later examples must make clear that loaded symmetric matrices become fully populated in the public matrix shell. | Day 8/9 should keep this wording explicit. |
| Pattern matrices | Existing docs correctly state value `1.0`; examples should not imply pattern values are caller-configurable. | Day 8/9 should retain this boundary. |
| `sparse_errno()` | README and header mention errno, but Matrix Market docs do not state reset-on-success or parse-versus-I/O split. | Day 9 should document error classes and reset behavior. |
| Compressed-first constructors | README/tutorial/examples mention them, but Day 3-4 guide should make simple-versus-diagnostic constructor choice explicit. | Day 4 guide and Day 6 examples. |
| Repeated-run support | README/tutorial/examples correctly limit iterative handles to CG/GMRES/MINRES; the new guide must preserve that limit. | Day 3-4 guide. |
| Benchmark evidence | Benchmark docs and README already caution against portable timing claims, but the path is dense. | Day 11 should create a shorter interpretation layer. |
| Planning evidence | Planning artifacts are extensive and useful, but user docs should summarize stable outcomes instead of routing users into sprint artifacts. | Days 10 and 12. |

## Validation Needs for Later Changes

| Planned Change Type | Files Likely Touched | Required Validation |
|---|---|---|
| New solver-selection guide | `docs/*.md`, README/examples links | `git diff --check`; trailing-whitespace scan over touched docs. |
| Matrix Market docs update | `docs/matrix_market.md`, possible README/examples references | `git diff --check`; trailing-whitespace scan; claim review against `include/sparse_matrix.h` and `src/sparse_matrix_io.c`. |
| Compressed-first example update | `examples/*.c`, `examples/README.md` | focused example build or `make examples`; `git diff --check`. |
| Matrix Market example addition | `examples/*.c`, build metadata if a compiled example is added, `examples/README.md` | `make examples`; source-list/build metadata checks if the example build set changes; `git diff --check`. |
| Public header wording update | `include/*.h` | public API/install-header review; if `.h` changes are code-adjacent, run checks required by project policy. |
| Benchmark interpretation docs | `benchmarks/README.md`, optional guide links | `git diff --check`; claim review against current benchmark commands and documented artifacts. |
| Maintainer/user split cleanup | README, docs, examples README, maintainer guide | `git diff --check`; link/reference review. |

## Prioritized Source of Truth for Days 3-4

The solver-selection guide should use this precedence order:

1. Public headers under `include/` for API names, options, ownership, errors,
   and lifecycle contracts.
2. Shipped examples for copyable public workflow patterns.
3. README and tutorial for current adoption wording that should be consolidated
   or linked.
4. Benchmark docs for measurement caveats, not correctness or universal
   performance guarantees.
5. Planning artifacts only for summarized evidence boundaries, not direct
   adoption instructions.

## Completion Criteria Status

- Documentation risks are tied to concrete files and sections.
- No planned guide claim requires behavior beyond the current public headers or
  implementation.
- Example-impacting changes have validation expectations.
- Days 3-4 have a prioritized source of truth for the solver-selection guide.
