# Sprint 101 Day 10 Compatibility Documentation Design

## Purpose

Day 10 designs the public wording that makes compressed CSR/CSC construction
the obvious front door for compressed-input callers while preserving
`SparseMatrix` insertion, mutation, and one-shot examples as supported
compatibility paths. This is a design artifact for Day 11 docs/examples work.
It does not change APIs, code, tests, examples, or public documentation.

## Public Wording Audit

| surface | current state after Day 9 | assessment |
|---|---|---|
| `include/sparse_csr.h` | clearly describes simple and diagnostic compressed-first constructors, copy ownership, and matrix-shell entry | aligned |
| `include/sparse_matrix.h` | describes `SparseMatrix` as mutable construction and one-shot direct compatibility shell | aligned |
| README capability list | names CSR/CSC export plus compressed-first construction | aligned |
| README workflow chooser | tells compressed-input callers to start with `sparse_create_from_csr/csc` | aligned |
| README quick start | still uses insertion-based construction for the smallest example | acceptable if labeled as smallest hand-written example, not the only front door |
| README API reference | now describes `sparse_from_csr/csc` as diagnostic compressed-first constructors | aligned |
| tutorial workflow chooser | now lists compressed-first construction before occasional solver paths | aligned |
| tutorial creation section | starts with insertion and then explains compressed input | acceptable but Day 11 should make the section title/body less insertion-centric |
| `examples/README.md` start map | lists basic, repeated-run, iterative, eigs, install, and tutorial routes; no compressed-input route | needs Day 11 edit |
| shipped example programs | no dedicated compressed-input construction example | likely Day 11 candidate if the project wants an executable reference |

## Remaining Linked-List-First Pressure

| pressure | source | Day 11 handling |
|---|---|---|
| Quick Start uses `sparse_create` and `sparse_insert` immediately | `README.md` | keep it for hand-written tiny matrices, but add one sentence that CSR/CSC callers should skip insertion and use compressed constructors |
| Tutorial section title says "Creating and Manipulating Sparse Matrices" and opens with insertion | `docs/tutorial.md` | rename or preface the section so insertion is one construction option, not the product center |
| Examples map has no compressed-input route | `examples/README.md` | add a "Have CSR/CSC input?" route pointing to the tutorial and any new example |
| No executable compressed-input teaching example exists | `examples/` | add a small example only if Day 11 can validate it with `make examples` and avoid broad solver-parity claims |

## Compressed-First Narrative Draft

Use this wording shape for Day 11:

> If your matrix already exists as CSR or CSC arrays, start with
> `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)`. The library
> validates and copies those arrays into a normal caller-owned `SparseMatrix`
> shell, then the existing solver, analysis, factor, iterative, and eigensolver
> APIs apply. Use `sparse_from_csr(...)` or `sparse_from_csc(...)` when the call
> site needs explicit `sparse_err_t` diagnostics.

Avoid wording that says:

- CSR/CSC constructors avoid all `SparseMatrix` ownership;
- solvers accept CSR/CSC objects directly;
- the mutable matrix shell is deprecated;
- compressed construction proves broad solver parity;
- matrix-free iterative callbacks are the default CSR/CSC path.

## Compatibility-Shell Wording Rules

| rule | rationale |
|---|---|
| Call insertion-based construction "best for tiny hand-written examples" or "mutable construction" | preserves the API without presenting it as the only product center |
| Keep `SparseMatrix` as the public solver object | matches current ABI and Day 8 lifecycle model |
| State that compressed constructors copy caller arrays | prevents adopt/no-copy assumptions |
| State that returned matrices are freed with `sparse_free(...)` | keeps ownership rules concrete |
| Treat repeated direct, iterative handles, and eigensolver handles as later lifecycle choices | avoids overloading compressed construction with solver-reuse claims |
| Treat matrix-free callbacks as expert escape hatches | avoids making unsupported built-in CSR/CSC adapters sound shipped |

## Day 11 Docs and Example Edit List

| priority | edit | target | validation |
|---:|---|---|---|
| 1 | Add a compact compressed-input route to the examples start map | `examples/README.md` | docs hygiene |
| 2 | Add a brief compressed-input note before or inside README Quick Start | `README.md` | docs hygiene |
| 3 | Reword the tutorial creation section so insertion is one option and compressed construction is visible before the first solver transition | `docs/tutorial.md` | docs hygiene |
| 4 | Consider adding `examples/example_compressed_input.c` that builds a small CSR matrix, solves through normal LU, and frees all owners | `examples/`, `Makefile`, possibly `CMakeLists.txt` | `make examples`; full C quality gate if `.c` or `.h` files change |
| 5 | If no executable example is added, explicitly record that Day 12's regression tests remain the executable proof | Day 11 artifact | docs hygiene |

## Non-Claims

Day 10 does not claim:

- replacement or deprecation of the mutable `SparseMatrix` shell;
- direct CSR/CSC solver entry APIs;
- no-copy/adopt constructors;
- compressed-native Matrix Market publication;
- broad compressed parity across LU, Cholesky, LDL^T, QR, SVD, iterative, and
  eigensolver families;
- portable performance superiority from compressed construction alone.

## Day 10 Conclusion

The public wording is mostly aligned after Day 9. The remaining Day 11 work is
to make the examples route and tutorial/Quick Start framing match the earned
implementation evidence. A small executable compressed-input example would be
valuable, but it should land only with normal example registration and the full
C quality gate because it would add a `.c` file.
