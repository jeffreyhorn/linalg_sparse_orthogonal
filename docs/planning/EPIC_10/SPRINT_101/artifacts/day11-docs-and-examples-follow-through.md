# Sprint 101 Day 11 Docs and Examples Follow-Through

## Purpose

Day 11 implements the docs and example edits selected by the Day 10
compatibility documentation design. The goal is to make the earned
compressed-first product model visible in public docs and examples without
claiming direct CSR/CSC solver APIs or deprecating mutable `SparseMatrix`
construction.

## Changed Files

| file | change |
|---|---|
| `README.md` | reframed Quick Start as the tiny hand-written matrix path and pointed CSR/CSC callers to compressed constructors |
| `docs/tutorial.md` | renamed the first construction section so insertion is one construction path rather than the conceptual center |
| `examples/README.md` | added the compressed-input route and program documentation |
| `examples/example_compressed_input.c` | added a small CSR-to-public-matrix-shell one-shot LU example |
| `CMakeLists.txt` | registered `example_compressed_input` for CMake example builds |
| `docs/planning/EPIC_10/SPRINT_101/WORKING_NOTES.md` | recorded Day 11 actions, validation expectations, and exit state |
| `docs/planning/EPIC_10/SPRINT_101/artifacts/day11-docs-and-examples-follow-through.md` | recorded docs/example follow-through evidence |

## Public Wording Updates

| surface | Day 11 result |
|---|---|
| README Quick Start | insertion-based construction is framed as a tiny hand-written matrix path; CSR/CSC callers are told to use compressed constructors |
| tutorial construction section | section title now starts from construction-path choice instead of only matrix mutation |
| examples start map | callers with CSR/CSC arrays now have an explicit example route |
| examples program list | `example_compressed_input` explains copy ownership and normal solver entry through the public matrix shell |

## Example Behavior

`example_compressed_input.c`:

- defines a small tridiagonal matrix in caller-owned CSR arrays;
- builds a public matrix shell with `sparse_from_csr(...)` to demonstrate the
  diagnostic constructor path;
- mutates one caller-owned CSR value after construction and prints that the
  matrix value remains unchanged;
- copies the matrix before one-shot LU factorization;
- solves the same known system as the basic tridiagonal example;
- computes and prints the residual;
- frees the working LU copy and the constructed matrix with `sparse_free(...)`.

## Claim Boundaries Preserved

Day 11 does not claim:

- direct CSR/CSC solver entry APIs;
- no-copy/adopt construction;
- deprecation of insertion-based `SparseMatrix` construction;
- broad compressed solver parity;
- performance superiority from compressed construction alone.

The executable example proves a user workflow, not a new solver family or a
new storage owner model.

## Validation Requirements

Day 11 added a `.c` example and modified `CMakeLists.txt`, so validation must
include:

```bash
make format
make examples
make lint
make test
git diff --check
rg -n "[ \t]+$" README.md docs/tutorial.md examples docs/planning/EPIC_10/SPRINT_101
```

The sprint-level required chain remains:

```bash
make format && make lint && make test
```

## Validation Results

| command | result |
|---|---|
| `make format` | passed |
| `make examples` | passed; built 13 example binaries including `example_compressed_input` |
| `make lint` | passed |
| `make test` | passed |
| `cmake -S . -B build/cmake-sprint101-day11 && cmake --build build/cmake-sprint101-day11 --target example_compressed_input` | passed |
| `./build/example_compressed_input` | passed; printed unchanged `A(0,0)` after CSR array mutation, exact expected solution, and zero residual |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

## Day 11 Conclusion

The compressed-first story now has a public docs route and an executable
example. CSR/CSC callers can see how to move from caller-owned compressed
arrays into the normal public matrix shell, then continue through existing
solver APIs with explicit ownership and compatibility boundaries.
