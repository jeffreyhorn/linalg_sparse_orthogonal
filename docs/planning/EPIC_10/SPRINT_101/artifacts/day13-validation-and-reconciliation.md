# Sprint 101 Day 13 Validation and Product-Model Reconciliation

## Purpose

Day 13 reconciles Sprint 101 implementation, tests, examples, documentation,
and claim wording before closeout. It ties the compressed-first product-model
claim to evidence and records which adjacent claims remain deferred or
explicitly out of scope.

## Evidence Inputs

| input | use |
|---|---|
| Sprint 100 Day 12 public claim audit | claim boundaries and wording discipline |
| Sprint 100 Day 13 handoff package | earned/candidate/non-goal state model |
| Sprint 101 Day 4 API design | selected compressed-first API behavior |
| Sprint 101 Day 6 constructor/import batch | header contract and first implementation proof |
| Sprint 101 Day 8 lifecycle design | ownership, mutation, and repeated-run rules |
| Sprint 101 Day 9 lifecycle batch | README/tutorial ownership wording |
| Sprint 101 Day 11 docs/examples batch | public workflow route and executable compressed-input example |
| Sprint 101 Day 12 regression proof | focused constructor, ownership, and solver-entry tests |

## Public Wording Reconciliation

| surface | reconciled state |
|---|---|
| `include/sparse_csr.h` | simple and diagnostic CSR/CSC constructors are documented as compressed-first entry points that validate and copy caller-owned arrays |
| `include/sparse_matrix.h` | `SparseMatrix` remains the mutable construction and one-shot direct compatibility shell |
| `README.md` | workflow chooser and Quick Start route compressed-input callers to CSR/CSC constructors without deprecating insertion |
| `docs/tutorial.md` | the first matrix section is framed as choosing a construction path; compressed construction is visible before solver transition |
| `examples/README.md` | examples map includes an explicit compressed-input route |
| `examples/example_compressed_input.c` | executable adoption proof demonstrates diagnostic CSR construction, copy ownership, normal LU entry, and cleanup |

## Earned Sprint 101 Claims

| claim | evidence |
|---|---|
| compressed CSR/CSC data has a clear public front door | `sparse_create_from_csr/csc`, `sparse_from_csr/csc`, header comments, README/tutorial/examples wording |
| compressed constructor ownership is copy/build, not adopt | `include/sparse_csr.h`, Day 8 design, Day 9 docs, `test_csr_constructor_copies_caller_owned_arrays`, `test_csc_constructor_copies_caller_owned_arrays`, `example_compressed_input.c` |
| invalid compressed input has focused diagnostic coverage | `test_csr_diagnostic_constructor_rejects_bad_inputs`, `test_csc_diagnostic_constructor_rejects_bad_inputs` |
| compressed-built matrices enter normal solver workflows | CSR-to-LU and CSC-to-Cholesky tests, plus `example_compressed_input.c` |
| mutable `SparseMatrix` construction remains supported compatibility | no signatures removed, insertion docs/examples retained, `include/sparse_matrix.h` compatibility-shell wording preserved |
| examples and docs match implemented behavior | README/tutorial/examples updates plus Day 11 example build and execution |

## Deferred Claims

| deferred item | next owner | reason |
|---|---|---|
| broader direct-solver external oracle evidence | Sprint 102 | outside compressed constructor scope; needs fixture taxonomy and oracle rules |
| solver-family-wide external comparison architecture | Sprint 103 | requires dedicated comparison fixtures and residual criteria |
| backend/runtime observability and performance sentinels | Sprint 104 | not part of constructor/front-door claim |
| reorder/fill and graph evidence clarity | Sprint 105 | unrelated to CSR/CSC construction ownership |
| large source and giant-test maintainability reductions | Sprint 106 | separate maintainability sprint |
| broad solver-selection public guidance | Sprint 107 | should build on Sprint 101-106 evidence |
| first-class support tier publication and ABI decision | Sprint 108 | platform/package scope remains unchanged in Sprint 101 |

## Non-Claims Preserved

Sprint 101 does not claim:

- full replacement or deprecation of the mutable `SparseMatrix` shell;
- direct solver APIs that accept `SparseCsr` or `SparseCsc` directly;
- no-copy or adopt constructors;
- Matrix Market compressed-object publication;
- broad compressed parity across every solver family;
- portable performance superiority;
- broad state-of-the-art replacement status.

## Sprint 102 Dependency Notes

Sprint 102 can rely on:

- a validated compressed-input-to-public-matrix-shell workflow;
- clear constructor ownership and diagnostics;
- focused CSR/CSC regression tests inside the existing `test_csr` executable;
- one executable compressed-input adoption example;
- explicit non-claims around direct CSR/CSC solver APIs and broad solver
  parity.

Sprint 102 should not assume:

- direct CSR/CSC solver entry objects exist;
- compressed constructors avoid `SparseMatrix` ownership;
- all solver families have compressed-input parity proof;
- external oracle coverage is universal.

## Validation Plan

Because Sprint 101 changed `.c` and `.h` files, Day 13 must rerun:

```bash
make format
make lint
make test
```

Additional focused checks for touched surfaces:

```bash
make examples
cmake -S . -B build/cmake-sprint101-day13
cmake --build build/cmake-sprint101-day13 --target example_compressed_input test_csr
./build/example_compressed_input
git diff --check
rg -n "[ \t]+$" README.md docs/tutorial.md examples docs/planning/EPIC_10/SPRINT_101 include/sparse_csr.h tests/test_csr.c
```

## Validation Results

| check | result |
|---|---|
| `make format` | passed |
| `make examples` | passed; all 13 example binaries built |
| `make lint` | passed |
| `make test` | passed |
| `cmake -S . -B build/cmake-sprint101-day13` | passed |
| `cmake --build build/cmake-sprint101-day13 --target example_compressed_input test_csr` | passed |
| `./build/example_compressed_input` | passed; reported retained `A(0,0) = 4.0`, all-ones solution, and zero residual |
| `git diff --check` | passed |
| trailing-whitespace scan | passed; no matches in touched docs, examples, planning artifacts, header, or test file |

## Day 13 Conclusion

Sprint 101 has enough implementation, tests, docs, examples, and validation
evidence to treat the bounded compressed-first product model as earned:
caller-owned CSR/CSC arrays can enter the library through validated
constructors, are copied into caller-owned `SparseMatrix` shells, and then use
existing solver workflows. Adjacent direct-solver, performance, platform, ABI,
and state-of-the-art claims remain deferred or explicit non-claims.
