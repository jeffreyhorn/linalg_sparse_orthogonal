# Day 2 External Reference Inventory

## Purpose

Day 2 collects and classifies adoption-facing external references before any
network validation or documentation edits. Day 3 can use this artifact as the
fixed candidate list for link QA instead of rediscovering references.

## Adoption Documents Scanned

| Document | Literal external URLs found | Named external resources found |
|---|---:|---|
| `README.md` | 0 | Matrix Market, OpenMP, CMake, Make, Linux, macOS, Windows, Homebrew, GCC, Clang, MSVC, ThreadSanitizer, AMD, COLAMD, ND |
| `INSTALL.md` | 0 | CMake, Make, Linux, macOS, Windows, Homebrew, GCC, Clang, MSVC, OpenMP, `pkg-config`, lcov, ThreadSanitizer |
| `docs/tutorial.md` | 0 | Matrix Market |
| `docs/solver_selection.md` | 0 | Matrix Market, CMake, AMD, COLAMD, RCM, ND |
| `docs/matrix_market.md` | 3 | Matrix Market, SuiteSparse Matrix Collection |
| `docs/algorithm.md` | 0 | SuiteSparse, BLAS, LAPACK, OpenMP, AMD, COLAMD, METIS, CMake, macOS, Homebrew |
| `benchmarks/README.md` | 0 | SuiteSparse, BLAS, OpenMP, Linux, macOS, Windows, AMD, COLAMD |
| `examples/README.md` | 0 | Matrix Market, SuiteSparse, CMake, AMD, COLAMD, ND |

## Literal Link Inventory

| ID | File | Line context | URL | Category | Workflow role | Day 3 action |
|---|---|---|---|---|---|---|
| L1 | `docs/matrix_market.md` | Matrix Market format overview | `https://math.nist.gov/MatrixMarket/formats.html` | Matrix Market | Informational reference required to explain the supported format contract | Network-check and keep, replace, or fence if stale. |
| L2 | `docs/matrix_market.md` | SuiteSparse Matrix Collection mention near top-level overview | `https://sparse.tamu.edu/` | SuiteSparse | Informational source for obtaining external `.mtx` inputs | Network-check and keep, replace, or fence if stale. |
| L3 | `docs/matrix_market.md` | SuiteSparse matrices usage section | `https://sparse.tamu.edu/` | SuiteSparse | Workflow-adjacent source for user-provided Matrix Market fixtures | Network-check and keep, replace, or fence if stale. |

## Named External Resource Classification

| Resource | Primary files | Category | Workflow role | Network-check disposition |
|---|---|---|---|---|
| Matrix Market | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/matrix_market.md`, `examples/README.md` | Matrix format | Public load/save format and adoption workflow | Covered by L1; no extra network target unless docs add another URL. |
| SuiteSparse Matrix Collection | `docs/matrix_market.md`, `docs/algorithm.md`, `benchmarks/README.md`, `examples/README.md` | External corpus | Fixture/source reference and benchmark context | Covered by L2/L3 where URL appears; named-only mentions are not network targets. |
| CMake | `README.md`, `INSTALL.md`, `docs/solver_selection.md`, `examples/README.md` | Build/toolchain | Maintained build and consumer path | Do not network-check; command and toolchain references only. |
| Make | `README.md`, `INSTALL.md` | Build/toolchain | Recommended local build path and Unix install path | Do not network-check; command references only. |
| OpenMP | `README.md`, `INSTALL.md`, `docs/algorithm.md`, `benchmarks/README.md` | Runtime/toolchain | Optional parallel runtime and performance context | Do not network-check; no external URL in adoption docs. |
| GCC, Clang, MSVC | `README.md`, `INSTALL.md` | Compiler/toolchain | Supported compiler context and reviewed-lane wording | Do not network-check; no external URL in adoption docs. |
| Linux, macOS, Windows | `README.md`, `INSTALL.md`, `benchmarks/README.md` | Platform | Reviewed/support-tier claim context | Do not network-check; platform names are claim guardrails, not URLs. |
| Homebrew | `README.md`, `INSTALL.md`, `docs/algorithm.md` | Toolchain/package ecosystem | macOS GCC/libomp/lcov context; no package-manager support claim | Do not network-check; no install recipe or support URL is present. |
| `pkg-config` | `INSTALL.md` | Build integration tool | Unix-side Make install consumer metadata | Do not network-check; command/reference only. |
| lcov | `INSTALL.md` | Coverage tool | Coverage backend documentation | Do not network-check; command/reference only. |
| ThreadSanitizer | `README.md`, `INSTALL.md` | Toolchain sanitizer | Reviewed Linux lane and macOS non-claim context | Do not network-check; no external URL in adoption docs. |
| Valgrind, perf, gprof | `benchmarks/README.md` if present in future edits | Profiling tools | Potential benchmark interpretation context | No current literal URL target from Day 2 scan. |
| AMD, COLAMD, RCM, ND, METIS | `README.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `benchmarks/README.md`, `examples/README.md` | Algorithms/libraries | Reordering and algorithm-reference context | Do not network-check; named algorithm/library references only. |
| BLAS, LAPACK | `docs/algorithm.md`, `benchmarks/README.md` | Numeric backend context | Performance and algorithm background | Do not network-check; no external URL in adoption docs. |

## Network-Check Candidate List for Day 3

| Candidate | URL | Reason |
|---|---|---|
| Matrix Market format page | `https://math.nist.gov/MatrixMarket/formats.html` | Adoption docs use it as the public definition of the `.mtx` coordinate format. |
| SuiteSparse Matrix Collection landing page | `https://sparse.tamu.edu/` | Adoption docs direct users toward external Matrix Market matrices and benchmark context. |

`https://sparse.tamu.edu/` appears twice in `docs/matrix_market.md`; Day 3
should validate the URL once and apply the result to both references.

## Excluded-Link Rationale

| Exclusion | Rationale |
|---|---|
| Local Markdown links such as `docs/matrix_market.md` or `benchmarks/README.md` | Day 2 is external-reference inventory; local relative-link checks belong to documentation hygiene when those files change. |
| Build commands such as `make`, `cmake`, `ctest`, `pkg-config`, and `brew` | They are commands or tool names, not external URL references. |
| Platform names such as Linux, macOS, and Windows | These are support-claim boundaries, not network-checkable references. |
| Compiler/runtime names such as GCC, Clang, MSVC, OpenMP, lcov, and ThreadSanitizer | The adoption docs contain no external URLs for these names. |
| Algorithm names such as AMD, COLAMD, RCM, ND, and METIS | The current adoption docs use these as algorithm or implementation context without external URLs. |
| Benchmark, planning, and fixture artifact paths | These are repository-local evidence references, not external links. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Day 3 can run link QA without rediscovering sources | Complete; two unique external URL targets are listed. |
| All adoption-facing external references have an owner or exclusion reason | Complete; literal URLs have Day 3 actions, named resources have exclusion or coverage rationale. |
| No documentation content is changed before validation | Complete; Day 2 only adds Sprint 116 planning artifacts. |

## Validation Notes

- Day 2 changed Sprint 116 planning documentation only.
- No adoption-facing documentation content was edited.
- No `.c` or `.h` files were modified.
