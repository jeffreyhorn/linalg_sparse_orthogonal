# Sprint 187 Day 10: Maintainability and Reliability Gates

## Purpose

Define measurable acceptance gates for Sprint 193 selected large review-surface
reduction and Sprint 195 selected reliability/failure-path proof. These gates
make code-quality work reviewable without silently changing behavior, public
API, support claims, or validation scope.

## Current Code-Quality Baseline

Current large review surfaces by line count include:

| Candidate | Current size signal | Planning interpretation |
| --- | ---: | --- |
| `tests/test_qr.c` | 3970 lines | Highest test review surface; solver semantics and oracle tolerances make broad extraction risky. |
| `tests/test_ldlt_csc.c` | 3469 lines | Recently reduced through Sprint 185 helper extraction; has existing guard model. |
| `tests/test_integration.c` | 3279 lines | Broad public lifecycle coverage; extraction risk is high because many solver workflows meet here. |
| `tests/test_svd.c` | 3029 lines | Large numerical proof owner with rank/vector/sign claim boundaries. |
| `tests/test_ldlt.c` | 3006 lines | Direct LDLT proof surface with ordering, solve, and error-path semantics. |
| `tests/test_etree.c` | 2962 lines | Large structural/reordering proof surface. |
| `tests/test_iterative.c` | 2929 lines | Existing allocation-failure owner for iterative repeated-run handles. |
| `tests/test_graph.c` | 2764 lines | Large graph/reorder structural guardrail owner. |
| `src/sparse_ldlt_csc.c` | 2095 lines | Largest implementation candidate; algorithmic risk is higher than test-helper extraction. |
| `src/sparse_lu_csr.c` | 1594 lines | Large implementation candidate with allocation/fill-in complexity. |

Current maintained gate patterns:

- `make ldlt-csc-helper-guard` protects Sprint 185 helper header extraction.
- `make source-list-check` reconciles `build-metadata/library_sources.txt`,
  `Makefile` `LIB_SRCS`, and CMake `add_library(...)` membership.
- `make iterative-allocation-failure-gate` proves selected iterative
  repeated-run handle allocation failures.
- `make matmul-allocation-failure-gate` proves selected `sparse_matmul()`
  stale-output and retry behavior.
- `make quality-review-full` is the strongest local reviewed baseline.

## Sprint 193 Gate: Selected Large Review-Surface Reduction

Sprint 193 must select exactly one source or test cluster. It must reduce
review surface with behavior-preserving edits and a focused guard.

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Candidate ranking | Candidates are ranked by line count, algorithm risk, helper ownership, current tests, registration impact, user-facing importance, and prior review history. | Selection is based only on file size or convenience. |
| Single cluster selection | Exactly one cluster is selected with named owner files and out-of-scope neighbors. | Multiple clusters are partially touched or ownership remains vague. |
| No-behavior-change record | Public declarations, status codes, diagnostics, numerical tolerances, fixtures, test names, `RUN_TEST(...)` ordering where meaningful, and support claims are explicitly preserved. | Refactor changes behavior without a separate selected implementation item. |
| Boundary design | Helper/source boundaries define ownership, include direction, internal linkage, cleanup ownership, global override restoration, and registration rules. | Extracted helpers become ad hoc shared APIs or duplicate ownership. |
| Focused implementation | Edits are the minimum needed to move repeated setup/helper code or split one implementation owner while preserving call sites. | The change becomes a broad rewrite, style sweep, or architecture redesign. |
| Guard | A focused guard verifies helper/source presence, include ownership, Make/CMake registration, and source-list boundaries. | Review-surface reduction can drift silently after merge. |
| Documentation | Maintainer guidance names the new boundary and the validation required before changing it. | Future contributors cannot tell where new helpers or test cases belong. |

## Sprint 193 Ranking Criteria

Use this scoring model before selecting the cluster:

| Criterion | High score means |
| --- | --- |
| Review surface size | The file or cluster is large enough that extraction materially reduces reviewer load. |
| Algorithm risk | The code touches numerical decisions, ordering, factorization, ownership, or cleanup paths. |
| Helper reuse | Repeated setup/helper logic can move behind a clear family-local boundary. |
| Existing proof coverage | A focused proof-owner binary already exists and can detect behavior drift. |
| Registration impact | Make/CMake/source-list changes are either unnecessary or mechanically guardable. |
| User-facing importance | The cluster affects maintained public workflows rather than obscure internals. |
| Extraction feasibility | One 14-day sprint can finish the boundary, guard, docs, and validation. |

Preferred target shape is a test-helper extraction or one narrow implementation
split. Broad storage replacement, solver redesign, public API changes, and
cross-family helper frameworks are rejected for Sprint 193.

## Sprint 193 No-Behavior-Change Invariants

Sprint 193 must preserve:

- public header declarations and documented public semantics;
- status/error-code precedence;
- diagnostic strings unless the sprint explicitly records a docs/test update;
- numerical fixture values and tolerances;
- external reference helper behavior and skip semantics;
- `RUN_TEST(...)` coverage and proof-owner registration;
- CMake and Makefile test-registration parity;
- library source membership in `build-metadata/library_sources.txt`,
  `Makefile`, and CMake;
- package, platform, ABI, comparison, performance, and state-of-the-art
  non-claims;
- process-global state restoration for any test override or environment
  variable touched by the selected cluster.

Any intentional behavior change must be split into its own reviewed item with
tests and claim wording; it cannot ride along as review-surface reduction.

## Sprint 193 Required Validation

Minimum validation for any selected cluster:

```sh
make format
make lint
make test
```

Additional required checks by surface:

| Surface touched | Additional required validation |
| --- | --- |
| Library `.c` source list | `make source-list-check`; CMake configure/build or `make quality-review-cmake-compile`. |
| Test registration | `make source-list-check` if source lists are touched; CMake `ctest -N` parity through `make quality-review-cmake-compile`. |
| LDLT CSC helper layout | `make ldlt-csc-helper-guard`; focused `build/test_ldlt_csc` execution. |
| Large matrix/reorder guardrail owners | `make large-matrix-guardrails` when graph/reorder guardrail behavior changes. |
| Public headers | Header docs guards where applicable plus Doxygen/API docs checks selected by the change. |
| Workflow or package side effects | The matching Day 7, Day 8, or Day 9 gate commands. |

## Sprint 195 Gate: Selected Reliability And Failure-Path Proof

Sprint 195 must select exactly one reliability owner. It must add deterministic
failure-path proof for that owner without implying exhaustive reliability
coverage.

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Owner selection | One allocation-heavy or failure-prone owner is selected by allocation density, cleanup complexity, user impact, current test gaps, and deterministic hook availability. | Reliability work is spread across several owners without complete proof. |
| Invariant record | Cleanup ownership, publication points, stale-output behavior, retry semantics, and global-state restoration are documented before implementation. | Tests are added without stating what failure semantics they prove. |
| Failure injection | Existing `sparse_alloc_test_fail_after(...)`/`sparse_alloc_test_reset()` hooks are used or an owner-local deterministic fail-at-count mechanism is added. | Failure tests depend on real allocator exhaustion or flaky timing. |
| Failed allocation proof | Tests force each selected allocation point to fail and verify the returned error/status. | Only the first allocation site or happy path is tested. |
| Cleanup proof | Tests prove partially allocated state is released or remains caller-owned as documented. | Failed paths leak ownership or leave ambiguous state. |
| Stale-output suppression | Tests prove output pointers, handles, or result buffers are cleared or left unchanged according to the owner contract. | A failed call can publish partial stale success-looking output. |
| Retry proof | Tests prove a subsequent call succeeds after reset or after caller-managed cleanup. | Failure poison persists into later calls. |
| Global-state restoration | Any process-global override, environment variable, fail hook, backend selector, or kernel override is restored before assertion early returns can exit the helper. | Failure leaves process-global state contaminated for subsequent tests. |
| Focused gate | A Make target, CTest label, or guard script owns the selected reliability proof. | The proof is only reachable through the full test suite and can drift silently. |

## Sprint 195 Owner Selection Criteria

Preferred owners have:

- multiple internal allocations or growth paths;
- public output publication points;
- cleanup labels or partial-state transitions;
- user-visible retry behavior;
- deterministic fail-at-count coverage potential;
- an existing focused test binary or easy CTest label;
- no need for broad architectural redesign.

Existing proof models:

| Existing owner | Proof model |
| --- | --- |
| `tests/test_iterative.c` | CG, GMRES, and MINRES repeated-run handle prepare/growth allocation failures through `make iterative-allocation-failure-gate`. |
| `tests/test_matmul.c` | `sparse_matmul()` workspace allocation failure, stale-output clearing, and retry recovery through `make matmul-allocation-failure-gate`. |
| `tests/test_integration.c` | Public lifecycle refactor retry behavior for LU, Cholesky CSC, and LDLT workflows. |

Sprint 195 should choose a new owner or a clearly narrower uncovered lane
inside an existing owner; it should not relabel the existing iterative or
matmul proof as new evidence.

## Sprint 195 Required Validation

Minimum validation:

```sh
make format
make lint
make test
```

Additional required checks by owner:

| Owner touched | Additional required validation |
| --- | --- |
| Iterative repeated-run handles | `make iterative-allocation-failure-gate`; matching CTest allocation-failure label when using CMake. |
| `sparse_matmul()` | `make matmul-allocation-failure-gate`; `python3 tests/test_matmul_allocation_failure_gate_registration.py`. |
| New focused reliability owner | New focused Make target or CTest label plus a registration guard comparable to the matmul gate. |
| Library source additions | `make source-list-check`; CMake parity through `make quality-review-cmake-compile`. |
| Process-global overrides or environment variables | Focused tests that force early-return/failure paths and prove restoration before subsequent assertions. |
| Public docs or headers | Relevant docs/header checks selected by the changed surface. |

## Global-State Restoration Rule

Any helper that changes process-global state must restore that state before a
test assertion macro or helper can return early. Acceptable patterns include:

- store the operation status in a local variable;
- restore the global override or environment variable;
- then assert or return the stored status;
- use cleanup labels for multi-step tests;
- reset allocation fail hooks in every failure and success path.

This rule applies to kernel overrides, backend selectors, environment
variables, allocation fail counters, report-generator globals, and temporary
working-directory state.

## Focused Gate Requirements

Any new Sprint 193 or Sprint 195 focused gate must:

- be callable from `make`;
- print a clear pass/fail owner name;
- fail when the selected proof owner is deregistered;
- fail when helper/source files drift into the wrong registration surface;
- fail when required tests are removed from the proof-owner binary;
- document whether CMake/CTest parity is required;
- be referenced from `docs/maintainer_guide.md` or a selected owner artifact.

## Retained Non-Claims

Sprint 193 does not claim:

- new solver behavior;
- broader numerical correctness;
- public API or ABI changes;
- package or platform support;
- performance improvement;
- external-library parity;
- state-of-the-art status.

Sprint 195 does not claim:

- exhaustive allocation-failure coverage;
- all-solver reliability;
- concurrency safety;
- global lifecycle correctness;
- memory-sanitizer cleanliness beyond the selected validation;
- package, platform, ABI, performance, comparison, or state-of-the-art proof.

## Completion Gates

Sprint 193 is complete when one selected review-surface cluster is reduced,
no-behavior-change invariants are preserved, focused guard coverage is added
or updated, source-list/registration parity is validated, maintainer guidance
is updated, and the required C quality gate passes.

Sprint 195 is complete when one selected reliability owner has deterministic
failure injection, failed-allocation checks, cleanup proof, stale-output
suppression, retry proof, global-state restoration evidence where applicable,
a focused gate, claim-safe documentation, and the required C quality gate
passes.

Either sprint must stop if validation fails, if a behavior/API/support claim is
expanded without explicit selection, or if a focused gate cannot prove the new
owner boundary.

## Validation

Day 10 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
