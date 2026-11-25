# Epic 21 Project Plan: Arithmetic, Numerical Conformance and Evidence-Backed Static Adoption

**Sprints:** 227–236 (10 sprints, 14 days each). **Planning capacity:** 168 hours maximum per sprint; 140 hours of estimated work and 28 hours of contingency each. **Total estimated work:** 1,400 hours across 140 days, below the 1,680-hour ceiling. Hours are engineering estimates, not elapsed CI or approval wait time.

This plan follows the [2026-11-23 code review](reviews/review-codex-2026-11-23.md) and its [dependency-ordered gap checklist](reviews/todo-codex-2026-11-23.md). It carries forward Epic 20's blocked external cell, static candidate and narrow timing interpretation without relabeling them complete. A sprint is complete only when its named capability and acceptance evidence pass. Failed hosted runs, absent approval, historical SHA evidence and merely drafted policies remain blocked. Public package scope stays static-only; generated API HTML stays local-only.

## Sprint 227: Sorted Matrix Arithmetic and Permutation Safety

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close the `sparse_add` repeated-scan cost and make arithmetic on unsupported permutation states fail explicitly or behave logically by a documented contract.

### Prerequisites from Previous Sprints

- Sprint 218's sorted matmul output builder and allocation-failure tests.
- Sprint 219's width-aware build and existing arithmetic public header contract.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 227.1 | Contract Freeze | Record physical/logical index, aliasing, drop threshold, ownership and failure semantics with dense-oracle baseline cases. | 18 |
| 227.2 | Permutation Guard | Implement one compatible arithmetic policy for nonidentity permutations across add, in-place add and matmul; test both operands. | 30 |
| 227.3 | Sorted Output Builder | Replace repeated `sparse_insert` calls in sorted add output with append/link construction and preserve row/column invariants. | 28 |
| 227.4 | Failure And Alias Matrix | Sweep allocations, cancellation, rectangular/wide rows, aliases and partial-update behavior; add independent dense comparisons. | 26 |
| 227.5 | Scaling Evidence | Retain pinned wide-row time and peak-memory before/after results without portable speed claims. | 20 |
| 227.6 | Quality Closure | Pass C format/lint/test, CMake parity, sanitizer and affected API documentation checks. | 18 |

### Deliverables

- Sorted addition output path, explicit permutation behavior and failure/alias regression corpus.
- Reproducible selected arithmetic profile and updated public contract.

### Completion Criteria

No repeated head scan per sorted output node; unsupported permutation use cannot return a misleading success; numerical and failure semantics match the frozen oracle; all gates pass.

### Total Estimate

140 hours.

## Sprint 228: Default Low-Rank Reconstruction Memory

**Duration:** 14 days, approximately 140 hours.

**Goal:** Remove the additional `m×n` reconstruction accumulator from the default sparse low-rank output path while preserving results and error behavior.

### Prerequisites from Previous Sprints

- Sprint 227's output/failure invariant tests; existing optional SVD outer-product implementation and corpus.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 228.1 | Baseline And Contract | Pin shapes, ranks, tolerances, output ordering, RSS and time for dense and optional reconstruction modes. | 20 |
| 228.2 | Differential Corpus | Compare zero, repeated/clustered, rank-deficient and sparse-output cases with an independent dense reference. | 26 |
| 228.3 | Default Path Promotion | Correct the outer route as needed, make it the maintained default and retain a controlled compatibility selector. | 30 |
| 228.4 | Failure Lifecycle | Sweep allocation and numeric failures; check result ownership, cleanup and repeatability. | 24 |
| 228.5 | Memory Accounting | Record reconstruction scratch separately from dense SVD factors and clarify remaining memory limits in docs. | 20 |
| 228.6 | Quality Closure | Pass full C quality, focused SVD corpus, CMake parity and sanitizer checks. | 20 |

### Deliverables

- Lower-memory default reconstruction with differential/failure evidence.
- Precisely scoped memory and output contract documentation.

### Completion Criteria

Default reconstruction has no `m×n` accumulator; verified output and error behavior matches the accepted contract; full SVD's remaining dense storage is stated plainly.

### Total Estimate

140 hours.

## Sprint 229: CG Family Numerical Contract

**Duration:** 14 days, approximately 140 hours.

**Goal:** Complete one coherent CG status, iteration and residual contract across assembled one-shot and reusable-handle, matrix-free and block entry points.

### Prerequisites from Previous Sprints

- Sprint 217's scaled-norm/finite diagnostics and PR #241 numerical regressions.
- Sprint 228's completed C quality baseline.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 229.1 | Status Specification | Freeze initial/post-update convergence, zero budget, breakdown, callback/history, reusable-handle and block aggregation semantics. | 18 |
| 229.2 | Independent CG Oracle | Build SPD, extreme-scale and invalid-preconditioner corpus with status-qualified residual/output expectations and expected iteration counts. | 26 |
| 229.3 | Full-Surface Repair | Fix one-shot, reusable-handle, matrix-free and block discrepancies, sharing helpers only where contracts match. | 32 |
| 229.4 | Handle, Failure And Thread Cases | Test explicit preparation, underprepared growth, repeated reuse after success and failure, allocation cleanup, reentrancy and callbacks without hidden state leakage. | 26 |
| 229.5 | Hosted Compiler Matrix | Register focused corpus on GCC, Apple Clang and MSVC and retain exact-head results. | 20 |
| 229.6 | Quality Closure | Run full C quality, sanitizer, CMake parity and document limits. | 18 |

### Deliverables

- CG-family conformance corpus and corrected implementation where needed.
- Public status contract with cross-compiler/failure evidence.

### Completion Criteria

The one-shot, reusable-handle, matrix-free and block CG entry points satisfy the declared initial, completed-step and final-verification rules on the frozen corpus. Residual and output assertions follow each return status: selected computed numeric failures require `converged=0` and `residual_norm=NaN`, invalid input and other hard errors receive only their documented guarantees, and completed selected solves receive the declared true-residual checks. Prepared, underprepared and grown handles remain reusable after both successful and failed solves; no claim is made for other Krylov families.

### Total Estimate

140 hours.

## Sprint 230: MINRES Breakdown and Residual Contract

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close MINRES's numerical status and true-residual handling across one-shot, reusable-handle and block entry points as a separate complete solver-family owner.

### Prerequisites from Previous Sprints

- Sprint 229's oracle/status vocabulary and callback test helpers.
- Sprint 217 MINRES finite checks and PR #241 zero-budget/final-verification review cases.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 230.1 | Public Contract | Specify initial convergence, zero budget, preconditioner BADARG, Lanczos breakdown, reusable-handle, block aggregation, counts and residual fields. | 18 |
| 230.2 | Symmetric Oracle | Build independent indefinite/semidefinite, exact-breakdown and extreme-scale reference cases for one-shot, handle and mixed-column block solves. | 28 |
| 230.3 | Recurrence And Exit Repair | Remove unused products on exit and correct numeric/breakdown and final-verification paths across the public MINRES surface as tests require. | 30 |
| 230.4 | Handle And Block Lifecycle Tests | Exercise explicit preparation, underprepared growth, reuse after failures, mixed converged/non-converged columns, aggregate count/residual rules, callbacks, histories, allocation failures and retries. | 24 |
| 230.5 | Portability Proof | Retain focused GCC/Clang/MSVC and CMake/Make output for exact source head. | 22 |
| 230.6 | Quality Closure | Pass format/lint/test, sanitizer and updated API/solver guidance checks. | 18 |

### Deliverables

- MINRES oracle and stable status/count/residual contract.
- Cross-build and sanitizer evidence for the selected family.

### Completion Criteria

The one-shot and prepared, underprepared and repeatedly reused handle paths satisfy the same status-qualified contract, including reuse after failures. Block MINRES satisfies per-column convergence plus documented aggregate count and residual rules for mixed outcomes. The frozen corpus passes with consistent public diagnostics; GMRES and BiCGSTAB breadth remains explicitly outside this sprint.

### Total Estimate

140 hours.

## Sprint 231: One Complete 64-Bit Input-to-Solve Route

**Duration:** 14 days, approximately 140 hours.

**Goal:** Prove Matrix Market input through CSC conversion and one direct solve under a genuine 64-bit build and installed consumer.

### Prerequisites from Previous Sprints

- Sprint 219 installed width guard, metadata and selected 64-bit lanes.
- Sprints 229–230 numerical/failure helpers; existing Matrix Market and CSC tests.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 231.1 | Route Inventory | Map every index conversion, pointer count and allocation on the chosen parse/CSC/solve path. | 20 |
| 231.2 | Overflow Rejection | Add large-metadata and malformed-index cases that exercise 64-bit arithmetic without impossible allocations. | 28 |
| 231.3 | Numerical Route | Test moderate actual 64-bit matrices against independent solutions and 32-bit behavior. | 26 |
| 231.4 | Installed Consumers | Validate clean-prefix Make/CMake static consumers at both widths and opposite-width rejection. | 24 |
| 231.5 | Hosted Scope | Run named Linux/macOS and maintained Windows subset as supported; record exact source and exclusions. | 22 |
| 231.6 | Quality Closure | Pass full C chain, CMake parity, sanitizer and width-specific docs checks. | 20 |

### Deliverables

- One positively and negatively tested 64-bit input-to-solve route.
- Installed width and platform evidence with explicit algorithm exclusions.

### Completion Criteria

The named route works with real 64-bit types and rejects unrepresentable metadata safely; no all-algorithm or arbitrary huge-matrix claim follows.

### Total Estimate

140 hours.

## Sprint 232: C And Evidence Tool Ownership with CI Inventory

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close one bounded C implementation/test ownership seam and make one oversized report/policy tool easier to maintain while preserving public compatibility, fail-closed output and complete hosted test registration.

### Prerequisites from Previous Sprints

- Sprint 220 structured workflow guard and Sprint 221 v1 external validator.
- Sprint 231 stable hosted and installed evidence inventory.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 232.1 | Compatibility Freeze | Freeze the selected C seam's public ABI/source behavior and independent invariants plus the selected tool's CLI/JSON/exit codes, real inputs and mutation verdicts. | 20 |
| 232.2 | C Ownership Extraction | Move one named responsibility from an oversized C implementation and its matching oversized test source into a bounded internal module and focused test owner; preserve public ABI/source compatibility. | 28 |
| 232.3 | Evidence Module Extraction | Separate the selected tool's parser/validator/renderer responsibilities without changing accepted evidence semantics. | 26 |
| 232.4 | Inventory And Adversarial Contract | Replace manual workflow-count assumptions with a declared checked registry and add stale provenance, renamed workflow, malformed input and diagnostic stability cases. | 26 |
| 232.5 | CI Wiring | Register affected Python suites and local-only guard in Linux/macOS lanes with pinned dependencies. | 20 |
| 232.6 | Quality Closure | Run C format/lint/test, Make/CMake parity, focused sanitizer and independent invariant tests plus policy/differential/mutation suites, docs-check and exact-head hosted checks. | 20 |

### Deliverables

- One bounded C implementation/test owner with independent invariant fixtures.
- Smaller bounded policy/evidence modules, frozen compatibility fixtures and CI inventory proof.

### Completion Criteria

The selected C responsibility has one named internal module and focused test owner, its invariants are tested independently of implementation details, the original oversized implementation/test pair no longer owns that seam, public ABI/source behavior is unchanged, and parity plus sanitizer gates pass. The selected tool's accepted/rejected outputs match the frozen contract; every affected test is run by CI and missing inventory items fail closed.

### Total Estimate

140 hours.

## Sprint 233: One Accepted Two-Sided CHOLMOD Cell

**Duration:** 14 days, approximately 140 hours.

**Goal:** Resolve Sprint 222's provenance and rights gap for exactly one independently reviewed two-sided correctness cell.

### Prerequisites from Previous Sprints

- Sprint 221 v1 schema and Sprint 222 blocked pilot/raw numerical records.
- Sprint 232 tool/CI contract; an available CHOLMOD environment and attribution reviewer.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 233.1 | Cell And Rights Intake | Pin matrix/RHS/conversion bytes, exact claim, attribution and retention decision before public output. | 18 |
| 233.2 | Provenance Capture | Capture actually loaded CHOLMOD/BLAS/transitive versions and hashes, compiler, flags, ordering, threads and host. | 22 |
| 233.3 | Two Fresh Executions | Run and retain two same-cell project/CHOLMOD pairs with original streams, statuses, solutions and failure attempts. | 30 |
| 233.4 | Independent Numerical Check | Compute residual/backward error, validate complete canonical IDs and test provenance tampering. | 28 |
| 233.5 | Independent Acceptance | Obtain raw-bundle/method and rights review; approve only the exact one-cell wording or record a blocker. | 22 |
| 233.6 | Quality Closure | Run pilot, validator, documentation and affected CI gates; retain artifacts and hashes. | 20 |

### Deliverables

- Validator-accepted observed cell, provenance and rights record if all gates pass; otherwise an explicit blocked packet.

### Completion Criteria

The cell is closed only with repeated two-sided numerical success, complete verified environment, independent review and rights clearance. A blocked packet is not a correctness result.

### Total Estimate

140 hours.

## Sprint 234: Exact Static Source-Candidate Package

**Duration:** 14 days, approximately 140 hours.

**Goal:** Produce and independently verify one immutable static source candidate and downstream install route, subject to named release-owner approval.

### Prerequisites from Previous Sprints

- Sprints 227–232 merged or explicitly excluded from the selected release scope.
- Sprint 233 outcome declared; Sprint 214 static checklist and Sprint 226 exploratory archive/install procedure.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 234.1 | Candidate Freeze | Approve version/changelog, static platform tier, owners/rollback and clean exact SHA. | 20 |
| 234.2 | Independent Archives | Build two source archives from that tree, compare bytes and review source/license manifest. | 24 |
| 234.3 | Extracted Install | Install Make/CMake static artifacts from archives in clean prefixes and prove 32/64 consumers and mismatch rejection. | 28 |
| 234.4 | Hosted Candidate Matrix | Collect Linux/macOS/Windows required checks, test counts and failure receipts for the exact candidate. | 28 |
| 234.5 | Security/Release And Timing Review | Obtain required signers, supply-chain/rollback evidence and a go/no-go record. Timing/product owners must independently accept exact-candidate s225-v5 evidence or sign a narrower static scope that keeps timing blocked and makes no speed claim; keep unsigned results blocked. | 20 |
| 234.6 | Quality Closure | Reconcile archive/installed manifests, docs and final candidate-specific gates. | 20 |

### Deliverables

- Reproducible candidate source archive, installed consumers and exact-SHA platform packet.
- Named signed readiness or explicit blocked decision, without automatic publication.

### Completion Criteria

One SHA passes archive, install, platform and approval gates before `release-ready` wording. For that exact candidate, timing/product owners also either independently accept the s225-v5 methodology, original bundles and recomputed decision, or sign a narrower static scope that leaves timing blocked and makes no speed claim. Sprint 235's later kernel comparison is separate evidence and cannot substitute for this inherited decision. A source CI pass or exploratory archive alone is insufficient.

### Total Estimate

140 hours.

## Sprint 235: Differing-Source Kernel Performance Evidence

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close the gap between the current identical-binary timing smoke test and one meaningful, correctness-qualified kernel comparison.

### Prerequisites from Previous Sprints

- Sprint 225 v5 protocol and PR #250 raw review; Sprint 234 exact-candidate/source identity rule.
- A selected kernel changed in Sprints 227–231 and a pinned pre-change baseline; Sprint 235 does not introduce a new code change after Sprint 234's freeze.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 235.1 | Claim And Workload Lock | Pre-register changed source, matrix set, correctness check, host class and primary metric. | 18 |
| 235.2 | Binary Provenance | Retain executable bytes or reproducible build attestation, source/build hashes and identical-harness proof. | 26 |
| 235.3 | Controlled Collection | Run warmup and alternating repeated pairs on named hosts; retain raw streams, failures, time and memory. | 30 |
| 235.4 | Statistical Decision | Apply locked median/dispersion thresholds and show a counterexample or neutral workload. | 24 |
| 235.5 | Independent Review | Recompute bundles, compare correctness and approve only named kernel/host wording. | 22 |
| 235.6 | Quality Closure | Run affected C/benchmark/CI/docs gates and archive the accepted or blocked decision. | 20 |

### Deliverables

- One differing-source comparison with complete raw/binary provenance and independent decision.

### Completion Criteria

Measured candidate and baseline differ in the intended kernel; correctness and predeclared evidence gates pass. Identical binaries or a noisy run do not count as a speed result. Any corrective source or checked-in documentation edit requires a new candidate freeze and affected Sprint 234 proof.

### Total Estimate

140 hours.

## Sprint 236: Installed First Use and Epic 21 Closeout

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close the supported static installed first-use route from the selected artifact and reconcile every Epic 21 capability/claim against its evidence.

### Prerequisites from Previous Sprints

- Sprints 227–235 merged, completed or explicitly blocked with retained evidence.
- Sprint 234 candidate archive and owner decision; Sprint 233/235 claim boundaries.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 236.1 | Clean-User Replay | Exercise archive-to-install-to-Matrix Market/CSR/CSC first solve on supported hosts and widths. | 20 |
| 236.2 | Current Guidance | Deduplicate README/INSTALL/cookbook/solver selection into a checked capability/support table. | 26 |
| 236.3 | API/Example Audit | Test links, all installed headers, maintained examples, docs policy and ownership/error guidance. | 24 |
| 236.4 | Evidence Reconciliation | Map every sprint item to exact SHA/raw receipt, status, signer and non-claim; resolve contradictory wording. | 24 |
| 236.5 | Final Candidate Decision | Rerun affected gates after edits; decide release-ready versus blocked on one immutable source and archive. | 26 |
| 236.6 | Epic Closeout | Publish retrospective, residual queue, rollback owner and review packet with final checks. | 20 |

### Deliverables

- Replayed static installed first-use path and concise current support guidance.
- Epic 21 evidence ledger, retrospective and signed release decision or explicit blocked handoff.

### Completion Criteria

The first-use route and docs close only after exact-artifact replay and checks. Release status is `ready` only if all candidate and independent approvals pass on the same SHA; otherwise it remains `BLOCKED` with named missing evidence.

### Total Estimate

140 hours.

## Portfolio Boundaries And Exit Rule

Ten sprints × 140 estimated hours = **1,400 hours**; ten × 168 maximum = **1,680 hours**. Each sprint has 28 hours of contingency for CI variance, numerical investigation and review. If Sprint 233 external rights or Sprint 234 release approval is unavailable, preserve the blocked result and spend remaining time on an already-funded regression or adoption check. Do not expand the portfolio to shared-library ABI, binary/package-manager publication, generated-HTML hosting, GPU/distributed operation, broad external solver parity or general state-of-the-art ranking without a new plan and acceptance program. A successful sprint checklist is a bounded technical result; only separately reviewed, exact-scope evidence may change public claims.
