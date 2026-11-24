# Epic 21 Gap-Closure Checklist (2026-11-23)

Companion to the [code review](review-codex-2026-11-23.md). Steps are in dependency order and map to Sprints 227–236. A step is **closed** only when its named capability, focused negative tests, required quality/hosted evidence and public wording agree. A proposed design, a successful schema check, or a blocked receipt is not a substitute. Each sprint budgets 140 engineering hours and leaves 28 hours below the 168-hour ceiling. External owner decisions are gates, never inferred from agent output.

## 1. Finish sorted matrix arithmetic (Sprint 227; R21-01/02)

1. Pin `sparse_add`, `sparse_add_inplace` and `sparse_matmul` semantics for physical versus logical indices, drop threshold, aliases, output ownership and failure state. Record dense-oracle results and wide-row scaling before edits.
2. Decide the compatible contract: either reject nonidentity permutations before mutation or implement logical arithmetic consistently for all three entry points. Add explicit negative/positive fixtures. Preserve current public ABI.
3. Move sorted `sparse_add` output through an append/link builder, preserving row and column order, duplicate combination and zero dropping. Audit `sparse_add_inplace` aliasing and rollback; make failures atomic if feasible under its documented API or state the remaining partial-update contract precisely.
4. Sweep each new allocation failure and compare randomized, rectangular, cancellation and permuted cases to an independent dense oracle. Record wide-row time and peak memory with repeatable commands.
5. Pass `make format && make lint && make test`, CMake/CTest parity and affected sanitizer/installed-consumer checks.

**Done when:** addition no longer restarts a row/column scan for every merged output and arithmetic cannot silently succeed on an unsupported permuted state. No unrelated matrix operation or universal speed claim is implied.

## 2. Remove the default low-rank reconstruction accumulator (Sprint 228; R21-03)

1. Measure default and optional outer paths for at least three pinned shapes, ranks and drop tolerances, including all-zero and rank-deficient cases; retain raw RSS/time and exact build settings.
2. Specify output equality/tolerance, node ordering, NaN/overflow and allocation-failure behavior. Check that the optional path really satisfies the same public contract.
3. Promote the lower-memory path as default only after differential tests and failure sweeps; keep an explicit compatibility escape hatch if needed and document its cost.
4. Measure reconstruction scratch separately from full SVD factor storage. Update the header, benchmark explanation and first-use memory guidance.
5. Pass focused corpus, sanitizer, full C quality and CMake parity gates.

**Done when:** the default sparse low-rank reconstruction does not allocate an `m×n` accumulator and its output/error contract is verified. Dense factor storage remains honestly documented.

## 3. Close CG-family status and extreme-scale conformance (Sprint 229; R21-04)

1. Freeze assembled one-shot and reusable-handle, matrix-free and block CG status, iteration-count, initial-convergence, zero-budget, callback/history and final true-residual contracts.
2. Build an independent SPD oracle with representable extreme scales, well/ill-conditioned cases, finite and invalid preconditioner outputs, and post-update overflow traps. Define residual and output expectations by return status, including required `NaN` diagnostics for selected computed numeric failures and the lack of a residual guarantee on invalid input or other hard errors.
3. Repair discrepancies across the public CG surface through shared helpers only where semantics truly match; preserve completed-step counts and aggregate block behavior.
4. Exercise explicit handle preparation, underprepared growth, repeated reuse after successful and failed solves, allocation failures and thread/reentrant cases; add focused MSVC/Clang/GCC hosted registration.
5. Run full C quality, sanitizers, CMake parity and record exact pass/fail evidence.

**Done when:** every CG entry point returns the specified status and count plus the residual/output state promised for that status on the frozen corpus. Completed selected solves satisfy true-residual expectations; selected computed numeric failures report `converged=0` and `residual_norm=NaN`; invalid input and other hard errors are not assigned guarantees absent from the public contract. Prepared, grown and reused handles remain valid after both successful and failed solves. This is a complete CG-family contract, not all Krylov solvers.

## 4. Close MINRES breakdown and residual conformance (Sprint 230; R21-04)

1. Freeze one-shot, reusable-handle and block MINRES initial residual, zero-budget, Lanczos/preconditioner breakdown, callback/history, final true-residual, numeric-failure and aggregate block semantics in the public header.
2. Construct an independent symmetric-indefinite/semidefinite oracle and extreme-scale cases, including the PR #241 failure shapes and mixed converged/non-converged block columns.
3. Refactor only the recurrence and final-verification seams required by failures; check no unused product is computed after tolerance or budget exit across the public MINRES surface.
4. Test explicit handle preparation, underprepared growth, repeated reuse after success and failure, deterministic block count/residual aggregation, and cleanup with/without preconditioner; include CMake, Make and hosted compilers.
5. Run full C quality and sanitizers; retain numerical diagnostics and bounded limitations.

**Done when:** one-shot and reusable-handle MINRES satisfy the same declared status/iteration/residual contract, handles remain reusable after failure, and block MINRES satisfies documented per-column and aggregate count/residual behavior across the frozen corpus. GMRES and BiCGSTAB remain separate future conformance owners.

## 5. Prove one genuinely 64-bit input-to-solve route (Sprint 231; R21-05)

1. Select Matrix Market to CSC and one direct solve as the bounded route. Inventory every index conversion, row pointer, allocation multiplication and public output on that route.
2. Add large-index *metadata* and overflow-rejection cases that do not allocate impossible matrices, plus real 64-bit installed-consumer and moderate numerical fixtures.
3. Audit Make/CMake width propagation and mismatch negatives in an extracted install prefix; test both 32- and 64-bit builds.
4. Validate on Linux/macOS and the maintained Windows MSVC subset where feasible; do not add a Windows breadth claim without a named run.
5. Pass full C quality, focused CMake/CTest and sanitizer gates, and update the width-specific support table.

**Done when:** one named 64-bit parse/convert/solve/installed-consumer route has passing positive and negative evidence; other algorithms remain unclaimed.

## 6. Consolidate evidence-policy ownership (Sprint 232; R21-06/10)

1. Freeze JSON/CLI/exit-code and generated-report outputs of `normalize_report_index.py`, API-doc policy inputs and external evidence validation with golden fixtures.
2. Extract bounded parser, validation and report-rendering modules from one oversized script at a time without changing accepted records or public wording.
3. Replace brittle manually counted workflow inventory assertions with a declared, machine-checked inventory; register all affected Python tests in Linux/macOS CI.
4. Add mutation tests for missing/mismatched provenance, stale claims and changed workflow names; retain failure diagnostics.
5. Run all affected Python suites, API-doc local-only guard, docs-check and the CI lanes that execute them.

**Done when:** the selected tool's ownership surface is smaller, output contracts are unchanged and CI detects an omitted test or workflow. This does not claim arbitrary YAML/shell parsing is solved.

## 7. Complete one validator-accepted CHOLMOD correctness cell (Sprint 233; R21-08)

1. Freeze the tiny SPD matrix/RHS, canonical cell identity and exact public claim; get fixture/module attribution and artifact-retention review before publication.
2. Capture full project and actually loaded CHOLMOD/BLAS/transitive binary hashes, compiler flags, ordering, threads, host and run identity; test provenance capture by deliberate tampering.
3. Execute two fresh same-cell project/CHOLMOD solve pairs and preserve original streams, solutions, failure records and independent residual/backward-error calculations.
4. Validate both as `correctness_observed` under v1, independently review raw bundles and rights, then approve only the exact one-cell wording.
5. Pass pilot tests and affected CI/docs gates. If provenance or rights cannot be independently approved, retain a blocked result and do not count the capability closed.

**Done when:** one reproducible, validator-accepted, reviewed and rights-cleared two-sided cell exists; no general external parity claim follows.

## 8. Produce a candidate-specific static source package (Sprint 234; R21-07)

1. Name release/product/API/security/rollback owner roles, choose version/changelog and static-only support scope, and select one clean candidate SHA after required code work.
2. Independently generate two source archives from that exact tree; compare bytes, content/license manifest and exported package metadata.
3. Extract the archive twice into clean environments and run Make and CMake static install plus 32/64 width-mismatch/positive consumer probes.
4. Collect Linux/macOS/Windows exact-candidate hosted proof for the claimed routes, with test counts and failure receipts; reject stale-head evidence.
5. Obtain named approvals and a rollback plan before calling the package release-ready. Timing/product owners must also either independently accept the exact-candidate s225-v5 methodology, original bundles and recomputed decision, or sign a narrower static release scope that keeps timing blocked and makes no speed claim. Sprint 235's later kernel comparison is not a substitute for this inherited decision. No tag or publication is implied by this sprint plan.

**Done when:** the exact candidate has reproducible source and installed-first-use evidence plus required signatures, including the explicit timing/product-owner acceptance or narrower-scope decision. If an owner is unavailable or neither timing decision is signed, label release readiness blocked and leave unsigned artifacts as technical evidence only.

## 9. Make one performance comparison meaningful (Sprint 235; R21-09)

1. Choose a kernel/source change completed before Sprint 234's candidate freeze that actually differs from its baseline; pin source, workload, representative matrices and correctness gates before timing. A corrective code edit requires a new candidate and repeated package proof.
2. Require either archived executable bytes plus hashes or an independently reproducible build attestation. Reject identical benchmark binaries when the claim is improvement.
3. Use matched serial build settings, randomized/alternating pair order, warmups, multiple trials, raw streams and predeclared noise/regression thresholds on controlled hosts.
4. Report time and peak memory with uncertainty, negative/noisy attempts and a counterexample workload; compare only the named kernel/hosts.
5. Recompute independently and obtain method/artifact review before any speed wording; run affected CI and docs gates.

**Done when:** one differing-source candidate has accepted, reproducible, correctness-qualified comparative evidence. A no-regression smoke test alone is not completion.

## 10. Close the installed adoption path and Epic ledger (Sprint 236; R21-07/10)

1. Test a new user's full path from the selected archive to first Matrix Market or CSR/CSC solve on a clean prefix, at the supported widths and platforms.
2. Deduplicate current README/INSTALL/cookbook/solver-selection guidance into a single source-of-truth support and capability table; keep historical proofs in planning artifacts.
3. Run link, header, example, docs-policy and installed-consumer checks on the exact candidate; compare current wording against signed release and evidence decisions.
4. Publish the Epic 21 reconciliation with per-item status, raw artifact index, blocked reasons, owners and explicit remaining non-claims. Decide release versus block using the same candidate; any tracked edit after freeze requires affected proof to rerun.
5. Run all required final gates. Keep generated API HTML local-only and shared ABI/package-manager distribution outside this scope.

**Done when:** the static installed route is reproducible from the named artifact and current guidance is coherent. Release acceptance is closed only if Sprint 234's independent approvals and exact-candidate gates are complete; otherwise state `BLOCKED` without treating documentation completion as a release.

## Portfolio rule

The ten sprints fund named closures, not ten broad research programs. Sprints 227–232 can close deterministic source/test gaps without external approval. Sprints 233–235 have explicit provenance, approval or controlled-measurement gates; failure of those gates is an honest blocked outcome, and remaining capacity returns to an already-funded regression or documentation task rather than expanding to a second unverified capability. Shared ABI, package-provider publication, GPU/distributed execution, broad external parity and general state-of-the-art ranking remain outside this 1,400-hour plan.
