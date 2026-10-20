# Epic 19 Residual Queue

This is the **Sprint 216 prioritized next-epic handoff**, not a funded
roadmap or a new support claim. The [Day 14 closeout](artifacts/day14-integrated-validation-closeout.md)
retained all 21 entries without a new evidence delta. A completed
Sprint 207-215 *decision, design, or blocked fallback* does not close the
capability it considered. The [Day 6](artifacts/day6-project-plan-status-207-211.md)
and [Day 7](artifacts/day7-project-plan-status-212-215.md) item ledgers retain
the exact selected outcomes. Owners are functional review roles, not assigned
people. Every claim below remains **unearned** until its acceptance evidence
is reviewed and the public wording is approved.

## Priority Basis

| Rank | Class | Why this order |
| --- | --- | --- |
| RQ-01 to RQ-03 | P0: evidence integrity | The absent external v1 validator blocks a meaningful pilot. A two-sided result and publication approval must follow in order; none can be inferred from `attempt.json`. |
| RQ-04 to RQ-10 | P1: bounded closure | Selected Windows, candidate release/provider, one benchmark, one reliability owner, and one review surface have named existing fixtures or policies. Order favors correctness and claim risk, then user impact and achievable scoped proof. |
| RQ-11 to RQ-21 | P2: explicit long-horizon decisions | External breadth, shared ABI, broad platform/provider, performance architecture, conformance, and docs/report tooling need new scope or sustained evidence. Revisit only when the named prerequisite or product decision is available. |

Priorities are a reviewed first order, not a promise to execute every entry in
one sprint. P0 is prerequisite-ordered: validator, one actual cell, then public
approval. P1 ranks selected Windows correctness/claim coherence before release
and provider adoption, followed by one-row/one-owner/one-cluster work. P2 needs
new product scope or a representative evidence program. Keep source
install/static support, selected freshness, and local-only generated API policy
unchanged while these gates are open. The
[Day 8 crosswalk](artifacts/day8-residual-queue-intake.md) explains which
older sprint residuals were merged, retained, or closed at a narrower scope.

## P0: Evidence Integrity

### RQ-01: External v1 Evidence Validator

- **Owner:** Report/corpus maintainers.
- **Dependencies:** Approve the isolated v1 object and complete-cell-ID rules in the [Sprint 215 blueprint](../SPRINT_215/STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md).
- **Acceptance evidence:** Add `tests/corpus/schemas/external_baseline_evidence_v1.json`, `scripts/validate_external_baseline_evidence.py`, a self-test, a canonical cell-ID vector, positive planning/blocked/observed fixtures, and negative cases for duplicate keys, statuses/units, incomplete provenance, unsafe paths, and hash drift. Demonstrate the validator rejects unearned pass claims without changing selected manifests or the existing normalizer.
- **Stop condition:** A missing semantic field, unsafe path, hash mismatch, or failed negative fixture keeps the pilot blocked; do not bypass validation to run solvers.
- **Claim unlocked only if closed:** A valid v1 evidence *format*, not numerical correctness or external parity. [Source](../SPRINT_215/RESIDUAL_QUEUE.md).

### RQ-02: One Two-Sided CHOLMOD Correctness Cell

- **Owner:** Solver/pilot owner with report reviewer.
- **Dependencies:** RQ-01; approved exact fixture bytes and runtime provenance plan. The [Day 10 blocked attempt](../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) is the baseline, not a pass.
- **Acceptance evidence:** Fresh attempt ID; matrix/RHS and conversion hashes; project binary/flags; actual loaded CHOLMOD and transitive-library hashes; effective ordering/backend/threads; raw project and CHOLMOD factor/solve status, full solution vectors, Day 4 residual/backward-error metrics, and validator-accepted complete cell ID for the pinned tiny SPD case.
- **Stop condition:** Either solver does not execute, provenance is incomplete, numerical gates fail, or v1 validation rejects the cell. Preserve blocked/failed logs without rewriting them as pass.
- **Claim unlocked only if closed:** Correctness for one named local matrix/solver/version/build cell, not parity across matrices, platforms, or performance. [Source](../SPRINT_215/RESIDUAL_QUEUE.md).

### RQ-03: External Attribution And Publication Approval

- **Owner:** Corpus owner and publication reviewer.
- **Dependencies:** RQ-01; RQ-02 before publishing a result. Fixture and CHOLMOD-module terms require review even for a small pilot.
- **Acceptance evidence:** Written source attribution and redistribution decision for the exact matrix/RHS, module/dependency license review, artifact retention/access policy, approved claim text and rollback owner for the exact validated bundle.
- **Stop condition:** Unclear fixture authorship, module terms, artifact ownership, or public claim scope means local evaluation only and no external-result publication.
- **Claim unlocked only if closed:** Publication of the specifically approved evidence bundle, not broader external superiority. [Source](../SPRINT_215/RESIDUAL_QUEUE.md).

## P1: Bounded Closure Candidates

### RQ-04: Selected Windows Cholesky Freshness

- **Owner:** Windows/report owner.
- **Dependencies:** Re-inspect an exact hosted run/job/artifact and resolve the generated support-tier contradiction in the [Sprint 208 decision](../SPRINT_208/artifacts/day5-promotion-decision.md).
- **Acceptance evidence:** Six selected rows/files with current commit, Windows path and artifact membership; aligned generator support tier/non-claims, selected manifest Windows workflow metadata, docs and negative guard fixtures; hosted and local checks on the same proposed claim.
- **Stop condition:** `local_only`, `no hosted CI proof`, or missing/wrong hosted files remain in generated evidence. Keep manifest Linux/macOS-only and selected Windows freshness re-deferred.
- **Claim unlocked only if closed:** Freshness for that exact Windows Cholesky target, not broad Windows support. [Source](../SPRINT_208/RETROSPECTIVE.md).

### RQ-05: Selected Windows QR Incompatible Freshness

- **Owner:** Windows QR/CI and manifest owners.
- **Dependencies:** Run and inspect the bounded Sprint 209 collection lane; the [Day 8 query](../SPRINT_209/artifacts/day8-manifest-decision.md) found no hosted artifact.
- **Acceptance evidence:** Exact hosted six-file `qr-incompatible-ls` artifact, row/path/freshness checks, generated support-tier and non-claim alignment, selected manifest Windows metadata, docs and PowerShell/workflow/normalizer regressions.
- **Stop condition:** No inspectable hosted artifact or mismatched row/support metadata leaves selected QR `local_only` and Linux/macOS-only.
- **Claim unlocked only if closed:** Selected Windows QR incompatible freshness only, not broad QR or least-squares parity. [Source](../SPRINT_209/RETROSPECTIVE.md).

### RQ-06: Candidate-Specific Static Source Release

- **Owner:** Release owner with platform and security reviewers.
- **Dependencies:** Explicit release candidate and publication decision; retain Sprint 214's static-first boundary.
- **Acceptance evidence:** [Release checklist](../SPRINT_214/RELEASE_READINESS_CHECKLIST.md) applied to a clean reviewed commit, changelog, immutable tag/signing policy, tracked-source manifest, two independent matching archive hashes/extraction builds, source-to-installed-file provenance, candidate-specific hosted jobs for every claimed platform, approvals and rollback plan.
- **Stop condition:** Missing candidate, mismatched archive, uninspected hosted result, or absent approval blocks release language and publication.
- **Claim unlocked only if closed:** Readiness of that exact static source candidate on its validated platforms, not shared ABI or provider packages. [Source](../SPRINT_214/RESIDUAL_QUEUE.md).

### RQ-07: One Public Source Package Provider

- **Owner:** Package/provider owner.
- **Dependencies:** Product selects a provider tier; stable archive/checksum provenance from RQ-06 or an equivalently reviewed immutable source artifact.
- **Acceptance evidence:** Provider-owned source recipe/formula, license and checksum review, non-local render/audit/install/test/uninstall and cleanup proof on a named platform, downstream static consumer, docs/support-tier update and overclaim guards.
- **Stop condition:** Developer-mode local Homebrew proof alone, unpublished formula ownership, stale archive/hash, or missing uninstall evidence keeps public-provider support deferred.
- **Claim unlocked only if closed:** The exact named provider/source/platform tier, not Homebrew/core, bottles, Linuxbrew or other providers. [Source](../SPRINT_207/RETROSPECTIVE.md).

### RQ-08: Selected Benchmark Timing Policy

- **Owner:** Benchmark methodology owner.
- **Dependencies:** New reviewed policy for one selected row; the Sprint 212 `--repeat 1`, `threshold=n/a` [decision](../SPRINT_212/artifacts/day5-product-policy-decision.md) remains current until then.
- **Acceptance evidence:** Stable runner/compiler class, warmup and repeat counts, variance/dispersion rule, pinned baseline provenance, threshold value, retained raw samples/artifacts, pass/fail implementation, negative metadata tests and scoped claim review.
- **Stop condition:** Single-sample freshness, host drift, absent variance/baseline or missing artifacts cannot become a timing gate or speed claim.
- **Claim unlocked only if closed:** A selected timing-regression policy for one runner/workload, not portable speed or Linux/macOS parity. [Source](../SPRINT_212/RETROSPECTIVE.md).

### RQ-09: Next Allocation-Failure Owner

- **Owner:** Solver/test reliability owner.
- **Dependencies:** Rank/select one owner outside the no-reorder linked-list LDLT [Sprint 210 proof](../SPRINT_210/artifacts/day14-closeout-review.md); CSC LDLT, reordered LDLT, Cholesky numeric, matrix construction/IO, and QR/SVD/eigensolver workspaces remain distinct candidates.
- **Acceptance evidence:** Named allocation lifecycle, deterministic fail-after sweep, cleanup, stale-output, caller-input/option preservation, retry, focused Make/CTest gate, registration guard, claim-safe docs and full C quality chain.
- **Stop condition:** Uncovered allocation sites or failing cleanup/retry prevent even the selected owner claim; no broad OS OOM or concurrent-hook inference.
- **Claim unlocked only if closed:** Deterministic failure handling for that selected owner only. [Source](../SPRINT_210/RETROSPECTIVE.md).

### RQ-10: Next Bounded Review-Surface Extraction

- **Owner:** Review/test owner.
- **Dependencies:** Re-rank current large files after the LDLT CSC native-parity [Sprint 211 extraction](../SPRINT_211/artifacts/day14-closeout-review.md); select one family-local cluster.
- **Acceptance evidence:** Before/after line and ownership map, unchanged public behavior/registration/order, helper include/source wiring, active-code ownership guard, focused behavior regression and applicable full C/CMake gate.
- **Stop condition:** Test registration or behavior drift, unowned helper dependencies, or incomplete source-list/guard proof stops the extraction.
- **Claim unlocked only if closed:** Reduced review burden for that one cluster, not a solver feature or broad maintainability claim. [Source](../SPRINT_211/RETROSPECTIVE.md).

## P2: Long-Horizon Decisions

Revisit a P2 entry only when its listed trigger is met. Otherwise retain the
current non-claim rather than scheduling a vague implementation task.

### RQ-11: External Matrix And Solver Breadth

- **Owner:** Corpus, solver and dependency owners.
- **Dependencies/revisit trigger:** RQ-01 and RQ-02 pass; matrix attribution and new-baseline dependency lock are available.
- **Acceptance evidence:** Separately hashed and licensed matrices (`tridiagonal_20.mtx`, `nos4.mtx` if approved), family-specific tolerance/phase rules, plus an independent pinned solver family (for example SciPy/SuperLU) on a genuinely nonsymmetric fixture with transitive dependency hashes and validated raw cells.
- **Stop condition:** Reused 4x4 tolerances, missing source terms, noncanonical fixture bytes or incomparable operations prevent combined rows.
- **Claim unlocked only if closed:** Only the named additional cells; no broad matrix-family or solver parity. [Source](../SPRINT_215/RESIDUAL_QUEUE.md).

### RQ-12: Hosted External And Optional Timing Matrix

- **Owner:** CI/benchmark and product claim reviewers.
- **Dependencies/revisit trigger:** RQ-01 to RQ-03 and at least one accepted local cell; a named platform/performance question is approved.
- **Acceptance evidence:** Exact-commit hosted run/job/artifact digests, runner/compiler/dependency hashes and same-cell validated correctness per claimed platform. Timing additionally requires paired warmup/repeat/clock/dispersion/host controls, retained invalid trials and a separate measured record; claim matrix must name coverage and limitations.
- **Stop condition:** Workflow registration, provisional attempts, unmatched environments or freshness-only measurements cannot support parity, speed or state-of-the-art claims.
- **Claim unlocked only if closed:** Only named hosted cells or measured conditions after publication approval; broad state-of-the-art status needs separately reviewed representative coverage. [Source](../SPRINT_215/RESIDUAL_QUEUE.md).

### RQ-13: Named-Platform Shared Library And ABI

- **Owner:** ABI/build/release owner.
- **Dependencies/revisit trigger:** New product decision selects public subset, platform/toolchain and static/shared package semantics; Sprint 214's [design](../SPRINT_214/artifacts/day2-abi-release-scope-decision.md) is accepted as a specification, not proof.
- **Acceptance evidence:** Hidden-by-default exact export allowlist, ABI epoch/ownership/layout/index-width rules, shared artifact and installed metadata, platform loader inspection (ELF SONAME, Mach-O install-name/RPATH or PE DLL/import library), executed installed dynamic consumer and missing/wrong-library negatives in hosted CI. Cross-version promise additionally needs old/new retained artifacts and consumers.
- **Stop condition:** `BUILD_SHARED_LIBS=ON` still rejected, uncurated exports, missing loader evidence or no consumer execution means static-first remains the only supported path.
- **Claim unlocked only if closed:** One named shared platform/ABI subset; cross-version or other-platform support needs its own proof. [Source](../SPRINT_214/RESIDUAL_QUEUE.md).

### RQ-14: Generated API Publication Choice

- **Owner:** API/docs publication owner.
- **Dependencies/revisit trigger:** Product chooses one hosted, retained-artifact or committed-HTML policy and its audience; local-only [Sprint 213 decision](../SPRINT_213/artifacts/day5-product-policy-decision.md) remains active until then.
- **Acceptance evidence:** Exact destination or artifact name, access/retention, generated freshness and public-header coverage, route/link validation, workflow staging safeguards, stale-output rollback, user/maintainer docs and claim-guard update. A structured workflow parser may be selected separately under RQ-19.
- **Stop condition:** Unknown destination/retention, broken route, stale HTML or unreviewed workflow publication keeps `docs/api/html/` ignored and local-only.
- **Claim unlocked only if closed:** The selected generated-doc publication path, not API completeness, ABI stability or release status. [Source](../SPRINT_213/RETROSPECTIVE.md).

### RQ-15: Broader Windows Support And Benchmark Evidence

- **Owner:** Windows/platform owner.
- **Dependencies/revisit trigger:** RQ-04/05 decisions and a separate support-tier definition for each additional lane.
- **Acceptance evidence:** Target-by-target hosted Windows artifacts and generated-claim/manifest alignment; separately executed Windows-native Make or `pkg-config` consumers if claimed; Windows benchmark lane with retained runner/methodology evidence if benchmark freshness is claimed.
- **Stop condition:** CMake/MSVC static proof or selected workflow registration cannot stand in for Windows Make, `pkg-config`, broad report freshness or timing parity.
- **Claim unlocked only if closed:** Only individually evidenced Windows lanes; no blanket platform parity. [Sources](../SPRINT_208/RETROSPECTIVE.md) and [209](../SPRINT_209/RETROSPECTIVE.md).

### RQ-16: Other Providers, Binary Packages And Release Distribution

- **Owner:** Provider/release product owner.
- **Dependencies/revisit trigger:** RQ-06 candidate provenance and explicit provider-specific decision; RQ-07 public source tier where applicable.
- **Acceptance evidence:** For each selected Homebrew/core, bottles/Linuxbrew, vcpkg, Conan, pkgsrc, distro or binary path: owned recipe/artifact, provider lint/install/consumer/uninstall, per-platform provenance, publication/maintenance policy, docs and negative guards.
- **Stop condition:** A source formula or one provider's proof cannot authorize another provider, a bottle, a binary package or package-manager release readiness.
- **Claim unlocked only if closed:** Only the exact provider and artifact/platform tier reviewed. [Source](../SPRINT_207/RETROSPECTIVE.md).

### RQ-17: Compressed-First Performance Architecture

- **Owner:** Numerical architecture and benchmark owners.
- **Dependencies/revisit trigger:** A named kernel/workload and measurement question are chosen; RQ-08 methodology is available for any speed claim.
- **Acceptance evidence:** Profile linked-list versus compressed dispatch, memory footprint and cache behavior on hash-pinned matrices; choose one compressed-kernel ownership change, preserve numerical tests, measure retained before/after samples on a controlled runner and document when users should remain compressed-first.
- **Stop condition:** Anecdotal pointer-chasing claims, one local timing, or a conversion-only change without numerical/measurement proof cannot support performance language.
- **Claim unlocked only if closed:** A bounded architecture or measured kernel result, not portable speed. [Source](../reviews/review-codex-2026-09-20.md).

### RQ-18: Broader Numerical Conformance And Robustness

- **Owner:** Test/oracle and numerical owners.
- **Dependencies/revisit trigger:** Select one solver family and an external or independently derived oracle with fixture provenance; RQ-01/02 where external evidence is published.
- **Acceptance evidence:** Matrix-family coverage plan, deterministic property/random seeds, edge/failure cases, numerical tolerances, independent oracle results, platform/toolchain identities, retained raw failures and regression fixtures. Repeat per family before a broad claim.
- **Stop condition:** Existing selected helper fixtures, one external cell or a broad test count do not establish all-family correctness, stability, memory or ecosystem parity.
- **Claim unlocked only if closed:** Conformance for the exact selected family/matrix set. [Source](../reviews/review-codex-2026-09-20.md).

### RQ-19: Structured Workflow And Guard Maintenance

- **Owner:** CI/guard maintainers.
- **Dependencies/revisit trigger:** The next workflow-guard change exposes parser complexity or a bypass; preserve current local-only policy.
- **Acceptance evidence:** Structured YAML/step and shell-aware checks (or a fail-closed narrower contract) that preserve every existing encoded/anchor/flow/multiline/PowerShell regression and step boundary; differential fixtures for allowed and forbidden publication cases. Factor comment stripping only with equivalent guard tests.
- **Stop condition:** A parser rewrite that misses a current bypass fixture or falsely rejects benign workflows cannot replace the existing guard.
- **Claim unlocked only if closed:** Guard maintainability for the selected workflow contract, not generated HTML publication. [Sources](../SPRINT_213/RETROSPECTIVE.md) and [211](../SPRINT_211/RETROSPECTIVE.md).

### RQ-20: API Completeness And Installed First-Use Route

- **Owner:** API/docs and developer-experience owners.
- **Dependencies/revisit trigger:** Choose an intended external-user route and decide whether generated-header and public-header coverage should expand; RQ-07 or RQ-14 only if a provider/site path is selected.
- **Acceptance evidence:** Audit checked-in versus generated header inventory, expand Doxygen inputs only by policy, validate page/link coverage, run an installed static first-use consumer from a clean prefix, and user-test the README/INSTALL/API navigation with current support tiers.
- **Stop condition:** More generated pages or a shorter README alone cannot imply public package availability, online docs, API stability or a release.
- **Claim unlocked only if closed:** Clearer bounded first-use/API coverage for the validated route. [Sources](../SPRINT_213/RETROSPECTIVE.md) and [review](../reviews/review-codex-2026-09-20.md).

### RQ-21: Benchmark Report Publication And Release Evidence

- **Owner:** Benchmark publication and release owners.
- **Dependencies/revisit trigger:** RQ-08 methodology and an explicit public report or release-benchmark decision; RQ-06 for a named release candidate.
- **Acceptance evidence:** Select exact report destination and retention, pin generated-report provenance and raw benchmark artifacts, validate publication routes/freshness and stale-output rollback, and, for release wording, run a release-scoped corpus with reproducibility instructions and reviewed notes/claim guards.
- **Stop condition:** Source-controlled selected freshness rows without retained raw samples, publication policy or release candidate remain local/selected evidence, not a public benchmark or release timing claim.
- **Claim unlocked only if closed:** Publication or release evidence for the exact selected report/candidate, not portable performance or backend superiority. [Source](../SPRINT_212/RETROSPECTIVE.md).

## Handoff Rule

For every promoted claim, record a dated decision, exact artifacts/checks,
platform and scope, validation result, reviewer approval and updated non-claim
boundary. A failed, skipped, missing or not-run gate remains visible. The
[Day 9 validation matrix](artifacts/day9-residual-decision-validation-matrix.md)
maps current changed surfaces to executable checks. Days 10-14 may change this
queue only through a recorded evidence delta.

## State-Of-The-Art Decision

**Decision: no state-of-the-art sparse linear algebra claim is earned.** The
[Sprint 215 attempt](../SPRINT_215/artifacts/day10-selected-pilot-outcome.md)
stopped before either solver ran; no accepted external v1 evidence, two-sided
numerical result, comparative timing sample, or publication approval exists.
The selected benchmark freshness row is threshold-free, not speed proof.

The earliest *bounded* external correctness wording would require RQ-01 to
RQ-03: one validator-accepted, hash-pinned, two-sided cell on a named build and
platform, complete actual runtime provenance, numerical gates, fixture/module
rights and explicit publication approval. That would permit only the exact
cell's result, not an external parity or performance claim.

Reconsidering a broad state-of-the-art assessment requires a pre-registered
representative matrix/operation matrix, not a fixed row count: at minimum
distinct SPD, nonsymmetric, and rectangular systems, plus separately selected
iterative/eigen/SVD cells if those families appear in the claim; independent
pinned baselines per claimed family; more than one matrix size/structure per
family; validated correctness and failure cases; and controlled performance
samples with variance on at least two named hosted platform/toolchain classes.
Include Windows only after separate Windows evidence if the wording covers
Windows. Every cell needs raw retained artifacts, runtime/build provenance,
licensing/attribution and publication approval. Even meeting this entry bar
would trigger a product/reviewer decision, **not** automatic state-of-the-art
status. RQ-11, RQ-12, RQ-17 and RQ-18 describe the remaining evidence work.
