# Epic 20 Project Plan: Bounded Correctness, Evidence, Performance and Static Release

**Sprints:** 217-226 (10 sprints, 14 days each)
**Planning capacity:** 168 hours maximum per sprint; 140 hours estimated per
sprint and 28 hours unallocated contingency. **Total planned work: 1,400
hours** over 140 days. Estimates are engineering hours, not elapsed CI time.

This plan implements the dependency-ordered [gap-closure checklist](reviews/todo-codex-2026-10-19.md)
from the [2026-10-19 code review](reviews/review-codex-2026-10-19.md).
It deliberately closes selected problems instead of claiming partial progress
on every Epic 19 residual. A design or guarded deferral is not a completed
capability. Each sprint ends with retained evidence and exact-scope wording;
any failed or unavailable required hosted/approval gate blocks promotion.
Static source installation, local-only generated HTML, and threshold-free
canonical freshness remain the defaults until their named gates change.

## Sprint 217: Reliable Norms And ILUT Numerical Diagnostics

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close the shared vector-norm overflow/non-finite behavior gap and
make the `west0067` ILUT result a meaningful numerical regression.

### Prerequisites from Previous Sprints

- Epic 19 static-first/selected-claim boundary and current C test harness.
- Existing ILUT `west0067` fixture and iterative solver tests.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 217.1 | Contract And Call-Site Audit | Inventory norm consumers and define finite, NaN, infinity, zero-vector and convergence semantics. | 20 |
| 217.2 | Scaled Norm Implementation | Replace naive square/sum norm with scaled accumulation and explicit non-finite behavior; review `vec_norminf`. | 28 |
| 217.3 | ILUT And Iterative Hardening | Distinguish norm overflow from invalid ILUT output; enforce finite/residual gates and reject false convergence in affected iterative paths. | 28 |
| 217.4 | Extreme-Scale Regression | Add exponent, NaN, infinity, underflow, and seeded iterative fixtures with independently computed expectations. | 24 |
| 217.5 | Public Contract Update | Clarify vector/solver diagnostic behavior and scoped limitations in headers and user docs. | 20 |
| 217.6 | Validation And Evidence | Run focused tests, sanitizer, `make format && make lint && make test`, and retain before/after results. | 20 |

### Deliverables

- Stable `vec_norm2`/`vec_norminf` behavior and finite-aware solver diagnostics.
- ILUT fixture with checked solution and residual, not only `norm > 0`.
- Focused test and full C-quality records.

### Completion Criteria

Finite representable norms remain finite; invalid inputs/outputs cannot pass
as successful convergence; all required gates pass. This closes the selected
shared diagnostic owner, not broad solver-family conformance.

### Total Estimate

140 hours.

## Sprint 218: Linear-Time Matmul Output Construction

**Duration:** 14 days, approximately 140 hours.

**Goal:** Remove `sparse_matmul`'s repeated sorted-row head scans while
preserving observable results, ownership and failure behavior.

### Prerequisites from Previous Sprints

- Sprint 217 numerical diagnostics and current matmul allocation-failure gate.
- Existing CSR/CSC conversion and dense-oracle test helpers.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 218.1 | Baseline And Semantics | Pin wide-row and rectangular fixtures; record output order, zero dropping, time and peak memory before edits. | 20 |
| 218.2 | Output Builder | Implement append/compressed construction without `sparse_insert` for every accumulated value; document row/column ownership. | 32 |
| 218.3 | Correctness Matrix | Compare against independent dense products for cancellation, duplicates, zero rows and seeded shapes. | 28 |
| 218.4 | Failure Lifecycle | Sweep new allocations and verify null output, unchanged inputs, cleanup and retry. | 24 |
| 218.5 | Controlled Measurement | Retain raw before/after samples and memory observations for pinned fixtures; avoid portable speed wording. | 18 |
| 218.6 | Quality Closure | Run full C chain, CMake/CTest parity, sanitizer and focused matmul gates. | 18 |

### Deliverables

- Matmul output path without quadratic row flush, plus deterministic numerical
  and failure regressions.
- Reproducible selected-kernel before/after profile.

### Completion Criteria

The repeated head-scan is absent, numerical/failure contracts are preserved,
and retained measurement plus quality gates pass. Only selected-kernel
performance may be discussed.

### Total Estimate

140 hours.

## Sprint 219: Enforced Installed Index-Width Contract

**Duration:** 14 days, approximately 140 hours.

**Goal:** Make 32/64-bit `idx_t` selection explicit and safe for installed
static consumers, rather than relying on a header warning.

### Prerequisites from Previous Sprints

- Sprint 218 source/consumer baseline; Sprint 214 static install/export tests.
- Public `SPARSE_IDX_BITS` typedef and both installed-consumer routes.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 219.1 | ABI Width Inventory | Map public index-bearing structures/functions and installed metadata for width mismatch risk. | 20 |
| 219.2 | Build Configuration | Add validated 32/64 selection to CMake and Make; propagate chosen width to library and tests. | 30 |
| 219.3 | Consumer Propagation | Export CMake usage requirements, pkg-config cflags and an executable package-width check. | 26 |
| 219.4 | Positive/Negative Consumers | Test clean-prefix installed consumers at both widths and intentional mismatch rejection. | 24 |
| 219.5 | Selected 64-Bit Lane | Run chosen solver, IO and conversion cases under a genuine 64-bit build; add hosted registration. | 20 |
| 219.6 | Contract Validation | Update docs and run install, CMake, source and full C quality gates. | 20 |

### Deliverables

- Width-aware static package metadata, mismatch guard and tested 32/64-bit
  installed consumers.
- Scoped documentation of tested widths and ABI exclusions.

### Completion Criteria

Consumers receive the correct width or fail before unsafe use; both widths
have passing build/consumer evidence. No shared or cross-version ABI claim.

### Total Estimate

140 hours.

## Sprint 220: Structured Local-Only Workflow Guard

**Duration:** 14 days, approximately 140 hours.

**Goal:** Replace fragile YAML normalization ownership while retaining the
current generated-API local-only publication rule.

### Prerequisites from Previous Sprints

- Sprint 213's local-only decision and current workflow guard fixture corpus.
- Sprints 217-219 quality and installed-contract baseline.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 220.1 | Fixture Freeze | Catalog accepted/rejected anchor, flow, quote, multiline, shell and PowerShell cases and real workflow inputs. | 22 |
| 220.2 | Structured Parser | Select supported YAML parser or fail-closed grammar and extract workflow step/run values with shell identity intact. | 34 |
| 220.3 | Policy Engine | Preserve shell-aware staging/upload detection and canonical paths without losing argument or step boundaries. | 28 |
| 220.4 | Differential Regression | Run old/new guards across fixtures, add false-positive and bypass mutations, explain discrepancies. | 20 |
| 220.5 | Maintainer Guide | Document parser dependency, supported syntax and fail-closed escalation path. | 18 |
| 220.6 | Gate And Rollout | Run workflow, docs and applicable quality checks; remove superseded parsing only after parity. | 18 |

### Deliverables

- Smaller structured guard with full preserved regression corpus and current
  workflow validation.
- Unchanged local-only generated HTML policy.

### Completion Criteria

All known allowed/forbidden fixtures and current workflows classify
correctly; ambiguous publication syntax fails closed. No HTML publication is
introduced.

### Total Estimate

140 hours.

## Sprint 221: External Evidence V1 Validator

**Duration:** 14 days, approximately 140 hours.

**Goal:** Close the missing schema/validator prerequisite that currently
blocks the optional independent baseline pilot.

### Prerequisites from Previous Sprints

- Sprint 215 external-evidence blueprint and blocked pilot record.
- Sprint 220 guard/retention conventions; current corpus manifest boundaries.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 221.1 | Schema Freeze | Finalize v1 status, units, provenance, paths, hashes, cell-ID and claim-transition contract. | 24 |
| 221.2 | Validator Implementation | Implement strict duplicate-key rejection, schema and semantic cross-field checks. | 30 |
| 221.3 | Fixture Matrix | Add positive planning/blocked/observed and negative missing/unsafe/duplicate/false-pass fixtures. | 28 |
| 221.4 | Pilot Preflight | Wire self-test into optional pilot without relaxing fail-closed behavior or changing selected reports. | 22 |
| 221.5 | Mutation Proof | Show syntax-valid corrupt evidence fails on hash, status, path and canonical ID mutations. | 18 |
| 221.6 | Docs And Validation | Document v1 versioning/retention and pass Python, docs and applicable full quality gates. | 18 |

### Deliverables

- `external_baseline_evidence_v1.json`, validator, self-test, and canonical
  positive/negative evidence fixtures.
- Passing preflight that still rejects incomplete pilot evidence.

### Completion Criteria

The optional pilot can proceed only with a valid v1 format; no solver or
parity result is inferred from validation alone.

### Total Estimate

140 hours.

## Sprint 222: One Two-Sided CHOLMOD Correctness Cell

**Duration:** 14 days, approximately 140 hours.

**Goal:** Execute and validate one genuinely independent, exact-scope SPD
solve against CHOLMOD, with provenance and publication rights kept separate.

### Prerequisites from Previous Sprints

- Sprint 221 v1 validator and preflight pass.
- Pinned Sprint 215 pilot fixture/operation and a usable CHOLMOD build.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 222.1 | Fixture And Rights | Pin matrix/RHS/conversion bytes; verify attribution, CHOLMOD module licenses and retention scope. | 20 |
| 222.2 | Runtime Provenance | Capture exact project/CHOLMOD builds, flags, loaded/transitive libraries, hashes, host, ordering and threads. | 28 |
| 222.3 | Two-Sided Execution | Run both solvers, retain raw statuses/streams/vectors, and compute independent residual/backward-error metrics. | 34 |
| 222.4 | Validation And Repeat | Validate complete v1 cell ID, repeat from frozen inputs, and retain failed attempts verbatim. | 24 |
| 222.5 | Claim Review | Obtain exact-bundle attribution/publication and wording decision; update docs only for approved scope. | 18 |
| 222.6 | Test Closure | Run pilot/validator negatives, affected documentation checks and quality gates. | 16 |

### Deliverables

- Validator-accepted two-sided local CHOLMOD correctness cell with complete
  raw data, hashes and repeat record.
- Written rights, retention and exact-claim decision.

### Completion Criteria

Both solvers ran on the same pinned problem; numerical gates and provenance
pass; publication status is explicit. Rights denial keeps results private and
blocks a public claim. A tiny cell never authorizes broad external parity.

### Total Estimate

140 hours.

## Sprint 223: Selected Windows Cholesky Freshness

**Duration:** 14 days, approximately 140 hours.

**Goal:** Complete the previously re-deferred selected Windows Cholesky
freshness promotion for one exact target, using current hosted artifacts.

### Prerequisites from Previous Sprints

- Sprint 208/216 six-file artifact and promotion criteria.
- Sprint 220 workflow guard; selected report normalizer/manifest unchanged
  unless promoted together.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 223.1 | Hosted Run Intake | Run/inspect exact-commit MSVC Cholesky job and six uploaded files with run/artifact IDs. | 20 |
| 223.2 | Row And Artifact Audit | Check target rows, hashes, paths, generator metadata and selected manifest agreement. | 30 |
| 223.3 | Atomic Promotion | Align selected Windows manifest, support tier, generated non-claims and public docs in one change. | 26 |
| 223.4 | Negative Guards | Reject stale, missing, renamed, wrong-platform and contradictory metadata fixtures. | 24 |
| 223.5 | Hosted Recheck | Repeat proposed lane on final commit and retain reviewer-visible job/artifact evidence. | 22 |
| 223.6 | Scope Sign-Off | Run local affected tests and approve exact selected-freshness wording only. | 18 |

### Deliverables

- Current, inspectable six-file Cholesky artifact and aligned selected target
  metadata/wording.
- Negative regression and hosted validation record.

### Completion Criteria

The named Windows Cholesky row is fresh and all contract surfaces agree. A
missing hosted artifact blocks promotion; no broad Windows solver claim.

### Total Estimate

140 hours.

## Sprint 224: Selected Windows QR Incompatible Freshness

**Duration:** 14 days, approximately 140 hours.

**Goal:** Complete selected Windows QR incompatible least-squares freshness
only after an actual hosted MSVC six-file artifact is inspected.

### Prerequisites from Previous Sprints

- Sprint 223 selected Windows promotion pattern and claim contract.
- Sprint 209 QR collection workflow/PowerShell guards and manifest deferral.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 224.1 | Hosted QR Collection | Run exact-commit QR job and inspect six-file artifact, path forms and diagnostics. | 20 |
| 224.2 | Selected Row Audit | Verify identities, hashes, normalizer result and generator metadata against manifest. | 30 |
| 224.3 | Atomic Metadata Update | Promote only `qr-incompatible-ls` in manifest, support tier and bounded public wording. | 26 |
| 224.4 | Failure Fixtures | Add missing/stale/wrong-row and PowerShell/workflow contradiction regressions. | 24 |
| 224.5 | Hosted Recheck | Repeat final hosted lane and retain run/job/artifact IDs and files. | 22 |
| 224.6 | Exact-Scope Review | Pass affected local checks and approve selected QR freshness non-claims. | 18 |

### Deliverables

- Inspected six-file hosted QR artifact and synchronized selected metadata.
- Negative guard, final hosted check and claim-review record.

### Completion Criteria

QR selected Windows freshness is promoted only for the named target with
actual hosted proof; missing evidence is a blocker, not a completed task.

### Total Estimate

140 hours.

## Sprint 225: One Controlled Canonical Timing Gate

**Duration:** 14 days, approximately 140 hours.

**Goal:** Turn one canonical `bench_refactor_csc`/`nos4` workload into a
reviewed runner-specific timing regression gate, separate from freshness.

### Prerequisites from Previous Sprints

- Sprint 218 selected-kernel measurement lessons; Sprint 212 threshold-free
  canonical row and current hosted artifact format.
- A stable, named CI runner/compiler class and retained raw-sample route.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 225.1 | Methodology Lock | Pre-register workload, backend, runner, compiler, warmup, repeat, clock and invalid-trial handling. | 24 |
| 225.2 | Baseline Collection | Collect repeated pinned baseline samples, variance and versioned provenance on the named runner. | 28 |
| 225.3 | Timing Gate | Implement separate threshold/dispersion pass-fail logic and retain all raw candidate samples. | 28 |
| 225.4 | Negative Cases | Reject changed workload, missing/old baseline, single sample, unknown host and variance/threshold overclaim. | 22 |
| 225.5 | Hosted Repetition | Run multiple hosted trials, evaluate noise and adjust only by documented policy review. | 20 |
| 225.6 | Claim Calibration | Update benchmark docs and pass report, metadata and applicable quality checks. | 18 |

### Deliverables

- Versioned selected baseline, repeated raw samples, documented dispersion
  policy and one enforced CI timing gate.
- Explicit non-claims for portable speed and platform parity.

### Completion Criteria

The gate produces reproducible runner/workload-specific pass/fail evidence
with retained raw data. Existing threshold-free freshness remains distinct.

### Total Estimate

140 hours.

## Sprint 226: Static Candidate And Installed First-Use Closeout

**Duration:** 14 days, approximately 140 hours.

**Goal:** Validate one exact static source-release candidate and close the
external user's installed first-use route; reconcile Epic 20 claims.

### Prerequisites from Previous Sprints

- Sprints 217-225 changes merged or explicitly blocked with recorded evidence.
- Sprint 214 static release checklist and Sprint 219 width-aware installed
  package; a selected candidate commit and release owner.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 226.1 | Candidate Freeze | Select clean commit, version/changelog, source/license manifest, platform tier and rollback owner. | 20 |
| 226.2 | Reproducible Source Archive | Build two independent archives, compare hashes, extract and map source to installed files. | 28 |
| 226.3 | Candidate Platform Proof | Run exact-candidate static installed consumers and required quality/hosted gates for every claimed platform. | 26 |
| 226.4 | First-Use Audit | Exercise clean-prefix README/INSTALL/example/API path; trim duplicated historical prose and validate links/header coverage. | 24 |
| 226.5 | Readiness Decision | Review artifact provenance, checklist, approvals, rollback and explicit static-only/release/provider boundaries. | 22 |
| 226.6 | Epic Closeout | Reconcile completed versus blocked items, publish validation/claim ledger and next-epic residual handoff. | 20 |

### Deliverables

- Candidate-specific static source archive/provenance and installed consumer
  evidence for named platforms.
- Tested first-use path, readiness decision and Epic 20 closeout ledger.

### Completion Criteria

All candidate gates and approvals pass before calling the exact candidate
release-ready. A missing hosted result or owner approval blocks that claim;
no tag, public release, shared ABI, provider package or broad SOTA wording is
inferred. First-use docs remain usable for the existing static source route.

### Total Estimate

140 hours.

## Portfolio Boundaries And Exit Rule

The ten sprints fund 1,400 hours, below the 1,680-hour ceiling (10 x 168).
Each sprint reserves 28 hours against CI variance, numerical investigation
and review. If a hosted/rights/release approval remains unavailable after
reasonable retries, retain the blocker and use remaining capacity on an
already-funded test or documentation gate; **do not** relabel a blocked
promotion complete. Public provider packages, shared-library/dynamic ABI,
hosted Doxygen HTML, broad Windows or external solver parity, all-family
conformance, and state-of-the-art performance ranking remain explicit future
decisions. The [todo](reviews/todo-codex-2026-10-19.md) gives their entry bars.
