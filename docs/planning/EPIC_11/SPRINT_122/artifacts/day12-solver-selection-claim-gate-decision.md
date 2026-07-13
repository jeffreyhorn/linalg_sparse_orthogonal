# Sprint 122 Day 12 Solver-Selection Claim Gate Decision

## Purpose

Day 12 converts the Day 11 solver-selection claim inventory into explicit
claim gates. It decides whether Sprint 122's bounded SVD, QR, partial-SVD, and
helper-ownership evidence justifies public wording expansion.

## Decision

Do not expand public solver-selection wording in Sprint 122.

The current public surfaces already route users to the right workflows without
overclaiming external parity, platform parity, performance portability, package
support, ABI stability, or state-of-the-art behavior. Sprint 122 adds useful
bounded evidence, but the evidence is not broad enough to change public
solver-selection promises.

Allowed Day 12 output is therefore this claim-gate decision artifact and the
Sprint 122 working-notes update. No README, solver-selection, tutorial,
examples, benchmark, install, public header, package, CMake, Makefile, or CI
wording changes are made.

## Unchanged Wording Rationale

| Surface | Decision | Rationale |
| --- | --- | --- |
| `README.md` | Unchanged | Current capability wording is workflow and availability oriented; it does not claim broad external parity or portable performance. |
| `docs/solver_selection.md` | Unchanged | The guide chooses by problem shape and already fences benchmarks as local measurement and state-of-the-art parity as unclaimed. |
| `docs/tutorial.md` | Unchanged | Tutorial wording teaches supported API workflows and keeps repeated-run support narrow. |
| `examples/README.md` | Unchanged | Examples remain teaching surfaces, not oracle, benchmark, or parity owners. |
| `benchmarks/README.md` | Unchanged | Benchmark rows are already described as local measurement artifacts, not portable performance guarantees. |
| `INSTALL.md` | Unchanged | Platform, package, static-first install, Windows CMake subset, and ABI caveats are already explicit. |
| `docs/maintainer_guide.md` | Unchanged in Sprint 122 | Older Sprint 102/103 evidence tables are historical snapshots. A future maintainer-guide refresh may add a Sprint 122 snapshot, but no public wording needs to depend on it now. |

## Claim Gates

| Claim Area | Gate Before Wording Can Expand | Current Sprint 122 Status |
| --- | --- | --- |
| SVD external evidence | Multiple bounded external fixtures covering full-rank, rank-deficient, rectangular, rank-threshold, pseudoinverse, low-rank, vector/orthogonality, tolerance, and failure interpretation. | Not met. Sprint 122 adds one rank-deficient singular-value fixture beyond the prior full-rank pilot. |
| QR external evidence | External dense-reference fixtures covering compatible/incompatible least-squares, rank-deficient least-squares, underdetermined minimum-norm, residual semantics, Q/economy semantics, skip policy, and unsupported cases. | Not met. Sprint 122 adds one incompatible 4x2 least-squares fixture. |
| Partial-SVD external evidence | Top-k values, vector residuals, sign/subspace handling, convergence budgets, repeated/clustered spectra, rectangular/rank-deficient fixtures, and failure interpretation. | Not met. Sprint 122 adds one diagonal top-k value fixture only. |
| Minimum-norm helper ownership | Behavior-specific helper migration with QR/COLAMD/SVD-pinv/refinement/fallback/SuiteSparse scenario ownership preserved and full C validation. | Not met. Sprint 122 explicitly defers migration. |
| Bidiagonal/Golub-Kahan helper ownership | Dedicated bidiagonal/GK helper owner with transposed-wide semantics, implicit Householder reconstruction, explicit U/V reconstruction, and bidiagonal QR iteration preserved. | Not met. Sprint 122 rejects general SVD-helper consolidation and allows only future limited extraction. |
| Cross-solver parity | Corpus-spanning generated-RHS/external-reference matrix across direct and iterative solvers, with solver assumptions, tolerance bands, unsupported cases, and platform effects recorded. | Not met. Sprint 120 has one small SPD generated-RHS LU/Cholesky/QR/CG pilot. |
| Support-level platform wording | Reviewed Linux/macOS/Windows surfaces covering the same build, test, CTest, install, thread/fuzz/property, and consumer lanes or explicit staged exclusions retired. | Not met. Existing platform tiers remain the source of truth. |
| Performance/state-of-the-art wording | Portable benchmark protocol with controlled compiler/platform/backend/thread settings, fixture corpus, repeats/statistics, thresholds, and reviewed artifact ownership. | Not met. Benchmarks remain local measurement. |
| Package/ABI wording | Product decision, shared-library build/install proof, runtime loader proof, ABI policy/tests, package metadata, and platform-specific consumer proof. | Not met. Static-first install and ABI non-claims remain. |

## External Evidence Threshold Table

| Evidence Type | Minimum Threshold For Public Claim | Current Evidence |
| --- | --- | --- |
| Bounded fixture claim | Named fixture, fixture matrix/RHS, reference source, tolerance, skip behavior, failure interpretation, and focused validation. | Met for selected Sprint 122 SVD, QR, and partial-SVD fixtures. |
| Family-local external confidence | At least one fixture per important behavior class plus family-local non-claim register. | Not met for SVD, QR, or partial SVD. |
| Broad external parity | Maintained external reference or package comparison over representative matrix families, vectors/subspaces where applicable, platform policy, and recurring validation. | Not met. |
| Public solver-selection change | Evidence-linked wording, unchanged unsupported-case fences, no implied API/build/platform expansion, and docs validation. | Not needed in Sprint 122. |
| Support-level promotion | Reviewed CI/build/test/install evidence and updated maintainer/platform truth. | Not met. |

## Non-Claim Update

Sprint 122 explicitly preserves these non-claims:

- no broad LAPACK, NumPy, SciPy, PETSc, Trilinos, Eigen, SuiteSparse, ARPACK,
  or vendor-backend parity;
- no broad external dense-library parity for SVD, QR, partial SVD, direct
  solvers, iterative solvers, or eigensolvers;
- no singular-vector, Q-basis, Ritz-vector, or subspace external parity;
- no broad minimum-norm or low-rank global optimality claim;
- no broad partial-SVD convergence-budget or vector/subspace claim;
- no broad cross-solver equivalence or solver superiority claim;
- no portable performance, scalability, memory, fill-reduction, or
  state-of-the-art claim;
- no package-manager distribution support;
- no shared-library or dynamic ABI stability;
- no equal Linux/macOS/Windows reviewed support;
- no Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity;
- no public API expansion from test-helper, external-reference, or
  maintainer-only proof work.

## Adoption-Surface Handoff Requirements

| Future Surface | Required Input Before Editing |
| --- | --- |
| README capability wording | Final earned-claim table from a closeout sprint plus confirmation that capability wording remains API/workflow oriented. |
| Solver-selection guide | Family-local claim table proving any stronger SVD, QR, partial-SVD, cross-solver, or support-level wording. |
| Tutorial and examples | Public API workflow examples only; no example should become an oracle, benchmark, or parity owner. |
| Benchmarks docs | Benchmark artifact context, repeat/statistics policy, platform/compiler/backend/thread fields, and no portable-performance reading. |
| Maintainer guide | Current evidence snapshot with named test owners, trust boundaries, and non-claims if future docs need to cite Sprint 122 evidence. |
| Install/platform docs | Reviewed CI/platform/package evidence, exact staged exclusions, and product decisions for ABI or package-manager claims. |

## Residual Owners

| Residual | Future Owner |
| --- | --- |
| Broader SVD external fixture matrix and dense-reference trust model. | Future SVD oracle/corpus sprint. |
| QR external rank-deficient, compatible, underdetermined/minimum-norm, and Q/economy semantics. | Future QR oracle sprint. |
| Partial-SVD vector/subspace/convergence external semantics. | Future partial-SVD numerical oracle sprint. |
| Minimum-norm helper migration. | Future QR solve / minimum-norm consolidation owner. |
| Bidiagonal/Golub-Kahan helper extraction. | Future Bidiagonal/GK-specific maintainability owner. |
| Public solver-selection wording refresh. | Future adoption or final claim-recalibration sprint after broader evidence lands. |

## Validation Notes

Day 12 changed documentation only. Required validation is `git diff --check`
and a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.
The branch passed `make format`, `make lint`, and `make test` after the Day 8
C/header changes; Day 12 adds no new code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 6 is complete. | Complete | Day 11 created the inventory; Day 12 records the claim-gate decision and thresholds. |
| Future public wording has explicit evidence thresholds. | Complete | Claim gates, threshold table, and adoption-surface handoff requirements are recorded. |
| No current public docs are expanded beyond validated Sprint 122 evidence. | Complete | Public docs are unchanged; Sprint 122 evidence remains bounded in this artifact. |
