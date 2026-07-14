# Sprint 122 Day 2 Residual Dedupe and Owner Map

## Purpose

Day 2 completes Sprint 122 Item 1 by deduplicating Sprint 121 residual deferred
debt and assigning every remaining residual to a concrete Sprint 122 owner. Each
entry is classified as active, duplicate, non-claim constraint, or future-sprint
handoff so later days do not re-audit completed work or imply unsupported public
claims.

## Residual Classification Summary

| Classification | Count | Meaning |
| --- | ---: | --- |
| Active Sprint 122 residual | 6 | Work explicitly owned by Days 3-12 and validated during Days 13-14. |
| Duplicate of completed Sprint 121 work | 11 | Work to use as input, not redo. |
| Non-claim constraint | 11 | Boundary statements that must remain true unless future evidence changes. |
| Future-sprint handoff candidate | 4 | Work to record only if Sprint 122 decides not to implement or update now. |

## Active Residual Owner Map

| Residual | Sprint 122 Owner | Decision Type | Prerequisites | Proof Gate |
| --- | --- | --- | --- | --- |
| Additional SVD external fixtures beyond `svd_rect_fullrank_6x4` | Days 3-4, SVD external oracle owner | Implement bounded fixture or explicitly defer | Sprint 121 SVD pilot, taxonomy labels, skip behavior, tolerance baseline | Fixture adds evidence not already covered by the 6x4 pilot; tolerance, skip path, failure interpretation, and non-claim wording are explicit. |
| QR external dense-reference lane | Days 5-6, QR external oracle owner | Design, implement, or explicitly defer | Day 3 QR audit, Day 4 taxonomy, Day 9 deterministic QR/LS fixtures | Fixture size, reference values, tolerance, skip behavior, unsupported-platform behavior, and failure interpretation are explicit before code changes. |
| Partial-SVD external parity design | Days 7-8, partial-SVD external oracle owner | Design, implement, or explicitly defer | Day 2 SVD audit, Day 10 partial-SVD fixture expansion, internal full-SVD reference baseline | Vector, subspace, ordering, convergence, tolerance, and non-claim semantics are separate from full-SVD singular-value parity. |
| Minimum-norm helper ownership migration | Day 9, QR/minimum-norm helper owner | Move helper ownership or preserve current ownership with rationale | Day 3 QR audit, Day 5 helper plan, Day 7 QR helper extraction, Day 9 least-squares expansion | A clearer QR/minimum-norm owner exists and movement does not hide COLAMD/reordering or minimum-norm scenario semantics. |
| Bidiagonal/Golub-Kahan helper extraction boundary | Day 10, low-level SVD helper owner | Extract, preserve local ownership, or defer | Day 2 SVD audit, Day 5 helper plan, Day 6 SVD helper extraction | Specialized transpose, reconstruction, and bidiagonal semantics stay visible; no generic SVD helper hides low-level proof meaning. |
| Solver-selection claim gate | Days 11-12, docs/product claim owner | Update public wording or explicitly keep current wording | Decisions from Days 3-10 and Sprint 121 non-claim register | External/support-level evidence justifies any wording change; otherwise no-update rationale preserves current claim boundary. |

## Duplicate Fence

| Sprint 121 Work | Day 2 Disposition | Rationale |
| --- | --- | --- |
| SVD, partial-SVD, low-rank, rank, and pseudoinverse audit | Duplicate; use as input | Sprint 121 Day 2 already inventoried proof owners and gaps. |
| QR, least-squares, rank-deficient, and minimum-norm audit | Duplicate; use as input | Sprint 121 Day 3 already inventoried QR and minimum-norm ownership. |
| Matrix taxonomy design | Duplicate; use as input | Sprint 121 Day 4 already defined fixture and evidence classes. |
| Bounded helper extraction plan | Duplicate; use as input | Sprint 121 Day 5 already set helper constraints and candidate boundaries. |
| First SVD helper extraction batch | Duplicate unless Day 10 proves specialized boundary change | Sprint 121 Day 6 completed selected SVD helper movement. |
| First QR helper extraction batch | Duplicate unless Day 9 proves minimum-norm ownership change | Sprint 121 Day 7 completed selected QR helper movement. |
| Deterministic rank-deficient fixture expansion | Duplicate; use as internal baseline | Sprint 121 Day 8 already expanded deterministic rank-deficient evidence. |
| Least-squares and pseudoinverse fixture expansion | Duplicate; use as internal baseline | Sprint 121 Day 9 already expanded compatible, incompatible, and minimum-norm evidence. |
| Low-rank and partial-SVD fixture expansion | Duplicate; use as internal baseline | Sprint 121 Day 10 already expanded bounded low-rank and partial-SVD evidence. |
| Bounded SVD external-reference pilot | Duplicate; use as baseline | Sprint 121 Days 11-12 already designed and implemented `svd_rect_fullrank_6x4`. |
| Final validation and closeout package | Duplicate; use as residual source | Sprint 121 Days 13-14 already validated and recorded residuals. |

## Non-Claim Constraints

These are not Sprint 122 implementation tasks. They are constraints that must
remain true unless a later sprint creates sufficient evidence and explicitly
updates the claim boundary.

| Constraint | Sprint 122 Handling |
| --- | --- |
| No LAPACK parity claim | Preserve in all SVD, QR, and partial-SVD decisions. |
| No SciPy or NumPy parity claim | Preserve unless a future product decision adds a maintained external dependency lane. |
| No SuiteSparse, PETSc, Trilinos, or Eigen parity claim | Preserve; SuiteSparse fixture loading remains smoke/corpus evidence, not parity. |
| No broad external dense-library parity claim | Preserve; bounded fixtures do not imply broad dense parity. |
| No singular-vector or subspace external parity claim | Preserve unless Day 7-8 explicitly designs bounded semantics. |
| No QR external parity claim | Preserve unless Day 5-6 implements and validates a bounded QR external lane. |
| No partial-SVD external parity claim | Preserve unless Day 7-8 implements and validates bounded partial-SVD external evidence. |
| No low-rank or pseudoinverse global optimality claim | Preserve; current evidence remains deterministic fixture coverage. |
| No package, install, platform, or ABI claim | Out of Sprint 122 scope. |
| No performance or scalability claim | Out of Sprint 122 scope. |
| No state-of-the-art or public API claim | Out of Sprint 122 scope. |

## Dependency Order

| Dependency | Required Order | Reason |
| --- | --- | --- |
| Additional SVD fixture decision before solver-selection wording | Days 3-4 before Days 11-12 | Public wording cannot cite broader SVD evidence until the additional fixture decision is known. |
| QR external lane decision before solver-selection wording | Days 5-6 before Days 11-12 | QR support claims depend on whether external evidence was added or explicitly deferred. |
| Partial-SVD external parity decision before solver-selection wording | Days 7-8 before Days 11-12 | Partial-SVD wording must know whether external parity is unsupported, deferred, or bounded. |
| Minimum-norm helper decision before solver-selection wording | Day 9 before Days 11-12 | Minimum-norm ownership affects whether docs can point to a coherent QR/minimum-norm proof owner. |
| Bidiagonal/Golub-Kahan boundary before final validation | Day 10 before Days 13-14 | Validation and retrospective must know whether specialized helper debt remains deferred. |
| All active residual decisions before closeout | Days 3-12 before Days 13-14 | Closeout must report final validation, non-claims, and future-sprint handoffs accurately. |

## Proof-Gate Checklist

| Gate | Applies To | Required Evidence |
| --- | --- | --- |
| Fixture uniqueness | SVD, QR, partial-SVD | The fixture covers a behavior not already proven by Sprint 121 deterministic fixtures or the 6x4 SVD pilot. |
| Tolerance ownership | SVD, QR, partial-SVD, helpers | Tolerance is fixture-specific and does not hide solver, vector, or subspace semantics. |
| Skip-path clarity | External-reference lanes | Unsupported optional inputs produce deterministic skips or explicit no-op rationale. |
| Failure interpretation | External-reference lanes | A failure message identifies whether the issue is reference mismatch, algorithm regression, unsupported fixture, or unsupported optional dependency. |
| Helper semantic preservation | Minimum-norm and Bidiagonal/Golub-Kahan helpers | Moved helpers keep scenario ownership and specialized reconstruction, transpose, norm, and residual semantics visible. |
| Claim traceability | Solver-selection wording | Every public wording change maps to concrete validated evidence and retains explicit non-claims. |
| Test-surface accounting | Any code or build change | Test membership, CMake/CTest counts, and platform effects are known before commit. |

## Future-Sprint Handoff Candidates

| Candidate | Handoff Trigger | Future Owner |
| --- | --- | --- |
| Additional SVD fixture expansion beyond bounded Sprint 122 scope | Day 3-4 finds useful fixture families but too much implementation scope | Future SVD oracle or corpus sprint. |
| Broad QR external parity | Day 5-6 keeps Sprint 122 to design or one bounded lane | Future QR oracle sprint after fixture and reference model mature. |
| Broad partial-SVD subspace parity | Day 7-8 determines vector/subspace semantics need more design than one sprint can safely own | Future partial-SVD numerical oracle sprint. |
| Public solver-selection rewrite | Days 11-12 conclude evidence is still internal or too narrow | Future docs/adoption sprint after broader support-level evidence lands. |

## Validation Notes

Day 2 changed documentation only. Required validation is `git diff --check` and
a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 1 is complete. | Complete | Active residual owner map, duplicate fence, dependencies, and proof gates are recorded. |
| Every residual is owned, deferred, or explicitly rejected as duplicate. | Complete | See residual classification summary, active owner map, duplicate fence, and future handoff table. |
| No residual depends on later work without a documented prerequisite. | Complete | See dependency order and proof-gate checklist. |
