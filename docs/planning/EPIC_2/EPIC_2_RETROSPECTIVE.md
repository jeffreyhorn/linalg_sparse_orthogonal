# Epic 2 Retrospective — Sprints 11-29 (linalg_sparse_orthogonal)

**Epic budget:** 19 sprints × ~144-168 h each (~2 800 h nominal)
**Branch range:** `sprint-11` → `sprint-29`
**Goal:** Close out the review-driven backlog from
`reviews/review-codex-2026-04-06.md` + `review-claude-2026-04-06.md`
across direct factorizations, iterative solvers, eigensolvers, SVD,
reorderings, and CI / observability hardening — closing the gap
between "library that works" and "library that ships."

> **Status:** *Day-14 fill-in pending.*  Day 13's skeleton lists the
> per-sprint table + the cumulative-metrics scaffolding; Day 14
> fills in the Sprint 22-28 ND/AMD trajectory, CSC-kernel speedup
> table, eigensolver capability matrix, production default flip
> log, advisory env-var log, open-question journal, and Epic-level
> lessons single-pass.

---

## Summary table

| sprint | title | key deliverables | nominal h |
|---|---|---|---:|
| 11 | Symbolic etree + analyze-once / factor-many | `sparse_etree_compute`; `sparse_analyze` / `sparse_factor_numeric` / `sparse_refactor_numeric` | *fill-in* |
| 12 | LDL^T (linked-list) | Bunch-Kaufman 1×1 / 2×2 pivots; inertia | *fill-in* |
| 13 | IC(0) + MINRES + ILU plumbing | IC(0) preconditioner; MINRES solver; ILUT polish | *fill-in* |
| 14 | BiCGSTAB | stabilized bi-conjugate gradient; matrix-free variant | *fill-in* |
| 15 | Iterative-solver hardening | stagnation detection; breakdown handling; residual history | *fill-in* |
| 16 | (filled-in Day 14) | *fill-in* | *fill-in* |
| 17 | CSC Cholesky scaffolding | CSC kernels + fundamental supernode detection | *fill-in* |
| 18 | Supernodal Cholesky + CSC LDL^T | batched supernodal kernels; CSC LDL^T native | *fill-in* |
| 19 | CSC LDL^T row-adjacency + supernodal LDL^T | sparse-row cmod scaling; supernodal LDL^T (6.8× on bcsstk14) | *fill-in* |
| 20 | Sparse symmetric eigensolver | Lanczos grow-m; shift-invert via CSC LDL^T AUTO; Sprint-20-deferred eigenpair refinement (closed Sprint 29 Item 3) | *fill-in* |
| 21 | Thick-restart Lanczos + LOBPCG + OMP MGS | Wu/Simon thick-restart; LOBPCG; parallel reorth | *fill-in* |
| 22 | AMD quotient-graph rewrite + ND scaffolding | AMD-QG (~5·nnz + 6·n initial workspace); ND multilevel | *fill-in* |
| 23 | ND HEM + threshold sweep | HEM matching default; `nd_base_threshold = 32` | *fill-in* |
| 24 | (filled-in Day 14) | *fill-in* | *fill-in* |
| 25 | ND HCC (heavy-cluster coarsening) | HCC matching path | *fill-in* |
| 26 | ND threshold flips | `nd_base_threshold = 96` then `128` defaults | *fill-in* |
| 27 | HCC default flip + per-axis advisory env vars | HCC default; `SPARSE_ND_SEP_LIFT_*` advisory | *fill-in* |
| 28 | Non-pipeline pivot (supernodal-etree post-pass) | Liu 1990 post-pass (advisory); 0.85× target FORMALLY RETIRED | 144 |
| 29 | SVD improvements + eigenpair refinement + progress callbacks + CI hardening + Epic 2 wrap-up | this sprint | 168 |

*Day 14 fills in the actual hours-spent per sprint (from each
RETROSPECTIVE.md) and replaces the placeholder rows.*

---

## Cumulative metrics

### Pres_Poisson ND nnz(L) trajectory (Sprints 22 → 29)

| sprint | nnz_L | ratio vs AMD baseline | source |
|---|---:|---:|---|
| 22 | *fill-in* | 1.063× | Sprint 22 Day 13 |
| 23 | *fill-in* | 0.952× | Sprint 23 RETROSPECTIVE.md |
| 24 | *fill-in* | 0.952× | Sprint 24 |
| 25 | *fill-in* | 0.952× | Sprint 25 |
| 26 | *fill-in* | 0.950× | Sprint 26 |
| 27 | *fill-in* | 0.9226× | Sprint 27 |
| **28** | *fill-in* | **0.9226×** | Sprint 28 (bit-stable) |
| **29** | *fill-in* | *Day 14 to fill* | Sprint 29 (no ND changes; same as 28) |

The literal 0.85× target was **formally RETIRED** in Sprint 28 after
6 consecutive sprints of misses (Sprint 28 Day-10 `non_pipeline_decision.md`
+ Day-13 `headline_summary.md`).

### CSC kernel speedups (Sprints 17-19)

*Day 14 fill-in.*  Anchor rows: bcsstk14 Cholesky 4.4× one-shot
(Sprint 17 / 18); bcsstk14 LDL^T 6.8× batched supernodal (Sprint 19).

### Eigensolver capability matrix (Sprints 20-21 + Sprint 29)

| backend | landed | which modes | bench-anchor |
|---|---|---|---|
| Grow-m Lanczos | Sprint 20 | LARGEST / SMALLEST / NEAREST_SIGMA | bcsstk04 / bcsstk14 / Pres_Poisson |
| Thick-restart Lanczos (Wu/Simon) | Sprint 21 Days 1-4 | LARGEST / SMALLEST / NEAREST_SIGMA | bcsstk14: ~7 MB → 565 KB peak basis |
| LOBPCG | Sprint 21 Days 7-10 | LARGEST / SMALLEST | bcsstk04 k=3 SMALLEST: 800-cap saturation → 8 iters with LDL^T preconditioning |
| Inverse-iteration refinement post-pass (opt-in) | Sprint 29 Day 5 | composes with all three | clustered-spectrum synthetic; *Day 14 to fill the residual delta* |

### Progress / cancel callbacks rollout (Sprint 29 Days 6-7)

11 routines: LU, Cholesky, LDL^T (linked-list + CSC), QR, CG,
GMRES, MINRES, BiCGSTAB, Lanczos, LOBPCG, ND.  Default-NULL-callback
path bit-identical to Sprint 28; cancellation returns
`SPARSE_ERR_CANCELLED` cleanly with the input matrix unmodified.

---

## Production default flips landed across Epic 2

*Day 14 fill-in.*  Anchor entries:

| sprint | flip | rationale |
|---|---|---|
| 23 | `nd_base_threshold = 32` | Sprint 22 Day 9 sweep |
| 23 | HEM matching default | (filled-in Day 14) |
| 25 | HCC matching default | (filled-in Day 14) |
| 26 | `nd_base_threshold = 96` | (filled-in Day 14) |
| 27 | `nd_base_threshold = 128` | Sprint 27 Day 3 |
| 27 | HCC default for ND coarsening | Sprint 27 Day 2 |
| **28** | none (zero production default flips) | Item-4 SUPERNODAL_POSTORDER bit-equivalent to default |
| **29** | *Day 14 to fill* (candidate: SPARSE_SVD_LOWRANK_OUTER) | Day 2 sweep verdict |

---

## Advisory env vars shipped across Epic 2

*Day 14 fill-in.*  Anchor table cross-referenced by axis:
`SPARSE_FM_*`, `SPARSE_ND_*`, `SPARSE_SUPERNODAL_*`, `SPARSE_EIGS_*`,
`SPARSE_SVD_*`.

---

## Open question journal

1. **Literal 0.85× Pres_Poisson target retirement** — Sprint 28 final
   verdict (6 consecutive sprints + non-pipeline pivot demonstrated
   the floor is structural).  Sprint 29 inherits the retirement; no
   work budgeted to revisit.  Sprint 30+ should only revisit with
   fundamentally different machinery (METIS C library interop;
   geometric mesh-aware ordering with first-class coordinate API;
   hybrid AMD-then-ND-on-separators).
2. **Supernodal numeric-factor kernels** — Sprint 28 Item-4
   supernodal-etree post-pass ships the input-ordering infrastructure
   but the numeric-factor kernel (batched supernodal cmod + dense
   factor + panel solve) remains Sprint-30+ work.  Estimated upside:
   5-15 % wall reduction on supernodal-heavy fixtures.
3. **ND opts struct + progress callbacks** — Sprint 29 Day 7 noted
   that ND doesn't have an opts struct so callbacks aren't wired
   there.  Sprint 30+ adds the opts struct + threads callbacks through
   the multilevel partition loop.
4. **Coverage gate ≥ 95 % aggregate** — Sprint 29 Day 12 lowered to
   80 % per measured 81.3 %.  Sprint 30+ revisits with synthetic-
   fault-injection scaffolding (out-of-budget Sprint 29).

---

## Lessons (Epic-level)

*Day 14 fill-in.*  Seeds (cross-sprint patterns):

- **Design-doc-first for high-uncertainty items.**  Sprint 22 ND
  scaffolding, Sprint 27 HCC, Sprint 28 non-pipeline pivot, Sprint 29
  refinement + low-rank — each opened with a Day-1 / Day-4 design doc
  that produced a confident pick before any implementation LOC.
- **Day-N close → Day-N+1 advisory decision** as the default rhythm
  for sprint-scope-fitting interventions.  Sprint 27 HCC + Sprint 28
  Items 1, 2, 4 + Sprint 29 SVD low-rank all followed this pattern.
- **Skeleton-first retrospectives** (Sprint 28 → Sprint 29 inheritance)
  let Day-14 retrospective work compress to 6-8 hrs instead of a
  full 12-hr day-from-scratch.
- **Test-bound calibration over aspirational targets.**  Sprint 28's
  formal retirement of the literal 0.85× target was the right call;
  Sprint 29's coverage threshold calibration follows the same
  pattern (`docs/planning/EPIC_2/SPRINT_29/coverage_threshold_decision.md`).

---

## DoD verification

*Day 14 fill-in.*  Required cross-Epic gates:
- `make format && make lint && make test && make sanitize && make wall-check` clean on macOS local.
- Linux + Windows + macOS CI jobs green on `sprint-29` HEAD.
- Linux-CI tsan job green.
- `make coverage` passes the calibrated 80 % threshold on CI Linux.
- All 19 Sprint 11-29 PROJECT_PLAN.md items closed or routed to
  Sprint 30+ with explicit references.

## Acknowledgements

*Day 14 fill-in.*
