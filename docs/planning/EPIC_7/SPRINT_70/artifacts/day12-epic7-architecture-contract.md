# Sprint 70 Day 12: Epic 7 Architecture Contract

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Convert the Sprint 70 audits and Day 11 target synthesis into one exact Epic 7
architecture contract so later sprints start from preserved boundaries,
cross-cutting non-goals, and a planning map that matches the live repo and the
Epic 7 project plan.

## Preserved Boundaries by Major Lane

Product-model convergence:

- first center:
  - public direct-workflow boundary
  - owned-factor and repeated-run workflow clarity
- explicitly not first center:
  - broad `SparseMatrix` rewrite
  - repo-wide generic matrix arithmetic redesign
  - conversion/writeback redesign across every compressed path at once

Capability modernization:

- first lane:
  - index-width modernization path
- second lane:
  - scalar-surface preparation and broadening
- later lane:
  - unsymmetric sparse eigensolver expansion
- explicitly not allowed:
  - solving width, scalar breadth, and algorithm-family expansion in one batch

Configuration modernization:

- target:
  - reduce residual env-var and compatibility-only control pressure on the
    highest-value public seams
- preserved rule:
  - typed or internal policy should widen only where precedence and ownership
    stay explicit
- explicitly not allowed:
  - another broad env-var cleanup pass disconnected from public or proof value

Backend/performance work:

- target:
  - deepen maintained compressed-backend maturity on the strongest product
    paths
- preserved rule:
  - benchmark-governed proof remains bounded and threshold free unless a new
    narrower gate is explicitly justified
- explicitly not allowed:
  - performance-story widening that outruns benchmark ownership or fallback
    proof

Benchmark governance:

- target:
  - preserve the canonical maintained surface and improve observability only in
    bounded reviewed ways
- preserved rule:
  - `make bench-canonical-report` remains artifact reporting, not a timing gate
- explicitly not allowed:
  - portable performance claims from local benchmark rows alone

Packaging/platform truthfulness:

- target:
  - preserve the static-first, evidence-based, explicitly asymmetric reviewed
    contract while tightening only what later proof actually supports
- preserved rule:
  - supplemental proof stays supplemental until new reviewed evidence exists
- explicitly not allowed:
  - shared-library maturity marketing, inflated Windows parity claims, or broad
    reviewed install-validation claims without new proof

## Cross-Cutting Non-Goals

Sprint 70 Day 12 fixes the following non-goals for later Epic 7 work:

- no broad rewrite of the library core
- no generic matrix-model redesign that tries to solve all ownership problems
  at once
- no unsupported product marketing claims
- no capability promises without end-to-end proof
- no benchmark or example wording that steals test-owned guarantees
- no drift from maintained reviewed validation truth
- no fake platform/release maturity beyond static-first, evidence-backed
  reality
- no giant-test or large-source breakup wave unless ownership clarity or
  assurance value justifies the proof cost

## Planning Implications for Sprints 71-79

Sprint 71:

- reduce chronology and policy spill from the strongest public surfaces
- do not reopen product-model or platform-contract work unless cleanup truly
  forces it

Sprint 72:

- be the first real product-model convergence sprint
- center the direct-workflow ownership seam before broader matrix-model
  ambitions

Sprint 73:

- continue configuration modernization only on ranked residual controls
- do not become a generic "remove env vars everywhere" pass

Sprint 74:

- land the first real capability seam on index width
- treat scalar breadth as preparation/support, not the same sprint's co-equal
  primary target

Sprint 75:

- deepen backend/performance architecture only where fallback, observability,
  and benchmark proof remain bounded

Sprint 76:

- extend benchmark and proof surfaces only where the widened product or
  backend story needs stronger evidence

Sprint 77:

- tighten packaging/platform/install convergence without widening the reviewed
  platform contract beyond shipped proof

Sprint 78:

- spend large-source, giant-test, and header cleanup budget only on the
  highest-value residual hotspots

Sprint 79:

- be integration, final validation, and closeout
- not another broad feature or architecture sprint

## Plan Alignment Check

The Sprint 70 contract remains aligned with the Epic 7 project plan:

- Sprint 70:
  - baseline, target, and architecture contract only
- Sprint 71:
  - public-surface cleanup first
- Sprint 72:
  - product-model convergence first
- Sprint 73:
  - residual configuration modernization
- Sprint 74:
  - bounded capability modernization led by index width
- Sprint 75+:
  - backend, benchmark, packaging/platform, cleanup, and closeout in that
    order

## Exit State

Sprint 70 Day 12 closes with one explicit Epic 7 architecture contract:

1. preserved implementation boundaries:
   - product-model, capability, configuration, backend/performance, benchmark,
     and packaging/platform lanes each have an allowed center and an explicit
     widening fence
2. cross-cutting non-goals:
   - no broad rewrite, no unsupported marketing, no proofless capability
     promises, and no drift from reviewed validation truth
3. planning implications:
   - Sprints 71-79 now inherit a contract-driven execution map rather than
     broad modernization themes
