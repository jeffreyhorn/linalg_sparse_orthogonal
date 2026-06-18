# Sprint 79 Day 4 - First Assurance Boundary

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Freeze the first final-assurance fence so Sprint 79 spends its final implementation budget on the strongest lifecycle/property seam without reopening broader proof churn or support-surface cleanup.

## Main Result
Sprint 79 now has one explicit first final-assurance fence instead of a generic numerical-assurance backlog.

Required first landing:
- `tests/test_integration.c`
- `tests/test_fuzz.c`

Support only if the first landing forces it:
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- `docs/maintainer_guide.md`
- `README.md`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Explicitly deferred:
- `tests/test_qr.c`
- `tests/test_reorder_nd.c`
- benchmark/reporting surfaces
- install/export proof scripts
- workflow YAML surfaces

## First-Lane Interpretation
The strongest first Sprint 79 lane is now fixed as:
- public lifecycle/property assurance

The first batch should therefore:
- target the public callback/cancel and repeated-run lifecycle truth seam
- use `tests/test_integration.c` as the public oracle owner
- use `tests/test_fuzz.c` as the bounded seeded property owner
- move family-local direct-solver tests only if the public/property landing cannot be expressed cleanly without them

## Why This Is First
This boundary wins the Day 4 rerank because it offers the best combination of:
- closeout payoff
- proof clarity
- bounded landing value
- cross-surface truth payoff

It is stronger than the alternatives because:
- platform-confidence-limited property coverage is real, but the current docs already state the Windows fuzz boundary truthfully
- family-local differential/oracle follow-through remains valuable, but reads as later assurance expansion rather than the strongest unresolved contradiction center
- support-surface wording drift is weaker still because the current docs already carry the strongest caveats directly

## Preserved Non-Goals
Sprint 79's first assurance landing must preserve these limits:
- no broad proof campaign across all solver families
- no new benchmark, package, or workflow mechanics
- no public-surface rewrite unless the first landing forces narrow wording follow-through
- no summary or retrospective work before the first proof gap is bounded

## Exit State
- The first Sprint 79 final-assurance lane is explicit.
- First-batch, support-only, and deferred surfaces are separated in writing.
- Day 5 can now design one bounded implementation/proof contract from a fixed boundary.
