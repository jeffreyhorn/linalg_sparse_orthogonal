# Day 11 Performance Wording Follow-Through

## Purpose

Day 11 applies the bounded wording cleanup identified in the Day 10
performance evidence audit. The goal is to remove broad performance language
that is not fenced by local evidence while preserving useful benchmark and
algorithm reference context.

## Applied Documentation Update

| File | Location | Before | After | Reason |
|---|---|---|---|---|
| `docs/algorithm.md` | Preconditioner table, ILU(0) row | `Good (3-1000x speedup)` | `Workload-dependent acceleration; benchmark locally` | The old wording read like a broad performance guarantee without naming fixture, workload, machine, compiler, thread count, or benchmark context. |

## No-Edit Decisions

| Area | Decision | Rationale |
|---|---|---|
| `README.md` performance handoff | No edit | Already says benchmark rows are branch-local measurement artifacts, not portable performance guarantees. |
| `docs/solver_selection.md` benchmark and preconditioner wording | No edit | Already states benchmark output is branch-local and preconditioners are not universal guarantees. |
| `benchmarks/README.md` report semantics | No edit | Already distinguishes threshold-free reports, hard gates, local evidence, and portability limits. |
| `INSTALL.md` support/platform wording | No edit | Describes reviewed and supplemental support lanes, not performance claims. |
| Broader `docs/algorithm.md` historical measurement tables | No edit | The Day 9 positioning note frames the document as technical background, and most measurements are tied to named fixtures or sprint artifacts. |

## Claim-Boundary Validation

| Guardrail | State after Day 11 |
|---|---|
| Public performance wording avoids universal speed claims | Complete. |
| Changed claim remains tied to local evidence | Complete; the replacement explicitly says to benchmark locally. |
| No benchmark command, report bundle, CI lane, or quality gate changed | Preserved. |
| No package/platform or ABI claim introduced | Preserved. |
| Performance residuals are suitable for Sprint 117 closeout | Complete; broader algorithm-doc splitting remains optional future work. |

## Remaining Performance Residuals

- Consider a future split of `docs/algorithm.md` into a concise public
  algorithm reference plus a historical measurement appendix if scanability
  becomes a recurring issue.
- Consider generated benchmark-artifact indexes in a future benchmark sprint,
  but do not add new benchmark commands or report semantics in Sprint 116.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Public performance wording avoids universal speed claims | Complete. |
| All changed claims remain tied to local evidence | Complete. |
| Touched documentation passes hygiene checks | Complete. |

## Validation Notes

- Day 11 touched `docs/algorithm.md` and Sprint 116 planning documentation.
- `git diff --check` passed.
- Focused trailing-whitespace scan over `README.md`,
  `benchmarks/README.md`, `docs/algorithm.md`, and
  `docs/planning/EPIC_10/SPRINT_116` passed.
- Focused scan confirmed the old `3-1000x` / `3-1000x speedup` wording is no
  longer present in `docs/algorithm.md`.
- No `.c` or `.h` files were modified.
- No code, workflow, Make/CMake, benchmark, package, install, or ABI behavior
  changed.
