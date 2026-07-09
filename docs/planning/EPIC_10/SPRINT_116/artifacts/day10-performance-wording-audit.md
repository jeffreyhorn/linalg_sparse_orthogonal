# Day 10 Performance Wording Evidence Audit

## Purpose

Day 10 audits public performance wording across the adoption surface and maps
meaningful claims to local evidence, existing caveats, or a Day 11 cleanup
candidate.

## Inputs Reviewed

| Input | Reviewed for |
|---|---|
| `README.md` | High-level performance story, benchmark handoff, CI/runtime wording, and package/platform caveats. |
| `docs/solver_selection.md` | Solver-family guidance, benchmark handoff, preconditioner language, state-of-the-art non-claims. |
| `benchmarks/README.md` | Benchmark interpretation, report bundles, lane semantics, and portable-performance caveats. |
| `docs/algorithm.md` | Technical/historical performance statements and whether they need extra fencing after Day 9 positioning. |
| `INSTALL.md` | Whether platform/toolchain/install wording is being used as a performance claim. |

## Evidence Map

| Surface | Performance wording | Evidence or boundary | Day 11 decision |
|---|---|---|---|
| `README.md` workflow handoff | Benchmarks measure retained workflow/performance behavior on current machine, compiler, dependency, fixture, and configuration. | Explicitly local and configuration-bound. | No edit. |
| `README.md` performance section | Benchmark rows are branch-local measurement artifacts, not portable performance guarantees. | Direct caveat plus handoff to `benchmarks/README.md`. | No edit. |
| `README.md` direct kernel wording | Dispatch-backed CSR LU and CSC Cholesky/LDL^T avoid linked-list pointer chasing on large matrix workloads. | High-level architecture claim; benchmark details live in benchmark docs. | No edit. |
| `docs/solver_selection.md` benchmark handoff | Benchmark output is branch-local and configuration-sensitive, not a portable timing guarantee. | Direct caveat. | No edit. |
| `docs/solver_selection.md` preconditioner guidance | Preconditioners are acceleration tools, not universal guarantees. | Direct caveat and solver-assumption guidance. | No edit. |
| `docs/solver_selection.md` eigensolver boundary | No nonsymmetric eigensolver support or portable state-of-the-art parity claim. | Direct non-claim. | No edit. |
| `benchmarks/README.md` result interpretation | Benchmarks do not prove portable performance across machine/compiler/OS/backend/runtime/thread/corpus/build dimensions. | Strong direct caveat and result-reading workflow. | No edit. |
| `benchmarks/README.md` report bundles | `bench-canonical-report` is threshold-free reporting; `performance-sentinels` keeps hard pass/fail to `wall-check`; large-matrix guardrails separate structural lanes from supplemental reports. | Explicit local evidence and lane semantics. | No edit. |
| `docs/algorithm.md` top positioning note | Algorithm doc is technical background, not a portable performance guarantee. | Added Day 9. | No edit. |
| `docs/algorithm.md` fixture tables and historical measurements | Most named values point to specific fixtures, sprint artifacts, or local benchmark captures. | Technical background after Day 9 note. | No broad rewrite. |
| `docs/algorithm.md` preconditioner table | ILU(0) quality says "Good (3-1000x speedup)". | Too broad for a standalone table entry; not clearly tied to named local fixtures. | Edit Day 11. |
| `INSTALL.md` platform/toolchain wording | CI/platform rows describe reviewed and supplemental support lanes. | Support truth, not performance claim. | No edit. |

## Unsupported Performance-Claim Candidates

| Candidate | File | Risk | Day 11 action |
|---|---|---|---|
| `Good (3-1000x speedup)` for ILU(0) | `docs/algorithm.md` preconditioner table | Reads like a broad performance guarantee without local evidence or fixture scope. | Downgrade to qualitative, evidence-bounded wording such as "Workload-dependent acceleration; benchmark locally." |

No public README, solver-selection, benchmark, or install wording requires a
Day 11 edit from this audit.

## Downgrade Or No-Edit Checklist

| Item | Decision | Rationale |
|---|---|---|
| Downgrade `docs/algorithm.md` ILU(0) `3-1000x speedup` wording | Edit | Broad range lacks local-evidence fence in the table itself. |
| Leave README performance and benchmark handoff wording unchanged | No edit | Already branch-local and non-portable. |
| Leave solver-selection benchmark/preconditioner caveats unchanged | No edit | Already states acceleration is not universal. |
| Leave benchmark report semantics unchanged | No edit | Already distinguishes threshold-free reports, hard gates, local evidence, and portability limits. |
| Leave install/support wording unchanged | No edit | Does not imply performance support. |
| Defer broader algorithm-doc performance restructuring | No edit for Sprint 116 | Day 9 positioning note fences the document role; Days 10-11 should not rewrite historical evidence. |

## Sprint 117 Closeout Notes

- A future sprint can split `docs/algorithm.md` into public algorithm
  reference and historical sprint evidence if scanability becomes a larger
  concern.
- A future benchmark sprint can add generated indexes from benchmark artifacts
  back into docs, but Sprint 116 should not create new benchmark commands or
  report semantics.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Performance wording is tied to evidence or marked for cleanup | Complete. |
| No universal benchmark claim is left unexamined | Complete. |
| Day 11 can apply bounded wording fixes | Complete; one focused algorithm-doc wording edit is queued. |

## Validation Notes

- Day 10 changed Sprint 116 planning documentation only.
- Performance-facing docs were inspected but not edited.
- No `.c` or `.h` files were modified.
