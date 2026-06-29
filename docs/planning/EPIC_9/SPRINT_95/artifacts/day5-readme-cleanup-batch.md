# Sprint 95 Day 5: README Cleanup Batch

## Purpose

Day 5 lands the first README rewrite batch from the Day 4 boundary plan. This
batch focuses on the highest-value public cleanup: remove sprint-ledger content,
collapse duplicated proof and support policy, and keep the README as the
adoption front door.

## Changed README Areas

| Area | Change | Owner after cleanup |
|---|---|---|
| Capability heading | `Features` became `Current Capabilities`. | README |
| Progress/cancel callbacks | Rewritten as current behavior without sprint/day provenance. | Headers for exact contracts |
| Symmetric eigensolver API map | Removed sprint labels from the section title and shift-invert note. | README summary plus headers |
| Performance characteristics | Replaced long speedup tables and CSC/LDL^T chronology with a compact benchmark summary. | `benchmarks/README.md` |
| Testing and quality | Replaced the sprint-named test ledger, dead-code detail, reviewed-quality policy, CI matrix, and readiness checklist with a compact command map. | `docs/maintainer_guide.md`, Makefile, tests |
| Installation | Shortened to a quick install summary and downstream consumer commands. | `INSTALL.md` |
| Documentation links | Added explicit tutorial, examples, and benchmarks links. | Respective docs |

## Preserved README Responsibilities

- project identity
- current capability summary
- workflow chooser
- build and local command map
- quick start code path
- API overview
- thread-safety and limitations
- compact testing/quality pointer
- compact install pointer
- documentation owner links

## Removed From README Front Door

- detailed CSC Cholesky and LDL^T sprint history
- historical benchmark tables and end-of-sprint benchmark links
- long proof-owner lists under performance sections
- full sprint-named test inventory
- dead-code workflow execution detail
- reviewed-quality policy interpretation
- cross-platform CI matrix detail
- quality readiness checklist
- install-validation proof detail

## Retained Claims And Destinations

| Retained claim | README handling | Destination for detail |
|---|---|---|
| Dispatch-backed CSR/CSC paths exist for large workloads. | Kept as a high-level benchmark/capability note. | `benchmarks/README.md`, public headers |
| Repeated-run direct, iterative, and eigensolver workflows exist. | Kept in capability and workflow sections. | tutorial, examples, benchmarks, headers |
| `make quality-review-full` is the strongest local reviewed baseline. | Kept as a compact testing/quality pointer. | maintainer guide and Makefile |
| Install supports Makefile and CMake downstream consumer paths. | Kept as a compact installation summary. | `INSTALL.md` |
| Historical measurements and old sprint evidence belong in planning artifacts. | Kept as an explicit archive-boundary rule. | `docs/planning/**` |

## Residual README Follow-Up

- Day 6 should check tutorial and quick-start alignment now that README is
  shorter.
- Day 7 should review algorithm/reference docs for chronology that no longer
  belongs in README.
- Day 8 should clean public header comments that still expose sprint history;
  that will require full quality checks because headers are `.h` files.
- Day 10-11 should decide whether README needs any updates after proof-owner
  naming cleanup.

## Day 5 Result

The README now reads more like product documentation and less like a sprint
archive. The most detailed benchmark, proof, install, and maintainer-policy
content is behind the intended owner links, while the README still preserves the
current public capability and workflow story.
