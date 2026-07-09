# Day 5 README Follow-Through

## Purpose

Day 5 applies only the necessary README wording fixes from the Day 4 boundary
audit. The goal is to preserve a compact adoption-facing README while keeping
CI, install, package/platform, ABI, and benchmark claims evidence-bounded.

## Applied README Update

| File | Location | Before | After | Reason |
|---|---|---|---|---|
| `README.md` | Build section before "With Make" | "cross-platform install, downstream-consumer, or package-manager detail" | "cross-platform install, downstream-consumer, or install-support detail" | Avoid implying package-manager support exists. Sprint 115 explicitly deferred package-manager support until actual recipes and install/consumer proof exist. |

## No-Edit Decisions

| README area | Decision | Rationale |
|---|---|---|
| Start Here | No edit | The section routes first-use adoption to the right deeper surfaces without becoming maintainer policy. |
| Continuous integration paragraph | No edit | The paragraph keeps Linux, macOS, Windows, reviewed/staged, ThreadSanitizer, and benchmark-signal boundaries explicit. |
| Installation section | No edit | It accurately describes `pkg-config` and `find_package(Sparse)` against the maintained static package surface and explicitly defers shared-library packaging. |
| Performance Characteristics | No edit | It keeps benchmark commands in `benchmarks/README.md` and says emitted rows are branch-local measurement artifacts, not portable performance guarantees. |
| API Overview | No edit | Long, but still a public reference map rather than maintainer policy or an unsupported claim. |

## Claim-Boundary Validation

| Guardrail | README state after Day 5 |
|---|---|
| No package-manager support claim | Preserved; the ambiguous "package-manager detail" wording was removed. |
| No shared-library package support claim | Preserved; README still says shared-library packaging is intentionally deferred. |
| No dynamic ABI guarantee | Preserved; README does not advertise ABI stability. |
| No Windows install-validation parity claim | Preserved; README keeps Windows to the reviewed CMake subset and CMake-first consumer story. |
| No full macOS install/export parity claim | Preserved; README describes reviewed Apple Clang plus supplemental Homebrew GCC/static-first evidence. |
| No universal performance claim | Preserved; README treats benchmark rows as branch-local measurement artifacts. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| README support and CI wording is compact and evidence-bounded | Complete. |
| No unsupported package/platform, ABI, or benchmark claim is introduced | Complete. |
| Touched documentation passes hygiene checks | Complete. |

## Validation Notes

- Day 5 touched `README.md` and Sprint 116 planning documentation.
- `git diff --check` passed.
- Focused trailing-whitespace scan over `README.md` and
  `docs/planning/EPIC_10/SPRINT_116` passed.
- Focused scan confirmed the old "package-manager detail" phrase is no longer
  present in `README.md`.
- No `.c` or `.h` files were modified.
- No code, workflow, Make/CMake, script, install, package, or ABI behavior was
  changed.
