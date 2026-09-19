# Day 7: Support Truth Consolidation Batch

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Centralize support/readiness truth and replace repeated caveats with
safe links where the Day 5 architecture allows.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 7 implements the first support-truth consolidation batch after the quick
reference landed. The work reduces repeated caveats in public docs where an
authoritative owner link is safer than duplicated prose, while preserving
inline warnings in locations where users could otherwise infer unsupported
package, ABI, platform, generated API, benchmark, release, or state-of-the-art
claims.

## Changed Surfaces

| File | Change | Boundary retained |
| --- | --- | --- |
| `README.md` | Shortened generated API policy wording and routed users to `docs/api_reference.md`; shortened report/Windows selected-target prose and routed to benchmark/support owners; shortened installation package caveats. | Local-only generated HTML; no release evidence; unpromoted Windows selected freshness; static-first install; no shared-library/dynamic ABI/package-manager/Homebrew support. |
| `docs/tutorial.md` | Consolidated install/support handoff into a single static-first support-matrix route; added explicit benchmark local/selected evidence wording in advanced handoff. | Tutorial remains local build-tree learning path; benchmark output is not portable performance proof. |
| `examples/README.md` | Added benchmark measurement caveat to repeated-run measurement handoff. | Examples are not timing harnesses; benchmark output is local/selected evidence, not portable performance proof. |
| `docs/maintainer_guide.md` | Added support truth and quick-reference routing guidance for future maintainers. | INSTALL remains public support truth; cookbook quick reference remains routing only; owner docs retain proof interpretation. |

## Before/After Caveat Ledger

| Area | Before | After | Decision |
| --- | --- | --- | --- |
| README generated API | Repeated Sprint 179/Sprint 204 local-only explanation and routing rationale. | Short local-only statement plus link to `docs/api_reference.md`. | Safe to shorten because API reference and API docs guards own detail. |
| README report/Windows | Long selected oracle/comparison/benchmark lane explanation plus Windows QR/Cholesky detail. | Short selected-target-only and Windows-unpromoted statement plus links to manifest/benchmark/support owners. | Safe to shorten while retaining no promoted Windows selected freshness. |
| README installation | Long package/ABI/Windows/Homebrew proof detail. | Short static-first summary with explicit unsupported package-manager, ABI, shared-library, and Homebrew boundaries. | Safe to shorten because INSTALL owns support matrix and proof rows. |
| Tutorial install handoff | Repeated static package ownership after install links. | Merged support matrix ownership into the handoff paragraph. | Safe to shorten because tutorial does not claim install support directly. |
| Tutorial benchmark handoff | Listed benchmark docs without a local/selected evidence reminder in the bullet. | Added concise non-portable-performance warning in the benchmark bullet. | Inline warning retained because benchmark commands are high-risk overclaim surfaces. |
| Examples measurement handoff | Listed benchmark binaries without an immediate no-portable-performance reminder. | Added concise local/selected measurement caveat. | Inline warning retained because examples can be mistaken for timing proof. |
| Maintainer guide routing | No Sprint 205-specific support-truth routing section. | Added owner model for support matrix, quick reference, and proof docs. | Needed so future edits preserve the consolidation model. |

## Support Truth Owner Confirmation

| Owner | Day 7 role |
| --- | --- |
| `INSTALL.md#support-readiness-matrix` | Public support/readiness truth for package, platform, ABI, generated API, benchmark, and Windows support status. |
| `docs/cookbook.md#problem-shape-quick-reference` | Compact user routing table only. |
| `docs/solver_selection.md` | Detailed solver-family guidance and selected evidence boundaries. |
| `docs/api_reference.md` | Source-controlled API declaration route and local-only generated API policy. |
| `benchmarks/README.md` | Benchmark/report interpretation and no-portable-performance policy. |
| `docs/maintainer_guide.md` | Maintainer proof interpretation and routing policy. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected report target identity, workflow metadata, claim scopes, and non-claims. |

## Preserved Inline Warnings

Day 7 deliberately retained inline warnings for:

- generated Doxygen HTML is local-only ignored output and not hosted,
  retained, source-controlled, or release evidence;
- Windows selected freshness remains unpromoted unless manifest metadata,
  generated support tier, and non-claim wording move together;
- installed package contract is static-first;
- shared-library packaging and dynamic ABI support are not claimed;
- broad package-manager distribution, Homebrew/core, bottles, Linuxbrew,
  public tap maintenance, binary packages, and user-facing Homebrew install
  paths are not claimed;
- benchmark output is local/selected evidence, not portable performance proof.

## Non-Changes

- No public API, ABI, source, header, Makefile, CMake, workflow, manifest, or
  generated-output behavior changed.
- No selected target metadata, support tier, claim scope, or non-claim list was
  changed.
- No support status was promoted.
- No package-manager, shared-library, dynamic ABI, broad Windows, hosted API,
  portable performance, release, external-library parity, or state-of-the-art
  claim was added.

## Validation And Hygiene

| Check | Day 7 result |
| --- | --- |
| Public docs edited | Yes: README, tutorial, examples. |
| Maintainer docs edited | Yes: maintainer guide routing note. |
| User-facing support claim changed | No new support status added; repeated caveats were shortened with owner links and retained inline warnings. |
| `.c` or `.h` files changed | No. Full C gate not required for Day 7. |
| Generated output changed | No generated output intentionally created. |
| `git diff --check` | Planned after artifact creation. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.3 has landed in public and maintainer docs. | In progress and satisfied for the Day 7 batch. Public docs and maintainer routing guidance were updated; Day 8 remains for examples/workflow link polish. |
| Repeated caveats are reduced without losing necessary local warning text. | Met. README/tutorial/examples caveats were shortened where owner links now cover detail, while high-risk non-claims remain inline. |
| Support truth routing is clear enough for future sprint updates. | Met. Maintainer guide now records the routing model and owner docs. |

## Day 7 Disposition

Day 7 is complete. Day 8 should perform the examples/workflow link pass to
ensure tutorial, cookbook, examples, and solver-selection routes are coherent
after the quick-reference and support-truth consolidation edits.
