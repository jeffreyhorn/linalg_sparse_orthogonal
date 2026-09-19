# Sprint 205 Day 8: Example And Workflow Routing

**Date:** 2026-09-19  
**Sprint item:** 205.3 Support Truth Consolidation  
**Theme:** Align examples, tutorial, cookbook, and solver-selection routing
with the quick-reference and support-truth model.

## Purpose

Day 8 completed the route-polish portion of item 205.3. The goal was to make
the runnable example surface clearer without turning examples into another
support matrix, benchmark authority, or package evidence ledger.

## Changed Surfaces

| Surface | Change | Boundary retained |
| --- | --- | --- |
| `examples/README.md` | Added a route-interpretation table that separates local build-tree examples, compressed-input examples, installed CMake consumers, benchmarks, and support-status lookup. | Examples remain local workflow references; support/readiness stays in `INSTALL.md`; benchmarks are not portable performance proof. |
| `examples/README.md` | Added support/readiness matrix handoff to the installed CMake consumer section. | Installed consumer example remains static-first and does not imply package-manager, shared-library, dynamic ABI, or broad platform support. |
| `docs/solver_selection.md` | Added a short interpretation note before the example handoff table. | Solver selection remains a workflow guide, not package, support, or measurement authority. |
| `docs/solver_selection.md` | Converted example handoff rows from raw program names to examples documentation routes. | Users can move from solver choice to runnable examples without copying support caveats into the solver guide. |

## Route Model After Day 8

| User question | Owner route | Notes |
| --- | --- | --- |
| "Which example should I run first?" | `examples/README.md#start-here` | Build-tree local examples only. |
| "What problem shape do I have?" | `docs/cookbook.md#problem-shape-quick-reference` and `docs/solver_selection.md` | Cookbook is the compact route; solver selection owns detail. |
| "How do I interpret this example output?" | `examples/README.md#diagnostics-handoff`, then `docs/solver_selection.md#diagnostics-handoff` | Diagnostics stay tied to the workflow that produced them. |
| "How do I use this after install?" | `examples/cmake_example/` and `INSTALL.md#installed-consumer-tutorial` | Installed support status remains in `INSTALL.md#support-readiness-matrix`. |
| "How do I measure it?" | `benchmarks/README.md` | Benchmarks are local or selected evidence, not portable performance proof. |

## Claim-Boundary Review

Day 8 did not add or promote claims for:

- package-manager distribution, Homebrew/core, bottles, Linuxbrew, or public
  taps;
- shared-library packaging, runtime loader behavior, or dynamic ABI support;
- broad Windows selected freshness or Windows package parity;
- hosted generated API documentation, retained generated artifacts, or release
  evidence;
- portable performance, external-library superiority, or state-of-the-art
  status.

The only installed-consumer language added points back to INSTALL and keeps the
static-first support boundary explicit.

## Validation Notes

| Check | Result |
| --- | --- |
| Example routing reviewed against Day 6 quick reference | Passed |
| Installed-consumer route reviewed against support/readiness owner | Passed |
| Solver-selection handoff reviewed for duplicate caveat expansion | Passed |
| Production source/header changes | None |
| Build, Makefile, CMake, workflow, manifest, or generated-output changes | None |

## Completion Criteria Review

- Users can move between the quick reference, solver-selection detail, and
  examples without circular or stale routes.
- Example docs distinguish local source builds, installed consumers, and
  measurement handoffs.
- Example docs do not imply package-manager, ABI, broad platform, hosted API,
  release, portable-performance, or state-of-the-art support.
- Item 205.3 is complete for the Sprint 205 branch-local support-truth
  consolidation scope.
