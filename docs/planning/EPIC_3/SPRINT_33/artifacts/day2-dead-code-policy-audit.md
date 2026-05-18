# Sprint 33 Day 2 Dead-Code Policy Audit

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Define the policy boundaries Sprint 33 must preserve before dead-code tooling or cleanup begins: active versus opt-in test coverage, historical evidence versus live code, public/documented surface versus internal-only surface, and comment debt versus actual dead code.

## Audit Summary

Day 2 confirms that Sprint 33 inherits Sprint 32’s truthfulness cleanup successfully:

- commented-out `RUN_TEST(...)` registrations in `tests/`: `0`
- active `RUN_TEST(...)` call sites: `1725`
- active `RUN_TEST_SLOW(...)` call sites: `2`
- active `RUN_TEST_EXPERIMENTAL(...)` call sites: `2`
- `SKIP_TEST(...)` call sites: `1`

The key consequence is that Sprint 33’s dead-code policy cannot treat opt-in execution paths or truthfulness-support scaffolding as dormant code. Those are live, deliberate parts of the regression model.

## Boundary 1: Live Tests Versus Historical Evidence

`tests/test_framework_optin.c` is the canonical Day 2 proof:

- it is compiled
- it is registered in CTest
- it exercises `RUN_TEST_SLOW(...)`, `RUN_TEST_EXPERIMENTAL(...)`, and `SKIP_TEST(...)`

That means:

- `tests/test_framework_optin.c` is live policy coverage, not cleanup fodder
- the opt-in wrappers in `tests/test_framework.h` are supporting live behavior, not suspicious dead branches

The repo-wide search also found `0` commented-out `RUN_TEST(...)` registrations, so the specific Sprint 32 dormant-scaffold anti-pattern is currently closed.

## Boundary 2: Comment-Level “Stub” Language Is Not Dead Code

Several active tests still carry sprint-history comments using words like
“stub”, “future”, or “Day N replaces this”. Representative samples:

- `tests/test_graph.c::test_graph_subgraph_is_stub`
- `tests/test_sprint20_integration.c` eigs validation comments
- `tests/test_eigs.c` refinement-post-pass comments
- `tests/test_reorder_nd.c` comments explaining that the historical dormant trio was retired into docs

These are not Day 32-style dormant registrations:

- the functions are still actively registered
- the assertions describe current behavior
- the comments preserve development history

Day 2 policy conclusion:

- active registration plus current assertions outrank historical wording in comments
- those sites are documentation/comment cleanup candidates at most, not dead-code candidates

## Boundary 3: Public/Documented Surface Versus Internal Cleanup Surface

Current public/documented surface counts:

- public installed headers: `18`
- example source files: `12`
- documented benchmark entries in `benchmarks/README.md`: `12`
- explicit example sections in `examples/README.md`: `5`

The most important audit result is the mismatch between documentation and the
current compilation database:

- `bench_svd` is documented and built by the Makefile, but absent from the current CMake `compile_commands.json`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

are all real example programs present in the repo and documented in project materials, but absent from the current CMake `compile_commands.json`.

Day 2 policy conclusion:

- absence from `compile_commands.json` is not a dead-code signal
- documented examples, documented benchmark binaries, and installed headers require manual public-surface review
- they are outside the first “definitely-unused internal code” deletion pass unless stronger evidence appears

## Boundary 4: Internal Candidates Need Stronger Evidence

The likely first-pass cleanup surface remains:

- unexported `static` helpers in `tests/`, `benchmarks/`, `examples/`, and `src/*.c`
- private declarations in `src/*internal*.h`

But Day 2 does not certify any specific function as definitely-unused yet,
because:

- `deadcode` tooling is not implemented
- `xunused` is missing locally
- `compile_commands.json` under-covers part of the tooling surface

So Day 2’s output is policy, not a deletion list.

## Required Evidence Hierarchy

Sprint 33 should follow this ordering when classifying findings:

1. Active registration and documented public surface override “looks unused”.
2. Historical or sprint-era comments do not by themselves make code dead.
3. Definitely-unused internal code needs stronger evidence than local grep absence.
4. Candidate public API or documented tooling code goes to a separate manual-review queue.

## Day 2 Conclusion

The main false-positive risks for Sprint 33 are now explicit:

1. live opt-in or active test code being mislabeled as dead because it is non-default or wrapped
2. documented examples/benchmarks being mislabeled as dead because the current compilation database does not cover them
3. active tests with historical comment language being mislabeled as dead because the comments mention an earlier stub phase

That makes the Day 3 job clear: formalize these boundaries into a repository dead-code policy and limitations note before any tooling or deletion work begins.
