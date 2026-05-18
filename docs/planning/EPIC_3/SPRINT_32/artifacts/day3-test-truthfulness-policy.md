# Sprint 32 Day 3 Test Truthfulness Policy

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Choose a project-level representation for active, slow, experimental, and historical tests that fits the current C test harness and gives Sprint 32 a concrete rule for removing dormant scaffold without weakening real coverage.

## Current Harness Reality

The repository’s test model is simple and important to preserve:

- each `tests/test_*.c` file builds as its own executable
- `make test` runs every built test binary directly
- CMake/CTest mirrors the same one-binary-per-file structure
- inside each binary, `main()` enumerates `RUN_TEST(...)` calls

This means Sprint 32 should not design around a hypothetical richer runner. The policy has to fit the current model.

### Current gap in the framework

`tests/test_framework.h` has:

- pass/fail accounting
- ordinary assertions
- `REQUIRE_OK(...)`

It does **not** have:

- explicit skip accounting
- first-class opt-in test registration
- xfail tracking

Tests can early-return after printing a local "skipped" message, but from the framework’s perspective that still looks like a pass. That is good enough for fixture-unavailable conditions inside active tests, but not good enough for representing a deliberate non-default test category.

## Chosen Category Model

Sprint 32 adopts four categories:

### 1. Active tests

Definition:

- run under default `make test` and `ctest`
- assert a current supported behavior, invariant, or measured bound
- are part of the project’s ordinary green path

Rule:

- active tests must pass in the normal suite

### 2. Slow opt-in tests

Definition:

- assert a current supported behavior or bound
- are excluded from the default path only because runtime or fixture cost is too high

Rule:

- slow tests must pass when explicitly enabled
- slow is about execution cost, not stale expectations

### 3. Experimental opt-in tests

Definition:

- assert a current live contract for a non-default or under-evaluation path
- are not run by default because they are intentionally non-default and not yet part of the normal regression budget

Rule:

- experimental tests must also pass when explicitly enabled
- experimental is **not** a synonym for expected-fail

### 4. Historical evidence

Definition:

- documents missed targets, retired goals, or old exploratory branches
- preserves measurements, rationale, and references

Rule:

- historical evidence lives in docs and sprint artifacts
- it does not compile into normal suite files

## What Sprint 32 Explicitly Rejects

The following pattern is no longer acceptable in merged code:

- keep a `static void test_*` body in a normal suite file
- comment out its `RUN_TEST(...)`
- treat the comments as the only signal that the test is inactive

That pattern has three problems:

1. it leaves compiled code that implies coverage the suite does not actually execute
2. it creates `-Wunused-function` debt that obscures real cleanup work
3. it blurs the line between live opt-in coverage and historical narrative

Sprint 32 therefore rejects long-lived commented-out `RUN_TEST(...)` scaffolding as a representation mechanism.

## XFail Policy

Sprint 32 does **not** adopt xfail-style merged tests as the primary answer.

Reason:

- the current framework has no first-class xfail accounting
- xfail would add another long-lived non-green state to a harness that is currently binary: pass or fail
- for the concrete `test_reorder_nd.c` problem, the dormant trio is historical or retired, not live contracts waiting for the right toggle

If a future sprint wants xfail semantics, that should be an explicit separate design decision with its own reporting model. It is not needed for the Sprint 32 cleanup.

## Decision Framework

When deciding what to do with a test or test stub:

### Keep active when

- the contract is current
- the test is stable enough for normal execution
- runtime is acceptable

### Move to slow opt-in when

- the contract is current
- the only problem is runtime or fixture cost
- the test is still valuable as real rerunnable coverage

### Move to experimental opt-in when

- the contract is current
- the path is intentionally non-default
- rerunning it still provides ongoing signal

### Move to docs-only historical evidence when

- the target was explicitly missed or retired
- the supporting sprint docs already preserve the measurements
- keeping the code would overstate the current protection surface

## Minimal Day 4 Implementation Shape

The least invasive framework extension is:

1. add explicit skip accounting
2. add a skip macro for test bodies
3. add opt-in wrappers for top-level registration in `main()`

Recommended environment gates:

- `SPARSE_TEST_SLOW=1`
- `SPARSE_TEST_EXPERIMENTAL=1`

Recommended wrapper behavior:

- default off
- when off, print a visible `[SKIP]` line with the enabling hint
- increment skipped count
- do not count as pass
- when on, run the wrapped test normally through the same pass/fail path as `RUN_TEST(...)`

Why this shape fits the repo:

- no new runner
- no CMake/Makefile architecture change required for the category itself
- source remains easy to audit because category choice is visible exactly where the test is registered

## Concrete Implication For `tests/test_reorder_nd.c`

The Day 2 audit showed:

- `26` static test functions
- `23` active `RUN_TEST(...)`
- `3` commented-out `RUN_TEST(...)`
- all `3` `-Wunused-function` warnings map exactly to that dormant trio

Those three dormant helpers are:

1. `test_finest_fm_annealing_pres_poisson_close_to_target`
2. `test_nd_root_spectral_pres_poisson_close_to_target`
3. `test_non_pipeline_pres_poisson_close_to_target`

They do **not** qualify for the new opt-in categories because:

- two assert known-missed Sprint 27 close-to-target claims
- one asserts a Sprint 28 target that was formally retired
- the relevant decision docs already preserve the evidence

Therefore the chosen structural end state is:

- keep the existing active advisory smoke/parity tests active
- remove the dormant trio from suite code during Day 5
- retain their history in sprint artifacts and decision docs

## Repo-Wide Implication

A repository-wide grep shows the commented-out `RUN_TEST(...)` anti-pattern is currently localized to `tests/test_reorder_nd.c`.

That is good news:

- Sprint 32 can close the concrete debt in one focused file
- the new policy can still apply repo-wide to prevent recurrence

The policy also gives future sprint work a cleaner temporary-development rule:

- failing scaffolding may exist locally while a day’s implementation is in progress
- before the work is considered landed, it must become active runnable coverage, a real opt-in test, or docs-only evidence

## Final Decision

Sprint 32’s truthfulness model is:

- active by default for live current contracts
- explicit opt-in only for live slow or experimental contracts
- docs-only for historical or retired targets

That model is specific enough for Day 4 framework support and Day 5 `test_reorder_nd.c` cleanup, and it matches the actual capabilities of the current Makefile/CMake test harness instead of assuming infrastructure the repo does not have.
