# Sprint 33 Day 3 Dead-Code Policy And Limitations

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Turn the Day 2 audit into an explicit repository dead-code policy for Sprint 33: define the code classes, state the known and expected tool limitations, and set the threshold that code must meet before it is treated as definitely-unused internal code.

## Policy Summary

Sprint 33 is not trying to prove that nobody anywhere could ever call a piece of code.
It is trying to prove a narrower, auditable claim:

- this code is internal-only
- it has no active or documented role
- the available tooling and manual review both say it is unused
- removing it does not regress the validated repository state

That narrower claim is the only one Sprint 33 will use for first-pass deletion.

## Code Classes

### 1. Active code

Definition:

- part of the normal validated path
- active `RUN_TEST(...)` registration
- normal example, benchmark, build, or library flow

Rule:

- not a dead-code candidate

### 2. Live opt-in code

Definition:

- current supported non-default behavior
- exercised through explicit wrappers, env vars, or documented opt-in flows

Examples:

- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`
- `tests/test_framework_optin.c`

Rule:

- not dormant scaffold
- not a dead-code candidate just because it is non-default

### 3. Historical evidence

Definition:

- retired-target data
- sprint artifacts
- historical measurements
- planning documentation

Rule:

- belongs in `docs/planning/`
- not in compiled dormant registrations

### 4. Documented public surface

Definition:

- installed headers under `include/`
- documented examples
- documented benchmark entry points
- README-described CLI/API behavior

Rule:

- not part of the first deletion pass
- route to manual public-surface review if questioned

### 5. Candidate unused public API

Definition:

- exported or documented surface that appears to have no in-repo callers

Rule:

- manual review only
- never delete in Sprint 33 solely from local reachability or scanner output

### 6. Definitely-unused internal code

Definition:

- internal-only code
- no active registration
- no documented/public contract
- enough corroborating evidence to delete safely

Rule:

- only class eligible for Sprint 33 first-pass removal

### 7. Comment/documentation debt

Definition:

- active code with outdated “stub”, “future”, or sprint-history wording

Rule:

- documentation cleanup candidate
- not dead code by itself

## Confirmed Day 3 Tool Limitations

### `cppcheck`

Local confirmation:

- `cppcheck --help` says `unusedFunction` is best enabled only when the whole program is scanned
- Sprint 33 plans to start from `cppcheck --enable=all --quiet src/`

Implication:

- raw `unusedFunction`-style findings from partial scans are evidence inputs, not auto-delete proof

### `compile_commands.json`

Current Day 1 / Day 3 confirmed coverage:

- `src`: `25`
- `tests`: `53`
- `benchmarks`: `13`
- `examples`: `6`

Known coverage gap versus repo source files:

- missing benchmark: `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- `xunused build/compile_commands.json` cannot be treated as full bench/example proof until Sprint 33 either improves the coverage or reports the gap explicitly

### `xunused`

Current Day 3 state:

- not installed locally

Implication:

- its exact local behavior is not yet exercised
- policy must assume it is a useful but bounded reachability input, not the sole arbiter

## Expected False-Positive Buckets

Sprint 33 should expect these findings to need manual review:

### 1. Under-covered bench/example code

Reason:

- absent from current `compile_commands.json`
- still built by Makefile and/or documented

### 2. Live opt-in test surface

Reason:

- non-default execution model can look suspicious to shallow tooling

### 3. Public/documented API surface

Reason:

- local callers are not the same thing as supported contract

### 4. Active tests with historical comments

Reason:

- comments may mention an earlier stub phase while the assertions are current and live

### 5. Scanner-only “unused” findings on partial scope

Reason:

- scanner scope can be narrower than repo behavior scope

## Deletion Threshold

Sprint 33 will only delete code when all of the following are true:

1. The code is internal-only.
2. It is outside installed headers and documented public behavior.
3. It is outside active and opt-in test registration.
4. There is at least one tooling or compile-surface signal supporting the claim that it is unused.
5. Manual review finds no live intended role.
6. It is not explained by a known coverage gap or false-positive bucket.
7. Post-removal validation still passes.

This is intentionally conservative.

### What is not enough by itself

None of the following is sufficient alone:

- a single scanner warning
- no local grep caller
- absence from the current `compile_commands.json`
- “stub” or “future” wording in comments

## Manual-Review Queue Rules

Any finding touching these surfaces is automatically routed out of the first cleanup pass:

- `include/*.h`
- documented example programs
- documented benchmark binaries
- README-described behavior
- test-truthfulness framework surface

These findings are still worth reporting, but they are category errors if treated as definitely-unused internal code.

## Day 4+ Guidance

### For tooling implementation

- `deadcode` should begin as evidence gathering
- reporting must preserve category boundaries
- coverage gaps must be visible in operator output

### For cleanup implementation

- remove internal-only code first
- keep public/documented findings separate
- do not count comment cleanup as dead-code closure unless compiled unused code is actually removed

## Day 3 Conclusion

Sprint 33’s dead-code policy is now deliberately narrower than a generic “unused code” sweep:

- preserve active and opt-in truthfulness
- preserve documented/public surface unless manually reviewed
- use tooling as evidence, not oracle
- only delete code that is demonstrably internal and safe to remove

That policy is specific enough to drive the Makefile/reporting work and the later first cleanup pass without reopening Sprint 32’s truthfulness problems.
