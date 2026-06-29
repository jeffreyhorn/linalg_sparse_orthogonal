# Sprint 97 Day 3: Convergence Architecture Design

## Purpose

Day 3 turns the Day 2 duplication ranking into a bounded build-convergence
architecture. The goal is to reduce source-list drift risk without weakening
reviewed Make, reviewed CMake, install/export, or platform proof.

No build topology is changed on Day 3.

## Architecture Principles

1. Preserve proof at the point of use.
2. Prefer checked convergence before hidden generation.
3. Keep Make and CMake readable to reviewers.
4. Treat source-list membership differently from test-platform eligibility.
5. Keep package-contract and platform-truth work in their planned Sprint 97
   lanes.

These principles intentionally reject a broad "generate everything" solution
for the first batch. Full generation could lower duplication, but it would also
hide build membership and platform exclusions behind tooling before the repo has
proved that such tooling is worth the maintenance burden.

## Selected First Target

### Library Source List

Selected target:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`

Why this is first:

- It is the highest-ranked Day 2 duplication candidate.
- It repeats 42 correctness-critical library source entries.
- It has fewer platform-specific semantics than test registration.
- It is frequently touched by source extraction work.
- A missing entry can make Make and CMake disagree about the built library.

Selected Sprint 97 architecture:

- Add one explicit library-source manifest as the durable source-membership
  reference.
- Add one small manifest parity checker or generator that can compare the
  manifest against Make and CMake.
- Keep Make and CMake source membership readable in the first implementation
  batch.
- Use the checker in the relevant reviewed or local quality path only after
  Day 4 freezes exact validation cost.

Preferred first implementation shape for Days 4-6:

| Piece | Proposed role |
|---|---|
| `build` or `scripts` manifest/check helper | parse the manifest and compare it to Make/CMake source membership |
| library-source manifest | list library sources once in a simple, reviewable text format |
| Makefile hook | optional first-batch parity check target or integration into reviewed compile-quality path |
| CMake hook | optional first-batch configure-time check or no hook until Make-side checker is proven |
| documentation artifact | record source-membership ownership and validation expectations |

Day 4 should choose the exact file names and hook depth. Day 3 only freezes the
architecture and selected target.

## Mechanism Evaluation

### Generated Source Lists

Benefits:

- removes manual duplication completely for the generated side
- makes source membership update a one-file operation

Risks:

- generated Make/CMake fragments can obscure reviewed source membership
- generated files can create churn if committed
- generation may be awkward in CMake configure paths if it depends on Python or
  shell assumptions

Day 3 decision:

- Do not start with full generation.
- Allow a later Sprint 97 batch to generate one side only if Day 5 proves a
  manifest and checker are simple and stable.

### Checked Manifest

Benefits:

- keeps Make and CMake readable
- makes drift explicit
- can be added with low behavioral risk
- gives Day 5 a concrete reduction in review risk even before full generation

Risks:

- does not remove every duplicated line
- introduces one more file to update unless the checker is enforced

Day 3 decision:

- Preferred first mechanism.
- Day 4 should freeze a minimal manifest format and checker contract.

### Shared Include Fragments

Benefits:

- can make one list consumed by both build systems
- avoids committed generated output if both systems can consume the same file

Risks:

- Make and CMake syntax differs enough that a shared fragment is likely to be
  awkward or fragile
- CMake `include()` and Make `include` have different path and quoting rules
- platform gating becomes harder if the same pattern is later applied to tests

Day 3 decision:

- Not the first implementation choice.
- Reconsider only if a simple manifest cannot support a checker.

### CMake Target Properties

Benefits:

- CMake can expose target sources internally after configuration
- useful for CMake-side assertions

Risks:

- does not help Make directly
- still requires parsing or invoking CMake to compare against Make

Day 3 decision:

- Useful as a checker implementation detail, not the architecture root.

### Make Variables

Benefits:

- Make already has explicit `LIB_SRCS` and derived `LIB_OBJS`
- easy for local shell/Python tooling to parse

Risks:

- Make syntax is not a clean manifest format
- CMake should not depend on parsing Makefile internals for source membership

Day 3 decision:

- Keep Make variables as build implementation.
- Do not make Makefile the long-term source-list authority.

### Workflow-Local Assertions

Benefits:

- CI logs can explain exactly which proof failed
- Windows expected count and staged exclusions are already visible this way

Risks:

- workflow-only assertions do not help local development until CI runs
- too much logic in YAML increases review cost

Day 3 decision:

- Preserve workflow-local assertions for platform proof.
- Do not put source-list convergence logic directly in workflow YAML.

### Documentation-Only Alignment

Benefits:

- low risk
- useful for package/story consistency

Risks:

- does not prevent Make/CMake source-list drift

Day 3 decision:

- Insufficient for the library source-list problem.
- Appropriate for package and platform claim work later in the sprint.

## Independent-Proof Preservation List

The following surfaces must remain explicit after Day 4-6 implementation:

- `make quality-review-cmake-compile` Make/CMake CTest-count parity assertion
- Windows `EXPECTED_WINDOWS_CTEST_COUNT`
- Windows staged exclusions:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- macOS Apple Clang reviewed path versus supplemental GCC lane
- macOS supplemental Make install/pkg-config confidence path
- Linux reviewed Makefile compile-quality path
- Linux reviewed CMake parity path
- Linux dead-code report/check path
- Make install/pkg-config proof separate from CMake install/find_package proof
- static-first package wording in CMake, README, and INSTALL until Day 7-8
  decides otherwise

## Deferred Targets

### Test Registration

Test registration is the second strongest duplication candidate, but Day 3
does not select it as the first landing target.

Reasons:

- tests encode platform eligibility, not only membership
- Windows has a smaller reviewed CTest surface than local Make/CMake
- Make's full local test list and CMake's platform gates are both meaningful
- current Make/CMake test-count parity is already an important proof assertion

Recommended later handling:

- add a test-registration manifest or checker only after the library-source
  checker pattern is proven
- keep platform-exclusion metadata explicit if a test manifest is introduced
- preserve CTest count assertions in local and Windows lanes

### Benchmarks

Benchmark registration remains residual for now.

Reasons:

- lower correctness risk than library/test registration
- specialized subsets such as `bench-fast` and canonical reports require
  separate curation
- POSIX-only benchmark gating is easier to review in local build files today

### Examples

Example registration remains residual for now.

Reasons:

- Make already uses a wildcard
- CMake's explicit targets keep user-facing example names readable
- example drift is less likely to break the core reviewed build

### Package And Platform Claims

Package and platform claim repetition is real, but it is not source-list
duplication.

Reasons:

- CMake, README, INSTALL, workflows, and install proof scripts speak to
  different audiences
- static-first wording should stay visible until Day 7-8 makes a package
  decision
- platform CI limitations should stay visible where the proof runs

## Validation Plan For Implementation Days

Day 4 boundary freeze should choose the exact validation gate. Day 3 recommends
the following validation split for the first source-list convergence batch:

| Change | Required validation |
|---|---|
| manifest/checker docs only | `git diff --check` and whitespace scan |
| adding a manifest/checker script | script self-test or direct checker run, plus docs hygiene |
| Makefile hook into source-list check | full `make format && make lint && make test`, plus the new check target |
| CMake hook into source-list check | full quality chain plus `make quality-review-cmake-compile` |
| any `.c` or `.h` changes discovered during landing | full `make format && make lint && make test` |
| workflow hook for source-list check | local equivalent command plus CI review after PR |

Preferred Day 4/5 first batch validation:

1. Run the source-list checker directly.
2. Run `make quality-review-cmake-compile` if Make/CMake registration or CMake
   configure logic changes.
3. Run `make format && make lint && make test` if any code, header, Makefile,
   or CMake build behavior changes.
4. Run `git diff --check` for all planning and build-script changes.

## Day 3 Result

Sprint 97 now has a bounded convergence architecture. The first selected
target is the library source list. The preferred first landing shape is a
simple manifest plus a parity checker, with Make and CMake readability
preserved until the checker proves useful. Test registration, benchmark
registration, example registration, package-surface claims, and platform
workflow language remain explicit follow-up lanes rather than Day 4's first
implementation target.
