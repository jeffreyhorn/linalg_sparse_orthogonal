# Sprint 40 Day 13: Validation Anchor & Refactor-Sprint Command Matrix

## Objective

Define the authoritative validation reference for the implementation-heavy Epic
4 sprints. This artifact distinguishes mandatory validation from targeted
follow-on validation and preserves the truthfulness constraints inherited from
Epic 3 so later refactors do not accidentally weaken or overstate the quality
baseline.

## Audit Inputs

This validation anchor is grounded in:

- the current `Makefile` quality-command surface
- the current README reviewed/dead-code/cross-platform contract
- the current reviewed CMake test registration count
- the Sprint 40 baseline/support artifacts already produced in Days 1-12

## Core Day 13 Conclusion

Later Epic 4 refactor sprints should treat validation as a layered model:

1. mandatory full C/C-header gate for all `*.c` / `*.h` changes
2. maintained reviewed-wrapper baseline for substantial local refactor proof
3. serial dead-code path as a separate completeness/reporting sibling
4. targeted benchmark/example/specialized reruns only when the touched surface
   justifies them

The goal is not “run everything always.” The goal is to preserve the strongest
validated baseline honestly while adding targeted checks where the change type
needs them.

## Authoritative Validation Commands

### Mandatory full C/C-header gate

For any change that touches `*.c` or `*.h`, the mandatory baseline remains:

```bash
make format
make lint
make test
```

Interpretation:

- this is the non-negotiable floor for C and header edits
- later Epic 4 implementation work should assume this gate always applies to
  code changes, even when additional targeted checks also run

### Reviewed local baseline

The strongest maintained local reviewed wrapper remains:

```bash
make quality-review-full
```

This expands to:

- `make quality-review`
- `make quality-review-cmake`

Interpretation:

- this is the strongest routine local reviewed baseline
- it is especially appropriate for substantial multi-surface refactors, not
  just small isolated edits

### Reviewed CMake parity path

The maintained parity path remains:

```bash
make quality-review-cmake-compile
make quality-review-cmake
```

Interpretation:

- use this path when the refactor could affect:
  - clean rebuild behavior
  - Makefile/CMake test-count parity
  - the registered active suite
  - broader CMake-visible source/test wiring

### Serial dead-code path

The maintained dead-code sibling path remains:

```bash
make deadcode-report
make deadcode-check
```

Interpretation:

- this path remains authoritative only when run serially
- it is a completeness/reporting workflow, not a zero-findings or removal-ready
  assertion

### Specialized validation surfaces

These remain targeted follow-ons rather than universal gates:

- `make tooling-build`
- `make examples`
- direct example binary reruns
- direct benchmark binary reruns
- `make wall-check`
- `make sanitize`
- `make warning-workflow WARNING_WORKFLOW_LABEL=label`

Interpretation:

- run them when the touched surface or claim actually depends on them
- do not silently treat them as universal baseline steps

## Mandatory vs Targeted Validation Split

### Always mandatory for `*.c` / `*.h` changes

- `make format`
- `make lint`
- `make test`

### Strongly expected for substantial refactor batches

- `make quality-review-full`

Use this especially when the change touches:

- multiple subsystems
- shared helper layers
- lifecycle/state machinery
- build/quality plumbing
- public API semantics that affect both Makefile and CMake validation paths

### Mandatory when dead-code surfaces are changed

If the change touches:

- dead-code scripts
- dead-code Makefile targets/rules
- dead-code report semantics
- dead-code compile-db source coverage

Then run serially:

- `make deadcode-report`
- `make deadcode-check`

### Mandatory when build/test registration surfaces are changed

If the change touches:

- `CMakeLists.txt`
- test registration wiring
- wrapper targets that affect reviewed CMake parity

Then run:

- `make quality-review-cmake-compile`
- and usually `make quality-review-cmake`

### Targeted when examples/benchmarks/tooling are directly affected

If the change touches:

- `examples/`
- benchmark CLI/help behavior
- example/benchmark build wiring
- docs whose truth depends on maintained examples

Then add the relevant targeted checks:

- `make tooling-build`
- `make examples`
- direct example reruns
- direct benchmark reruns

### Targeted when performance or sanitizer-sensitive paths are affected

If the change touches:

- concurrency-sensitive code
- sanitizer-sensitive code
- performance-regression gates
- platform-adjacent validation helpers

Then add the relevant targeted checks:

- `make wall-check`
- `make sanitize`

### Documentation-only changes

If only docs change and no `*.c` / `*.h` files are touched:

- the full code gate is not automatically required
- validation should be scoped to the truth surface being edited

Examples:

- README command-map edits -> re-read command surfaces / dry-run commands
- dead-code docs edits -> verify current report/check outputs and wording
- cross-platform wording edits -> re-read workflow YAML and current contract

## Change-Type Validation Matrix

| Change Type | Mandatory Validation | Targeted Follow-On Validation | Notes |
|---|---|---|---|
| `*.c` / `*.h` implementation edits | `make format`; `make lint`; `make test` | `make quality-review-full` for substantial batches | This remains the floor for all code changes |
| large multi-file refactor | `make format`; `make lint`; `make test`; `make quality-review-full` | benchmark/example reruns as needed | Use full reviewed baseline to preserve architectural truth |
| `CMakeLists.txt` / test-registration changes | `make format`; `make lint`; `make test` if code touched | `make quality-review-cmake-compile`; usually `make quality-review-cmake` | Preserve CTest parity truth |
| dead-code workflow/report changes | serial `make deadcode-report`; serial `make deadcode-check` | `python3 -m py_compile` / shell syntax checks as applicable | Dead-code remains a separate serialized sibling path |
| Makefile quality-wrapper changes | `make format`; `make lint`; `make test` if code touched | `make -n` dry runs; usually `make quality-review-full` | Preserve wrapper meaning and rerun guidance truth |
| examples/benchmark behavior changes | `make format`; `make lint`; `make test` if code touched | `make tooling-build`; `make examples`; direct reruns | Scope reruns to touched surfaces |
| workflow/docs-only contract changes | surface-specific sanity checks | YAML parse, command dry runs, artifact re-read | No automatic full C gate if no code changed |

## Preserved Truthfulness Constraints

Later Epic 4 refactor sprints should explicitly preserve these invariants.

### 1. CTest registration truth

The maintained reviewed CMake parity baseline currently remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Implication:

- later sprints must not casually claim suite growth/shrinkage
- if the count changes intentionally, they must document the reason explicitly

### 2. Makefile vs CMake test-count parity

The reviewed CMake compile wrapper still checks parity between:

- `ctest -N`
- `$(words $(TEST_BINS))`

Implication:

- later build/test registration work must preserve or intentionally reconcile
  that parity contract

### 3. Dead-code serialization

The dead-code path still shares:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- authoritative `deadcode-report` / `deadcode-check` execution remains serial
- parallel dead-code runs are not valid evidence

### 4. Dead-code meaning

`deadcode-check` still means:

- report generated
- findings categorized
- completeness invariants satisfied

It does not mean:

- zero findings
- removal-ready queue
- reachability proof

### 5. Cross-platform contract wording

Later sprints should preserve the current honest model:

- Linux = strongest enforced reviewed baseline
- macOS dead-code = staged
- Windows local Makefile reviewed-wrapper parity = staged
- Windows dead-code = excluded

### 6. Instrumented-build reset caveat

After:

- `sanitize`
- `asan`
- `sanitize-all`
- `tsan`
- `omp`
- `coverage*`

The maintained return path to normal reviewed validation still requires:

```bash
make clean
```

## Day 13 Decisions

1. The mandatory floor for code changes remains `make format`, `make lint`,
   and `make test`.
2. `make quality-review-full` is the strongest routine local reviewed baseline
   and should be the default substantial-refactor proof.
3. Dead-code validation remains separate and serialized, not folded into the
   universal mandatory code gate.
4. Targeted example/benchmark/sanitizer/performance reruns should be scoped by
   touched surface, not run reflexively on every change.
5. CTest count truth, Makefile/CMake parity, and dead-code serialization are
   explicit invariants later Epic 4 work must preserve.

## Day 13 Output for Later Sprints

Later Epic 4 implementation sprints now have:

- a stable mandatory-vs-targeted validation split
- a change-type command matrix
- a compact truthfulness checklist for CTest, dead-code, and platform wording
- a clearer default answer for “what do I need to run for this refactor?”

That should keep the later sprints both rigorous and efficient, without
quietly weakening the validated baseline or wasting time on unfocused blanket
validation.
