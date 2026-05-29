# Maintainer Guide

This guide is the maintainer-facing policy home for repository-wide quality
contract interpretation, documentation ownership, and a few stable norms that
should not keep getting re-explained inside `README.md`, tutorial prose, or
public headers.

It is intentionally narrower than a full developer handbook. It explains how
to read the maintained command surfaces and where policy lives. It does not
replace the executable truth in `Makefile`, scripts, CI workflows, or API-local
header contracts.

## Audience

This document is for:

- maintainers
- high-context contributors doing repo-wide cleanup
- reviewers evaluating quality-contract or documentation-ownership claims

This document is not the primary entry point for:

- first-time library users
- API consumers learning one solver
- benchmark/example users looking for command syntax

Those audiences should start with:

- [README](../README.md)
- [tutorial](tutorial.md)
- [benchmarks/README](../benchmarks/README.md)
- [examples/README](../examples/README.md)

## Authoritative Surfaces

Repository policy and executable truth are not the same thing.

Executable truth stays with:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- CI workflows under `.github/workflows/`
- public headers for API-local call-site caveats
- `tests/test_framework.h` for live opt-in test wrapper semantics

This guide owns:

- how to interpret those surfaces
- which surface is authoritative for which kind of claim
- where maintainer-only policy should live instead of spreading through README

Command-detail boundary:

- keep wrapper expansion, rerun guidance, build-tree paths, and other
  executable command detail in `Makefile`
- keep dead-code workflow execution detail in `Makefile`,
  `scripts/deadcode_workflow.sh`, and `scripts/deadcode_report.py`
- use this guide for repository-wide interpretation of those surfaces, not as a
  shadow command reference

## Reviewed Baseline and Warning Authority

### Strongest local reviewed baseline

The strongest maintained local reviewed baseline is:

```bash
make quality-review-full
```

Interpretation:

- this is the strongest local reviewed baseline command
- it composes the reviewed Makefile path and the reviewed CMake parity path
- it is the right default proof point for local “current branch is in the
  reviewed baseline” claims unless a narrower claim is being made
- exact wrapper expansion and rerun guidance should stay with the
  `Makefile` target help

### Reviewed CMake parity

The maintained shared parity surface is the reviewed CMake path:

```bash
make quality-review-cmake
ctest -N --test-dir build/quality-review-cmake
```

Interpretation:

- use `ctest -N` to confirm the maintained suite count when truthfulness about
  the active parity surface matters
- use the full reviewed CMake path when claiming CMake parity still passes
- keep configure/build/ctest command detail in the `Makefile` target help

### Repository-wide warning-clean claims

Repository-wide warning claims should use the Sprint 30 authoritative warning
docs and workflow:

- [Compile Hygiene Playbook](planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md)
- [Rebuild Workflow](planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md)

Interpretation:

- the Apple Clang CMake full-tree inventory remains the authoritative warning
  proof for repository-wide warning claims
- `Makefile all` remains a narrower library-build cross-check, not the
  repository-wide warning authority
- supported build surfaces define the warning-quality bar, not only the
  easiest local command

## Dead-Code Workflow Meaning

The dead-code workflow is separate from the normal lint and test surfaces:

```bash
make deadcode
make deadcode-report
make deadcode-check
```

Interpretation:

- `make deadcode` refreshes raw dead-code evidence
- `make deadcode-report` regenerates the classified report outputs
- `make deadcode-check` is a report-completeness gate, not a zero-findings
  claim
- keep exact emitted report wording and execution sequencing local to the
  `Makefile` and dead-code scripts

How to read the results:

- treat the workflow as conservative evidence rather than full reachability
  proof
- exported installed-header symbols remain manual-review items, not automatic
  deletion candidates
- dead-code noise and secondary static-analysis buckets are supporting context,
  not automatic cleanup authority by themselves

Operational constraint:

- run the `deadcode*` targets serially because they share
  `build/deadcode-cmake` and `build/deadcode/`

## Documentation Ownership Rules

Sprint 48 exists because too much maintainer policy drifted into user-facing
docs. Use these ownership rules going forward.

### `README.md`

`README.md` should stay the user/operator entry point.

It should keep:

- quick-start material
- build/test essentials
- high-level feature map
- concise operator-quality command map
- compact cross-platform quality table
- direct links to deeper docs

It should not become the full maintainer-policy home again.

### `docs/maintainer_guide.md`

This guide should own repository-wide maintainer policy such as:

- reviewed baseline interpretation
- warning authority
- dead-code meaning
- documentation ownership rules
- lifecycle/cancellation maintainer expectations
- stable style/norm reminders that affect multiple docs

### `docs/tutorial.md`

The tutorial should keep user-facing teaching flow and behavioral guidance
needed to use the library.

It should not carry long maintainer-policy blocks when a concise reference to
this guide is enough.

### Public headers

Public headers should keep concise API-local caveats needed at call sites.

They should not expand into long maintainer-policy explanations if the same
policy is already owned here.

### Local benchmark/example READMEs

`benchmarks/README.md` and `examples/README.md` should keep local usage details
and surface-specific notes.

They should not absorb repo-wide quality policy or warning-policy prose.

## Lifecycle and Cancellation Expectations

Maintainers should treat lifecycle and cancellation policy in two layers.

API-local truth:

- stays in the relevant public headers
- stays in focused tutorial prose when it teaches usage

Maintainer interpretation:

- belongs here when the point is policy ownership, documentation placement, or
  cross-surface consistency

Current stable interpretation:

- in-place direct factorization paths can legitimately carry cancellation caveat
  wording in local headers because users need that at the call site
- iterative solvers and eigensolvers generally do not need the same kind of
  input-mutation caveat because they do not factor into `A`
- long repeated lifecycle explanations across README, tutorial, and headers are
  a documentation smell; keep the concise local truth and move the broader
  policy explanation here

## Stable Repo Norms

### Non-default option examples

Use designated initializers in README/tutorial/header/example snippets when
teaching non-default option behavior.

Reason:

- evolving option structs stay clearer and less brittle when examples name the
  non-default fields explicitly

### Historical evidence vs live test truth

Do not keep retired targets, old measurements, or dormant experiment evidence
as commented-out active-suite scaffolding.

Put that material in:

- `docs/planning/`

Live non-default test semantics stay with:

- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`
- `SKIP_TEST(...)`

in:

- `tests/test_framework.h`

### Tree-mutating local modes

Some local modes intentionally rebuild the tree in an alternate configuration,
for example:

- `make sanitize`
- `make asan`
- `make sanitize-all`
- `make tsan`
- `make omp`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

When returning to the normal direct or reviewed path, reset with:

```bash
make clean
```

## Cross-Reference Guidance

When editing docs, prefer this pattern:

1. keep local truth where the user needs it
2. keep maintainer-only policy here
3. link rather than repeat when the repeated text is not locally necessary

Good examples:

- README linking here for maintainer policy
- tutorial linking here for policy interpretation while keeping user-facing
  behavior guidance local
- headers keeping short caveats while avoiding long repeated repo-policy blocks

Bad examples:

- restating the full reviewed-baseline contract in multiple user-facing docs
- duplicating dead-code interpretation in README, scripts, and guide prose
- using README as both quick-start and full maintainer handbook
