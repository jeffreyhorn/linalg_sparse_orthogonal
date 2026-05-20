# Sprint 35 Handoff

**Source sprint:** 35  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 35's public-doc, header-example, and API-usage
consistency work into explicit starting constraints for Sprint 36 and later
Epic 3 maintenance work.

## Starting State For Sprint 36

Sprint 35 does **not** hand off a broken public-doc surface, a stale
header-example queue, or a docs/examples mismatch that still needs immediate
cleanup.

Authoritative validated close state at Sprint 35 close:

- `make examples`: passed
- `make tooling-build`: passed
- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `make quality-review-compile`: passed
- `make quality-review`: passed
- `make quality-review-cmake-compile`: passed
- `ctest -N --test-dir build/quality-review-cmake`: `53` registered tests
- `ctest --test-dir build/quality-review-cmake --output-on-failure`: `53 / 53`
  passed

Validated timings captured on Day 13:

- `make lint`: `432.50 s`
- `make test`: `106.16 s`
- `make quality-review-compile`: `381.24 s`
- `make quality-review`: `560.29 s`
- `make quality-review-cmake-compile`: `73.68 s`
- full `ctest` real time on `build/quality-review-cmake`: `173.23 s`

## Public-Surface Contract Now In Force

Sprint 35 delivered a public-doc consistency contract, not just scattered file
edits.

### Public example style

- designated initializers are the normal public example style when teaching
  non-default option-struct behavior
- `NULL` remains the intended public teaching surface only for pure-default
  paths
- public snippets should use current shipped type names, current field names,
  and current supported behavior only

### Documentation ownership split

- installed headers are the authoritative API contract
- `README.md` is the concise entrypoint:
  - command map
  - short stable snippets
  - brief signposts
- `docs/tutorial.md` is the fuller usage-teaching surface:
  - iterative/precondition guidance
  - QR/SVD usage distinctions
  - matrix-state assumptions
- `INSTALL.md`, `examples/README.md`, and `benchmarks/README.md` are support
  docs and should point back to the canonical layers instead of drifting into
  competing guidance

## Highest-Value Shipped Corrections

Sprint 35 closed the highest-signal public drift surfaced on Day 1:

- stale iterative/ILU public type names were removed from
  `docs/tutorial.md`
- public SVD wording was reconciled to the shipped full-vs-economy behavior in
  `include/sparse_svd.h` and the tutorial
- QR public docs now state the real split between:
  - `sparse_qr_solve()`
  - `sparse_qr_solve_minnorm()`
- iterative/precondition docs now state the practical family split more
  directly:
  - IC(0) for SPD workflows
  - ILU(0) / ILUT for general or indefinite-system workflows
- support docs now use the current reviewed Sprint 34 command names where they
  refer to build/quality workflows

## Sprint 34 Baseline Still Preserved

Later Epic 3 work should preserve all of these:

- Sprint 34 reviewed Makefile wrappers still define the maintained local
  quality contract
- Sprint 34 reviewed CMake parity wrappers still define the maintained CMake
  parity contract
- `53` registered CTest tests remain the auditable active-suite count until
  intentionally changed
- `tests/test_framework_optin.c` remains live coverage for:
  - `SKIP_TEST`
  - `RUN_TEST_SLOW`
  - `RUN_TEST_EXPERIMENTAL`
- Sprint 35 public-doc changes should not be casually treated as "docs-only"
  drift if they alter command names, usage contracts, or matrix/precondition
  assumptions

## Residual Deferred Queue

Sprint 35 closes without a new cleanup backlog.

Not carried forward as residual Sprint 35 debt:

- stale installed-header positional example queue: none
- stale README/tutorial public type-name queue: none
- known example-build mismatch after the rewrite: none
- reviewed-quality regression from doc/example updates: none

The main later-sprint responsibility is regression prevention:

- keep public docs aligned with the current API as cross-platform and quality
  work continues
- preserve the ownership split between headers, README, tutorial, and support
  docs
- avoid reintroducing stale option-struct or workflow-name examples when later
  sprints touch examples or quality docs

## Suggested First-Fix Queue For Sprint 36+

Sprint 36 should start from cross-platform quality parity, not from reopening
Sprint 35 documentation cleanup.

Immediate later-sprint emphasis belongs here instead:

- Sprint 36:
  - preserve the Sprint 35 public-doc contract while extending reviewed
    quality parity across macOS and Windows
  - keep support-doc command naming aligned if cross-platform quality commands
    or caveats change
- Sprint 38:
  - preserve docs/examples consistency inside the broader readiness and
    quality-gate expansion work
  - avoid broadening compile/dead-code coverage in ways the public docs do not
    reflect

## Reproduction Commands

Use these commands before and after later Epic 3 docs/example work:

1. `make examples`
2. `make tooling-build`
3. `make format`
4. `make lint`
5. `make test`
6. `make quality-review-compile`
7. `make quality-review`
8. `make quality-review-cmake-compile`
9. `ctest --test-dir build/quality-review-cmake --output-on-failure`

Expected stable comparison targets at Sprint 35 close:

- `53` registered CTest tests
- full `ctest`: `53 / 53` passing
- example/tooling compile surfaces: green
- public docs teach current stable API names and designated-initializer usage

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day4-header-batch1.md](./artifacts/day4-header-batch1.md)
- [day5-header-batch2.md](./artifacts/day5-header-batch2.md)
- [day8-readme-tutorial-implementation.md](./artifacts/day8-readme-tutorial-implementation.md)
- [day10-api-precondition-implementation.md](./artifacts/day10-api-precondition-implementation.md)
- [day11-install-quality-docs-polish.md](./artifacts/day11-install-quality-docs-polish.md)
- [day12-example-build-validation.md](./artifacts/day12-example-build-validation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
