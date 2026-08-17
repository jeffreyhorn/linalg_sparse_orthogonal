# Sprint 162 Day 9 Downstream Consumer Evidence

## Scope

Day 9 reviews and labels the Windows downstream package evidence for the
selected retained non-claim decision. The supported Windows package consumer
surface remains installed CMake package metadata, not Windows Makefile or
`pkg-config` execution.

## Current Windows Downstream Evidence

| Evidence | Source | Boundary |
| --- | --- | --- |
| Generated CMake consumer | `.github/workflows/windows-ci.yml` creates a temporary CMake project with `find_package(Sparse REQUIRED)`. | Proves installed CMake package discovery, link target, headers, static `.lib`, and runtime execution. |
| Maintained CMake example | `.github/workflows/windows-ci.yml` configures `examples/cmake_example` with `CMAKE_PREFIX_PATH`. | Proves the maintained downstream example works from the installed CMake package. |
| Exact-version consumer | `.github/workflows/windows-ci.yml` creates a temporary CMake project with `find_package(Sparse <version> EXACT REQUIRED)`. | Proves exact-version CMake package acceptance and downstream execution. |
| Mismatch rejection | `.github/workflows/windows-ci.yml` creates a lower same-major version request and requires configure failure. | Proves the CMake package does not accept unsupported version ranges. |
| Static metadata checks | `.github/workflows/windows-ci.yml` checks static imported target metadata, installed `.lib`, `sparse.pc` static metadata, no DLLs, and no unsupported shared-selector metadata. | Proves package metadata remains static-first before downstream consumers run. |

## Day 9 Workflow Update

Day 9 added hosted-output diagnostics to the Windows install/downstream job:

- after `sparse.pc` checks, CI states that validation is metadata-only and does
  not run `pkg-config`;
- after the generated CMake consumer runs, CI states that it passed through
  installed CMake package metadata;
- after the maintained CMake example runs, CI states that it passed through
  installed CMake package metadata;
- after the exact-version consumer runs, CI states that it passed through
  installed CMake package metadata;
- after the mismatched-version configure fails, CI states that the mismatch was
  rejected as expected.

These changes do not alter workflow commands or package behavior. They make the
downstream evidence easier to review in hosted logs and keep it scoped to the
selected Windows CMake-first support surface.

## Explicit Non-Evidence

Day 9 does not add or claim:

- Windows `pkg-config --exists`;
- Windows `pkg-config --cflags`;
- Windows `pkg-config --libs`;
- Windows `pkg-config --modversion`;
- downstream Windows compile/link/run from `pkg-config` output;
- Windows `make install`;
- Windows `make uninstall`.

Those remain retained non-claims under the Sprint 162 decision.

## Static-First Consumer Boundary

The downstream consumers run only after package metadata checks confirm:

- installed static `.lib` exists;
- no installed DLL exists;
- installed headers and generated version header exist;
- CMake package target is `STATIC IMPORTED`;
- CMake package target points at the installed static `.lib`;
- CMake package metadata has no source/build path leaks;
- CMake package metadata has no shared imported target or loader metadata;
- `sparse.pc` uses the static archive package description;
- `sparse.pc` has no unsupported package or ABI wording.

## Focused Local Validation

Run during Day 9:

```sh
bash scripts/static_package_deferral_check.sh
rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh
git diff --check
git status --short -- '*.c' '*.h'
```

Expected result:

- static guard passes with Windows package non-claim wording and no unselected
  package execution checks;
- no trailing whitespace;
- no diff hygiene errors;
- no C or header modifications.

Observed static guard result:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

## Day 9 Conclusion

The Windows downstream evidence now matches the selected package decision in
both behavior and hosted-output wording. It proves installed CMake package
consumers and exact-version behavior while keeping Windows Makefile and
`pkg-config` execution parity out of the pass evidence.
