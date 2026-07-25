# Sprint 133 Day 12 - pkg-config Consumer Proof

## Purpose

Day 12 strengthens the downstream `pkg-config` consumer proof for the selected
static-first package contract. The proof now validates generated `.pc` fields,
link flag interpretation, current static/private dependency semantics, and
consumer compile/link/run behavior.

## Implemented Proof

| File | Change |
| --- | --- |
| `tests/test_install.sh` | Added exact `pkg-config` resolution, variable, flag, static-link, and unsupported-claim checks before the existing downstream consumer compile/run checks. |
| `docs/maintainer_guide.md` | Documented the expanded Make install and `pkg-config` proof responsibilities. |

No C source, public headers, package templates, install rules, workflows, or
package-manager files changed on Day 12.

## Added pkg-config Checks

| Check | Implemented behavior |
| --- | --- |
| Package resolution | Requires `pkg-config --print-errors --exists sparse` to pass. |
| Exact version resolution | Requires `pkg-config --exists "sparse = $VERSION"` to pass. |
| Installed variables | Requires `prefix`, `libdir`, and `includedir` to point at the temporary install prefix. |
| Installed include flags | Requires `pkg-config --cflags sparse` to emit the installed include path. |
| Installed link flags | Requires `pkg-config --libs sparse` to emit the installed static archive link flags: `-L.../lib -lsparse_lu_ortho -lm`. |
| Static link interpretation | Requires `pkg-config --libs --static sparse` to match the current self-contained link flags. |
| Private dependency semantics | Requires no `Libs.private` stanza in the current `sparse.pc`; dependencies needed by downstream consumers stay in `Libs` until a future product decision introduces a private/static split. |
| Unsupported claim fence | Scans `sparse.pc` for unsupported shared-library, ABI, or package-manager wording. |

The existing downstream proof remains in place: compile, link, and run both a
small installed-package consumer and the maintained example source with
`pkg-config` output.

## Validation Evidence

Successful focused run:

```text
  [PASS] pkg-config can resolve sparse
  [PASS] pkg-config exact version constraint works
  [PASS] pkg-config prefix points at install prefix
  [PASS] pkg-config libdir points at installed libdir
  [PASS] pkg-config includedir points at installed includedir
  [PASS] pkg-config --cflags returns installed include path
  [PASS] pkg-config --libs returns installed static archive link flags
  [PASS] pkg-config --static libs match current self-contained link flags
  [PASS] pkg-config file has no private dependency stanza
  [PASS] pkg-config file has no unsupported packaging or ABI claims
--- Summary ---
Passed: 22
Failed: 0
ALL INSTALL TESTS PASSED
```

The first focused runs exposed path-normalization assumptions in the new exact
flag checks. Different `pkg-config` implementations may preserve or collapse
repeated slashes in emitted compiler/linker flags. The test now keeps raw
`.pc` variable checks separate from emitted flag checks: emitted `-I` and `-L`
paths must resolve to the installed include and library directories, while the
remaining link flags stay exact.

## Support Boundary

This proof strengthens the local Unix-side Make install and `pkg-config`
consumer story for the selected static archive package surface. It does not
add shared-library packaging, dynamic ABI compatibility, package-manager
support, or reviewed platform install parity.

## Residual pkg-config Queue

| Item | Status |
| --- | --- |
| `Libs.private` split | Deferred. The current package has a self-contained public link surface, and `-lm` remains in `Libs` for downstream consumers. |
| Optional thread/OpenMP flags | Existing Make/CMake generation appends optional flags when those build modes are selected, but Day 12 validates the default installed package contract only. |
| CI promotion | `tests/test_install.sh` remains local install proof unless a future sprint promotes the full path to reviewed CI. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| An installed pkg-config consumer can prove the selected contract. | Complete | `tests/test_install.sh` compiles, links, and runs two installed consumers using `pkg-config` output. |
| Static/private dependency semantics are documented and tested. | Complete | The test requires no `Libs.private` stanza and matching `--libs`/`--libs --static` output for the current self-contained link surface. |
| Unsupported package-manager or ABI claims remain fenced. | Complete | The installed `sparse.pc` is scanned for unsupported shared-library, ABI, and package-manager wording. |
