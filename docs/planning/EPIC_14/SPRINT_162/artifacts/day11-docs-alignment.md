# Sprint 162 Day 11 Documentation Alignment

## Scope

Day 11 aligns public and maintainer documentation with the Sprint 162 retained
non-claim decision. The documentation must describe the Windows package surface
as reviewed CMake install/downstream proof while keeping Windows Makefile and
`pkg-config` execution parity out of supported claims.

## Documentation Updates

| File | Update | Reason |
| --- | --- | --- |
| `README.md` | Added a Windows install note that `sparse.pc` is installed and inspected as static package metadata by the reviewed CMake install/downstream lane. | Prevents readers from interpreting installed `sparse.pc` as Windows `pkg-config` command proof. |
| `INSTALL.md` | Clarified that Windows CI inspects installed `sparse.pc` as metadata only and does not run `pkg-config`. | Aligns the install contract with the retained non-claim decision. |
| `INSTALL.md` | Updated support-split and supported-platform wording to say Windows carries metadata-only `sparse.pc` inspection. | Keeps platform support tiers precise. |
| `docs/maintainer_guide.md` | Updated package/platform proof wording to metadata-only `sparse.pc` inspection. | Gives maintainers the same interpretation used by CI. |
| `docs/maintainer_guide.md` | Added a Sprint 162 note that Windows Makefile install/uninstall parity and Windows `pkg-config` command execution parity remain unsupported unless a future product decision adds proof. | Records the retained non-claim boundary in the package history. |

## Support-Tier Wording Checklist

| Surface | Documentation State |
| --- | --- |
| Windows CMake install/downstream | Reviewed and supported for the static-first package surface. |
| Windows installed `sparse.pc` | Metadata-only inspection. |
| Windows `pkg-config` command execution | Explicit retained non-claim. |
| Windows Makefile install/uninstall | Explicit retained non-claim. |
| Linux/macOS Make install and `pkg-config` | Reviewed package proof. |
| Shared-library packaging | Deferred and unsupported. |
| Dynamic ABI compatibility | Deferred and unsupported. |
| Runtime-loader behavior | Deferred and unsupported. |
| Package-manager distribution | Unsupported package claim. |
| Broad Windows parity | Unsupported platform claim. |

## Unsupported-Claim Scan Notes

The Day 11 documentation wording intentionally avoids:

- saying Windows supports `pkg-config`;
- saying Windows runs `pkg-config --exists`, `--cflags`, `--libs`, or
  `--modversion`;
- saying Windows supports `make install` or `make uninstall`;
- implying installed `sparse.pc` metadata is equivalent to command execution;
- implying shared-library support, dynamic ABI compatibility, runtime-loader
  behavior, package-manager distribution, or broad Windows parity.

## Sprint 163 Handoff Wording

Sprint 163 performance or publication work should not reinterpret Sprint 162
package evidence as runtime performance evidence or broader platform support.
The handoff boundary is:

- package proof may cite Windows CMake install/downstream validation;
- package proof may cite metadata-only Windows `sparse.pc` inspection;
- performance proof must stay separate from package proof;
- Windows Makefile and `pkg-config` execution parity remain retained
  non-claims.

## Focused Local Validation

Run during Day 11:

```sh
bash scripts/static_package_deferral_check.sh
rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 README.md INSTALL.md docs/maintainer_guide.md .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh
git diff --check
git status --short -- '*.c' '*.h'
```

Expected result:

- static guard passes with support wording, Windows package non-claim wording,
  and no unselected Windows package execution checks;
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

## Day 11 Conclusion

Documentation now matches the implemented Sprint 162 Windows package decision:
Windows package support is reviewed through CMake install/downstream
validation, installed `sparse.pc` is metadata-only inspection, and Windows
Makefile plus `pkg-config` execution parity remain explicit non-claims.
