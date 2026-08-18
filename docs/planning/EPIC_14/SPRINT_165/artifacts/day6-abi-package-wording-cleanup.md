# Sprint 165 Day 6 ABI And Package Wording Cleanup

## Purpose

Day 6 applies the Day 5 ABI wording cleanup and rechecks package docs for
static-first and dynamic ABI non-claim consistency. The goal is to remove stale
ABI-adjacent phrasing without changing supported install behavior.

## Change Applied

| File | Change | Reason |
| --- | --- | --- |
| `include/sparse_cholesky.h` | Replaced the older `@warning **ABI break in v2.0.0.**` wording with `@warning **Source rebuild required for v2.0.0 options layout.**` and preserved the zero-initialized source initializer guidance. | The old wording warned about a real compiled-object layout issue, but could imply the project otherwise maintains a dynamic ABI policy. The new wording matches `include/sparse_ldlt.h` and `include/sparse_eigs.h`. |

## Surfaces Reviewed Without Edits

| Surface | Result |
| --- | --- |
| `README.md` | Already keeps package evidence bounded and does not claim ABI guarantees, package-manager distribution, broad platform package parity, or shared-library support. |
| `INSTALL.md` | Already separates supported static archive install workflows from deferred shared-library, dynamic-loader, dynamic ABI, package-manager, and Windows Makefile/pkg-config execution parity. |
| `docs/maintainer_guide.md` | Already tells maintainers not to treat exact package version metadata as a dynamic ABI guarantee. |
| `CMakeLists.txt` | Already documents exact CMake package-version compatibility as package metadata, not a broad dynamic ABI guarantee. |
| `cmake/SparseConfig.cmake.in` and `sparse.pc.in` | No dynamic ABI or shared-library claims were introduced. |

## Package And ABI Boundary After Cleanup

The maintained package surface remains:

- installed static archive;
- installed public headers;
- installed CMake package metadata;
- installed `sparse.pc` metadata;
- downstream source rebuilds against installed headers and static archive.

The following remain non-claims:

- dynamic ABI compatibility;
- stable binary struct layout across previously compiled objects;
- shared-library support;
- runtime-loader behavior;
- Linux SONAME policy;
- macOS install-name/RPATH policy;
- Windows DLL/import-library behavior;
- package-manager distribution or upgrade policy;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` execution parity.

## Validation

Because a public header changed, Day 6 ran the required full quality gate:

```text
make format && make lint && make test
```

Result:

```text
All tests passed.
```

Additional checks:

```text
bash scripts/static_package_deferral_check.sh
```

Result: passed.

```text
git diff --check
```

Result: passed.

```text
rg -n "ABI break|ABI compatible|ABI stable|binary compatible|soname policy|ABI guarantee" \
  include/sparse_cholesky.h README.md INSTALL.md docs/maintainer_guide.md CMakeLists.txt
```

Result: the stale `ABI break` phrase is gone. Remaining hits are expected
non-claim language in README, CMake comments, and maintainer guidance.

## Completion Check

- The Day 5 public-header wording candidate was cleaned up.
- Exact-version package metadata remains clearly separated from dynamic ABI
  guarantees.
- Static-first install guidance remains unchanged and usable.
- Unsupported ABI, shared-library, runtime-loader, package-manager, and broad
  platform claims were not added.
