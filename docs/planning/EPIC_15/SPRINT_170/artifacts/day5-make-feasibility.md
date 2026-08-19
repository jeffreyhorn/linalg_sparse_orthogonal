# Sprint 170 Day 5: Makefile Static-Only Feasibility Review

## Purpose

Review the Makefile build, install, uninstall, and `pkg-config` behavior
against the Sprint 170 shared-library ABI product-decision criteria.

## Makefile Static-Only Inventory

The Makefile maintained library target is static archive only:

| Area | Current behavior |
| --- | --- |
| Library target | `LIB = $(BUILDDIR)/libsparse_lu_ortho.a` |
| Source ownership | `LIB_SRCS` enumerates the implementation sources and maps them to `LIB_OBJS`. |
| Default build | `all: $(LIB)` builds the static archive. |
| Compiler flags | C11 plus warning flags and optional `SPARSE_MUTEX` / `SPARSE_OPENMP` defines. |
| Link flags | `-lm`, optional `-ldl`, optional pthread/OpenMP flags for consumers and built tools. |
| Generated header | `$(BUILDDIR)/include/sparse_version.h` is generated from `VERSION` and `include/sparse_version.h.in`. |
| Shared target | None. |
| PIC policy | None dedicated to shared-library output. |
| SONAME/install-name/DLL policy | None. |
| Export/import macro integration | None. |

The Makefile therefore supports the current static archive product directly.
It does not contain a hidden or accidental shared-library lane that could be
promoted without explicit design work.

## Install And Uninstall Package Metadata Notes

Make install uses:

- `PREFIX ?= /usr/local`
- `INSTALL_LIB = $(DESTDIR)$(PREFIX)/lib`
- `INSTALL_INC = $(DESTDIR)$(PREFIX)/include/sparse`
- `INSTALL_PC = $(INSTALL_LIB)/pkgconfig`
- `HEADERS = $(wildcard include/*.h)`

The `install` target:

1. Builds `$(LIB)`.
2. Creates library, include, and pkg-config directories.
3. Installs `libsparse_lu_ortho.a`.
4. Installs checked-in public headers.
5. Installs generated `sparse_version.h`.
6. Generates `sparse.pc` from `sparse.pc.in`.

The `uninstall` target removes:

- `$(INSTALL_LIB)/libsparse_lu_ortho.a`
- `$(INSTALL_INC)`
- `$(INSTALL_PC)/sparse.pc`

No shared-library artifacts are installed by Make. There is no `lib*.so`,
`lib*.dylib`, `.dll`, import-library, runtime destination, or loader metadata
path in the Makefile install contract.

## `pkg-config` Static-First Review

`sparse.pc.in` describes a static archive package surface:

```pkgconfig
Name: sparse
Description: Static archive package metadata for sparse linear algebra
Version: @VERSION@
Cflags: -I${includedir}
Libs: -L${libdir} -lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@
```

The template intentionally does not contain:

- `Libs.private`
- shared/static selectors
- ABI metadata
- package-manager metadata
- SONAME, dylib, DLL, or runtime-loader claims

Make appends only selected build-option link flags to
`@SPARSE_PC_LIBS_EXTRA@`:

- `-pthread` when `SPARSE_MUTEX` is enabled.
- `-fopenmp` on non-Darwin OpenMP builds.
- `-L... -lomp` for Darwin OpenMP builds when Homebrew `libomp` is found.

This is compatible with the static-first package contract, but it also means a
future shared-library product must revisit dependency exposure. In particular,
dynamic consumers may need a deliberate public/private dependency split rather
than inheriting the current self-contained static link flags.

## Maintained Make Install Proof

`tests/test_install.sh` is the Make install/uninstall and `pkg-config`
validation owner. It verifies:

- static archive installation;
- absence of installed `.so`, `.so.*`, `.dylib`, and `.dll` artifacts;
- checked-in headers plus generated `sparse_version.h`;
- `pkg-config` package resolution and exact version resolution;
- installed `prefix`, `libdir`, and `includedir` variables;
- `pkg-config --cflags` and `pkg-config --libs` install-prefix behavior;
- `pkg-config --static` equivalence with current link flags;
- absence of `Libs.private`;
- static archive description in `sparse.pc`;
- absence of unsupported package/ABI wording in `sparse.pc`;
- downstream compile/link/run checks for both a minimal consumer and the
  maintained CMake example source;
- uninstall cleanup for library, headers, and `sparse.pc`.

This proof is strong for Unix-side static package behavior. It is not a
runtime-loader, shared-library, package-manager, or Windows Makefile parity
proof.

## Static Package Deferral Guard Coverage

`scripts/static_package_deferral_check.sh` reinforces the Make/CMake package
boundary by checking:

- CMake rejects `BUILD_SHARED_LIBS=ON`.
- CMake target remains explicitly `STATIC`.
- CMake install metadata remains archive-only.
- `sparse.pc.in` keeps the static archive package description.
- Public headers do not gain export/import or ABI selector macros.
- CMake does not gain unapproved shared ABI metadata such as `SOVERSION`,
  `WINDOWS_EXPORT_ALL_SYMBOLS`, visibility presets, install-name, or RPATH
  selectors.
- CMake and pkg-config package metadata do not gain shared/static selectors.
- README, INSTALL, maintainer guide, and Windows workflow wording preserve
  package, ABI, shared-library, runtime-loader, and Windows Make/pkg-config
  non-claims.

The guard mostly protects the CMake/shared metadata boundary, but it also
protects the Make-owned `sparse.pc.in` static description and selector absence.

## Shared-Library Makefile Feasibility Risks

| Risk | Severity | Required future work |
| --- | --- | --- |
| No shared target | High | Add an explicit opt-in target rather than changing `all` or static install behavior silently. |
| No PIC policy | High | Decide when objects are built with `-fPIC` and avoid mixing shared/static object assumptions accidentally. |
| No export list | High | Combine Day 4 symbol allowlist work with Make linker flags for Linux/macOS and a Windows strategy if Make-on-Windows is ever selected. |
| No loader metadata | High | Add Linux SONAME and macOS install-name/RPATH rules before any shared install proof. |
| No import-library/DLL policy | High | Make does not currently own a reviewed Windows package lane; DLL support would require a separate staged decision. |
| Static `pkg-config` flags | Medium | Decide whether dynamic consumers need different `Libs`, `Libs.private`, or package names. |
| Install artifact split | Medium | Add tests proving static and shared installs do not overwrite or ambiguously describe each other. |
| Uninstall artifact split | Medium | Uninstall must remove only selected artifacts and leave unrelated static/shared installs intact if both are ever supported. |

## Feasibility Finding

The Makefile is feasible and coherent for a static-first-only continuation.
It is not ready for staged shared-library exploration without first adding a
separate design for PIC, export control, loader metadata, install artifact
layout, pkg-config metadata, and validation.

For Sprint 170 decision synthesis, the Makefile evidence favors keeping the
supported Make package contract static-only unless the sprint explicitly
chooses to fund a new opt-in shared path with its own proof stack.

## Day 5 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Makefile static-only inventory | Complete | Mapped static archive, sources, generated header, flags, and absent shared controls. |
| Install/uninstall package metadata notes | Complete | Recorded installed archive, headers, generated version header, `sparse.pc`, and cleanup behavior. |
| `pkg-config` static-first review | Complete | Confirmed static archive description, link flags, and absence of unsupported selectors/claims. |
| Shared-library Makefile feasibility risks | Complete | Listed missing shared target, PIC, export, loader, install, and validation requirements. |
| Day 5 Make-feasibility artifact | Complete | This file. |

## Validation

Day 5 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Makefile package behavior is mapped to the decision needs. | Complete | Static archive build/install/pkg-config/uninstall behavior is documented. |
| Unsupported shared-library behavior remains guarded. | Complete | Existing deferral guard coverage and absent shared selectors are recorded without weakening them. |
| Feasibility notes are ready for decision synthesis. | Complete | The artifact recommends static-only continuation unless a separate shared proof stack is funded. |
