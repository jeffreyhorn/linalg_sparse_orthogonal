# Day 12 Packaging Documentation Alignment

## Purpose

Day 12 aligns public and maintainer documentation with the package, ABI, and
platform truth established by Sprint 112 Days 4-11. The goal is to keep
adoption docs concise while placing detailed proof ownership and non-claims in
maintainer-facing documentation.

## Source Evidence

| Source | Evidence used |
|---|---|
| Day 4 ABI decision | Static-first package tier selected; shared-library and dynamic ABI support remain non-claims. |
| Day 6 Make install proof | `bash tests/test_install.sh` passed with 14 passed, 0 failed. |
| Day 7 CMake install/export proof | `bash tests/test_cmake_install.sh` passed with 16 passed, 0 failed, 0 skipped. |
| Day 8 downstream consumer proof | Installed pkg-config and CMake consumers use public installed headers only. |
| Day 9 platform-tier contract | Linux, macOS, and Windows reviewed/supplemental/staged lanes are separated. |
| Day 10 Windows follow-through | Windows remains reviewed MSVC CMake-first subset with staged exclusions. |
| Day 11 macOS follow-through | macOS keeps reviewed Apple Clang lane plus supplemental package confidence. |

## Documentation Alignment Matrix

| Surface | Day 12 assessment | Action |
|---|---|---|
| `README.md` | Compact package summary already points to `INSTALL.md`, names pkg-config and CMake consumers, and preserves static-first/shared-library boundary. | No edit; keep user-facing front door concise. |
| `INSTALL.md` | Install contract, supported-platform table, verification section, macOS notes, and Windows notes already match Days 4-11 evidence. | No edit; avoid duplicating maintainer proof detail. |
| `docs/maintainer_guide.md` | Correct owner for detailed package/platform proof snapshot and non-claims. | Updated with Sprint 112 proof snapshot. |
| `CMakeLists.txt` comments | Static-first package comments and exact-version CMake package comments already match the support decision. | No edit. |
| `sparse.pc.in` | Metadata emits static package link flags and does not claim ABI compatibility. | No edit. |
| `.github/workflows/ci.yml` | Linux reviewed and supplemental lane comments already match Day 9. | No edit. |
| `.github/workflows/macos-ci.yml` | Header comments already distinguish reviewed Apple Clang, supplemental GCC, and supplemental install/pkg-config proof. | No edit. |
| `.github/workflows/windows-ci.yml` | Header comments and job output already preserve reviewed CMake subset and staged exclusions. | No edit. |

## Maintainer Documentation Update

Updated `docs/maintainer_guide.md` with a Sprint 112 package/platform proof
snapshot that records:

- static-first package tier as the selected support contract;
- local Make install proof coverage;
- local CMake install/export proof coverage;
- Linux as strongest reviewed source of truth;
- macOS reviewed Apple Clang plus supplemental package confidence;
- Windows reviewed MSVC CMake-first subset with 51 registered CTest tests;
- Windows staged exclusions for `test_threads`, `test_sprint4_integration`,
  and `test_fuzz`;
- explicit non-claims for shared-library support, ABI stability,
  package-manager support, runtime-loader behavior, Windows install-validation
  parity, and macOS full install/export parity.

## Public Documentation Decision

No README or INSTALL edit was made on Day 12. Those files already express the
validated support truth at the right level for adopters:

- README remains the first-use front door;
- INSTALL remains the operational install and validation guide;
- maintainer guide owns detailed reviewed/supplemental/staged interpretation;
- Sprint artifacts preserve the command-level proof record.

## Validation Requirements

Because Day 12 changed documentation only, required validation is:

- `git diff --check`;
- trailing-whitespace scan over touched Sprint 112 docs and maintainer docs;
- local relative Markdown link check for touched Markdown files.

No `.c` or `.h` file changed, so the C quality chain is not required for Day
12.

## Completion Criteria

- Package docs match actual validation evidence.
- Support tiers and non-claims are consistent across public and maintainer
  docs.
- Maintainer proof detail remains out of the first adoption path.
