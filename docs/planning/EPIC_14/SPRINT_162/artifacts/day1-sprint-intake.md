# Day 1 Sprint Intake And Package Surface Inventory

Day 1 establishes the Sprint 162 evidence baseline and separates the Windows
package parity decision from solver comparison evidence.

## Scope Source

Sprint 162 implements the current Epic 14 project-plan section:
`docs/planning/EPIC_14/PROJECT_PLAN.md`, Sprint 162. The prompt references an
older Epic 12 path; this artifact follows the current Epic 14 plan.

## Sprint Goal

Decide and close the remaining Windows package parity gap for `pkg-config` and
Makefile support without confusing it with CMake install validation.

## Starting Package Inventory

| Surface | File or Lane | Current Evidence | Day 1 Classification |
| --- | --- | --- | --- |
| Unix Make install | `Makefile`, `tests/test_install.sh` | Installs static archive, headers, generated version header, and `sparse.pc`; validates uninstall. | Maintained Unix-side proof. |
| Unix `pkg-config` execution | `tests/test_install.sh`, Linux/macOS CI | Validates `pkg-config --exists`, exact version, variables, cflags, libs, static libs, downstream compile/link/run, and maintained example. | Maintained Unix-side proof. |
| CMake install/export | `CMakeLists.txt`, `tests/test_cmake_install.sh` | Installs static archive and CMake package files; validates static imported target, no source/build leaks, exact-version behavior, downstream CMake example, and static-first `sparse.pc` metadata. | Maintained static-first proof. |
| Static deferral guard | `scripts/static_package_deferral_check.sh` | Rejects `BUILD_SHARED_LIBS=ON`; checks static target/install metadata, no export/import macros, no ABI loader metadata, and no static/shared package selector. | Maintained non-claim guard. |
| Linux reviewed package lane | `.github/workflows/ci.yml` | Runs Make install/`pkg-config`, CMake install/export, and static-first deferral checks. | Strongest reviewed Unix package proof. |
| macOS reviewed package lanes | `.github/workflows/macos-ci.yml` | Runs reviewed Make install/`pkg-config` and CMake install/export proof. | Reviewed macOS static-first proof. |
| Windows CMake consumer subset | `.github/workflows/windows-ci.yml` | Runs MSVC CMake configure/build/CTest and checks expected CTest count. | Reviewed CMake-first Windows test proof. |
| Windows CMake install/downstream | `.github/workflows/windows-ci.yml` | Installs static `.lib`, headers, CMake metadata, `sparse.pc` metadata, CMake downstream consumers, exact-version behavior, mismatch rejection, and unsupported shared metadata checks. | Reviewed Windows CMake install/downstream proof. |
| Windows Makefile parity | `.github/workflows/windows-ci.yml`, docs | Explicitly not claimed. | Open Sprint 162 decision. |
| Windows `pkg-config` execution parity | `.github/workflows/windows-ci.yml`, docs | `sparse.pc` metadata is installed, but `pkg-config` execution parity is explicitly not claimed. | Open Sprint 162 decision. |
| Public install docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | State static-first package shape and Windows CMake-first boundary. | Documentation baseline. |
| Downstream CMake example | `examples/cmake_example` | Maintained installed CMake consumer example. | Existing downstream proof owner. |

## Initial Parity Split

| Question | Current Answer | Sprint 162 Action |
| --- | --- | --- |
| Does Windows install static package metadata? | Yes, via CMake install/downstream workflow. | Preserve and audit. |
| Does Windows prove CMake downstream consumers? | Yes. | Preserve and strengthen only if selected. |
| Does Windows prove `pkg-config` execution? | No. `sparse.pc` is installed as metadata, but execution parity is not claimed. | Decide promote or retain non-claim. |
| Does Windows prove Makefile install parity? | No. | Decide promote or retain non-claim. |
| Does any package surface prove shared libraries or dynamic ABI? | No. | Keep guarded as explicit deferral. |

## Sprint 161 Handoff Incorporated

Sprint 161 closes solver comparison evidence and explicitly warns that package
claims must be earned through package/install evidence. Sprint 162 therefore
starts with this boundary:

- do not reuse selected QR or partial-SVD comparison freshness as package
  proof;
- do not infer Windows package parity from installed `sparse.pc` metadata
  alone;
- decide Windows `pkg-config` and Windows Makefile parity independently;
- keep static-first CMake package proof separate from runtime-loader,
  shared-library, and package-manager claims.

## Explicit Non-Goals

Sprint 162 does not prove:

- package-manager availability through Homebrew, apt, dnf, pacman, vcpkg,
  Conan, or similar tools;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selectors;
- broad Windows platform parity;
- Windows Makefile parity unless selected by the product decision;
- Windows `pkg-config` execution parity unless selected by the product
  decision;
- performance, release, or state-of-the-art evidence.

## Assumptions

- The Windows CMake install/downstream lane is green enough to be the starting
  reviewed Windows package proof.
- Linux/macOS Make install and `pkg-config` proof is the comparison baseline
  for Unix-style installed consumers.
- Retaining a non-claim is acceptable if the sprint adds clearer checks and
  docs that prevent accidental support wording.
- Any promoted Windows `pkg-config` path needs an explicit provider and
  downstream compile/link/run proof.
- Any promoted Windows Makefile path needs a reviewed command path and cannot
  be inferred from CMake install behavior.

## Stop Conditions

Stop and reassess if proposed implementation:

- treats `sparse.pc` file existence as `pkg-config` execution proof;
- treats CMake install/downstream proof as Windows Makefile parity;
- introduces shared-library, dynamic ABI, runtime-loader, or package-manager
  support wording;
- weakens Linux/macOS package validation or Windows CMake install proof;
- changes C or header files without running the full C quality gate.

## Day 1 Completion

Day 1 completed the Sprint 162 package surface inventory, created working
notes, recorded non-goals, and established the Day 2 audit path: compare
Windows CMake install proof against Unix Make install and `pkg-config` proof
with the parity questions separated.
