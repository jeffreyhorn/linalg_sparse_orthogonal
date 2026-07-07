# Sprint 112 Retrospective

**Sprint:** 112 - Packaging, ABI & Cross-Platform Validation Expansion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-112`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 112 started from Sprint 100 package/platform evidence templates,
      Sprint 110 no-public-header-drift evidence, and Sprint 111
      user-facing docs without inferring shared-library/ABI support or broader
      Windows coverage.
- [x] Make install, CMake install/export, pkg-config, examples, downstream
      consumers, versioning, exact-package behavior, README, INSTALL,
      maintainer docs, and CI workflow comments were audited before support
      claims changed.
- [x] The support-tier decision was made explicitly: preserve the maintained
      static-first package tier and keep shared-library packaging and dynamic
      ABI compatibility as non-claims.
- [x] Install/export proof was refreshed for the selected package tier:
  - `bash tests/test_install.sh` passed with 14 passed, 0 failed.
  - `bash tests/test_cmake_install.sh` passed with 16 passed, 0 failed,
    0 skipped.
- [x] Downstream consumer proof confirmed installed pkg-config and CMake
      consumers use public installed headers only.
- [x] Linux, macOS, and Windows support tiers were separated into reviewed,
      supplemental, staged, local-only, and unsupported lanes.
- [x] Windows stayed bounded to the reviewed MSVC CMake-first subset, with
      `test_threads`, `test_sprint4_integration`, and `test_fuzz` still
      staged.
- [x] The macOS scope stayed bounded to the reviewed Apple Clang lane plus
      supplemental Homebrew GCC and static-first Make install/`pkg-config`
      confidence.
- [x] Maintainer documentation now carries the Sprint 112 package/platform
      proof snapshot while README and INSTALL remain concise adoption surfaces.
- [x] Final validation passed:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - `git diff --check`
  - trailing-whitespace scans over touched docs and Sprint 112 artifacts
  - local relative Markdown link checks
- [x] No `.c`, `.h`, build-system, package metadata, workflow, public API,
      install-header, helper-target, or reviewed CTest-scope drift was
      introduced.
- [x] Residual deferred debt is dependency-ordered for Sprint 113 or later.

## What Went Well

1. **The package-support tier is now explicit.**
   Sprint 112 did not treat existing static install evidence as an implicit
   ABI promise. The sprint selected the static-first package tier, validated
   that tier, and kept shared-library packaging, runtime-loader behavior,
   symbol policy, SONAME/SOVERSION compatibility, DLL/import-library behavior,
   and dynamic ABI compatibility as explicit non-claims.

2. **Install/export proof is stronger and easier to cite.**
   The Make install and CMake install/export scripts now sit behind a clear
   evidence chain. The sprint recorded static archive install, 19 installed
   headers, no shared artifacts, pkg-config metadata, CMake config/version/
   target files, exact-version behavior, mismatched-version rejection, and
   installed downstream consumer compile/link/run proof.

3. **Platform wording became evidence-tiered.**
   Linux is the strongest reviewed source of truth. macOS has a reviewed Apple
   Clang lane plus supplemental Homebrew GCC and install/pkg-config confidence.
   Windows is the reviewed MSVC CMake-first subset only. This separation
   prevents local install scripts or no-header-drift evidence from silently
   becoming platform parity claims.

4. **Windows and macOS follow-through stayed honest.**
   The sprint reviewed whether exclusions could move into reviewed support and
   decided not to promote unsupported lanes without implementation or CI
   evidence. Windows staged exclusions remain visible, and macOS local
   CMake install/export proof is recorded without calling it reviewed macOS
   install/export parity.

5. **Documentation detail landed in the right place.**
   README and INSTALL already matched the support truth at user-facing depth,
   so they stayed concise. The Sprint 112 proof snapshot went into
   `docs/maintainer_guide.md`, where detailed reviewed/supplemental/staged
   interpretation belongs.

6. **Validation matched the changed surface.**
   The branch changed documentation and planning artifacts, plus maintainer
   documentation. It reran both install proof scripts and final Markdown/diff
   hygiene, while correctly avoiding the C quality chain because no `.c` or
   `.h` files changed.

## What Didn't Go Well

1. **Several platform improvements remain intentionally unpromoted.**
   Windows install validation, Windows pthread/fuzz parity, macOS CMake
   install/export parity, macOS coverage parity, and package-manager support
   still need new reviewed lanes before public claims can widen.

2. **The package story is still static-first.**
   That is the right current claim, but it leaves dynamic-linking consumers
   without a supported shared-library product story. A future shared-library
   effort needs build rules, metadata, loader behavior, symbol policy,
   versioning policy, and platform ownership before it can be advertised.

3. **Some evidence remains local rather than reviewed CI.**
   `tests/test_install.sh` and `tests/test_cmake_install.sh` are strong local
   proof surfaces, but Sprint 112 did not promote them to Linux reviewed CI
   lanes or full macOS/Windows install/export parity lanes.

4. **Exact-version package behavior is conservative.**
   Exact-version CMake package metadata avoids overclaiming ABI compatibility,
   but it also means consumers do not get broader version compatibility
   semantics. That remains correct until the project owns ABI compatibility.

## Final Metrics

### Validation

| Metric | Sprint 112 close state |
|---|---:|
| Make install/pkg-config proof | `bash tests/test_install.sh` passed, 14 passed / 0 failed |
| CMake install/export proof | `bash tests/test_cmake_install.sh` passed, 16 passed / 0 failed / 0 skipped |
| installed public headers checked by scripts | 19 |
| pkg-config version checked by scripts | `2.2.0` |
| CMake exact-version behavior | exact installed version accepted; lower mismatched version rejected |
| Windows reviewed CTest count | 51 registered tests |
| public/install header drift | 0 files |
| C source/test drift | 0 files |
| build/package/workflow drift | 0 files |
| full C quality chain required | no, docs-only branch |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched docs and Sprint 112 artifacts |
| local Markdown link check | passed |

### Package and Platform Support

| surface | Sprint 112 close state |
|---|---|
| package tier | static-first |
| shared-library package support | not claimed |
| dynamic ABI compatibility | not claimed |
| CMake package compatibility | exact-version only |
| Make install proof | local Unix-side proof |
| CMake install/export proof | local Unix-side proof |
| Linux platform tier | strongest reviewed source of truth |
| macOS platform tier | reviewed Apple Clang lane plus supplemental package and GCC confidence |
| Windows platform tier | reviewed MSVC CMake-first subset only |
| Windows staged exclusions | `test_threads`, `test_sprint4_integration`, `test_fuzz` |

### Sprint 112 Artifact Package

| Metric | Sprint 112 close state |
|---|---:|
| artifact files under `SPRINT_112/artifacts/` | 14 |
| planning and working-note files | 2 |
| retrospective files | 1 |
| maintainer docs updated | 1 |

Notes:

- scope, audit, and decision artifacts:
  - `day1-package-platform-evidence-baseline.md`
  - `day2-package-surface-audit.md`
  - `day3-abi-support-options.md`
  - `day4-abi-support-decision.md`
- package proof artifacts:
  - `day5-install-consumer-proof-design.md`
  - `day6-make-install-proof.md`
  - `day7-cmake-install-export-proof.md`
  - `day8-downstream-consumer-proof.md`
- platform and docs artifacts:
  - `day9-platform-tier-contract.md`
  - `day10-windows-follow-through.md`
  - `day11-macos-follow-through.md`
  - `day12-packaging-documentation-alignment.md`
- validation and closeout artifacts:
  - `day13-integrated-validation.md`
  - `day14-closeout-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Decide whether local Unix install scripts should become reviewed Linux CI
  lanes.
- Add a reviewed macOS CMake install/export lane before claiming full macOS
  install/export parity.
- Add a separate reviewed Windows install-validation lane before claiming
  Windows installed-package support.
- Add a Windows-native thread-test owner before promoting `test_threads` or
  `test_sprint4_integration`.
- Make `test_fuzz` portable and reviewed under MSVC before claiming Windows
  fuzz/property coverage.
- Promote macOS coverage parity only after backend/tooling behavior is stable
  enough to own as reviewed evidence.
- Revisit Homebrew GCC version assumptions when Homebrew's default GCC changes.
- Revisit macOS TSan only if the upstream dyld/runtime limitation is resolved
  and a reviewed lane is added.
- Add shared-library/dynamic ABI support only as a separate product contract
  with build rules, metadata, runtime-loader proof, symbol policy, versioning
  policy, and platform ownership.
- Add package-manager support only after package recipes and install/consumer
  proof exist.

Still consciously constrained rather than silently solved:

- no shared-library package claim;
- no dynamic ABI compatibility claim;
- no SONAME/SOVERSION or symbol export stability claim;
- no runtime-loader behavior claim;
- no Windows Makefile parity claim;
- no Windows separate reviewed install-validation lane;
- no macOS full reviewed install/export parity claim;
- no package-manager support claim;
- no public API or install-header change.

Not carried forward as unresolved Sprint 112 debt:

- package surface audit;
- static-first versus shared-library/ABI support decision;
- install/consumer proof design;
- Make install and pkg-config proof;
- CMake install/export proof;
- downstream consumer proof;
- Linux/macOS/Windows platform-tier contract;
- Windows reviewed-scope follow-through;
- macOS package/platform follow-through;
- packaging documentation alignment;
- integrated package/platform validation;
- Sprint 112 closeout and handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-package-platform-evidence-baseline.md](./artifacts/day1-package-platform-evidence-baseline.md)
- [day2-package-surface-audit.md](./artifacts/day2-package-surface-audit.md)
- [day3-abi-support-options.md](./artifacts/day3-abi-support-options.md)
- [day4-abi-support-decision.md](./artifacts/day4-abi-support-decision.md)
- [day5-install-consumer-proof-design.md](./artifacts/day5-install-consumer-proof-design.md)
- [day6-make-install-proof.md](./artifacts/day6-make-install-proof.md)
- [day7-cmake-install-export-proof.md](./artifacts/day7-cmake-install-export-proof.md)
- [day8-downstream-consumer-proof.md](./artifacts/day8-downstream-consumer-proof.md)
- [day9-platform-tier-contract.md](./artifacts/day9-platform-tier-contract.md)
- [day10-windows-follow-through.md](./artifacts/day10-windows-follow-through.md)
- [day11-macos-follow-through.md](./artifacts/day11-macos-follow-through.md)
- [day12-packaging-documentation-alignment.md](./artifacts/day12-packaging-documentation-alignment.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md)
