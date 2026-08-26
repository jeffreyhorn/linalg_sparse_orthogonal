# Day 14: Integrated Validation And Closeout

## Purpose

Day 14 closes Sprint 180 by validating that the package-manager provider
decision, selected artifact path, proof-script behavior, guards, public docs,
package metadata, install/export checks, and generated-output hygiene are
consistent.

## Sprint Decision Status

Sprint 180 selects a local Homebrew formula/tap proof as the first
package-manager provider implementation path. This is not a public Homebrew
support claim. The current repository state leaves the proof unclaimed because
there is no standalone `LICENSE`, `COPYING`, or `NOTICE` file that the formula
can cite as provider metadata.

The selected path is source-controlled through:

- `packaging/homebrew/sparse-lu-ortho.rb.in`
- `packaging/homebrew/README.md`
- `scripts/homebrew_local_formula_proof.sh`
- `scripts/package_manager_deferral_check.sh`
- README, INSTALL, maintainer-guide non-claim wording
- Sprint 180 artifacts and working notes

The selected path remains bounded to local proof material only. Homebrew/core,
bottles, Linuxbrew, public taps, provider-hosted binaries, broad
package-manager support, shared-library support, dynamic ABI support, and
static/shared selectors remain unsupported.

## Validation Results

| Check | Result | Notes |
| --- | --- | --- |
| `bash scripts/homebrew_local_formula_proof.sh` | Pass as claim-safe unavailable | Exit status `2`; creates a temporary source archive and stops before formula rendering because no standalone license metadata file exists. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Confirms Sprint 171 deferral record, unselected provider absence, selected Homebrew proof boundary, metadata neutrality, public non-claims, and claim-safe proof behavior. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Confirms static-first package contract and shared-library/dynamic ABI deferrals remain intact. |
| `bash -n scripts/homebrew_local_formula_proof.sh` | Pass | Proof script parses as shell. |
| `bash -n scripts/package_manager_deferral_check.sh` | Pass | Provider claim guard parses as shell. |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Pass | Formula template parses as Ruby despite render-time placeholders. |
| `python3 scripts/normalize_report_index.py --family package --check` | Pass | `7` package report rows normalize cleanly. |
| `bash tests/test_install.sh` | Pass | `23` Make install, pkg-config, downstream compile/run, version, and uninstall checks passed. |
| `bash tests/test_cmake_install.sh` | Pass | `27` CMake install/export, downstream `find_package`, exact-version, mismatch-version, pkg-config, and static metadata checks passed. |
| Generated Homebrew output hygiene check | Pass | No committed `Formula/` files, archives, logs, bottles, or generated tap output under `packaging/homebrew`. |
| `git diff --check` | Pass | No whitespace errors. |

## Consistency Review

| Surface | Closeout state |
| --- | --- |
| Product decision | Exactly one selected provider proof path: local Homebrew formula/tap proof. |
| Public support posture | Package-manager support remains unavailable; Homebrew proof remains unclaimed. |
| Provider artifacts | Template, provider notes, and proof script exist; no rendered formula or generated tap material is committed. |
| Guard behavior | The package-manager guard now allows the exact selected local-proof artifacts while still rejecting unselected provider recipes and broad package-manager claims. |
| Package metadata | `sparse.pc.in` and `cmake/SparseConfig.cmake.in` remain provider-neutral. |
| Static-first contract | Install/export validation and static-package guard pass; shared-library and dynamic ABI support remain deferred. |
| Docs | README, INSTALL, and maintainer guide describe the selected proof artifacts and missing-license blocker without claiming provider support. |

## Residual Risks

- The local Homebrew install and `brew test` path has not executed because
  formula rendering is intentionally blocked by absent standalone license
  metadata.
- Homebrew/core, bottles, Linuxbrew, public taps, hosted binary packages, and
  broad package-manager support remain unproven and unsupported.
- vcpkg, Conan, and pkgsrc remain rejected as first-provider Sprint 180 paths;
  revisiting them requires new tooling, recipe, license, checksum, and
  provider-specific proof evidence.

## Sprint 181 Handoff

Before citing Homebrew support or local Homebrew install proof:

1. Add an approved standalone `LICENSE`, `COPYING`, or `NOTICE` file.
2. Re-run `bash scripts/homebrew_local_formula_proof.sh` and require a
   successful render, local install, `brew test`, uninstall, and cleanup path.
3. Keep generated formula, tap, source archive, bottle, cache, log, and install
   output out of source control.
4. Update README, INSTALL, maintainer-guide wording, and
   `scripts/package_manager_deferral_check.sh` only after proof behavior
   changes.
5. Continue running `bash scripts/static_package_deferral_check.sh` and install
   validation before any package-manager claim change.

## Retrospective Inputs

- The sprint successfully converted a broad provider question into one bounded
  selected proof path with fail-closed guard behavior.
- The strongest practical blocker is license metadata, not recipe structure or
  local archive generation.
- Keeping the source-controlled artifact as a template avoided accidental
  public tap or formula claims.
- Guarding docs, metadata, generated-output hygiene, and proof-script behavior
  together kept the package-manager posture consistent across public surfaces.
