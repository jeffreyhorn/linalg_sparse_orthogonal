# Homebrew Local Formula Proof

This directory contains local proof material for the Sprint 180 Homebrew
provider decision. It is not a Homebrew/core formula, a tap, a bottle, a
Linuxbrew claim, or general package-manager support.

The source-controlled template is:

- `sparse-lu-ortho.rb.in`

Run the local proof command from the repository root:

```sh
SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh
```

Use the command result this way:

| Exit | Meaning | Support wording |
| ---: | --- | --- |
| `0` | Render, archive, checksum, install, installed-surface validation, `brew test`, uninstall, and cleanup passed. | Docs may mention only the exact local static source formula proof. |
| `2` | A required local prerequisite or approved license metadata is unavailable. | Homebrew support remains unclaimed. |
| Any other nonzero exit | The proof failed. | Fix the proof failure before changing support wording. |

The template is not installed directly. The proof script renders it into a
temporary local formula by injecting:

- the current checkout version from `VERSION`;
- a temporary `file://` source archive URL;
- that archive's SHA-256 checksum;
- accurate local-proof license metadata.

The license metadata must be a project-approved Homebrew license identifier
that matches a standalone root `LICENSE`, `COPYING`, or `NOTICE` file.
Placeholder values such as `NOASSERTION`, `UNKNOWN`, `TBD`, `TODO`, or
template placeholder text are blocker evidence, not proof metadata.

The proof validates license metadata before creating the temporary source
archive. Once approved metadata exists, the archive must include the selected
standalone license file along with the source, CMake/package metadata, and
examples needed by the local formula proof.

After install, the proof validates the static archive, installed headers,
CMake package files, and `sparse.pc`. Installed package metadata must not gain
provider wording, shared-library selectors, `Libs.private`, dynamic ABI
wording, SONAME/DLL/dylib policy, or static/shared selection knobs.

Before the proof reaches `brew test`, the script checks that the template's
`test do` block still builds an exact-version downstream CMake consumer, links
`Sparse::sparse_lu_ortho`, exercises installed public headers, asserts
successful executable output, verifies installed package metadata, and rejects
shared-library artifacts.

Generated formula files, local taps, source archives, logs, Homebrew caches,
build trees, install prefixes, and bottle outputs are proof outputs and must
not be committed.

The selected proof boundary is local source formula only, static archive only,
and macOS-local first. Homebrew/core, bottles, Linuxbrew, hosted binaries,
registry readiness, shared-library support, dynamic ABI support, static/shared
selectors, and broad package-manager support remain unsupported unless a later
product decision adds separate evidence.

Sprint 186 closeout keeps this directory in proof-only status. Until approved
standalone license metadata exists at the repository root and the proof script
completes render, install, `brew test`, uninstall, and cleanup successfully,
do not present this template as an available Homebrew install method.

Maintainers changing this directory should run:

```sh
scripts/package_manager_deferral_check.sh
scripts/static_package_deferral_check.sh
```

Also run install checks if CMake, Makefile, install metadata, or downstream
consumer wording changes. Run package report normalization checks only when
package report metadata changes.
