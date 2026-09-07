# Sprint 198 Day 5: Archive and Checksum Proof

## Purpose

Review and document the local Homebrew source archive and checksum proof path,
including the required archive entries, generated-output boundaries, and
metadata blocker behavior.

## Day 5 Disposition

Archive and checksum proof hardening remains gated by the Day 2 license
decision blocker. The proof script correctly stops before archive creation
while standalone root license metadata is absent, so Day 5 does not force an
archive/checksum run with guessed metadata.

The current claim-safe state is:

- no temporary source archive is created when no root `LICENSE`, `COPYING`, or
  `NOTICE` exists;
- no SHA-256 is calculated for an incomplete provider archive;
- no rendered formula receives guessed license metadata;
- proof output remains unavailable evidence, not Homebrew support evidence.

## Archive Source Inputs

When approved license metadata exists, the proof script creates a temporary
source archive from these baseline entries:

| Required entry | Purpose |
| --- | --- |
| `CMakeLists.txt` | CMake build/install entry point for the formula. |
| `Makefile` | Maintained source/build metadata included in the proof archive. |
| `VERSION` | Formula version source. |
| `sparse.pc.in` | Installed pkg-config metadata template. |
| `cmake/` | Installed CMake package metadata templates. |
| `include/` | Public headers for the installed package. |
| `src/` | Library implementation sources. |
| `examples/` | Downstream examples needed by current source package context. |
| approved root `LICENSE`, `COPYING`, or `NOTICE` | Provider license metadata required before formula proof. |

The script appends detected root metadata filenames from
`LICENSE_METADATA_ENTRIES` and then validates each required entry with
`verify_source_archive()`.

## Checksum and Formula Injection

The proof script calculates SHA-256 with `shasum -a 256` when available and
falls back to `sha256sum`. If neither tool exists, the proof exits unavailable
instead of continuing with missing checksum evidence.

After archive verification, the script injects:

- `SPARSE_FORMULA_URL=file://<temporary archive>`;
- `SPARSE_FORMULA_SHA256=<archive checksum>`;
- `SPARSE_VERSION=<VERSION>`;
- `SPARSE_HOMEBREW_HOMEPAGE=<local proof homepage or override>`;
- `SPARSE_HOMEBREW_LICENSE=<approved identifier>`.

Day 5 confirms that license injection remains blocked until the approved
identifier exists.

## Generated Output Boundary

The selected proof writes temporary archive, formula, tap, install log, and
test log files under a temporary root. `--keep-temp` may preserve those files
for diagnostics, but they remain proof outputs and must not be committed.

The package-manager guard rejects generated Homebrew proof output under
`packaging/homebrew`, including rendered `.rb` files, archives, logs, bottle
outputs, and nested `Formula` paths.

Day 5 found no generated Homebrew proof outputs under `packaging/homebrew`.

## Failure Diagnostics

| Failure point | Expected diagnostic |
| --- | --- |
| No root metadata file | `formula rendering blocked: no standalone LICENSE, COPYING, or NOTICE file exists for provider metadata` |
| Missing license identifier | `SPARSE_HOMEBREW_LICENSE is not set to accurate local-proof license metadata` |
| Placeholder license identifier | `SPARSE_HOMEBREW_LICENSE must be an accurate Homebrew license identifier, not placeholder metadata` |
| Missing archive entry after creation | `source archive is missing required entry: <entry>` |
| Missing SHA-256 tool | `no SHA-256 tool found; install shasum or sha256sum` |
| Archive creation failure | `could not create local Homebrew proof source archive: <archive>` |

## Day 6 Handoff

Day 6 should validate the render path while preserving the current metadata
blocker. If approved license inputs are still unavailable, render validation
should remain limited to template syntax, placeholder presence, and fail-fast
diagnostics before archive/render/install/test work.

## Validation

Day 5 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
