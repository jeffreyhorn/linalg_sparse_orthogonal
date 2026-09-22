# Sprint 207 Day 3: Formula And Metadata Baseline

## Purpose

Audit package formula inputs, metadata, archive behavior, and installed static
surface against the Sprint 207 provider options before proof-path
implementation or provider-scope decision work begins.

## Audited Surfaces

| Surface | Evidence reviewed |
| --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Local-proof formula template, placeholders, build/install commands, static archive checks, downstream `test do` block. |
| `packaging/homebrew/README.md` | Proof-only scope, MIT path command, unsupported provider list, generated-output warning. |
| `scripts/homebrew_local_formula_proof.sh` | License detection, source archive creation, SHA-256 calculation, temporary tap rendering, source install, installed static surface validation, downstream `brew test`, cleanup. |
| Root `LICENSE` | MIT license metadata. |
| `VERSION` | Current rendered version source: `2.2.0`. |
| `CMakeLists.txt` | Static archive install, public header install, generated version header install, CMake package config/export install, `sparse.pc` install. |
| `sparse.pc.in` | Static package pkg-config metadata. |
| `cmake/SparseConfig.cmake.in` | CMake package config import surface. |
| README and INSTALL package wording | Current user-facing source install and package-manager non-claim wording. |

## Formula Template Baseline

| Field or behavior | Current state | Baseline interpretation |
| --- | --- | --- |
| Formula class | `SparseLuOrthoLocal` | Explicitly local-proof oriented; not yet a public tap class/name. |
| Formula name | Rendered proof installs `sparse-lu-ortho-local` through a temporary tap. | Good for avoiding accidental provider claims; not a user-facing provider name. |
| `desc` | `Static sparse linear algebra library local formula proof` | Correctly local-proof scoped; would need rewrite for public provider support. |
| `homepage` | Placeholder `__SPARSE_HOMEBREW_HOMEPAGE__`, defaulted by the proof script to a local-proof GitHub placeholder URL. | Not provider-ready until a project-owned public homepage is selected. |
| `url` | Placeholder rendered to a temporary `file://` source archive. | Local proof only; public tap or Homebrew/core needs stable public archive provenance. |
| `sha256` | Placeholder rendered from the temporary local archive. | Reproducible for local proof run; not public-provider provenance. |
| `version` | Placeholder rendered from `VERSION`. | Current version source is clear. |
| `license` | Placeholder rendered from `SPARSE_HOMEBREW_LICENSE`; current approved path is `MIT`. | MIT license blocker is closed for local proof. |
| Dependency | `depends_on "cmake" => :build` | Suitable for source formula proof. |
| Build flags | CMake source build with OpenMP and mutex disabled. | Static serial package proof; not broad feature packaging. |
| Local CMake bindir | `ENV.prepend_path "PATH", "__SPARSE_LOCAL_CMAKE_BINDIR__"` | Practical for local proof; public provider support should avoid depending on local injected tool paths unless explicitly justified. |
| Static artifact check | Requires `lib/libsparse_lu_ortho.a`; rejects `.dylib`, `.so`, `.dll`. | Strong static-only guard. |
| `test do` block | Builds exact-version downstream CMake consumer and links `Sparse::sparse_lu_ortho`. | Strong installed static consumer proof. |

## Source Archive And Checksum Behavior

The proof script creates a local archive under a temporary root and requires
these entries:

- `CMakeLists.txt`;
- `Makefile`;
- `VERSION`;
- `sparse.pc.in`;
- `cmake`;
- `include`;
- `src`;
- `benchmarks`;
- `examples`;
- `tests`;
- detected root license metadata such as `LICENSE`.

The script then lists the archive, verifies each required entry, computes a
SHA-256 using `shasum -a 256` or `sha256sum`, and renders `url
"file://$ARCHIVE"` into the temporary formula.

### Archive Baseline Finding

This is strong local proof archive behavior. It is not yet a public provider
archive policy because the URL is temporary and local, not a stable release or
public source archive. Public tap or Homebrew/core readiness needs a separate
decision for stable source provenance.

## Installed Static Surface Inventory

| Installed surface | Source owner | Proof check |
| --- | --- | --- |
| Static archive | `CMakeLists.txt`, Makefile, build system | Proof requires `lib/libsparse_lu_ortho.a`; static package guard rejects shared-library install behavior. |
| Public headers | `include/*.h`, generated version header | CMake installs checked-in public headers under `include/sparse`; proof requires installed `include/sparse`. |
| Generated version header | `include/sparse_version.h.in`, `VERSION`, generated build header | CMake installs generated `sparse_version.h`; Make install also installs generated version header. |
| CMake package config | `cmake/SparseConfig.cmake.in`, CMake export | Proof requires `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, and `SparseTargets-noconfig.cmake`. |
| CMake imported target | CMake export | Proof requires `Sparse::sparse_lu_ortho STATIC IMPORTED` and archive path metadata. |
| pkg-config metadata | `sparse.pc.in` | Proof requires `lib/pkgconfig/sparse.pc`; guard rejects provider, shared-library, and ABI wording. |
| Downstream CMake consumer | Formula `test do` block | Proof builds a downstream executable and requires exact package version. |

## Metadata Compatibility Baseline

| Metadata | Current state | Promotion implication |
| --- | --- | --- |
| License | Root MIT license file exists and proof accepts `SPARSE_HOMEBREW_LICENSE=MIT`. | License metadata supports local MIT proof and likely public formula metadata, subject to provider formatting. |
| Version | `VERSION` is `2.2.0`. | Version can feed formula metadata; release provenance remains unproven. |
| Package description | `sparse.pc.in` says static archive package metadata. | Correctly avoids shared-library and provider claims. |
| CMake package | Minimal static package import. | Suitable for installed static consumer proof. |
| User docs | README and INSTALL keep package-manager distribution unclaimed and route users to Make/CMake source installs. | Claim-safe for current state; must change only after provider proof. |
| Maintainer docs | Maintainer guide names exact local proof and guard commands. | Good runbook baseline for future guard changes. |

## Gaps Before Provider Promotion

| Gap | Affects | Owner condition |
| --- | --- | --- |
| No public tap formula artifact. | Public Homebrew tap/source formula. | Add or explicitly design a public tap formula separate from local proof template. |
| No stable public archive URL. | Public tap and Homebrew/core readiness. | Define release/archive provenance and checksum policy. |
| Formula renders `file://` temp archive. | Public tap and Homebrew/core readiness. | Replace or supplement local proof with stable-source proof before promotion. |
| Formula name/class includes `Local`. | Public tap and Homebrew/core readiness. | Rename for provider formula if promotion is selected. |
| Formula injects local CMake bindir. | Public tap and Homebrew/core readiness. | Decide whether provider formula can rely on Homebrew-managed CMake without local path injection. |
| Homebrew/core audit readiness absent. | Homebrew/core readiness. | Add audit record and Homebrew/core-style metadata proof before any readiness claim. |
| No bottle or Linuxbrew evidence. | Bottles and Linuxbrew. | Retain non-claims unless separate evidence is added. |
| No shared-library package or ABI evidence. | Shared-library package support and ABI. | Retain static package boundary and static-package guard. |

## Provider Option Implications

| Option | Day 3 implication |
| --- | --- |
| Public Homebrew tap/source formula | Possible future path, but not yet earned. Needs stable source/archive policy, public formula naming/ownership, and proof that does not depend solely on temporary local artifacts. |
| Homebrew/core readiness | Not supported by current baseline. Requires release/archive discipline, Homebrew/core audit readiness, non-local formula behavior, and maintainer submission policy. |
| Continued deferral | Fully supported by current baseline if Sprint 207 strengthens guards and docs around the local-proof-only state. |

## Day 3 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.2 has current evidence for formula and metadata state. | Complete. |
| Static install contents are understood before proof changes. | Complete. |
| Missing metadata or reproducibility issues are recorded with an owner. | Complete. |

## Day 3 Outcome

Day 3 confirms that the current package machinery is coherent and guarded for
developer-mode local Homebrew static source formula proof. The same machinery
does not yet prove a public tap/source formula or Homebrew/core readiness
because public source archive provenance, public formula ownership, non-local
formula behavior, and audit-readiness evidence are missing.
