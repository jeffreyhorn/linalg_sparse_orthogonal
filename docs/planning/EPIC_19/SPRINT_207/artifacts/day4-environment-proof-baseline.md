# Sprint 207 Day 4: Environment And Proof Baseline

## Purpose

Reproduce or block the existing package proof and record exact environment
constraints before Sprint 207 decides whether to promote a provider path or
strengthen package deferral.

## Environment Inventory

| Field | Value |
| --- | --- |
| Host OS | macOS `15.7.9-x86_64` |
| Kernel | Darwin `24.6.0` |
| CPU | Intel `kabylake`, 16-core 64-bit |
| Homebrew | `6.0.22-231-gb5dc864` |
| Homebrew prefix | `/usr/local` |
| Command Line Tools | `26.3.0.0.1.1769666919` |
| Xcode | `N/A` |
| Selected developer directory | `/Library/Developer/CommandLineTools` |
| SDK path | `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` |
| C compiler | Apple clang `17.0.0 (clang-1700.6.4.2)` |
| CMake | `4.4.3` |
| Ruby | `2.6.10p210` |

## Proof Command

```sh
HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh
```

## Proof Result

The command passed with exit `0`.

Relevant output stages:

```text
homebrew-local-formula-proof: scope: local Homebrew formula proof only; no Homebrew/core, bottles, Linuxbrew, or broad package-manager support
homebrew-local-formula-proof: creating temporary local tap: sparse-lu-ortho/local-proof-53394
homebrew-local-formula-proof: temp root: /var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T//sparse-homebrew-proof.YVpPB2
homebrew-local-formula-proof: creating local source archive
homebrew-local-formula-proof: archive sha256: c8003d621ef13919c4e47c615b5a4ea877bc9ad05992a5e9b733b108416dc5ed
homebrew-local-formula-proof: rendering temporary formula: /usr/local/Homebrew/Library/Taps/sparse-lu-ortho/homebrew-local-proof-53394/Formula/sparse-lu-ortho-local.rb
homebrew-local-formula-proof: installing local formula from source
homebrew-local-formula-proof: checking static installed package surface
homebrew-local-formula-proof: running brew test for downstream CMake consumer
homebrew-local-formula-proof: uninstalling local formula
homebrew-local-formula-proof: passed: local Homebrew formula proof completed for static source formula scope only
```

## Stage Classification

| Stage | Status | Interpretation |
| --- | --- | --- |
| Tool availability | Passed | `brew`, `cmake`, `ruby`, `tar`, checksum tool, and C compiler were available. |
| License metadata | Passed | Root MIT license metadata and `SPARSE_HOMEBREW_LICENSE=MIT` were accepted. |
| Source archive creation | Passed | Proof archive was created from selected repository entries. |
| Source archive verification | Passed | Required archive entries and license metadata were present. |
| SHA-256 calculation | Passed | Archive SHA-256 was computed. |
| Temporary tap creation | Passed | Temporary local tap was created for formula rendering. |
| Formula rendering | Passed | Placeholder formula was rendered under the temporary tap. |
| Source install | Passed | Homebrew installed the local source formula. |
| Installed static surface validation | Passed | Static archive, headers, CMake package files, and `sparse.pc` were validated. |
| Downstream `brew test` | Passed | Exact-version downstream CMake consumer compiled, linked, and ran. |
| Uninstall | Passed | Local formula was uninstalled. |
| Script cleanup | Passed | Current-run temp root and current-run temporary tap were removed. |

## Cleanup And Side-Effect Audit

| Check | Result |
| --- | --- |
| Installed formula scan | No `sparse-lu-ortho-local` formula remained installed. |
| Temporary proof tap scan | No `sparse-lu-ortho/local-proof-*` tap remained after cleanup. |
| Current proof temp root | `/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T//sparse-homebrew-proof.YVpPB2` was removed. |
| Repository packaging output scan | No generated `.tar.gz`, `.tgz`, `.zip`, `.log`, `.rb`, `.bottle.*`, or `Formula/` output appeared under `packaging/homebrew`. |
| Git status | Only Sprint 207 planning artifacts are untracked/changed. |

During cleanup review, an older stale temporary tap
`sparse-lu-ortho/local-proof-12578` was present from prior local environment
state. It was removed with:

```sh
brew untap --force sparse-lu-ortho/local-proof-12578
```

After removal, the proof tap scan returned no `sparse-lu-ortho/local-proof-*`
entries.

## Failure Classification Baseline

No Day 4 proof failure occurred. If future runs fail, classify failures this
way:

| Failure type | Classification |
| --- | --- |
| Missing `brew`, `cmake`, `ruby`, `tar`, checksum tool, or C compiler | Environment blocker; package support remains unclaimed. |
| Missing standalone root license metadata or placeholder `SPARSE_HOMEBREW_LICENSE` | Metadata blocker; package support remains unclaimed. |
| Archive missing required source entry | Project proof defect. |
| Formula render syntax failure | Project proof defect. |
| Homebrew platform-support refusal without developer mode | Provider/environment limitation, not user-facing support evidence. |
| Static installed surface missing archive, headers, CMake package, or `sparse.pc` | Project install/proof defect. |
| Shared-library artifact appears | Static package boundary violation. |
| Downstream `brew test` failure | Project package proof defect unless caused by external Homebrew/toolchain outage. |
| Cleanup leaves current-run tap, temp root, or installed formula | Project cleanup defect. |

## Claim Boundary

Day 4 proves only the existing developer-mode local Homebrew static source
formula path on this host. It does not prove:

- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottle support;
- Linuxbrew support;
- hosted binary packages;
- vcpkg, Conan, pkgsrc, distro/system package, or other package-manager
  support;
- shared-library package support;
- dynamic ABI compatibility;
- broad package-manager distribution;
- release readiness;
- package ecosystem state-of-the-art status.

## Day 4 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.2 has reproducible package proof baseline evidence or a documented environment blocker. | Complete; proof passed. |
| Environment limitations are separated from project implementation defects. | Complete; host/tooling and failure classes are recorded. |
| Cleanup requirements are known before proof path implementation. | Complete; current-run cleanup passed and stale older tap was removed. |

## Day 4 Outcome

The local proof remains reproducible on the current host with
`HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`. This strengthens the
existing local-proof baseline but does not by itself justify a public tap,
Homebrew/core, bottle, Linuxbrew, binary package, shared-library, ABI, or
broad package-manager support claim.
