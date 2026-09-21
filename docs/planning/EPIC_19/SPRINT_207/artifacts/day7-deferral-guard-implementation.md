# Sprint 207 Day 7: Deferral Guard Implementation

## Purpose

Implement the first continued-deferral hardening batch selected by Day 5 and
designed by Day 6, while preserving the existing local Homebrew static source
formula proof behavior.

## Changed Surface

| File | Change |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Added Sprint 207 provider-decision guard coverage and broader forbidden-provider-claim scans. |
| `docs/planning/EPIC_19/SPRINT_207/WORKING_NOTES.md` | Recorded Day 7 implementation and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_207/artifacts/day7-deferral-guard-implementation.md` | Added this implementation record. |

No formula template, proof script behavior, install metadata, C source, or
public header changed.

## Package Guard Changes

### Sprint 207 Decision Guard

Added `check_sprint207_provider_decision()` to
`scripts/package_manager_deferral_check.sh`.

The check requires:

- Day 5 provider decision artifact exists;
- Day 6 proof/deferral design artifact exists;
- continued deferral with stronger guards remains the selected path;
- public tap promotion remains rejected;
- Homebrew/core readiness wording remains rejected;
- future public tap evidence includes stable source archive and SHA-256
  provenance;
- future Homebrew/core evidence includes Homebrew/core-style formula audit
  evidence;
- Day 6 maps successful proof only to developer-mode local static source
  formula proof;
- Day 6 retains regression targets for public tap and Homebrew/core wording.

### Unsupported Provider Claim Guard

Added `check_forbidden_public_provider_claims()` to scan these current support
surfaces:

- `README.md`;
- `INSTALL.md`;
- `packaging/homebrew/README.md`;
- `docs/maintainer_guide.md`.

The scan rejects positive support/availability/provider wording for:

- package-manager distribution;
- Homebrew/core readiness;
- Homebrew/core support;
- public tap support;
- bottles;
- bottle support;
- Linuxbrew;
- Linuxbrew support;
- vcpkg;
- Conan;
- pkgsrc;
- distro/system packages;
- binary packages;
- binary package support.

## Preserved Behavior

The existing local proof command is unchanged:

```sh
HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh
```

The package guard still runs the selected local proof path and still treats
successful proof as local static source formula evidence only.

## Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed | Shell syntax is valid. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Deferral record, Sprint 198 proof record, provider recipe absence, selected Homebrew local proof boundary, package metadata neutrality, Sprint 207 provider decision, forbidden public provider claims, and public non-claims all passed. |
| Installed formula scan | Passed | No `sparse-lu-ortho-local` formula remained installed after validation. |
| Temporary proof tap scan | Passed | No `sparse-lu-ortho/local-proof-*` tap remained after validation. |
| Generated Homebrew output scan | Passed | No generated `.tar.gz`, `.tgz`, `.zip`, `.log`, `.rb`, `.bottle.*`, or `Formula/` output appeared under `packaging/homebrew`. |

## Claim Boundary

Day 7 strengthens guard enforcement for the Day 5 decision. It does not
promote:

- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottles;
- Linuxbrew;
- vcpkg, Conan, pkgsrc, distro/system packages, or binary packages;
- shared-library package support;
- dynamic ABI compatibility;
- broad package-manager distribution.

## Day 7 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.3 has concrete implementation progress. | Complete; package guard now enforces Sprint 207 decision and provider non-claim scans. |
| Existing package proof behavior is preserved or intentionally replaced. | Complete; preserved. |
| Unsupported provider states fail clearly. | Complete for first batch; positive provider wording now fails through the package guard. |

## Day 7 Outcome

The first implementation batch is complete. Sprint 207 now has guard coverage
that ties package-manager support wording to the Day 5 continued-deferral
decision and rejects common unsupported provider claims in public and
maintainer-facing docs.
