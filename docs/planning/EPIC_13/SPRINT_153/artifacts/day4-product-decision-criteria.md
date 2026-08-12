# Sprint 153 Day 4 Product Decision Criteria

## Purpose

Day 4 converts the Day 2 ABI surface audit and Day 3 loader audit into
decision criteria for Day 5. The decision is between implementing a first
supported shared-library surface in Sprint 153 or strengthening the
static-first deferral with exact blockers and proof.

## Decision Inputs

| Input | Finding | Decision Impact |
| --- | --- | --- |
| Public header audit | The installed surface has `19` headers, many public concrete layouts, callback typedefs, error/version utilities, and allocator/lifetime contracts. | Any dynamic ABI claim must govern structs, callbacks, enums, allocator boundaries, and version metadata. |
| Accidental export audit | Internal non-static helpers would be exported by a naive shared build. | Shared support requires symbol curation before any supported artifact ships. |
| Linux loader audit | Linux needs `.so`, SONAME, exported-symbol filtering, dependency metadata, installed consumer proof, and loader inspection. | Linux is feasible only with a narrow, proof-backed target. |
| macOS loader audit | macOS needs `.dylib`, install-name/RPATH policy, exported-symbol filtering, dependency handling, and installed consumer proof. | macOS support cannot be inferred from existing static package proof. |
| Windows loader audit | Windows needs DLL/import-library layout, `SPARSE_API`, `__declspec` policy, runtime lookup proof, and allocator/C runtime review. | Windows is the largest risk and cannot be claimed without source/header changes. |
| Existing package proof | Linux, macOS, and Windows CI prove static-first install/export/downstream behavior and reject shared artifacts. | Static-first deferral is already maintained and can be strengthened with exact blockers. |

## Product Paths Under Consideration

### Path A: Implement Shared-Library Support

Implement a first supported shared-library package surface in Sprint 153. This
can only be accepted if its platform scope is explicit and all proofs for that
scope land in the same sprint.

### Path B: Strengthen Static-First Deferral

Keep the maintained package contract static-first, preserve
`BUILD_SHARED_LIBS=ON` rejection, and make the deferral stronger by tying it to
exact ABI, export, loader, metadata, and downstream proof blockers.

## Minimum Viable Shared-Library Support Requirements

If Day 5 selects shared-library implementation, the selected scope must satisfy
all hard gates below before merge.

| Gate | Requirement | Required Proof |
| --- | --- | --- |
| Scope gate | The supported platform set is explicit: Linux-only, POSIX-only, or Linux/macOS/Windows. | Product decision record names supported and unsupported platforms. |
| Build gate | CMake builds the selected shared artifact type without weakening the static archive target. | Focused CMake configure/build proof for shared and static modes. |
| Install gate | Install rules place shared artifacts in platform-appropriate locations without losing static install proof. | Install validation checks shared artifact presence and static artifact behavior for the selected scope. |
| Export gate | Only intentional public symbols are exported. | Symbol inspection allowlist using `nm`/`readelf`/`objdump`, `nm`/`otool`, or `dumpbin`/`.def` evidence as appropriate. |
| Header gate | Public headers expose a stable export/import macro policy. | Header review plus downstream compile proof using installed headers. |
| ABI gate | Public structs, enum values, callbacks, allocator boundaries, and error/version metadata have an explicit compatibility policy. | ABI decision section in maintainer/user docs plus test-backed examples for lifecycle boundaries. |
| CMake package gate | Installed CMake metadata describes the selected shared target correctly. | `find_package(Sparse)` installed consumer builds and runs against the installed shared artifact. |
| `pkg-config` gate | Unix `sparse.pc` semantics are explicit for shared versus static linking. | Installed `pkg-config` consumer builds/runs and `--libs`/`--static` outputs match the selected contract. |
| Loader gate | Runtime loader behavior is verified. | Platform inspection plus downstream execution from the installed prefix. |
| CI gate | The selected platform proof runs in CI or is explicitly documented as local-only and non-release-blocking. | Workflow update or explicit non-claim in the decision record. |
| Documentation gate | README, INSTALL, maintainer guide, CMake/package comments, and tests agree on support scope. | Text scan and focused doc review. |

## Minimum Viable Static-First Deferral Requirements

If Day 5 selects static-first deferral, the selected scope must satisfy all
hard gates below before merge.

| Gate | Requirement | Required Proof |
| --- | --- | --- |
| Rejection gate | `BUILD_SHARED_LIBS=ON` remains rejected with diagnostic wording naming exact blockers. | Focused configure-failure proof from `scripts/static_package_deferral_check.sh` or equivalent. |
| Artifact gate | Make and CMake install proofs continue to reject `.so`, `.so.*`, `.dylib`, and `.dll` artifacts. | `tests/test_install.sh` and `tests/test_cmake_install.sh` checks remain aligned. |
| Metadata gate | `sparse.pc` and CMake package files make no shared-library, dynamic ABI, loader, package-manager, or selector claims. | Existing static package tests and static deferral guard. |
| Blocker gate | Deferral docs list exact blockers: visibility/export policy, SONAME/install-name/import-library policy, downstream shared consumer proof, loader proof, and dynamic ABI policy. | README/INSTALL/maintainer guide or sprint artifact alignment. |
| Platform gate | Linux, macOS, and Windows evidence boundaries remain explicit. | CI comments and docs preserve static-first/platform-tier wording. |
| Handoff gate | Sprint 154 receives a clear comparison handoff that does not need to rediscover shared-library blockers. | Day 14 handoff includes exact residuals. |

## Scorecard

Scores use `1` as weak/high-cost and `5` as strong/low-cost. A path with any
failed hard gate is not mergeable regardless of total score.

| Criterion | Path A: Shared Support | Path B: Static Deferral | Rationale |
| --- | ---: | ---: | --- |
| Feasibility in remaining sprint days | 2 | 5 | Shared support needs build, install, visibility, package metadata, loader proof, and docs. Static deferral builds on existing proof. |
| ABI risk | 1 | 5 | Public concrete layouts and callbacks make shared ABI risky now. Static deferral avoids claiming binary compatibility. |
| Platform coverage clarity | 2 | 5 | Shared support would likely need a tier split. Static deferral already has Linux/macOS/Windows static-proof wording. |
| Test cost | 2 | 4 | Shared support adds artifact, symbol, loader, and downstream tests. Static deferral strengthens existing tests. |
| Documentation cost | 3 | 4 | Shared support requires broad docs changes. Static deferral requires targeted blocker wording. |
| User value | 3 | 3 | Shared support helps dynamic-link consumers, but an under-proven ABI would harm trust. Static deferral improves clarity and prevents false support assumptions. |
| Release confidence | 2 | 5 | Static archive package proof is mature; dynamic loader proof is absent today. |
| Maintenance burden | 2 | 5 | Shared ABI support creates ongoing compatibility obligations. Static deferral keeps current maintenance scope bounded. |

## Recommended Day 5 Decision Bias

Day 4 does not make the final product decision, but the criteria strongly bias
toward Path B unless Day 5 deliberately narrows Path A to a platform scope
small enough to finish with complete proof.

The main reason is not that shared libraries are unimportant. The issue is
that this repository currently lacks the minimum prerequisites for a credible
dynamic ABI claim:

- no public export/import macro;
- no symbol visibility/export map;
- no SONAME, install-name, or DLL/import-library policy;
- no installed shared consumer proof;
- no runtime loader proof;
- no compatibility policy for public concrete structs and callbacks.

## Rollback Rules

These rules apply to either product path.

| Trigger | Required Action |
| --- | --- |
| Shared artifact builds but exported-symbol allowlist is incomplete. | Revert shared support and keep static-first rejection. |
| Shared artifact installs but downstream consumer links the static archive or build-tree artifact. | Revert shared support or mark it experimental and uninstalled; no support claim. |
| Linux support works but macOS or Windows is unimplemented. | Documentation must name the platform split explicitly; otherwise revert the broad shared claim. |
| Windows DLL builds without verified import/export decoration and runtime lookup. | Do not claim Windows shared support. |
| `sparse.pc` or CMake metadata implies shared/static selectors without tests for both modes. | Remove selector wording or add proof before merge. |
| Public concrete struct compatibility is undocumented. | Do not claim dynamic ABI stability. |
| Static-first deferral wording is vague or omits exact blockers. | Do not close the product decision; add blocker-specific wording first. |
| Install/package tests fail after implementation. | Stop and fix before proceeding. |
| C or header files change. | Run `make format && make lint && make test` before proceeding. |

## Day 5 Decision Checklist

Day 5 should answer these questions in the decision record:

1. Is Sprint 153 implementing shared support or strengthening static-first
   deferral?
2. If shared support is selected, which platforms are supported in this sprint?
3. If shared support is selected, what is the exact exported public symbol set?
4. If shared support is selected, how are CMake and `pkg-config` semantics
   split between static and shared artifacts?
5. If static deferral is selected, what exact blockers are named in diagnostics
   and docs?
6. Which proof commands own the selected claim?
7. What work is intentionally handed off to Sprint 154 without being claimed?
