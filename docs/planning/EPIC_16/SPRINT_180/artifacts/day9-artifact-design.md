# Sprint 180 Day 9: Artifact Design

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Design the source-controlled Homebrew local formula proof material selected by
the Day 8 product decision before adding provider prototype files. Day 9
defines the artifact shape, ownership, proof behavior, generated-output
hygiene, claim boundaries, docs references, and guard treatment for Days
10-13.

## Selected Artifact Strategy

Sprint 180 should implement a **Homebrew local formula template** rather than
a live source-controlled tap.

| Artifact | Day 9 design |
| --- | --- |
| Source-controlled template | `packaging/homebrew/sparse-lu-ortho.rb.in` |
| Proof script | `scripts/homebrew_local_formula_proof.sh` |
| Generated formula | Created under a proof temp directory only. |
| Generated tap | Created under a proof temp directory only if `brew` needs a local tap layout. |
| Source archive | Created by the proof script from the current checkout. |
| Formula source URL | Injected as a local `file://` archive URL in the temporary formula. |
| Formula SHA-256 | Computed by the proof script and injected into the temporary formula. |
| Install path | CMake source build using the existing static archive install/export surface. |
| Formula test | Build and run an installed downstream consumer; check version and static-only metadata. |

This approach keeps the selected provider material source-controlled without
pretending the repository already contains a Homebrew/core formula, a tap, a
bottle, or a distribution-ready release formula.

## Files To Add On Day 10

| Path | Type | Purpose |
| --- | --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Formula template | Defines the local Homebrew formula prototype with placeholder fields for local source URL and SHA-256. |
| optional `packaging/homebrew/README.md` | Provider notes | Explains that the template is local proof material only and not Homebrew/core, bottle, Linuxbrew, or public package-manager support. |

Day 10 should not add a root `Formula/` directory or a committed generated
formula. The current guard rejects `*/Formula/*` paths; that remains useful
because generated local tap/formula outputs must not be committed.

## Files To Add Or Update On Days 11-13

| Path | Day | Purpose |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Day 11 or 12 | Generates a temporary source archive and formula, runs local `brew` install/test/uninstall checks, and cleans generated state. |
| `scripts/package_manager_deferral_check.sh` | Day 13 | Narrows provider guard behavior so only selected Homebrew template/proof files are allowed and all other provider artifacts remain fail-closed. |
| `README.md` | Day 13 | Adds only claim-safe wording for the proven local Homebrew proof level if validation succeeds. |
| `INSTALL.md` | Day 13 | Updates package-manager support split without implying Homebrew/core, bottles, Linuxbrew, registry readiness, or broad provider support. |
| `docs/maintainer_guide.md` | Day 13 | Documents how maintainers validate and preserve the local Homebrew proof boundary. |

## Template Design

The formula template should be intentionally local-proof-specific.

| Template area | Design requirement |
| --- | --- |
| Class name | Use a local-proof formula class name, such as `SparseLuOrthoLocal`, so users do not mistake it for an official Homebrew/core formula name. |
| Description | Describe the library briefly without claiming Homebrew support or broad package-manager distribution. |
| Homepage | Use the repository homepage if available, or a placeholder that Day 10 can justify locally. |
| URL | Use a placeholder replaced by the proof script with a temporary `file://` source archive URL. |
| SHA-256 | Use a placeholder replaced by the proof script with the generated archive checksum. |
| Version | Match `VERSION` from the checkout. |
| License | Use conservative local-proof metadata. If accurate license metadata cannot be represented, Day 10 must stop and renew deferral rather than add inaccurate provider metadata. |
| Dependencies | `depends_on "cmake" => :build` only unless implementation proves more is required. |
| Install block | Configure, build, and install with CMake into `prefix`; keep optional OpenMP and mutex defaults off. |
| Test block | Build and run a downstream CMake consumer with `find_package(Sparse)` against `prefix`; check installed version behavior where feasible. |
| Static-only checks | Confirm installed `libsparse_lu_ortho.a` exists and no `.dylib`, `.so`, `.dll`, install-name, RPATH, or static/shared selector artifacts are introduced. |
| Unsupported wording | Do not mention Homebrew/core, bottles, Linuxbrew, registry readiness, binary packages, shared libraries, dynamic ABI, or broad package-manager support as supported. |

## Source Archive Design

The proof script should create a deterministic-enough local archive for proof
purposes without treating it as release evidence.

| Requirement | Design |
| --- | --- |
| Input | Current checkout, excluding `.git`, build directories, generated docs, temporary proof output, and ignored provider caches. |
| Output | Temporary archive under the proof temp directory. |
| URL | Local `file://` URL used only by the generated temporary formula. |
| Checksum | SHA-256 computed from that temporary archive and injected into the generated formula. |
| Claim boundary | The archive is proof input only, not a release tarball, registry source, or distribution artifact. |
| Cleanup | Archive and generated formula are removed by trap-based cleanup unless a debug flag intentionally preserves temp output. |

## Proof Command Success Behavior

The future proof command should succeed only when all selected local proof
requirements pass:

1. Locate `brew`, `cmake`, C compiler, and checksum tools.
2. Create isolated temp directories for archive, generated formula, build,
   Homebrew cache/prefix state where feasible, and logs.
3. Generate local source archive from the current checkout.
4. Compute SHA-256 and render the formula template into a temp formula.
5. Run syntax or style checks that are feasible for a local template.
6. Install the formula from source with `brew install --build-from-source`.
7. Confirm static archive, headers, CMake package files, and `sparse.pc` were
   installed.
8. Confirm no shared-library artifacts were installed.
9. Run `brew test` or an equivalent formula test that builds and runs a
   downstream CMake consumer.
10. Query version or installed metadata and compare with `VERSION`.
11. Uninstall the formula and check cleanup.
12. Remove temp archive, generated formula, local tap state, build trees, and
   logs unless debug preservation is explicitly requested.

## Proof Command Failure Behavior

Failure behavior must preserve package-manager non-claims.

| Failure | Required behavior |
| --- | --- |
| `brew` missing | Exit with a clear skip/fail message that local Homebrew proof is unavailable; do not imply support. |
| `cmake` or compiler missing | Fail locally with setup details; do not treat the provider path as supported. |
| source archive creation fails | Fail before formula rendering and clean temp output. |
| SHA-256 injection fails | Fail before `brew install`. |
| formula install fails | Fail and report the log path; uninstall/cleanup any partial install if possible. |
| static archive missing | Fail; do not continue to docs or support wording. |
| shared artifact present | Fail hard because the selected path is static-only. |
| downstream consumer fails | Fail; do not claim Homebrew proof. |
| version mismatch | Fail and report expected versus actual version. |
| uninstall or cleanup fails | Fail or warn explicitly, depending on whether any persistent install remains. |

## Ownership And Update Frequency

| Artifact | Owner | Update trigger |
| --- | --- | --- |
| Formula template | Package-manager proof owner for Sprint 180 | Version changes, install layout changes, CMake package target changes, dependency changes, or Homebrew formula policy changes. |
| Proof script | Package-manager proof owner for Sprint 180 | Formula template changes, guard changes, Homebrew CLI behavior changes, or proof failure diagnostics. |
| Package metadata | Existing package owners | Only changes if CMake install/export or `sparse.pc` behavior changes; the Homebrew proof must consume metadata, not redefine provider-neutral metadata. |
| Docs wording | Maintainer docs owner | Updated only after proof and guard pass. |
| Guard | Package-manager guard owner | Updated when selected provider files are added and when unsupported provider patterns need new fail-closed coverage. |

## Relationship To Package Metadata

The Homebrew formula template should consume existing package metadata rather
than introducing new package identity:

| Metadata | Source |
| --- | --- |
| Version | `VERSION` |
| CMake package target | Installed `Sparse::sparse_lu_ortho` |
| Static archive | Installed `libsparse_lu_ortho.a` |
| Headers | Installed `include/sparse/*.h` |
| pkg-config metadata | Installed `lib/pkgconfig/sparse.pc` |
| CMake package files | Installed `lib/cmake/Sparse/*.cmake` |

The template must not edit `sparse.pc.in` or `cmake/SparseConfig.cmake.in` to
add Homebrew wording. Those files remain provider-neutral.

## Guard Design Checklist

Day 13 should update `scripts/package_manager_deferral_check.sh` to distinguish
selected local proof artifacts from unsupported provider recipes:

| Check | Required behavior |
| --- | --- |
| Formula template allowance | Allow only `packaging/homebrew/sparse-lu-ortho.rb.in` and optional `packaging/homebrew/README.md`. |
| Proof script allowance | Allow only `scripts/homebrew_local_formula_proof.sh` as the selected provider proof command. |
| Live formula rejection | Continue rejecting committed `*/Formula/*` paths outside ignored/temp areas. |
| Other provider rejection | Continue rejecting vcpkg, Conan, pkgsrc, Debian, Fedora, RPM, and unselected provider artifacts. |
| Metadata neutrality | Continue rejecting provider wording in `sparse.pc.in` and `cmake/SparseConfig.cmake.in`. |
| Public docs | Permit only wording that names local Homebrew proof level and unsupported boundaries after proof passes. |
| Generated output | Reject or ignore generated archives, local taps, formula render outputs, logs, bottle outputs, and Homebrew cache artifacts. |

## Docs Reference Checklist

Day 13 public docs should reference the selected proof only after the formula
template and proof script pass:

| File | Required wording |
| --- | --- |
| `README.md` | Mention only local Homebrew formula proof if validated; keep source install as the primary support path. |
| `INSTALL.md` | Replace blanket Homebrew unsupported wording with exact local-proof status only after proof passes; keep core/bottle/Linuxbrew/broad package-manager claims unsupported. |
| `docs/maintainer_guide.md` | Document proof command, guard command, generated-output hygiene, and claim boundary. |
| `sparse.pc.in` | No Homebrew wording. |
| `cmake/SparseConfig.cmake.in` | No Homebrew wording. |
| workflows | No Homebrew provider execution unless a later sprint or day adds CI proof with claim-safe scope. |

## Stop Conditions

Day 10 should renew formal deferral instead of adding the Homebrew template if:

- license metadata cannot be represented accurately even for a local formula
  template;
- the formula source URL and SHA-256 placeholders would be too ambiguous to
  validate safely;
- a formula template cannot be separated from a live Homebrew/core or tap
  claim;
- CMake source build cannot preserve the static-only install surface;
- the guard cannot be narrowed without allowing unrelated provider artifacts.

## Day 9 Decision

Proceed with a Homebrew local formula-template design for Day 10. The template
is the source-controlled provider artifact; the actual local formula, local
tap, source archive, and SHA-256 are generated by the proof script in a
temporary directory and must not be committed.

This design preserves the Sprint 171 public deferral posture until later
implementation and validation prove the selected local Homebrew path.

## Day 9 Deliverables

- provider prototype artifact design
- expected success and failure behavior
- artifact ownership notes
- docs and guard reference checklist
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day9-artifact-design.md`

## Validation

Day 9 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider formula template, proof script, or public
user-facing docs were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Source-controlled artifact work is designed before implementation. | Complete | Files-to-add and selected artifact strategy sections above. |
| Claim boundaries are explicit for both success and failure paths. | Complete | Support, failure, docs, guard, and stop-condition sections above. |
| Proof-script requirements are ready for implementation. | Complete | Proof command success and failure behavior sections above. |
