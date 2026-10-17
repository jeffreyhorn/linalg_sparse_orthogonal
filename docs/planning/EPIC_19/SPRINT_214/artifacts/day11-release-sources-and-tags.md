# Sprint 214 Day 11: Release Sources And Tags

## Scope And Decision

Item 214.5 is a **future release-readiness checklist**, not a release. The
selected Sprint 214 path remains static-first with shared-library, dynamic
ABI, package-provider, and release-readiness support deferred. Day 11 creates
`RELEASE_READINESS_CHECKLIST.md` as the living gate; Day 12 must add
provenance, CI evidence, approval, and rollback detail. No tag, source archive,
binary asset, or documentation asset was created or published.

## Repository Evidence Intake

| Surface inspected | Observation | Gate consequence |
| --- | --- | --- |
| `VERSION`, Make, CMake | `VERSION` is `2.2.0`; Make and CMake derive generated version metadata from it. CMake package compatibility is `ExactVersion`. | Version fields in source, installed headers, `sparse.pc`, and CMake config must agree for a release candidate. Package version is not a dynamic ABI epoch. |
| Changelog and tags | No `CHANGELOG.md` or other root changelog is tracked; `git tag --list` returned no tags. | Require a reviewed version entry and an approved `v<VERSION>` annotated-tag policy before any candidate is frozen. Do not mint a tag just to satisfy the checklist. |
| Source state | The `sprint-214` worktree has uncommitted tracked and untracked work. | A release source snapshot cannot be identified by this worktree; require one reviewed commit and a clean checkout. |
| Ignored generated files | `.gitignore` excludes `build/`, `docs/api/`, and generated `include/sparse_version.h`. `git check-ignore -v` confirmed representative paths; `git ls-files` showed none of those outputs tracked. | Tracked-source archive is a plausible candidate, but inspect its entire manifest; ignore rules alone are not release-content approval. |
| Local Homebrew source archive | `scripts/homebrew_local_formula_proof.sh` uses a curated `tar -czf` list, checks required entries, and computes one SHA-256 for local formula testing. | This is developer-mode provider proof, not a versioned release archive, two-build byte comparison, or public package-provider approval. Its file list omits several candidate release inputs. |
| Generated API documentation | `docs/api/html/` remains local-only under the Sprint 213 policy and is not tracked. Checked-in `docs/api_reference.md` is a source document. | Neither the candidate source asset nor a release/CI docs asset may silently retain generated HTML. |

These observations do not resolve legal approval, changelog content, tag
signing, archive manifest selection, platform binaries, or provenance. They
are blockers/inputs for the later release decision, not evidence of readiness.

## Candidate Source Archive Contract

The draft checklist chooses a **tracked-source snapshot at one approved
commit** as the conservative candidate. The minimum required content is the
license, public/install docs, `VERSION`, Make/CMake/pkg-config source, headers
and implementation, examples/tests/fixtures, scripts, benchmarks, Doxyfile,
and formatting/lint configuration. The full `git ls-tree` and resulting tar
manifest must be reviewed; `git archive` otherwise includes all tracked
files, including historical planning and developer-only material. There is no
approved `.gitattributes export-ignore` policy today, and Day 11 does not add
one without reviewing its effect on downstream source builds.

Exclude local/generated build directories, compiled outputs, caches, coverage,
generated `include/sparse_version.h`, and `docs/api/html/`. The generated
version header must be reconstructed from `VERSION` after extraction. A
candidate archive is rejected if a required source or legal input is absent,
if local-only output is present, or if an extracted clean build cannot run
the maintained static tests.

### Reproducibility Acceptance Procedure

1. Approve one commit and a non-moved annotated `v<VERSION>` tag whose value
   matches `VERSION`; record tag object and peeled commit hashes. Settle the
   signing identity/policy before creating the tag. Freeze the manifest.
2. In two independent clean checkouts of that commit, with pinned Git/gzip
   versions, run `git archive --format=tar --prefix=<name>/ <commit> | gzip -n`
   to the candidate `sparse-lu-ortho-<VERSION>-src.tar.gz`. Record OS, arch,
   Git/gzip versions, locale, timezone, compression flags, commit timestamp,
   and the chosen `SOURCE_DATE_EPOCH` policy. `git archive` from a commit uses
   commit metadata; gzip `-n` omits original name/time in its header.
3. Compare SHA-256 of the compressed files and sorted `tar -tzf` manifests.
   Reject duplicate/unexpected paths, path traversal, missing required files,
   or forbidden generated files. Extract one copy to an empty directory and
   run the static Make/CMake build and installed consumer tests there.
4. If hashes or manifests differ, stop and diagnose tooling, timestamps,
   ordering, or content. A same-host pass would establish only that declared
   environment; cross-OS reproducibility needs its own compared runs.

No pair of archives was built or compared on Day 11, so source
reproducibility is **unverified**. A one-off SHA-256 or `tar -tzf` check is
insufficient for that claim.

## Tag, Artifact, And Publication Policy

The proposed source name is `sparse-lu-ortho-<VERSION>-src.tar.gz`, retained
only after approval with immutable bytes, checksum, manifest, commit/tag, and
environment record. Checked-in docs are reviewed within that source snapshot.
No binary-release name or retention is approved because the current CI
archives are build/test outputs, not provenance-backed binary packages.
Future binary names must distinguish platform, architecture, static/shared
mode, ABI/index/scalar variant, and relevant toolchain/CRT. No standalone
documentation asset is selected; generated Doxygen HTML has no release name,
upload, or retention under the local-only policy. Existing bounded CI report
artifacts remain evidence under their own policies, not release payloads.

The proposed tag prefix `v` is a **policy draft**, not a historical project
convention inferred from existing tags. The release maintainer must secure
changelog, legal/license, platform/CI, and public-claim signoff before
tagging; a lightweight, mismatched, rewritten, or unreviewed tag blocks the
release. The Day 12 checklist will bind checksums, source-to-binary
traceability, CI evidence, provider claims, and rollback to this source
identity.

## Validation And Handoff

Day 11 performed repository inspection only: `git tag --list`, `git
ls-files`, `git check-ignore -v`, `VERSION` and build-rule inspection, and
review of the local Homebrew archive recipe and generated API policy. No
source archive or release command was executed. Only planning Markdown was
added or updated, so no C/header format-lint-test gate was required. Final
whitespace validation is recorded in `WORKING_NOTES.md`.

Day 12 must complete the checklist's provenance, CI, approval, and rollback
slots, and keep every current blocker visible. It must not mark the checklist
as passed merely because the static install/consumer tests pass locally.
