# Sprint 214 Release Readiness Checklist

Status: **checklist design complete; release evidence incomplete**. No release
is authorized, tagged, archived, or published. Days 11-12 define the gates;
Day 13 local static validation passed, but a later candidate-specific review
must supply release results.
The selected product path remains a static archive with shared-library,
dynamic ABI, package-provider, and release-readiness claims deferred.

## Candidate Identity And Source Gates

| Gate | Owner | Required command or inspection | Pass condition | Stop condition | Current state |
| --- | --- | --- | --- | --- | --- |
| Changelog entry | Release maintainer and API reviewer | Review a proposed `CHANGELOG.md` entry against merged commits, public API/package changes, and retained non-claims. | Version-specific changes, migrations, compatibility limits, and known issues are accurate and approved. | Missing changelog, unsupported shared/ABI/provider claim, or undocumented breaking change. | Blocked: no `CHANGELOG.md` exists. |
| Version coherence | Release maintainer | Inspect `VERSION`, Make/CMake-generated `sparse_version.h`, installed `sparse.pc`, and `SparseConfigVersion.cmake`; run the installed Make/CMake consumers for the candidate commit. | All identify one approved package version; CMake exact-version behavior matches it. | Any version drift or treating package `VERSION` as a dynamic ABI epoch. | `VERSION=2.2.0`; candidate not selected or tested as a release. |
| Candidate commit and clean tree | Release maintainer | `git status --porcelain`, `git rev-parse HEAD`, `git show --stat --oneline <commit>`, and review the candidate diff. | One reviewed, committed source snapshot is frozen with no uncommitted or untracked release input. | Dirty checkout, unresolved review, or a tag pointing at a different commit. | Blocked: `sprint-214` has uncommitted work. |
| Tag policy | Release maintainer | Propose an annotated `v<VERSION>` tag on the reviewed commit; inspect `git cat-file -t <tag>`, `git rev-parse <tag>^{commit}`, and the tag message/signature policy. | Tag name equals approved `VERSION`, resolves to the frozen commit, and is immutable after approval. Signing requirement and signing identity are decided before tag creation. | Lightweight, mismatched, moved, or unreviewed tag; missing signing decision. | Blocked: no existing tags or approved tag policy; do not create a tag during this sprint. |
| Source manifest | Build/release owner | Review `git ls-tree -r --name-only <commit>` and a candidate `tar -tzf` manifest; compare required inputs and forbidden paths below. | Archive has approved source, build recipes, license, checked-in docs, tests, and tooling, with no local output. | Missing build/test/legal input, unexpected generated output, path traversal, or unreviewed content. | Draft policy only; no candidate manifest approved. |
| Reproducible source bytes | Build/release owner and independent verifier | Produce archives twice from independent clean checkouts at the same commit with pinned Git/gzip versions; compare sorted listings and SHA-256 byte hashes; extract and build/test one copy. | Identical archive bytes and manifests in the declared environment, with successful extraction and source build. | Different bytes or contents, omitted required input, failed extraction/build, or missing environment record. | Not run; reproducibility unclaimed. |
| License and review signoff | Release maintainer, legal/license reviewer, CI/platform owner | Inspect `LICENSE` and candidate archive, plus signed-off checklist, release notes, and Day 12 CI/provenance evidence. | Named approvers accept exactly the static-first artifact and its residual non-claims. | Missing license, unassigned approver, or approval before evidence exists. | Pending; `LICENSE` is tracked, but no release approval exists. |

Candidate source contents: the tagged tracked tree must include `LICENSE`,
`README.md`, `INSTALL.md`, `VERSION`, `CMakeLists.txt`, `Makefile`,
`sparse.pc.in`, `Doxyfile`, `.clang-format`, `.clang-tidy`, `cmake/`, `include/`,
`src/`, `examples/`, `tests/` including checked-in fixtures, `scripts/`,
`benchmarks/`, and checked-in documentation such as `docs/api_reference.md`.
The exact **full** tracked-file manifest needs human review before release;
this list is a minimum, not a silent approval of every tracked path. A
`git archive` candidate includes all tracked files unless an approved
export-ignore policy changes that set.

Forbidden local/generated content includes `.git/`, `build/`,
`cmake-build/`, `CMakeFiles/`, `CMakeCache.txt`, object/static/shared outputs,
coverage data, Python caches, local reports, generated
`include/sparse_version.h`, and generated `docs/api/html/`. The version
header must be regenerated from `VERSION` during build. Checked-in source
fixtures and Markdown are different from ignored local generated output.
Reject an archive that accidentally contains local-only Doxygen HTML even if
it is otherwise reproducible.

## Candidate Archive Procedure (Not Executed)

The proposed source asset name is
`sparse-lu-ortho-<VERSION>-src.tar.gz`. A future release procedure should
resolve the approved `v<VERSION>` tag to its commit, record that commit and
tool versions, and run the same command in two independent clean checkouts:

```sh
version=$(cat VERSION)
tag="v${version}"
commit=$(git rev-parse "${tag}^{commit}")
git archive --format=tar --prefix="sparse-lu-ortho-${version}/" "$commit" |
  gzip -n > "sparse-lu-ortho-${version}-src.tar.gz"
```

This is a **candidate method**, not a demonstrated reproducible archive.
Use the same pinned Git and gzip versions and record OS, architecture,
locale, timezone, compression flags, commit timestamp, and `SOURCE_DATE_EPOCH`
policy. Compare `tar -tzf` listings, SHA-256 hashes of both compressed files,
and extracted trees. Build/test from the extracted tree, not the original
checkout. If byte hashes differ, diagnose tool metadata, file ordering,
timestamps, or manifest drift; do not call the artifact reproducible. Cross-OS
reproducibility requires its own compared runs and is not inferred from two
same-host checkouts.

## Artifact And Publication Boundary

| Surface | Owner | Proposed name and retention after approval | Required check | Stop condition / current policy |
| --- | --- | --- | --- | --- |
| Source archive | Build/release owner | `sparse-lu-ortho-<VERSION>-src.tar.gz`; retain immutable bytes, manifest, checksum, tag/commit, and build-environment record with an approved release record. | Two-build byte comparison, manifest/forbidden-path audit, extraction and static build/test. | Stop on any mismatch or missing proof. Candidate only; no archive generated or published by Day 12. |
| Binary package | CI/platform and release owners | No approved name or release retention. A future name must include version, OS, architecture, static/shared link mode, ABI/index/scalar variant, and toolchain/CRT where relevant. | Named platform build, installed consumer, provenance, dependency and ABI checks before any binary asset. | Stop if any named platform or ABI check is absent. Deferred; CI static `.a`/`.lib` outputs are not release packages. |
| Checked-in documentation | Documentation and release owners | `README.md`, `INSTALL.md`, `docs/api_reference.md`, and other approved source docs travel only inside a reviewed source snapshot for now. | Links, claim guard, and source manifest audit. | Stop on unsupported claims or unreviewed standalone upload. No standalone documentation release asset selected. |
| Generated API HTML | Documentation and CI owners | No release asset name or retention. `docs/api/html/` stays local-only and ignored. | `make api-docs-freshness` is for local inspection; ensure source/release manifest and CI uploads exclude HTML. | Stop on any committed/uploaded HTML; a separate policy decision is needed before publication. |
| Local benchmark/report/coverage output | CI/platform and release owners | No release asset name or retention by this checklist. | Exclude `build/`, coverage, and other generated paths from source/binary/docs publication. | Stop if output enters a release payload. Existing selected CI report artifacts have their own bounded evidence policy. |

The developer-mode Homebrew proof's `tar -czf` source archive and checksum
are **not** a release archive or reproducibility result. Its source list is
narrower than this candidate tracked-source snapshot, and its local formula
does not confer package-provider distribution support.

## Provenance And Package Origin Gates

An approved source candidate must have a single traceable chain: reviewed
commit -> annotated tag -> reproducible source archive -> archive checksum ->
clean extracted build -> installed static package -> installed consumer result.
The source checkout and a separately authorized source archive are the only
candidate origins. A local Homebrew formula, CI build artifact, public tap,
bottle, binary package, or package-manager registry is **not** an approved
origin or distribution channel. The optional binary gate below does not
authorize a binary release.

| Input | Owner | Required inspection and retained record | Pass / stop condition | State |
| --- | --- | --- | --- | --- |
| Source identity | Release maintainer | Record tag object, peeled commit, reviewed diff, full tracked-source manifest, and archive name. | All identities resolve to the same approved source; stop on an unreviewed, moved, or mismatched tag. | No candidate or tag. |
| Source checksum | Build/release owner and independent verifier | Record SHA-256 of both independently built compressed archives (for example `sha256sum` on Linux, `shasum -a 256` on macOS, `Get-FileHash -Algorithm SHA256` on Windows), manifests, tool versions, and environment. | Both archives and manifests agree byte-for-byte under the declared environment; stop on any mismatch. | No archives or checksums. |
| Build provenance | Platform build owner | From an extracted candidate archive, record build host/runner, OS/architecture, compiler/linker and versions, CMake/Make versions, flags, index/scalar settings, dependency versions, build command, and source digest. | Inputs reproduce the named static build; stop if any binary cannot be traced to the archived source and recorded inputs. | No release build record. |
| Static package contents | Platform build owner | Record installed-file manifest and SHA-256 hashes of archive, headers, `sparse.pc` where applicable, and CMake package files; inspect symbols, link dependencies, and absence of shared/import artifacts. | Installed files match the static-only manifest and a separate installed consumer links/runs; stop on missing, extra, or mismatched files. | Day 9-10 local consumers passed, but not from a release archive. |
| Optional binary provenance | Product/release and platform owners | Before considering a binary asset, define exact platform/toolchain/CRT, license/dependency inventory, build inputs, installed consumer, artifact hashes, retention, and signing/attestation policy. | Requires a separate explicit product decision and evidence for each binary; stop if any input or policy is absent. | Deferred; no binary asset selected. |
| Documentation provenance | Documentation owner | Audit checked-in docs against the tagged source manifest, public claim guards, and CI/release upload manifests. | Checked-in source docs match the approved snapshot and generated `docs/api/html/` is absent; stop on an unreviewed claim or generated HTML upload. | Local-only HTML policy retained; no release docs asset. |

Do not infer source-to-binary equivalence from an archive checksum alone.
Retain commands, machine-readable manifests, hashes, tool versions, logs, and
approver identities together with any approved release record. No signed
attestation or SBOM is claimed by this checklist; decide those requirements
before choosing a binary publication path.

## Candidate CI Evidence Matrix

Each required job must be tied to the **same candidate commit** with a run URL,
conclusion, runner/toolchain identity, command log, and artifact digest where
applicable. A workflow definition or prior local pass is not a hosted result.
If a platform is excluded from a future release, the public claim and asset
name must exclude it explicitly; do not silently treat a skipped job as a pass.

| Evidence | Owner and exact check | Acceptance / stop condition | Current state |
| --- | --- | --- | --- |
| Linux static install and consumer | CI owner: `.github/workflows/ci.yml` `package-contract`; `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, `python3 tests/test_static_install_contract.py`, `bash scripts/static_package_deferral_check.sh`, and `make package-manager-deferral-guard`. | All pass on candidate; installed Make/`pkg-config` and CMake consumers link/run, negative fixtures reject unsupported metadata/selectors. Stop on failure or absent run. | Registered; no Day 8 branch hosted result recorded here. |
| macOS static install and consumer | CI owner: `.github/workflows/macos-ci.yml` `install-and-pkgconfig` and `cmake-install-export`; same install scripts, CMake fixture test, and static deferral guard. | Both jobs pass on candidate; Make/`pkg-config` and CMake routes are separately evidenced. Stop on failure or absent run. | Registered; Day 9-10 local macOS proof is not hosted candidate proof. |
| Windows static install and consumer | CI owner: `.github/workflows/windows-ci.yml` `install-and-downstream`; `python tests/test_static_install_contract.py` and PowerShell CMake install, validator, and downstream consumer. | Job passes on candidate with static `.lib`, package metadata, and installed consumer. Stop on failure or absent run; Windows Make/`pkg-config` parity is not inferred. | Registered; no hosted candidate result recorded here. |
| Shared/ABI deferral | Product and CI owners: selected static guard in Linux/macOS jobs, Windows installed-prefix checks, `BUILD_SHARED_LIBS=ON` rejection, and no shared/import/loader metadata. | All unsupported selectors/artifacts/claims remain rejected. A shared proof requires a separate decision, export allowlist, platform loader and installed dynamic consumer tests; stop on accidental promotion. | Static-only path selected; no shared proof. |
| Documentation and provider claims | Documentation/CI owners: `bash scripts/static_package_deferral_check.sh`, `bash scripts/check_api_docs_local_only.sh`, relevant `make docs-check`, and the package-provider guard when claims or metadata change. | Claim guards and applicable docs validation pass for the candidate; generated HTML is not retained or published. Stop on unsupported wording or upload. | Day 13 local API-docs, support-docs, and provider guards passed; no candidate hosted result. |
| C and build quality | Build/CI owners: `make format && make lint && make test` when `*.c`/`*.h` changes; `make quality-review-compile` and `make quality-review-cmake` on reviewed Linux/macOS build lanes, plus Windows reviewed CMake/CTest subset. | Applicable checks pass on candidate with logs and no unresolved warning/error. Stop on failed or skipped required gate. | Day 13 local reviewed compile and CMake quality passed, including 60/60 CTest. No C/header files changed; no candidate hosted result. |

The Windows job proves only its named CMake/MSVC static subset. Linux/macOS
quality jobs are not substitutes for Windows evidence, nor does a static
consumer prove loader, dynamic ABI, package-provider, or binary-release
readiness.

## Approval, Blockers And Rollback

The release maintainer owns a go/no-go record that cites every row above and
the exact public wording. API/ABI, documentation, CI/platform, build/release,
license/legal, and product owners must each approve their part manually after
reviewing candidate-specific evidence. The product owner may authorize a
**static source-only** release without promoting Sprint 170 shared/ABI or
Sprint 207 provider deferrals; any shared, dynamic ABI, provider, or binary
claim instead needs a separate superseding decision and its own proof. No
approval is inferred from a green local test or an existing CI definition.

Current hard blockers are the absent changelog, unfrozen/dirty candidate,
unapproved tag/signing policy, absent tag and archive, unreviewed source
manifest, unrun two-build byte comparison, absent source-to-binary record,
missing candidate-specific hosted results, and missing manual approvals. A
failed required check, generated HTML publication, unsupported claim, missing
license, or provenance mismatch also blocks release. Mark each item resolved
only with retained evidence, not a prose assertion.

Before any publication, the release maintainer must record how to halt an
in-progress upload, disable downloads/links and notify consumers if a bad
asset escapes, and issue a corrected new version/tag rather than moving an
immutable tag or silently replacing bytes. Preserve the original digest and
incident record; mirrors may retain already downloaded bytes, so withdrawal
cannot be promised as erasure. If a release is cancelled before publication,
record the no-go decision and leave the candidate unpublished. Rollback owner,
communications channel, and correction procedure need explicit approval for
the actual release destination. No rollback action has been exercised here.
