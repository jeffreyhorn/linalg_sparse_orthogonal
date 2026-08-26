# Sprint 180 Day 11: Proof Script Design

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Design the local proof script for the Homebrew local formula template added on
Day 10. Day 11 specifies command flow, temporary directories, cleanup,
logging, failure messages, local-versus-CI behavior, and claim-safe boundaries
before Day 12 implements the script.

## External Homebrew References

Official Homebrew documentation consulted on 2026-08-25:

| Topic | Source |
| --- | --- |
| Formula Cookbook | <https://docs.brew.sh/Formula-Cookbook> |
| Formula Ruby API | <https://docs.brew.sh/rubydoc/Formula.html> |
| Adding software | <https://docs.brew.sh/Adding-Software-to-Homebrew> |
| Tap creation and local testing | <https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap> |
| Homebrew manpage | <https://docs.brew.sh/Manpage> |
| Taps | <https://docs.brew.sh/Taps> |

Relevant Day 11 takeaways:

- Homebrew formulae are Ruby classes whose class and file names must
  correspond.
- Formulae should have a meaningful `test do` block.
- Local tap formulae can be tested with `brew install --build-from-source`.
- Taps are formula repositories and can execute local code, so the proof must
  keep generated tap material temporary and explicit.
- Homebrew/core, bottle, and Linuxbrew readiness are separate evidence
  questions and are outside the selected Sprint 180 proof.

## Script Contract

Day 12 should add:

```sh
scripts/homebrew_local_formula_proof.sh
```

The script proves only the selected Day 8 scope:

- local Homebrew formula proof;
- source build from a temporary local archive;
- static archive package surface;
- downstream compile/link/run through the provider-installed package;
- version and installed-file checks;
- uninstall and cleanup;
- claim-safe failure when local prerequisites are unavailable.

It must not claim:

- Homebrew/core acceptance;
- bottles or hosted binaries;
- Linuxbrew support;
- general Homebrew support beyond the local proof;
- vcpkg, Conan, pkgsrc, Debian, Fedora, RPM, or system package support;
- shared-library support, dynamic ABI support, runtime-loader behavior, or
  static/shared selectors.

## Command Flow

The script should run these stages in order:

| Stage | Behavior |
| --- | --- |
| 1. Parse options | Support default cleanup and an explicit debug-preserve flag such as `--keep-temp`. |
| 2. Locate repository | Resolve `ROOT_DIR` from the script path; fail if template or `VERSION` is missing. |
| 3. Check tools | Require `brew`, `cmake`, Ruby, C compiler, `tar`, and a SHA-256 command such as `shasum -a 256` or `sha256sum`. |
| 4. Check template | Verify `packaging/homebrew/sparse-lu-ortho.rb.in` exists, Ruby syntax passes, and all required placeholders are present. |
| 5. Prepare temp root | Create one temp root with subdirectories for archive input, rendered formula, local tap/formula path, logs, and optional cache state. |
| 6. Create source archive | Archive the current checkout into the temp root while excluding `.git`, build directories, generated docs, temporary proof output, generated formula/tap output, caches, logs, and ignored package outputs. |
| 7. Compute checksum | Compute SHA-256 of the temporary archive. |
| 8. Render formula | Replace URL, SHA-256, version, homepage, and license placeholders in the template and write a temporary formula file with a class/file-name match. |
| 9. Formula syntax check | Run Ruby syntax check on the rendered formula; optionally run `brew audit --formula --strict` if it is feasible and can be interpreted as local style evidence only. |
| 10. Install from source | Run local `brew install --build-from-source` against the rendered local formula or local tap path. |
| 11. Static install checks | Check `libsparse_lu_ortho.a`, headers, CMake package files, and `sparse.pc`; fail if `.dylib`, `.so`, `.dll`, or shared-selector artifacts appear. |
| 12. Downstream test | Run `brew test` for the installed local formula so the formula `test do` compiles, links, and runs the installed CMake consumer. |
| 13. Version checks | Verify the injected version matches `VERSION`; rely on the formula `find_package(... EXACT ...)` test and add metadata checks where feasible. |
| 14. Uninstall | Run `brew uninstall --force` for the local formula name if installed. |
| 15. Cleanup | Remove temp archive, rendered formula, tap state, logs, caches, and build trees unless `--keep-temp` is set. |
| 16. Report | Print a concise pass/fail summary that names the proof level as local-only. |

## Temporary Directory And Generated Output Rules

| Output | Location | Cleanup rule |
| --- | --- | --- |
| source archive | proof temp root | Remove on exit unless `--keep-temp`. |
| rendered formula | proof temp root or temp tap `Formula/` path | Remove on exit unless `--keep-temp`. |
| local tap metadata | proof temp root only | Remove on exit unless `--keep-temp`. |
| Homebrew logs | proof temp root where controllable; otherwise referenced by path | Remove temp copies unless `--keep-temp`; do not commit. |
| build directories | Homebrew-managed temp/build paths and proof temp root | Uninstall and clean temp-owned paths. |
| installed formula | Homebrew cellar/prefix | Always attempt uninstall on exit if install stage started. |
| bottles/caches | Must not be generated intentionally | If created, report as unexpected output and clean if possible. |

The script must never write generated provider output into source-controlled
paths except for explicitly added future script files.

## Local Proof Behavior

The local proof is the authoritative Day 12 behavior. It may run on a macOS
developer host with Homebrew available.

Expected local success evidence:

- `brew --version` reports a local Homebrew installation;
- the source archive is generated from the current checkout;
- the formula is rendered with local `file://` URL and SHA-256;
- the formula installs from source;
- the installed package contains `libsparse_lu_ortho.a`;
- installed CMake package files and `sparse.pc` exist;
- no shared-library artifacts are installed;
- `brew test` builds and runs the downstream CMake consumer;
- version behavior matches `VERSION`;
- uninstall and cleanup complete.

Expected local unavailable behavior:

- if `brew` is missing, the script exits with a clear message that local
  Homebrew proof is unavailable on this host;
- missing `brew` does not count as Homebrew support, Homebrew failure, or
  public package-manager evidence.

## CI Behavior

Day 11 does not require adding CI execution.

If CI is added later, it must be a separate explicit decision or Day 13/14
scope change with these boundaries:

- macOS-only unless Linuxbrew is separately selected and proven;
- no bottle generation;
- no Homebrew/core submission or tap push;
- cache behavior documented and bounded;
- missing Homebrew or unsupported runner state fails or skips claim-safely;
- docs must distinguish hosted proof from public package-manager support.

## Failure Messages

Failure messages should be precise and claim-safe.

| Failure | Message requirement |
| --- | --- |
| missing `brew` | State that local Homebrew proof cannot run on this host; do not say Homebrew support failed. |
| missing `cmake` or compiler | State local proof prerequisites are missing. |
| missing template placeholders | State the source-controlled template is stale or incomplete. |
| missing license metadata | State formula rendering is blocked because accurate local-proof license metadata is unavailable. |
| archive creation failure | State local source archive proof input could not be created. |
| checksum failure | State local source archive checksum could not be computed. |
| formula render failure | State generated formula was not produced; do not invoke `brew`. |
| `brew install` failure | State local formula install proof failed and point to logs. |
| shared artifact found | State static-only proof failed because unsupported shared artifacts appeared. |
| downstream test failure | State installed local formula consumer proof failed. |
| version mismatch | State expected and actual versions. |
| uninstall failure | State cleanup may have left local Homebrew state and name the formula. |

## License Metadata Handling

The repository currently has no standalone `LICENSE`, `COPYING`, or `NOTICE`
file. Day 12 must not invent provider license metadata.

Acceptable Day 12 behavior:

- render license only if accurate local-proof metadata is already available;
- otherwise fail before `brew install` with a clear stop-condition message;
- document that Homebrew proof remains blocked by license metadata rather than
  widening claims or using inaccurate placeholders.

If no license metadata exists, the proof script may still implement
pre-render checks and stop safely. That is an acceptable claim-safe failure
path, not a support claim.

## Static-Only Checks

The proof script should duplicate the essential static-first checks outside
Homebrew where possible:

| Check | Required behavior |
| --- | --- |
| archive | `lib/libsparse_lu_ortho.a` exists after install. |
| headers | installed `include/sparse/*.h` exists. |
| CMake config | installed `lib/cmake/Sparse/SparseConfig.cmake` exists. |
| CMake target | installed target metadata contains `Sparse::sparse_lu_ortho` as static where feasible. |
| pkg-config | installed `lib/pkgconfig/sparse.pc` exists and has no provider or shared-library wording. |
| shared artifacts | fail on `.dylib`, `.so`, `.so.*`, `.dll`, import-library, or shared-selector evidence. |

## Day 12 Implementation Handoff

Day 12 should implement the script in small stages:

1. Add option parsing, tool checks, temp-root setup, and cleanup trap.
2. Add template placeholder and Ruby syntax checks.
3. Add source archive and SHA-256 generation.
4. Add formula rendering.
5. Add license-metadata stop condition.
6. Add `brew install`, static installed-file checks, `brew test`, uninstall,
   and cleanup if license metadata allows rendering.
7. Run the proof command and record whether it passes, stops safely on license,
   or stops safely on missing local tools.

## Day 11 Deliverables

- proof-script command-flow design
- cleanup and temporary-directory rules
- local-versus-CI proof split
- failure-message and non-claim requirements
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day11-proof-script-design.md`

## Validation

Day 11 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider script, or public user-facing docs were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Proof-script behavior is specified before implementation. | Complete | Command flow and Day 12 handoff sections above. |
| Cleanup and failure behavior are explicit. | Complete | Temporary directory, generated output, and failure message sections above. |
| Local proof does not depend on unavailable provider infrastructure. | Complete | Local proof uses only local Homebrew and temporary formula/archive inputs; CI and registry paths remain out of scope. |
