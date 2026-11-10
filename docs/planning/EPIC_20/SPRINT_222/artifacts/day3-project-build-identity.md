# Sprint 222 Day 3: Project Build Identity

## Outcome and boundary

The selected project's 32-bit-index static library and pilot helper compiled successfully on 2026-11-10. **No solver was executed.** The retained helper is a build artifact that contains the project-side entry point; it also declares a CHOLMOD dynamic link, whose actual loaded identity remains Day 4 work. This build does not establish an observed comparison or a v1 `correctness_observed` record.

The complete raw evidence is in the ignored local directory:

`build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day3-project-build-20261110T204153Z-hfgirZ/`

That directory was created uniquely and has not been reused. Its `project-build-manifest.json` has SHA-256 `fee0e326d854bdb1aca7fe90e55c415dd981fb95ea39eaff326cb29ec9e0f741`. It names the actual compile and archive commands, 49 source compilation units, source snapshot hashes, compiler, SDK, flags, expected controls, and byte-backed outputs. Keep the directory local per the Day 2 rights ledger.

## Source snapshot and reproducibility

| Field | Captured value / proof |
| --- | --- |
| Git HEAD | `dab7943c218e997cda6898c9cdd8d3572a6029c9`, branch `sprint-222`. |
| Prebuild worktree | Dirty only because four untracked Sprint 222 Markdown files existed: `PLAN.md`, `WORKING_NOTES.md`, Day 1 inventory, Day 2 rights ledger. No tracked change; `tracked.patch` is zero bytes with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. |
| Exact status and untracked bytes | `git-status.z` SHA-256 `3e6cc13b7b9a57b970af4f2f2a515d6725d54808b18e72aeab7170ab57e897f6`; `untracked-input-hashes.txt` SHA-256 `860c3324049ccdb18e5d635d1bf4505b4ae14caff0dcc9b588bfa196b4625635`. The four byte copies are under `prebuild-untracked/`. A reconstructible concatenated `git diff --no-index` patch, `prebuild-untracked.patch`, has SHA-256 `93af6201ca55cf38fc00f43f02f699d42d4355e9e4027b21c996d78971698d21`; `git apply --check` succeeds in an empty directory. |
| Tracked build inputs | `tracked-build-inputs.json` lists 93 tracked Makefile, version, source, header and helper inputs with byte hashes; SHA-256 `ee35d3e84e5b4203c9f41b2be305e6b1d7fdf024aad71ac282d0d0d8389e962d`. All compiler inputs came from this HEAD; none of the four untracked Markdown files was a build input. |
| Input fixture | The frozen `tests/data/symmetric_4.mtx` SHA-256 remains `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. The Day 1 inventory gives the independent matrix/CSC checks. |

This is the **prebuild** snapshot. Adding this Day 3 artifact and updating `WORKING_NOTES.md` changes the untracked set and bytes afterward. Before an actual solver run, recapture status and any dirty patch under a new attempt ID; do not reuse this snapshot to assert a later clean worktree or a complete v1 cell identity.

## Compiler, flags, archive, and helper

| Field | Captured value |
| --- | --- |
| Compiler | `/usr/bin/cc`, Apple clang 17.0.0 (`clang-1700.6.4.2`), target `x86_64-apple-darwin24.6.0`; full `cc --version` bytes retained. |
| SDK | `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` from `xcrun --show-sdk-path`. |
| Library build | `make -B build/libsparse_lu_ortho.a`, forcing regeneration and compilation of all 49 library sources. Each command used `cc -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -DSPARSE_IDX_BITS=32 -Ibuild/include -Iinclude -c ...`. Archive command is `ar rcs build/libsparse_lu_ortho.a` with all 49 object paths, captured verbatim in the raw log and manifest. |
| Build mode | Optimized `-O2`, no debug or sanitizer flag in the recorded compile commands. `SPARSE_IDX_BITS=32`; `build/.sparse-idx-bits` and retained generated `sparse_version.h` both state 32. No `SPARSE_MUTEX` or `SPARSE_OPENMP` macro/flag was present in the 49 recorded compile commands. Default library link flags are `-lm`; the archive step itself has no linker flags. |
| Helper build | Exact `helper-compile.argv.txt`: `cc -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -O2 -Iinclude -Ibuild/include -I/usr/local/Cellar/suite-sparse/7.12.2/include/suitesparse scripts/cholmod_pilot.c -Lbuild -lsparse_lu_ortho -L/usr/local/Cellar/suite-sparse/7.12.2/lib -lcholmod -lm -o <local-bundle>/cholmod_pilot`. The package flags were captured from `pkg-config`; stdout and stderr are retained separately and empty on success. |
| Project path in helper | Source inspection confirms `sparse_copy(a)`, `SPARSE_REORDER_NONE`, `SPARSE_CHOL_BACKEND_LINKED_LIST`, `sparse_cholesky_factor_opts`, `sparse_cholesky_solve`, and a required `used_csc_path=0` before returning success. Effective behavior is unobserved until execution. |
| Declared linkage | `otool -L` on the helper names `/usr/local/opt/suite-sparse/lib/libcholmod.5.dylib` and macOS `libSystem.B.dylib`. `helper-linkage.stdout` is retained. This is a linker declaration, not actual runtime load-path proof. |

## Retained output hashes

| Local file | Bytes | SHA-256 |
| --- | ---: | --- |
| `cholmod_pilot` | 239,616 | `ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f` |
| `libsparse_lu_ortho.a` | 492,960 | `f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3` |
| `sparse_version.h` | 1,021 | `849d4acbd12131aec3f712299f5345fb6de8aac333fd769196f103dbc0f3e13d` |
| `project-library-build.stdout-stderr.log` | 12,666 | `ab767b5f28eb7f3d2b682e7e310e3046f6ad5715cf54b75c3607f32807934914` |
| `helper-linkage.stdout` | retained | `bb5ef10d127bb86eb5548e2a7c37fa46d9b58e737d5fd9757aeef512c95a0bd6` |

Both build exit-code files contain `0`. The helper is the executable artifact and the archive is the linked project library artifact for this Day 3 build. The manifest and all raw logs remain ignored and local; these hashes are a pointer to exact bytes, not a published binary.

## Promotion gates and next owner

Day 4 must establish the **actually loaded** CHOLMOD binary and transitive paths, byte hashes, module/build options, ABI and effective factor controls. Day 5 must capture host and thread provenance. The solver attempt must obtain its own fresh source-status/patch snapshot, copied executable/archive identity, raw outputs and statuses. The v1 dirty-worktree path requires a nonempty `project_patch` artifact whose digest equals the recorded patch hash; the Day 3 prebuild untracked patch is evidence for this build only. If a later attempt cannot reproduce its source state, it remains blocked. No unknown project compile flag was found in this build's raw command log.
