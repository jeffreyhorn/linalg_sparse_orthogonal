# Sprint 153 Day 12 Integrated Package And ABI Validation

## Purpose

Day 12 runs integrated validation for the selected static-first package/ABI
decision. The validation covers install/export behavior, downstream consumers,
unsupported shared-artifact checks, static deferral diagnostics, and
report-index evidence meaning.

## Validation Summary

| Validation | Result | Evidence |
| --- | --- | --- |
| Static deferral guard | Pass | `bash scripts/static_package_deferral_check.sh` passed. |
| Make install/package proof | Pass | `bash tests/test_install.sh` passed with `23` checks and `0` failures. |
| CMake install/export proof | Pass | `bash tests/test_cmake_install.sh` passed with `27` checks, `0` failures, and `0` skips. |
| Package report-index structure | Pass | `python3 scripts/normalize_report_index.py --family package --check` reported `6` rows ok. |
| Package report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family package --check-freshness` reported freshness ok for `6` source-controlled rows. |
| Runtime backend report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` reported freshness ok for `1` source-controlled row. |
| Whitespace/diff hygiene | Pass | `git diff --check` passed. |

## Make Install And `pkg-config` Proof

`bash tests/test_install.sh` validated:

- static archive install;
- no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts;
- all `19` installed headers;
- installed `sparse.pc`;
- exact package version resolution;
- installed prefix/libdir/includedir variables;
- installed include and static archive link flags;
- no `Libs.private` stanza;
- static archive package metadata description;
- no unsupported package or ABI wording;
- generated and maintained `pkg-config` consumers compile/link/run;
- uninstall cleanup.

## CMake Install/Export Proof

`bash tests/test_cmake_install.sh` validated:

- CMake configure, build, and install;
- static archive install;
- no shared-library artifacts;
- installed `19` headers;
- installed CMake package files and `sparse.pc`;
- `Sparse::sparse_lu_ortho` remains `STATIC IMPORTED`;
- no shared imported metadata;
- no unsupported loader or shared-selector metadata;
- install-prefix include/archive paths;
- no source/build-tree path leaks;
- static archive `sparse.pc` metadata;
- installed maintained CMake example configure/build/run;
- exact-version consumer configure/build/run;
- mismatched-version rejection;
- `pkg-config --modversion` matches `VERSION`.

## Unsupported Shared-Artifact And Loader Proof

The selected decision defers shared-library support, so validation proves
absence and rejection rather than runtime loader behavior:

- shared artifacts are absent from Make and CMake install trees;
- installed CMake package metadata contains no shared imported target metadata;
- installed CMake package metadata contains no unsupported loader or
  static/shared selector metadata;
- `BUILD_SHARED_LIBS=ON` fails with exact blocker wording;
- `sparse.pc` contains no shared-library, dynamic ABI, package-manager, or
  static/shared selector claim.

## Report-Index Evidence Meaning

The report-index checks confirm source-controlled proof ownership and evidence
meaning:

- package rows are source-controlled proof-owner rows, not proof that an
  install command was just run;
- package freshness warnings are advisory and governed by schema/Git review;
- runtime backend governance rows remain source-controlled policy evidence;
- no generated report-index row is treated as package, CI, ABI, loader, or
  release proof.

There is no separate maintained `ci` report family in
`scripts/normalize_report_index.py`; CI scope is represented through package
proof ownership, workflow comments, and source-controlled policy rows.

## Day 13 Handoff

Day 13 should run the final quality/residual review:

- determine whether `.c` or public `.h` files changed;
- run full C gates only if required by changed files;
- rerun focused package/report/doc checks;
- review residual shared-library, ABI, package, platform, and loader debt;
- prepare the Sprint 154 handoff.
