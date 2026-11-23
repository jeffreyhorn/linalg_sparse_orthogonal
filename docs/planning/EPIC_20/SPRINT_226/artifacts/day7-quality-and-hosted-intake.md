# Sprint 226 Day 7: Local Quality and Historical Hosted Intake

## Decision

The extracted [exploratory base archive](day3-exploratory-source-archive.md) passed the local reviewed Make/CMake quality paths and the retained package-consumer checks. Hosted runs were located for the **same base commit**, `c42d6715fad7e895bae4941a006df3563f1afc38`, and their job results and retained logs were inspected. These runs are **historical base-commit evidence**, not release-candidate proof: [Day 2](day2-static-candidate-definition.md) left candidate freeze blocked, and Sprint 226's documentation is still uncommitted. Linux's hosted reviewed CMake parity job also failed after its runner received a shutdown signal before CTest finished. The separate Sprint 225 timing receipt is `blocked` for a missing accepted baseline.

**Gate verdict:** No platform is promoted as a claimed release-candidate tier. The required exact-candidate hosted matrix remains **blocked**, even though several base-commit jobs passed. No timing, release-ready, public provider, or broad platform claim follows.

## Local quality on extracted base source

Commands ran with `make -C build/sprint226/day3/sparse-lu-ortho-2.2.0` and an isolated `BUILDDIR` at `build/sprint226/day7/make`; CMake parity used `QUALITY_REVIEW_CMAKE_DIR=build/sprint226/day7/cmake` as an absolute path. All logs are retained locally under ignored `build/sprint226/day7/`. No C or header file was edited, so `format-check` preserved exact source bytes; the conditional source-edit `make format` requirement was not triggered.

| Local command or gate | Result | Retained log SHA-256 |
| --- | --- | --- |
| `make format-check` | **PASS**; clang-format dry run made no edits. | `format-check.log`: `a251cf3af0fb78d330a1a403d12f2d8915d0ddee8f450fd6d93806a4904188aa` |
| `make lint` | **PASS**; strict compile, clang-tidy, and cppcheck finished. Tool warnings were emitted but the gate exited 0. | `lint.log`: `0197401c6872289ed99cf6407c1398cf28785d69a853718d7075bdecc262ddc8` |
| `make test` | **PASS**; 62 Make test-binary banners, all tests passed. | `make-test.log`: `bfee8ae7802ed05226bed636ef84bbe20839218a58d3f0bd005646eaecd1b8d4` |
| `make quality-review-cmake` | **PASS**; clean CMake rebuild, `ctest -N` 63, Make 62 plus one focused CTest selector, and **63/63 CTests passed**. | `cmake-parity.log`: `7eaee2ef6b2400a5d701fa1fc5ef83cb3baa65b37a866864d38abf89e52ad42d` |
| `make source-list-check` | **PASS**, 49 library sources. | `source-list.log`: `d5e08f4d4d15529a2ed5b5f3993993b551522035e67908ea1fe9edcdda1a4079` |
| `make support-docs-guard` | **PASS**. | `support-docs.log`: `62be2a570e20eaf80747515e0322df278a7a4ba6043f4c738c39eb8d9428b2e3` |
| `make api-docs-local-only` | **PASS** with pinned `PyYAML==6.0.3` in ignored local venv; workflow policy checked 123 job/step records. Generated HTML remained in the ignored extracted tree. | `api-docs.log`: `88ee16e8a15b627b35b8e1f1ff3730d1202eabd6f0f712b36aa685c79a18d164` |
| `python3 tests/test_package_manager_deferral_guard.py` | **PASS** when run separately. | `provider-policy-tests.log` was empty (exit 0); SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. |
| `make package-manager-deferral-guard` | **INCOMPLETE / exit 2** after the optional local Homebrew proof was deliberately interrupted. The proof had spent over 16 minutes building a Homebrew CMake 4.4.5 dependency from source. This is not a passing guard or a diagnosed source regression. The temporary tap was removed by the proof cleanup. | `provider.log`: `d34e944eb374c4dcea9e3d7cf83b3de58cf285ce2be226f41604c0a6f95b3d64` |

The prior [Day 6](day6-local-installed-consumers.md) exploratory static install/consumer matrix also passed: Make install 26/26, CMake install 38/38, pkg-config width 40/40, CMake width 20/20, pre-use width 12/12, Make width 28/28, 64-bit functional 1/1, and static contract mutations 26 tests. Those are local macOS results, not an exact-candidate hosted run. The unfinished Homebrew proof is separate from the static source install and public provider deferral boundary; a final release checklist must classify and rerun any required gate on the selected candidate.

## Hosted base-commit run inventory

`gh run list --commit c42d6715fad7e895bae4941a006df3563f1afc38` and per-run job JSON were saved locally in `build/sprint226/day7/`. The four push runs below all report that exact base SHA. No Sprint 226 candidate SHA has been selected, and no run was triggered for an uncommitted worktree.

| Platform/workflow run | Inspected jobs and evidence | Historical base result | Candidate gate |
| --- | --- | --- | --- |
| [Linux CI run 44433727330](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727330) | `package-contract` job `135257250492` passed; its downloaded log shows 26 Make install, 38 CMake install, 40 pkg-config width, 20 CMake width, and 12 pre-use width passes. `lint` job `135257250453` passed `quality-review-compile`. Selected 64-bit job passed. Reviewed CMake parity job `135257250502` failed during CTest after 31/63 passed; the runner reported a shutdown signal and exit 143. Supplemental coverage also failed with no downloadable job log in this intake. | **Run failure**; package and compile subsets passed, but full reviewed CMake CTest did not complete. The log does not establish a code-level CTest failure. | **Blocked**: exact-candidate run absent, and the base-run parity result is incomplete. |
| [macOS CI run 44433727057](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727057) | Apple Clang quality job `135257249578` passed; AppleClang 21.0.0.21000101, CMake 63/63 CTests, and Make tests passed. Make/`pkg-config` install job `135257249654` passed 26 install and 40 width checks. CMake install job `135257249841` passed 38 install, 20 CMake width, and 12 pre-use checks. Selected 64-bit job passed. | **Run success** for this historical base SHA. | **Blocked**: not a frozen candidate SHA or owner-approved platform claim. |
| [Windows CI run 44433727133](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727133) | MSVC reviewed CMake job `135257249486` passed 63/63 CTests (MSVC 19.44.35230.0). `install-and-downstream` job `135257249826` passed the static install validator, generated and maintained CMake downstream consumers, exact version, and expected mismatch rejection. No Windows Make or pkg-config execution parity is inferred. | **Run success** for this historical base SHA and named Windows CMake subset. | **Blocked**: not a frozen candidate SHA or broader Windows claim. |
| [Sprint 225 timing run 44433727385](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727385) | Job `135257250247` failed its preflight. Downloaded artifact `14486279554` confirms `status=blocked`, missing accepted baseline pointer, null candidate source/timing values, and matching base source in host provenance. | **Timing blocked**, with no measured candidate. | **Blocked**: no accepted baseline or exact-candidate timing decision. |

The downloaded package and quality job logs are local under `build/sprint226/day7/`; the relevant log SHA-256 values are:

| Job log | SHA-256 |
| --- | --- |
| Linux package | `9d666b5a131c983acc471ca7f632624a71b0c4b94404bc273ba4c28eb9b46ab7` |
| Linux compile | `ae89f9c4f2ab2c389acaf21298492eb36521425bd483683c00916a6d87618838` |
| Linux interrupted CMake parity | `3e8614585d7054346f2d966de79610174b3b3e897600fd45cfa6959ea16f0124` |
| macOS Apple Clang quality | `5296f7fc75edec94626e348f1c3e98e0fd5a5dacf8fc0a4346deb595b5dfdc31` |
| macOS Make/`pkg-config` install | `0fe759fdcdcc215d1c9cb172a4f55cd1dca16cc4c8bbc370ec009d029b40287c` |
| macOS CMake install | `4e9f310d66996c3122b239c61a52cedf1fc9994c7b3d728a537b6c83d8f35c9a` |
| Windows CMake quality | `99c3920c9ad04c27fa51e274351850c88b216d744c3cfbaca24ed60819c14f12`; job JSON records success and 63/63 CTests. |
| Windows CMake install/downstream | `da019a8b0216be28d4e1e5d79bb8a6e61d3c2d5650fc0287b785e6b8ec216703` |

Artifact inventory for the base runs found selected-report and dead-code artifacts, but **no install-specific artifact ZIP** for Linux, macOS, or Windows. The installed-consumer counts were verified from the downloaded job logs, not from a nonexistent package ZIP. A final candidate proof must retain the exact-run logs and any required artifact digests under the release owner's retention policy.

## Separate timing receipt audit

The timing artifact ZIP was downloaded through the GitHub Actions artifact API. Artifact ID `14486279554`, ZIP size 1,466 bytes, SHA-256 `d89a5a06180b34e431a150daa71abf8412450b073eb65d48594657505b9353df`. Its two files were independently hashed after extraction:

| Member | SHA-256 | Relevant content |
| --- | --- | --- |
| `sprint225-host-provenance.json` | `dec786511ea0f9105d9887c76724327d31a02eaac1319936d63ad72f90468f62` | Source `c42d6715fad7e895bae4941a006df3563f1afc38`, run `44433727385`, attempt 1, macOS 26 arm64 virtual Apple M1/Apple Clang 21 preflight context. |
| `sprint225-timing-decision.json` | `de14ba87e8f2b2fd0ece17fa4d874a6e3a56f9390a1875ee1871ea1ac3b5b2a6` | `status=blocked`; accepted baseline pointer missing; baseline ID, candidate source, candidate samples, and candidate median are null. |

This receipt supports only a preflight blocker. It cannot be called a timing failure of the algorithm, a measured regression, or a passing speed gate. The existing selected performance freshness jobs are separate and threshold-free.

## Release gate disposition and Day 8 handoff

| Gate | Day 7 disposition |
| --- | --- |
| Local exploratory base quality and static consumers | Reviewed Make/CMake, docs and static installed-consumer paths passed as stated; provider aggregate remained incomplete after an interrupted optional Homebrew dependency build. |
| Exact-candidate Linux/macOS/Windows hosted evidence | **Absent** because no candidate was frozen. Historical base runs cannot be promoted. Linux base CMake parity also needs a clean hosted rerun after the runner shutdown if that base is ever proposed. |
| Sprint 225 timing | **Blocked** by absent accepted baseline and exact-candidate measurement. |
| Release readiness | **Blocked** by changelog/version, clean committed candidate, owner approvals, exact-candidate archive/install proof and hosted gates, plus the unresolved timing-scope decision. |

Day 8 should audit provenance only within the stated historical scope, and begin the first-use walkthrough from a fresh extraction or clean prefix so generated local quality output cannot be mistaken for source-archive contents. Any final candidate must be identified by one reviewed full SHA and retested after that selection.
