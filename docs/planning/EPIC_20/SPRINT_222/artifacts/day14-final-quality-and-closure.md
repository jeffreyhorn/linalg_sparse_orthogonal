# Sprint 222 Day 14: Final Quality and Closure Ledger

## Final disposition

**The local two-sided CHOLMOD correctness work passed its numerical gates twice. The proposed v1 observed cell remains blocked, and no external result is approved for publication.** The project and CHOLMOD processes used the frozen 4×4 SPD fixture and returned finite solutions within the independent residual and oracle tolerances. The repeat reproduced the first pair's solution streams and metrics. It did not reproduce a single v1 identity: the dirty source-patch digest changed, Day 6 lacks an execution-time host fingerprint, and complete baseline build/BLAS/thread provenance remains unavailable. The first and repeat observed candidates were rejected at `/environment`; their validator-accepted records are `blocked`, with null cell IDs. The independent rights/attribution decision remains pending under the [Day 12 no-publication disposition](day12-independent-rights-and-claim-decision.md).

The final ignored verification bundle is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day14-final-l7sd8pse/`. Its three-file `manifest.json` SHA-256 is `6f7cae4acc058df6591d2eeccb111448287cb1d6feac55ec20cb2dff549c836d`; `verification.json` SHA-256 is `668852a724bcbf94b2ba0b63c78548e86213b9176050cab3cbeff719ff99a56b`. The bundle retains `verify.py`, byte-backed index/record checks and four CLI transcripts. Every file in this final manifest passed a separate size/SHA-256 check. No earlier attempt or raw stream was rewritten.

## Attempt and identity reconciliation

| Evidence | SHA-256 of retained index/manifest | Outcome |
| --- | --- | --- |
| Day 3 project build | `fee0e326d854bdb1aca7fe90e55c415dd981fb95ea39eaff326cb29ec9e0f741` | Retained helper/archive and build identity. |
| Day 4 CHOLMOD runtime | `a9e349cd9310f58ce121f9b6185033872440106f6443a5848d93730d7ee1bcb9` | File-backed loaded dependencies recorded; full bottle/module and shared-cache BLAS identity remain open. |
| Day 5 host/setup | `56515f31f7afcaa804118f9eeefac6fac1fbd0846694442de07af27ce754a56b` | Host fingerprint and requested thread controls recorded; actual Accelerate BLAS threads unknown. |
| Day 6 first project solve | `6e4c6f28601c9acb696192add7f700723b22ad469f3001b2f7ed752940da7781` | Exit/status/numerical pass; no execution-time host fingerprint. |
| Day 7 first CHOLMOD solve | `a140595d6b5c7150f686749af0133d6e9fd3950cd3ef04305926c6f494fe6dee` | Exit/status/CSC/numerical pass. |
| Day 8 first-pair reconciliation | `7e20fb0e8fc4c91cd29834569624ce141456455d96d147246b8cf3165598df88` | 148/148 checks pass. |
| Day 9 first v1 record | `234bd44e3421c876ba7345de3876f5e2dfa6cea2418fb4616d39bd5df961677b` | Blocked record valid; observed candidate rejected. Earlier preparation failures retained. |
| Day 10 repeat and separate v1 validation | `bfc37b2c6419e7c69b1b964d6327d75c07e5e924a387ca44dc3b0ecde33dcb7e` / `cfdd54715c4f8484406fb5ef30ace6d7b6f989a0b9686015b13828904c59838b` | 219/219 repeat checks pass; blocked v1 record valid, observed candidate rejected. |
| Day 11 source index and review manifest | `392c6c4336718e94b2ca6a60f9a8c5f0d2a62cf105a1a37457cce949853d82bd` / `c40d01d46839b2660c95f836de1044e50a6fafda8947d81807f6b92d314b6205` | All 386 Day 3–10 source files and 39 review payloads passed byte-size/SHA-256 recheck. Includes earlier failed drafts. |
| Day 12 publication disposition | `be288c15ee87ae5f313fe6c105f9dbeea592fce440acfe96da974e530b549c62` | Pending attribution/publication review; empty approved artifact list and null approved claim. |

Day 14 independently rechecked the six final Day 6–10 manifests plus the Day 11 source and review indexes: eight indexes, with every listed size and SHA-256 matching. It reran the v1 CLI with the checked-in schema and each record's own bundle root. `s222-d9-provenance-blocked` and `s222-d10-repeat-blocked` exited `0` as `evidence_status=blocked`; the first and repeat attempted observed records each exited `1` with `missing_provenance /environment`. Day 11's six synthetic adversarial mutations still document rejection of missing raw side, changed loaded-library bytes, bad ABI, nonfinite metric, stale cell ID and unsupported claim scope. No synthetic mutation was promoted as external evidence.

A later documentation cross-check initially used the wrong Day 3 filename (`manifest.json`) and stopped at a missing file. The retained Day 3 file is `project-build-manifest.json`; the corrected cross-check verified all twelve cited local evidence digests and the separate Day 12 disposition digest. No evidence file or digest was changed to resolve that path error.

The frozen fixture SHA-256 is `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. Both pairs have project residual infinity norm `3.552713678800501e-15`, backward error `1.0449157878825003e-16`, oracle error `2.220446049250313e-16`; CHOLMOD's corresponding errors are `0`. The fixed residual and oracle limits are `1.4495071809506044e-12` and `1.380483029476766e-13`. The first and repeat paired solution differences are each `2.220446049250313e-16` against combined oracle limit `2.760966058953532e-13`. The repeat's host fingerprint matched Day 5/7 (`3f4db376401dc48cf182656aa1f31f2bf41234de468a7d65a2030777b40e7e70`), but Day 6 has no execution-time fingerprint. Its source patch SHA-256 `80113ecd42d4fa172c78a988baa91ac9204e0e04828c002c1d77a41a3bf0f3a2` differs from repeat `2d0f7a4e19c78b474734d3f1d6fe4b8a318825c68d35f2239afb66fb91d03993`. The four requested thread variables were `1`, while effective Accelerate BLAS threads remain unobserved. No timing samples or performance measurements were collected.

## Day 14 test and quality results

| Command or route | Outcome | Scope / limit |
| --- | --- | --- |
| `python3 tests/test_validate_external_baseline_evidence.py` | PASS, 67 tests. | V1 positive/negative, raw artifacts, identity, numerical, timing and claims. |
| `python3 tests/test_run_cholmod_pilot.py` | PASS, 19 tests. | Runner preflight, status, raw capture, timeout/process cleanup. |
| `python3 tests/test_external_baseline_claim_docs.py` | PASS, 5 tests. | Public-document nonclaims and mutation cases. |
| `make build/test_cholmod_pilot_csc` and `build/test_cholmod_pilot_csc` | PASS, `CSC pilot validation: PASS`. | Make CSC test. |
| `bash tests/test_install.sh`, in detached same-HEAD worktree | PASS, 26 assertions. | Make install, static/package contract, downstream consumers and uninstall. Isolated because the script runs `make clean`, which would delete the ignored local evidence in the active workspace. Temporary worktree removed after the check. |
| `bash tests/test_installed_width_check.sh` | PASS, 12 assertions. | Clean 32/64-bit CMake/pkg-config consumers and swapped-archive rejection. |
| `cmake -S . -B build/quality-review-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; `cmake --build build/quality-review-cmake --target test_cholmod_pilot_csc --parallel 1`; `ctest --test-dir build/quality-review-cmake -R '^test_cholmod_pilot_csc$' --output-on-failure` | PASS, 1/1 CTest. | Applicable focused CMake parity route on local macOS; no full CTest/hosted Windows lane claimed. |
| `make docs-check` | PASS; 18 public headers, 18 generated reference pages and 18 generated source pages. | Doxygen output is ignored local content. |
| `API_DOCS_PYTHON=/tmp/s222-api-docs-venv/bin/python bash scripts/check_api_docs_local_only.sh` | PASS after Doxygen generation; 111 workflow job/step records checked. | Pinned `PyYAML==6.0.3`; no generated API files staged/tracked. |
| `/tmp/s222-api-docs-venv/bin/python tests/test_api_docs_workflow_reader.py`, `...policy_inputs.py`, `...policy.py`, `...mutations.py`, `...differential.py`, `...test_api_docs_local_only_guard.py` | PASS: 20, 8, 50, 4, 6 tests; local-only guard suite silent exit `0`. | Workflow parser/policy positive/negative and local-only regressions. |
| `python3 scripts/normalize_report_index.py --no-generated --check` | PASS, 114 rows. | Existing selected index unchanged. |
| Day 14 `verify.py` and final manifest check | PASS. | Exact retained files, v1 CLI verdicts, source/claim boundaries. |

No `.c` or `.h` file changed on this branch during Sprint 222, so the conditional `make format && make lint && make test` gate in the Day 14 plan was **not triggered**. The only tracked code edit is the Python timeout-test fixture in `tests/test_run_cholmod_pilot.py`, which passed its full suite. The focused CMake test and install consumer routes passed locally. Full CMake/CTest, hosted Windows and other CI jobs were not run by this Day 14 local closure; no platform-wide pass is inferred.

`git ls-files docs/api build/external-baseline` returned no tracked files; the final verifier checked that the local bundles are ignored. Neither `selected_report_targets.tsv` nor `report_families.tsv` contains a Sprint 222 result, and the report normalizer stayed at 114 rows. Sprint 215's example remains a `planned` blueprint; both Sprint 222 run records remain `blocked`, with `timing.state=not_run`, `claim_scope=no_result`, null canonical IDs and pending publication/attribution. The generated Doxygen HTML remained local after `make docs-check`.

## Item closure and limits

| Item | Day 14 status |
| --- | --- |
| 222.1 Fixture and rights | Fixture/conversion bytes and license inventory complete; independent attribution and redistribution decision pending. |
| 222.2 Runtime provenance | Project binary, loaded CHOLMOD and ten file-backed dependencies, host and requested controls captured; exact bottle modules/options, Apple shared-cache BLAS identity and effective BLAS threads incomplete. |
| 222.3 Two-sided execution | Both solvers ran independently; statuses, CSC, raw streams and numerical checks passed locally. |
| 222.4 Validation and repeat | Fresh repeat and failed preparations preserved; local numerical repeat passed, while complete observed v1 cell ID and same-cell repeat are blocked. |
| 222.5 Claim review | Written no-publication disposition with reviewer identity unset; attribution/publication approval pending. |
| 222.6 Test closure | Applicable local focused, install, CMake and documentation checks passed. Hosted/platform-wide checks were not claimed. |

This closes the planned **16-hour test-closure scope** as a recorded local verification, not as 16 elapsed hours. The Sprint 222 plan's 140-hour estimate is a planning budget, not timing evidence or a statement that all proposed deliverables were achieved. The validator-accepted two-sided `correctness_observed` deliverable remains **blocked**. Keep all raw/binary attempts ignored and private; an independent rights/attribution decision and a newly complete, stable v1 cell are required before any external correctness statement. The separate [Sprint 223 handoff](day14-sprint223-handoff.md) preserves this boundary.
