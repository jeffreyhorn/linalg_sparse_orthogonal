# Sprint 224 Day 5: QR Promotion Eligibility Decision

## Decision

**Authorize preparation of a QR-only atomic promotion candidate; do not promote Windows QR freshness yet.** The mandatory hosted intake conditions are met for the current *intake* source revision: [Windows run 43762862057](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057) is a successful `master` push on `44e3b84dab97a10b2727cc026e7124a9a5415421`; [selected QR job 132961326705](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/job/132961326705) succeeded; [artifact 14181192246](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts/14181192246) is downloadable, has the expected six files, and its independently downloaded ZIP matches API digest `sha256:09755ddbdc837e2b5587e15a200836c47c3ae6bc105fd3a25cdc8989a6baa69b`. [Day 3](day3-qr-artifact-row-audit.md) confirms rows and observation traceability; [Day 4](day4-qr-numerical-and-provenance-audit.md) confirms numerical and cross-file consistency.

The artifact is **not** evidence that the current source-controlled manifest already selects Windows. All six generated QR rows say `local_only`, and their summary denies hosted and Windows freshness. These are explicit promotion blockers to be fixed together, not reasons to discard the successful intake. No manifest, generator, workflow, guard, schema, README or INSTALL promotion edit is made on Day 5. Days 6–10 may prepare the candidate below; Days 11–14 must inspect a new hosted artifact from the final changed source head. If the final exact-head run fails or the cross-file contract cannot be made coherent, re-defer QR and retain Linux/macOS-only selection.

## Selected-target freshness check on downloaded bytes

The six Day 2 files were copied byte for byte to a temporary `build/comparison/qr_incompatible_ls/` tree, then checked with:

```sh
python3 scripts/normalize_report_index.py \
  --build-root /tmp/sprint224-day5-normalizer/build \
  --family comparison --require-generated comparison \
  --check-freshness --selected-target qr-incompatible-ls
```

Result: `normalize-report-index: freshness ok (17 rows)`. The six selected generated rows were individually reported fresh because their `source_commit` equals current local HEAD `44e3b84dab97a10b2727cc026e7124a9a5415421`; the additional rows in the 17-row index include source-controlled comparison contracts. The check confirms target-filtered row freshness and required files for this intake revision. It does **not** override `local_only` or approve a Windows claim. Day 3 independently compared the six ordered QR row IDs against `SRT-COMP-QR-INCOMPATIBLE-LS`.

## Thirteen-gate decision on the intake revision

The Sprint 208 criteria are adapted to the QR target and to the actual run event. `source_branch=master` is correct for this push run; a future pull-request run must instead prove its actual source branch and head, not reuse `master` mechanically.

| Gate | Status at Day 5 | Evidence or remaining work |
| ---: | --- | --- |
| 1. Current hosted run | **Pass for intake** | Run `43762862057`, `push`/`master`, head `44e3b84d…`, timestamps and success recorded on Day 2. |
| 2. Exact selected job | **Pass for intake** | QR MSVC job `132961326705` succeeded; log shows checkout, configure, build, generator, selected normalizer and upload. |
| 3. Named artifact available | **Pass for intake** | QR artifact `14181192246`, 4620 bytes, unexpired, downloaded; ID, expiry and digest recorded. |
| 4. Exact archive membership | **Pass for intake** | Downloaded ZIP digest matches API; exactly six required flat files with matching extracted bytes and individual hashes. |
| 5. Exact row identity | **Pass for intake** | Six unique, ordered `comparison_qr_overdetermined_incompatible_4x2_*` IDs equal the selected manifest row. |
| 6. Row status and provenance | **Pass for intake** | All six rows pass with expected reasons; SHA, `master`, clean worktree, timestamp, observations and source paths agree. |
| 7. Bounded platform/compiler | **Pass for intake** | Rows, manifest and summary say `windows-amd64`, VS 2022 Release CMake probe; Day 4 verified source-controlled dense baseline and 4×2 fixture numerics. |
| 8. Negative path/row guards | **Partial** | Existing QR path, stale, duplicate, unexpected-row and missing-required-file tests pass, as do workflow and PowerShell structural suites. Days 8–10 must add or verify exact QR hosted cross-file, wrong-artifact, renamed/extra-file, PR-head and contradictory-claim rejection before promotion. |
| 9. Selected manifest Windows tuple | **Fail for promotion** | `SRT-COMP-QR-INCOMPATIBLE-LS` lists Linux/macOS workflow, job, artifact and platform only. Add the QR Windows tuple in aligned third positions only after candidate review. |
| 10. Support and claim alignment | **Fail for promotion** | Manifest and generated QR rows/configuration say `local_only`; a Windows-selected manifest alone would contradict the output. |
| 11. Generated wording | **Fail for promotion** | QR rows and summary still contain `no hosted CI proof`, `no Windows report freshness`, and local-only scope. |
| 12. Public/maintainer wording | **Fail for promotion** | Current docs accurately describe evidence collection and re-deferral, not a promoted Windows-selected QR row. Align all maintained surfaces only with the atomic candidate. |
| 13. Broad non-claims | **Pass for current boundary** | Current manifest and generated non-claims exclude broad Windows/QR/least-squares parity, packages, ABI, performance and state-of-the-art status. Preserve them while narrowing only the named target's Windows exclusion. |

**Tally:** Seven intake passes (1–7), one partial guard gate (8), four promotion failures (9–12), and one current-boundary pass (13). All thirteen must pass on the same final changed head before approving a Windows QR freshness sentence. The observed local-only fields are not silently counted as hosted-selected evidence.

## Frozen QR-only candidate change map

| Surface and owner | Required candidate change or verification |
| --- | --- |
| `.github/workflows/windows-ci.yml` — workflow owner | Keep the exact QR MSVC/CMake build, generator, selected normalizer and six-file upload. Add PR-head checkout and source SHA/branch wiring analogous to the reviewed Cholesky lane so later PR artifacts identify the source commit rather than a synthetic merge checkout. Verify a push run remains valid. |
| `scripts/run_external_comparison.py` — generator owner | Admit `hosted_selected` only for the exact QR job, Windows runner, repository, event, source SHA/branch and clean worktree. Emit a narrowly hosted QR scope/summary and remove only the contradictory hosted/Windows-denial phrases for that verified context; retain `local_only` and original non-claims for local or other-platform runs. |
| `scripts/normalize_report_index.py` — report owner | Verify QR hosted rows, manifest and summary agree on target, six files, row IDs, paths, support tier, source and baseline/runtime provenance. Preserve QR selected-target isolation from unrelated artifacts and the existing Cholesky contract. |
| `tests/corpus/manifests/selected_report_targets.tsv` — report owner | Update only `SRT-COMP-QR-INCOMPATIBLE-LS`: append `.github/workflows/windows-ci.yml`, `selected-qr-incompatible-comparison-freshness`, `sprint209-windows-selected-comparison-qr-incompatible`, and `windows` in matched third positions; set hosted support and a bounded three-platform selected-freshness claim while retaining broad non-claims. Leave other QR and Cholesky rows untouched. |
| Manifest, generator, normalizer, workflow and PowerShell tests — guard owner | Replace QR re-deferral assertions with exact positive tuple and negative mutations. Cover PR-head mismatch, six-file/row/path contradictions, baseline/runtime conflict, hosted/local-only conflict and claim drift; keep valid local-only output accepted outside the exact hosted job. |
| README, INSTALL, `docs/maintainer_guide.md`, `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md`, Epic 20 planning notes — docs owner | Replace QR re-deferral language only after candidate fields agree; document one named fixture, source-controlled dense helper, exact evidence links and explicit exclusions. Preserve the Sprint 220 generated-API local-only boundary. |

**Proposed exact public sentence, conditional on final proof:** “Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense QR reference helper.” This sentence does not assert broad solver correctness, Windows package support, external-package parity or performance superiority.

## Day 5 checks and handoff

| Command or check | Result |
| --- | --- |
| Downloaded-bundle selected-target normalizer | `freshness ok (17 rows)`; six generated QR rows fresh for intake HEAD. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass in current QR re-deferral state. |
| `python3 tests/test_selected_comparison_workflow.py` | Pass for current selected workflow ownership. |
| `python3 tests/test_validate_windows_powershell.py` | Pass for current structural guard and QR re-deferral boundary. |
| `python3 tests/test_normalize_report_index.py` | Pass for current normalizer and selected QR cases. |
| Day 3 row and Day 4 numerical/provenance audits | Pass for the downloaded intake artifact. |

Item 224.2's planned 30-hour audit allocation is closed by Days 3–5; this is planned scope accounting, not measured elapsed labor. Day 6 begins the conditional atomic candidate. **Stop and re-defer** if any mandatory hosted field cannot be reconciled, a guard cannot reject contradictions, a local-only run is accidentally marked hosted, or the final-head hosted job/artifact is unavailable. The intake ZIP becomes historical as soon as the branch gains a new source commit.
