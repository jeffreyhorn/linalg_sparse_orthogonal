# Sprint 224 Day 14: Selected Windows QR decision and handoff

## Decision rule and final attestation

**Approve only `SRT-COMP-QR-INCOMPATIBLE-LS` / `qr-incompatible-ls` for selected Windows QR incompatible least-squares report freshness when the final committed PR head has a successful exact-head Windows/MSVC QR job and independently audited six-file artifact.** The authoritative final-head run, job, artifact IDs, ZIP and file hashes, and gate outcome are recorded in the [PR #248 final-head attestation](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248). This file is part of the commit being tested, so it deliberately does not assert a future run ID or a self-referential commit SHA. If the PR attestation is absent, reports a different SHA, or records any failed gate, this decision is **blocked** and the accepted wording must not be treated as approved.

The named sentence is: “Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.” It concerns one source-controlled `qr_overdetermined_incompatible_4x2` comparison report and its six generated rows. It does not establish broad Windows QR solver correctness or external-package parity. README, INSTALL, maintainer, corpus and schema text now carry the same bounded sentence and point to the PR for final-head evidence. A later source commit requires a new hosted audit.

The [Day 11 candidate audit](day11-hosted-qr-candidate-audit.md) and [Day 12 proposed-head recheck](day12-final-proposed-head-qr-recheck.md) passed at their respective historical SHAs. Their artifacts cannot substitute for the final PR-head artifact after this closeout commit. Day 13's [local validation transcript](day13-hosted-local-validation-closeout.md) passed before the final documentation update; the final local gates must also pass before the commit is pushed.

## Sprint 208 gate application

| Gate | Required final-head evidence and decision boundary |
| ---: | --- |
| 1. Current hosted run | PR #248 attestation names a successful `pull_request` run whose head equals the final committed PR head. |
| 2. Exact selected job | `selected-qr-incompatible-comparison-freshness` on `windows-2022` succeeds after explicit PR-head checkout, VS 2022 x64 configure/build, QR generator, selected normalizer and upload. |
| 3. Named artifact available | `sprint209-windows-selected-comparison-qr-incompatible` has a downloadable, unexpired artifact ID and recorded creation/expiry times. |
| 4. Exact archive membership | Independently downloaded ZIP SHA-256 matches API/upload digest; CRC and bytes pass; six flat required files appear once each, with recorded per-file hashes. |
| 5. Exact row identity | Six ordered `comparison_qr_overdetermined_incompatible_4x2_*_v1` IDs equal the selected manifest, with no missing, duplicate, extra or Cholesky row. |
| 6. Row status and provenance | Six study rows pass; commit, source branch, clean worktree, timestamp, artifact path and project/baseline observation values agree across the bundle. |
| 7. Bounded platform/compiler | Rows, manifest, summary and logs agree on `windows-amd64`, VS 2022 Release CMake, the 4×2 fixture and the source-controlled dense QR reference helper; numerical deltas meet `1e-10`. |
| 8. Negative guards | Missing/renamed/extra files, stale SHA, wrong/duplicate/out-of-order rows, path forms, baseline/runtime, hosted/local-only contradiction, broad claim, job/checkout and upload mutations reject in the Day 9–10 suites. |
| 9. Selected manifest tuple | Only `SRT-COMP-QR-INCOMPATIBLE-LS` gains the Windows workflow/job/artifact/platform tuple and `hosted_selected` support; Cholesky and other QR target rows remain independent. |
| 10. Support and claim alignment | Hosted QR rows, configuration and manifest agree on `hosted_selected` and the one named target. Local QR output remains `local_only`. |
| 11. Generated wording | Hosted study and summary use the selected Windows fixture-local scope and retain broad non-claims; neither says `no hosted CI proof` nor `no Windows report freshness`. |
| 12. Public/maintainer wording | README, INSTALL, maintainer, corpus and schema carry the same named sentence, final-head PR evidence route, and later-commit recheck rule. |
| 13. Broad non-claims | No broad QR/least-squares or Windows report parity, raw basis identity, global rank policy, NumPy/SciPy/package parity, package manager, shared-library ABI, performance superiority or state-of-the-art claim follows from this artifact. |

All thirteen gates must be marked PASS in the final PR attestation for approval to take effect. A missing, stale, failed or contradictory final-head artifact yields **re-deferral**, even if every local check passes. This keeps the manifest and docs' named target separate from a general Windows report or solver promise.

## Planned 140-hour scope ledger

These are the project-plan estimates, not a claim of measured elapsed labor. The Sprint 224 plan allocates 10 hours per day for 14 days, with no day over 12 hours.

| Item | Planned hours | Day allocation and result |
| --- | ---: | --- |
| 224.1 Hosted QR Collection | 20 | Days 1–2; exact intake contract and downloaded hosted QR bundle. |
| 224.2 Selected Row Audit | 30 | Days 3–5; ZIP, rows, observations, numerical/provenance audit and eligibility decision. |
| 224.3 Atomic Metadata Update | 26 | Days 6–8; one selected manifest row, guarded generator, exact-head job, docs and strict hosted normalizer. |
| 224.4 Failure Fixtures | 24 | Day 8 (4 h), Days 9–10 (10 h each); artifact, row, path, provenance and claim regressions. |
| 224.5 Hosted Recheck | 22 | Days 11–12 (10 h each), Day 13 (2 h); candidate and proposed-head jobs and retained six-file audits. Final closeout commit still requires its own run. |
| 224.6 Exact-Scope Review | 18 | Day 13 (8 h), Day 14 (10 h); local gates, claim calibration, final attestation and handoff. |
| **Total** | **140** | **14 days × 10 h; each item completely accounted for.** |

## Quality, failure log and residual queue

The selected manifest, comparison generator, selected workflow, full normalizer, Windows PowerShell structural, support-doc and API-doc checks are the affected local gates. The exact commands and pass results are retained in [Day 13](day13-hosted-local-validation-closeout.md) and the final PR attestation. No `.c` or `.h` file changed during Sprint 224, so the C format/lint/test trigger did not apply. The successful Day 11 and Day 12 selected Windows runs and ZIP audits remain historical evidence; they are not counted as final-head proof. The only residual before final approval is the final commit's hosted QR run and artifact audit. Local PowerShell execution is unavailable on this macOS workstation and is not counted as hosted Windows proof; the hosted Windows PowerShell validation job is separate.

Sprint 225's planned one-workload canonical timing gate is independent of this threshold-free QR report freshness result. Carry forward the exact-head rerun rule and the absence of broad Windows QR, external-package, package/ABI, performance and state-of-the-art claims. Other QR targets and Cholesky retain their separate evidence owners. A subsequent source change must invalidate this Sprint 224 final-head attestation until its own hosted artifact is reviewed.
