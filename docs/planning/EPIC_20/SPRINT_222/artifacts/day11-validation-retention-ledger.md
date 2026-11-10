# Sprint 222 Day 11: Validation and Retention Ledger

## Disposition

**Local numerical repeat: passed. V1 observed cell: blocked. Publication: pending.** Day 11 completed a read-only audit of the first pair, repeat pair, validator outcomes and preserved failed preparations. It did not change solver outputs, construct a canonical cell ID or promote a claim. The Day 8 first-pair reconciliation still has 148 passing checks; the Day 10 repeat still has 219 passing checks. Both accepted v1 records remain `evidence_status=blocked` with `cell_id=null`; their attempted observed candidates were rejected for missing `/environment` provenance.

The ignored review directory is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day11-review-ftih1s_0/`. `source-artifact-index.json` SHA-256 `392c6c4336718e94b2ca6a60f9a8c5f0d2a62cf105a1a37457cce949853d82bd` lists **386 files** under Day 3–10 directories. Every indexed file's size and SHA-256 was checked twice. `evidence-audit.json` SHA-256 `90c04145cd965cbb0f0e8712dd22bafc4c100f92d4ce0d9a6d33fb2d8db71b08` verifies all four solver command manifests, raw stdout/stderr streams, status fields and solution vectors are present and hashes their exact bytes. The 39-payload `review-manifest.json` SHA-256 is `c40d01d46839b2660c95f836de1044e50a6fafda8947d81807f6b92d314b6205`. The review directory retains its checker source as `review-runner.py` and every mutation input/output. These are **local evidence indexes**, not a redistributable artifact.

## Attempt and identity audit

| Source | Retained result | Identity and failure treatment |
| --- | --- | --- |
| Day 6 project, Day 7 CHOLMOD | Separate process commands, stdout/stderr, fixture copies, source snapshots, phase statuses, solutions and independent metrics; Day 8 paired ledger | Numeric gates pass. Day 6 lacks an execution-time host fingerprint. The first sides' dirty patches differ after Day 6 documentation was added. |
| Day 10 project and CHOLMOD | New process commands and raw streams for each side, independent metrics, both preflight logs and per-side host fingerprints | All 219 checks pass; stdout and numerical fields match the first pair. Stderr loader traces differ in raw bytes. The repeat's dirty-patch digest differs from the first pair, so it is a **different potential v1 cell**. |
| Day 9 and Day 10 v1 validation | Actual observed candidates and accepted blocked records, all CLI transcripts | Candidate attempts rejected for missing `/environment`; accepted blocked records did not claim a side pass. Canonical cell IDs stay null. |
| Earlier Day 8 and Day 9 drafts | Two Day 8 read-only reconciliation drafts and two Day 9 preparation failures are included in the 386-file source index | Day 8 drafts had incomplete checks or incorrect symbolic matcher expectations; Day 9 drafts failed before validation. None replaced a solver attempt or a final record. |

Frozen matrix bytes, helper/archive bytes, 32-bit project/CHOLMOD_INT ABI, requested one-thread variables, factor controls and ten file-backed loaded dependencies were checked in the prior execution bundles. Day 10's host fingerprints match Day 5/7. Complete bottle build/module settings, an approved Apple BLAS/LAPACK shared-cache identity, effective BLAS thread observation, first-side execution-time host proof and one stable first-versus-repeat dirty-patch key remain unresolved. An OpenMP team-size-one probe is **not** an observation of Accelerate BLAS threads. No timing samples were taken.

## Adversarial validator transcript

The retained `mutation-transcripts.json` tests a **synthetic, fully staged valid v1 observed fixture** as a positive control, then independently mutates one condition per case and invokes the production CLI with the checked-in schema. This checks the validator's behavior without representing any synthetic record as external evidence. The six altered records and their staged local artifacts remain in `mutations/`. Both **actual** blocked records were separately revalidated.

| Case | CLI exit and decisive diagnostic |
| --- | --- |
| Positive synthetic observed control | `0`, valid `correctness_observed`. |
| Missing baseline raw side | `1`, `artifact_path_or_hash /artifacts/1/path`. |
| Changed loaded library bytes | `1`, `artifact_path_or_hash /artifacts/3/path` (SHA-256 differs). |
| Bad baseline index ABI | `1`, `unsupported_enum /cell/problem/baseline_index_abi`. |
| Nonfinite project residual | `1`, `nonfinite_or_missing_value_abuse /correctness/project/residual_inf`. JSON parsing stops before a record ID can be read. |
| Well-formed but stale canonical cell ID | `1`, `cell_id_mismatch /cell_id`. |
| Unsupported portable-speed claim scope | `1`, `unsupported_enum /claim_scope`. |
| Actual first and repeat blocked records | Both `0`, valid with `evidence_status=blocked`. |

The first local Day 11 mutation draft is also retained as `day11-review-z_xapbt6/`; its stale-ID mutation used a malformed string and was rejected at syntax rather than semantic identity comparison. The final `day11-review-ftih1s_0/` run uses a syntactically valid stale SHA-256 ID and reaches `cell_id_mismatch`. This corrected the **test input**, not the validator or any solver output. `mutation-transcripts.json` retains exact argv, exit codes and raw stream digests for every final case.

## Retention and privacy boundary

`privacy-scan.json` records indicator counts and file names without publishing raw excerpts. It found `/Users/` paths in **12 raw stderr files**, chiefly dyld traces and their retained copies; no tested token prefix was seen. The scan is only an indicator search, not a complete privacy clearance. All raw bytes must remain in ignored `build/` until a reviewer explicitly approves exact artifact scope, redaction and licenses. If disclosure is approved, create a separately hashed derivative; preserve the original local evidence and its hashes. The local index includes retained helper, CHOLMOD dylib and dependency metadata, but its presence does not grant redistribution rights.

Item 222.4's planned 24-hour validation and repeat scope is accounted for by the Day 9 record attempt, Day 10 repeat and this Day 11 audit. That is planned scope, not elapsed time or a complete observed v1 cell. The missing provenance remains a blocked outcome, and the review packet must keep numerical validation separate from independent rights/wording approval.
