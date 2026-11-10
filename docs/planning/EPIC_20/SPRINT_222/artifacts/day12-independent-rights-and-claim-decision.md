# Sprint 222 Day 12: Rights and Claim Decision

## Decision

**No publication of the external comparison or local evidence bundle.** This is a documented disposition while independent review and v1 execution provenance remain incomplete. It is **not** an independent rights approval or a finding that any license forbids release. `publication_review=pending`, fixture `attribution_review=pending`, approved external claim `null`, and approved artifact allowlist empty. The separate machine-readable [Day 12 disposition](day12-publication-disposition.json) has SHA-256 `be288c15ee87ae5f313fe6c105f9dbeea592fce440acfe96da974e530b549c62`. It identifies the exact review scope and leaves independent reviewer fields `null`; Codex prepared the evidence record as a steward and cannot supply a maintainer or rights reviewer decision.

The Day 11 [exact-bundle review packet](day11-rights-and-wording-review-packet.md) remains the requested reviewer handoff. Its local source index SHA-256 is `392c6c4336718e94b2ca6a60f9a8c5f0d2a62cf105a1a37457cce949853d82bd` (386 Day 3–10 files); its review manifest SHA-256 is `c40d01d46839b2660c95f836de1044e50a6fafda8947d81807f6b92d314b6205` (39 payloads). The exact first and repeat v1 blocked records have SHA-256 `515e4264d0d542a2a5d33337bbd5599a799ef64fe4d15410fb590c94152f0236` and `270fd30358237993073f975e1db643b53ab02250e21784a941b3ec09904773c6`. The disposition's scope is limited to those bytes and the frozen `symmetric_4.mtx` fixture SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. Any changed bundle, redacted derivative or claim needs a separately hashed review.

## Evidence recheck and separation of gates

| Gate | Day 12 check | Disposition |
| --- | --- | --- |
| Index integrity | Re-read and SHA-256 checked all 386 source files and all 39 review payloads against their indexed sizes and digests. | PASS for retained local bytes; this is not approval to disclose them. |
| First v1 record | Ran `python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --input build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day9-v1-record-_tcqsm4j/blocked-record.json --bundle-root build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day9-v1-record-_tcqsm4j`; exit `0`, `valid record_id=s222-d9-provenance-blocked evidence_status=blocked`. | Valid blocked record, no canonical cell ID or observed external result. |
| Repeat v1 record | Ran the same validator/schema with `--input` and `--bundle-root` pointing to `day10-v1-validation-yqlyl1z8/repeat-blocked.json` and its directory; exit `0`, `valid record_id=s222-d10-repeat-blocked evidence_status=blocked`. | Valid blocked record; the source patch differs from the first pair. |
| Local numerical work | Day 8 first pair: 148 passing checks; Day 10 repeat: 219 passing checks. Both local pairs met residual and oracle limits and the paired solution-difference gate. | Numerical pass does not change v1 claim or rights status. |
| Independent reviewer | No signed or attributable decision naming an exact bundle, approved files, attribution and sentence is present in the branch or review packet. | Reviewer identity `null`; approval unavailable. |

The first and repeat observed candidates were rejected at `/environment`, and both accepted records carry `cell_id=null`. Missing CHOLMOD bottle build/module settings, approved Apple BLAS/LAPACK identity, effective BLAS thread observation, Day 6 execution-time host proof and differing dirty-patch hashes prevent a complete one-cell identity. These are execution evidence gaps, separate from rights. No timing protocol was executed; the retained wall durations are only command metadata.

## Attribution, rights and retention decision

The repository history establishes that Jeffrey Horn introduced the 114-byte `tests/data/symmetric_4.mtx` fixture in commit `aa1c04eab3536031b9a92ee92587260520ce1913`, and the repository has an MIT license notice. It does **not** independently establish authorship of every matrix value or approved public attribution wording. The maintainer attribution decision remains **pending**.

The selected local CHOLMOD 5.3.4 library is from SuiteSparse 7.12.2. The [Day 2 rights ledger](day2-rights-retention-and-attempt-layout.md) and [Day 4 runtime inventory](day4-cholmod-runtime-identity.md) identify loaded file paths and module/license questions. Exported optional-module symbols do not prove which terms apply to the exact installed binary; requested simplicial execution does not settle compiled-module composition. SuiteSparse_config, OpenMP, GCC runtime and any optional modules require an exact component and notice review. The 12 raw stderr files flagged with home-directory paths need manual privacy review before any proposed disclosure. No independent rights reviewer or approved release allowlist is recorded. These facts warrant a local hold; they are not a legal conclusion about what could eventually be distributed.

The original raw streams, binaries, candidate records, rejected candidates, mutation transcripts and manifests remain under ignored `build/external-baseline/`. Do not upload them, attach them to a PR, add them to a selected report, or copy them into generated API documentation under the current disposition. A metadata-only or redacted publication would be a **new exact artifact** requiring its own digest and approval. The Day 12 JSON itself records the hold and may be kept in planning docs; it does not confer release rights on the indexed evidence.

## Wording decision

**Current status wording, suitable for internal planning:**

> Sprint 222 retained local numerical and validation evidence for a single candidate 4×4 project/CHOLMOD comparison. Required v1 execution provenance and publication review are incomplete; the records remain blocked, with no approved external correctness or performance result.

The Day 11 one-cell correctness sentence is a **conditional proposal only**. It cannot be approved on Day 12 without a complete v1 observed cell and a separate attributable decision for the exact disclosure. No speedup, ranking, winner, hosted, cross-platform, state-of-the-art or broad solver-parity claim is authorized. The no-publication disposition applies even though the local numerical checks passed.

## Required reviewer handoff

The repository maintainer must confirm matrix origin, exact attribution and permitted fixture-copy terms. A maintainer-designated rights reviewer must name the exact index/manifest digests, assess loaded binary/module/dependency obligations and flagged raw streams, specify an artifact allowlist or denial, and record their identity, date and exact approved wording. A future evidence owner must also supply a validated observed v1 record and canonical cell ID before any one-cell correctness statement can be promoted. If review is unavailable, keep the JSON statuses pending and this no-publication disposition in effect; do not infer approval from elapsed time or from the numerical pass.
