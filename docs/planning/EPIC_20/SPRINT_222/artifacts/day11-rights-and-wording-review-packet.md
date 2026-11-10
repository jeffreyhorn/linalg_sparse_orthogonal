# Sprint 222 Day 11: Exact-Bundle Rights and Wording Review Packet

## Decision requested for Day 12

**Current disposition: no publication.** The local first and repeat pairs pass numerical checks, but neither has a valid `correctness_observed` v1 cell, and no rights or attribution reviewer has approved release of the exact evidence. This packet prepares a decision; it records no approval and sends no bundle outside the ignored workspace. The responsible repository maintainer should identify the fixture author/source and approve or deny attribution and proposed wording. A rights reviewer designated by the maintainer should evaluate the exact proposed disclosure, loaded SuiteSparse binary/modules, transitive notices, retained raw streams and any redaction. Their written decision must identify the reviewer, exact bundle digest, allowed artifacts, wording and date. A generic license opinion would not authorize another bundle.

**Exact local review target:** `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day11-review-ftih1s_0/`. The 386-file Day 3–10 source index has SHA-256 `392c6c4336718e94b2ca6a60f9a8c5f0d2a62cf105a1a37457cce949853d82bd`. The 39-payload review manifest has SHA-256 `c40d01d46839b2660c95f836de1044e50a6fafda8947d81807f6b92d314b6205`. The Day 9 first-record manifest SHA-256 is `234bd44e3421c876ba7345de3876f5e2dfa6cea2418fb4616d39bd5df961677b`; the Day 10 repeat and validation manifest SHA-256 values are `bfc37b2c6419e7c69b1b964d6327d75c07e5e924a387ca44dc3b0ecde33dcb7e` and `cfdd54715c4f8484406fb5ef30ace6d7b6f989a0b9686015b13828904c59838b`. See the [Day 11 validation ledger](day11-validation-retention-ledger.md) for the exact checks and mutation transcript. These hashes fix the **local review scope**; any later evidence or redacted derivative needs a new digest and decision.

## Evidence and claim eligibility

| Gate | Evidence | Decision state |
| --- | --- | --- |
| Numerical correctness | Both separate local project and CHOLMOD solves passed residual and oracle gates; paired infinity difference was `2.220446049250313e-16` against `2.760966058953532e-13`. The repeat matched both original solutions and metrics. | Local numeric pass only. |
| V1 execution identity | Both actual observed candidates were rejected at `/environment`; accepted first and repeat records are `blocked` with null cell IDs. Changed dirty-patch digest makes the repeat a different potential v1 cell. | **Blocked** for an observed one-cell claim. |
| Timing/performance | No timed trial or validated timing window. Command durations were captured only as execution metadata. | No speed, rank, winner, hosted, cross-platform or state-of-the-art claim. |
| Fixture origin | `tests/data/symmetric_4.mtx`, 114 bytes, SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`, introduced by Jeffrey Horn in `aa1c04eab3536031b9a92ee92587260520ce1913`; repository MIT notice exists. | Independent authorship/source confirmation and exact attribution wording **pending**. |
| External code and binary rights | Loaded Homebrew SuiteSparse 7.12.2 / CHOLMOD 5.3.4 dylib SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`. Day 4 recorded exported optional-module symbols and ten non-system dependencies. Installed license bundle and tagged upstream sources were inventoried on Day 2. | Exact compiled modules, optional GPL code, SuiteSparse_config, OpenMP and GCC runtime terms/notice obligations **pending**. Requested simplicial mode alone does not decide compiled-module rights. |
| Privacy/retention | Raw outputs and copied binary bytes stay under ignored `build/`. Indicator scan found home-directory paths in 12 raw stderr files. | Manual raw-stream review, redaction decision and artifact-level release scope **pending**. |

## Wording for review

**Proposed one-cell correctness-only wording, conditional on a *future* complete v1 observed record and separate exact-bundle rights approval:**

> On the single 4×4 symmetric positive-definite repository fixture `tests/data/symmetric_4.mtx`, the project linked-list Cholesky path and a locally installed SuiteSparse CHOLMOD 5.3.4 simplicial path each returned a solution within the stated residual and oracle tolerances under the recorded local configuration. This observation applies only to the identified local cell and does not establish timing or broader solver parity.

This text is **not approved for publication now**. Even if fixture attribution and binary terms are approved, the missing environment, baseline build/module and BLAS/thread provenance prevents its presentation as a validated v1 `correctness_observed` cell. If the reviewer authorizes only metadata publication, the statement and exact metadata derivative must still be reviewed as a new hashed artifact. No raw binary or trace is implicitly included.

**Current no-publication wording:**

> Sprint 222 retained local numerical and validation evidence for a single candidate 4×4 project/CHOLMOD comparison. Required v1 execution provenance and publication review are incomplete; the records remain blocked, with no approved external correctness or performance result.

The proposed public wording makes no assertion that a 4×4 example represents other matrices, platforms, installations, solver modes or performance. The repeat does not cure the original dirty-patch identity difference. Do not use a ratio, speedup, winner, hosted threshold or “state of the art” formulation in either disposition.

## Reviewer checklist and decision record fields

1. **Maintainer attribution:** Confirm independent creation or identify any external matrix source; approve exact credit and MIT notice handling for any fixture copy. Record reviewer name and date.
2. **Binary and component rights:** Establish actual installed CHOLMOD build flags/module composition, including GPL modules and optional dependencies; review exact loaded file hashes, SuiteSparse_config, libomp, GCC runtimes and applicable notices/source obligations. Specify whether binaries, raw traces, metadata only, or none may be shared.
3. **Privacy:** Inspect the exact 12 flagged raw stderr files and all other proposed files for home paths, host identifiers or secrets. Approve an exact original or separately hashed redacted derivative; preserve the original local evidence.
4. **Claim and provenance:** Verify a **new** complete observed v1 record and canonical cell ID before approving the conditional claim. Record the exact approved sentence, or the no-publication decision and reason. Rights approval and numerical/provenance validation are independent decisions.
5. **Decision artifact:** Store reviewer identity, review date, selected index/manifest digests above, artifact allowlist, approved wording and any conditions in a separate hashed Day 12 artifact. If no independent reviewer is available, keep `publication_review=pending` and the no-publication disposition.

The packet itself is planning documentation and is safe to review inside the repository. It does not attach or copy the ignored local binary/raw evidence into a public report, PR or generated API documentation.
