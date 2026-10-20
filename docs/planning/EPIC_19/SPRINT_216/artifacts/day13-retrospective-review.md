# Sprint 216 Day 13: Retrospective Claim Review

## Decision And Scope

The [Epic 19 retrospective](../../EPIC_19_RETROSPECTIVE.md) is review-ready,
not an Epic closeout. **No state-of-the-art claim is earned.** This audit
compares its outcome, validation, lesson and non-claim statements with the
[Day 3 claim inventory](day3-claim-inventory.md), [Day 5 final claim matrix](day5-benchmark-api-corpus-claims.md),
[Day 6](day6-project-plan-status-207-211.md) and
[Day 7](day7-project-plan-status-212-215.md) item ledgers, the
[residual queue](../RESIDUAL_QUEUE.md), and the
[Day 10](day10-focused-validation.md) and
[Day 11](day11-quality-gate-and-retro-outline.md) current-tree records.
There is no new hosted artifact review or external solver execution on Day 13.
Sprint 216 item statuses remain unassigned until Day 14.

## Statement Audit

| Retrospective statement group | Cross-check and disposition |
| --- | --- |
| Objective, 54 prior-sprint selected-item dispositions, pending 216 closeout | Day 6 has 30 Sprint 207-211 items and Day 7 has 24 Sprint 212-215 items. The [project-plan snapshot](../../PROJECT_PLAN.md#epic-19-current-status-snapshot) distinguishes merged Sprint 215 from in-progress 216. `complete` means the selected decision, design or fallback, not the broader capability. C16/C17 stay bounded. |
| Sprint 207 package and Sprint 214 ABI/release outcomes | C05/C10 and the ledgers support only the local developer-mode Homebrew source proof, static installed consumers and future checklist. Day 11 reran static contracts, Make and CMake consumers locally. RQ-06/RQ-07/RQ-13/RQ-16 retain candidate, provider and shared-ABI gates. No release or public package claim. |
| Sprint 208/209 Windows outcomes | C13 and both ledgers distinguish historical inspected Cholesky files from unsupported selected freshness, and a QR workflow from an inspected hosted QR result. RQ-04/RQ-05/RQ-15 retain artifact and metadata checks. Neither Day 10 nor Day 11 reran hosted Windows jobs. |
| Sprint 210/211 reliability and review outcomes | Ledgers support one 25-site no-reorder linked-list LDLT failure owner and one 295-line LDLT CSC test extraction. RQ-09/RQ-10/RQ-18/RQ-19 retain broader work. Lessons about ownership and reviewability do not assert new solver behavior or broad reliability. |
| Sprint 212 benchmark outcome | C01-C03/C12 and Day 10 establish selected `--repeat 1`, `baseline=n/a`, `threshold=n/a` freshness only. RQ-08/RQ-12/RQ-21 retain timing methodology and publication decisions. No comparison or platform parity inferred. |
| Sprint 213 generated API outcome | C08/C09/C14/C15 and Day 10's 18-header local generation establish source-controlled routing and ignored local HTML. RQ-14/RQ-20 retain publication and first-use gates. The retrospective does not describe local generation as hosted publication. |
| Sprint 215 blueprint and blocked pilot | C04/C06/C07/C11/C12, [blocked attempt](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md), and RQ-01/RQ-02/RQ-03 agree: v1 validation unavailable, neither solver ran, no accepted `evidence.json`. Day 10's 13 synthetic tests and 40 source-controlled helper comparison rows are different local checks. No independent solver parity inferred. |
| Validation chronology and lessons | Day 10's 14 commands passed locally; Day 11's installed-static tests passed locally and `make clean` removed Day 10 generated reports. Historical Sprint 214 60/60 CTest is not a Day 11 rerun. C17 and the queue keep process lessons separate from capability acceptance. Day 14 still owns final integrated results. |

The review found no retrospective statement requiring a stronger support
claim. It expanded the state-of-the-art section to expose the exact blocked
cell, missing numerical and timing evidence, publication gate, and the
difference between bounded correctness and a broad assessment. It also
lists the final non-claims explicitly. No prior-sprint item disposition or
queue priority changed.

## State-Of-The-Art Evidence Assessment

| Dimension | Current status | Future gate |
| --- | --- | --- |
| Selected baseline | The [Sprint 215 blueprint](../../SPRINT_215/STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md) specifies one float64 natural-order 4-by-4 SPD solve against CHOLMOD. Local preflight found CHOLMOD `5.3.4`/SuiteSparse `7.12.2` and a candidate dylib hash, not actual loaded-library identity. | RQ-01 validates the v1 record; RQ-02 records actual runtime/transitive-library provenance and both solvers' raw outputs. |
| Validated matrix/platform coverage | Zero accepted external cells. The planned fixture hash was checked, but no solver-side conversion or numerical outcome was recorded. The preflight host was Darwin x86_64. Historical Windows selected Cholesky freshness evidence is unrelated to external-baseline comparison. | RQ-02 accepts one exact cell. RQ-11 broadens to separately selected systems, sizes, structures and independent baselines. RQ-12 adds named hosted classes. |
| Independent correctness and failure comparisons | Neither side ran; no solution, residual, backward error or matching failure behavior exists. Day 10 local comparison rows use project source-controlled dense helpers; optional NumPy/SciPy rows defer. | Complete the two-sided numerical gates for the named cell, then pre-register and validate representative SPD, nonsymmetric and rectangular families; include iterative/eigen/SVD only when claimed. |
| Timing and ranking | The pilot has `timing=not_run`; the selected canonical benchmark has `threshold=n/a` and no paired external samples. | RQ-12 requires controlled paired samples, warmups, raw streams, dispersion, host/binary provenance and per-trial correctness on at least two named hosted toolchain/platform classes for a broad assessment. RQ-17/RQ-18 address architecture and numerical breadth. |
| Publication permission | Exact fixture attribution, CHOLMOD-module/dependency license, artifact retention/access, approved text and rollback owner are not recorded as approved. | RQ-03 requires written review for any exact-cell publication. Broader claims need per-cell rights and a separate product/reviewer decision. |

The [queue decision](../RESIDUAL_QUEUE.md#state-of-the-art-decision) is the
promotion boundary. RQ-01/RQ-02/RQ-03 are prerequisites for **one bounded
correctness statement**, not sufficient evidence for external parity or
speed. RQ-11/RQ-12/RQ-17/RQ-18 describe the broader program. Multiple
representative matrix/operation classes, more than one size/structure per
family, independent pinned baselines, validated correctness and failures,
controlled comparative timing, at least two hosted classes (Windows if
claimed), complete retained provenance, rights, and explicit approval are
necessary before reconsidering state-of-the-art wording. Passing that entry
bar triggers review; it does not automatically earn a ranking.

## Reviewer Questions

1. Who owns approval of the v1 semantic validator and its positive/negative
   fixtures, and which immutable complete-cell ID will be accepted? Until
   answered, RQ-01 blocks even the tiny pilot.
2. Which exact matrix/RHS source, CHOLMOD module/dependency terms, retention
   policy and public wording can the corpus/publication reviewers approve?
   Until answered, an observed cell remains local-only.
3. Which solver families, operations, matrix structures/sizes and independent
   baselines are in scope for any proposed broad assessment? Do not use a
   single tiny SPD cell as a proxy for the project.
4. Which named hosted platform/toolchain classes and controlled timing
   protocol would support a proposed comparative claim, and is Windows
   explicitly in scope? Preserve per-cell raw artifacts and failure cases.
5. Who makes the final product/claim decision and owns rollback if a
   validated published bundle later fails provenance or reproducibility?

These are future gate questions, not missing inputs for today's no-claim
decision.

## Final Non-Claim List

- No public package provider, binary distribution, shared library, dynamic
  ABI compatibility guarantee, approved release candidate or release-ready
  status from local static-consumer checks.
- No promoted selected Windows Cholesky/QR freshness, broad Windows parity,
  broad allocation-failure coverage or repo-wide review-surface completion.
- No timing threshold, portable/comparative speed, hosted Linux/macOS timing
  parity, or published/retained generated API HTML.
- No accepted two-sided external correctness cell, independent-package
  parity, external-result publication, or state-of-the-art ranking.

## Day 14 Handoff

Rerun checks affected by the retrospective/Day 13 edits, verify all links,
then reconcile the final 216.1-216.6 statuses with the current tree and any
new evidence. Do not treat local docs checks or this review as hosted,
external, release or C-source validation.

## Day 13 Verification

On local Darwin x86_64, all checks below passed:

| Check | Result |
| --- | --- |
| Local relative-link existence scan of the retrospective, project plan, working notes and this audit | PASS, no missing targets. |
| `python3 -m unittest tests.test_external_baseline_claim_docs` | PASS, 5 tests. |
| `python3 tests/test_selected_performance_docs.py` | PASS. |
| `make support-docs-guard` | PASS. |
| `make docs-check` | PASS, 18 checked-in public headers with 18 generated reference and source pages each. Generated HTML remains ignored and local-only. |
| Trailing-whitespace scan of the same four documents; `git diff --check` | PASS. |
| Tracked and untracked `.c`/`.h` change inventory | None; the conditional `make format && make lint && make test` source chain was not triggered. |

These checks validate documentation consistency on this tree, not hosted
platforms, actual external solvers, package publication or release readiness.
