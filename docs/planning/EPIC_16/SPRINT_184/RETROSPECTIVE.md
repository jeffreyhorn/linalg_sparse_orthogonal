# Sprint 184 Retrospective

**Sprint:** 184 - Public Header Coherence Batch 3
**Duration:** 14 days (Days 1-14 landed on branch `sprint-184`)
**Status:** Complete

## Source Artifact Note

Sprint 184 was executed from the Epic 16 project-plan section for Sprint 184
and lives under `docs/planning/EPIC_16/SPRINT_184/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 184 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reviewed prior declaration-preserving header cleanup batches, Sprint 177
      residual evidence, and Sprint 164 header selection criteria.
- [x] Inventoried QR, SVD/partial SVD, and LDLT candidate public headers with
      related docs, examples, and test surfaces.
- [x] Captured declaration baselines for QR, SVD/partial SVD, and LDLT before
      selecting the Sprint 184 family.
- [x] Selected exactly one family for Sprint 184: `include/sparse_qr.h`.
- [x] Normalized QR lifecycle, ownership, error-code, tolerance, workspace,
      option/result, cancellation, rank, nullspace, diagnostics, and
      minimum-norm contract wording.
- [x] Organized QR declarations into coherent sections with bounded
      organization-only movement and declaration-set preservation evidence.
- [x] Aligned QR-facing README, tutorial, solver-selection, cookbook, API
      reference, and examples README wording with the cleaned header contract.
- [x] Added `scripts/check_qr_header_docs_guard.sh` and
      `make qr-header-docs-guard` to protect QR section headings, declaration
      tokens, unsupported-claim boundaries, and docs alignment wording.
- [x] Preserved non-claims for broad QR correctness, global rank policy, broad
      minimum-norm behavior, external-library parity, package/platform/ABI
      support, Windows report freshness, generated API publication,
      performance guarantees, release readiness, and state-of-the-art status.
- [x] Ran focused validation, example smoke checks, API docs validation,
      declaration-set checks, whitespace checks, and the full
      `make format && make lint && make test` gate after header changes.
- [x] Confirmed no implementation `.c` files were changed and no generated API
      HTML files were staged.

## What Went Well

1. **The sprint selected a single high-impact header family.** QR had the
   highest current combination of docs visibility, rank/minimum-norm claim
   sensitivity, and feasible declaration-preserving cleanup.

2. **The cleanup stayed API-local.** Header comments now clarify lifecycle,
   ownership, tolerance, workspace, diagnostics, cancellation, and
   minimum-norm behavior without changing signatures, structs, enum values,
   include guards, or implementation behavior.

3. **Declaration organization was bounded and reviewable.** Only
   `sparse_qr_free()` and the minimum-norm solve/refine declarations moved, and
   the sorted comment-stripped declaration-set diff against `HEAD` remained
   empty.

4. **Documentation was aligned without widening claims.** README, tutorial,
   solver-selection, cookbook, API reference, and examples README text now
   describe the same QR lifecycle and evidence boundaries as the cleaned
   header.

5. **COLAMD wording became clearer.** QR docs no longer imply AMD is the
   primary QR column-ordering path; `sparse_qr_factor_opts()` now points users
   at COLAMD for unsymmetric/QR workflows.

6. **The new guard covers the highest-risk drift points.** The QR guard checks
   section headings, declaration tokens, unsupported-claim absence, and
   required docs alignment text in one focused target.

7. **The full validation pass was run after header changes.** Day 13 completed
   `make format && make lint && make test`, `make qr-header-docs-guard`,
   `make api-docs-validate`, `git diff --check`, and the declaration-set diff.

## What Didn't Go Well

1. **The Day 2 line-numbered checksum was too sensitive to comment edits.**
   Comment cleanup shifted declaration line numbers, so later days switched to
   comment-stripped declaration comparisons as the more appropriate
   preservation check.

2. **The header organization change required extra evidence.** Moving
   declarations improves scanability, but it also increases review load. The
   sprint needed explicit before/after order notes and sorted declaration-set
   checks to keep the change defensible.

3. **The docs alignment surface was broader than the code change.** QR wording
   appeared in README, tutorial, solver-selection, cookbook, API reference, and
   examples README, so a small header cleanup still required careful docs
   review.

4. **Full lint remains expensive.** `make lint` runs strict compile,
   `clang-tidy`, and `cppcheck` across a large surface. It passed, but it is a
   long-running part of header-touching validation.

5. **SVD and LDLT remain unresolved header-coherence candidates.** Sprint 184
   intentionally selected QR only, so the other high-value families need future
   dedicated cleanup sprints rather than opportunistic edits.

## Final Metrics

### Validation

| Metric | Sprint 184 close state |
| --- | --- |
| QR header/docs guard | passed: `make qr-header-docs-guard` |
| API docs validation | passed: `make api-docs-validate` |
| docs checks | passed during focused documentation days |
| examples build | passed: `make examples-build` |
| QR least-squares example smoke | passed: `./build/example_least_squares` |
| QR minimum-norm example smoke | passed: `./build/example_minnorm` |
| QR COLAMD example smoke | passed: `./build/example_colamd` |
| formatting | passed: `make format` |
| lint | passed: `make lint` |
| full test suite | passed: `make test` |
| final `git diff --check` | passed |
| sorted comment-stripped QR declaration-set diff against `HEAD` | passed with no output |
| generated API HTML staged | 0 files |
| implementation `.c` files changed | 0 files |

### Changed Surface

| Metric | Sprint 184 close state |
| --- | ---: |
| selected public header families | 1 |
| candidate public header families inventoried | 3 |
| QR public header files changed | 1 |
| implementation `.c` files changed | 0 |
| QR declarations added | 0 |
| QR declarations removed | 0 |
| QR declarations with changed signature | 0 |
| bounded QR declaration moves | 3 |
| QR docs/example narrative files changed | 6 |
| guard scripts added | 1 |
| Makefile targets added | 1 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |

### Claim Governance

| Metric | Sprint 184 close state |
| --- | ---: |
| broad QR correctness claims added | 0 |
| global rank-policy claims added | 0 |
| broad minimum-norm behavior claims added | 0 |
| external-library parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI claims added | 0 |
| Windows report freshness promotions added | 0 |
| generated API publication claims added | 0 |
| portable performance guarantees added | 0 |
| release readiness claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 184 closes this Epic 16 public-header coherence claim:

The selected QR public header family has clearer lifecycle, ownership,
error-code, tolerance, workspace, option/result, cancellation, rank,
nullspace, diagnostic, and minimum-norm contract wording; its declarations are
organized into coherent sections without public declaration-set drift; and its
QR-facing documentation is aligned with a focused guard that protects the
cleaned contract and unsupported-claim boundaries.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-declaration-baseline.md](./artifacts/day2-declaration-baseline.md);
- [day3-family-selection-and-contract-map.md](./artifacts/day3-family-selection-and-contract-map.md);
- [day4-core-contract-cleanup.md](./artifacts/day4-core-contract-cleanup.md);
- [day5-advanced-contract-cleanup.md](./artifacts/day5-advanced-contract-cleanup.md);
- [day6-organization-guardrail-design.md](./artifacts/day6-organization-guardrail-design.md);
- [day7-coherent-header-sections.md](./artifacts/day7-coherent-header-sections.md);
- [day8-documentation-alignment-map.md](./artifacts/day8-documentation-alignment-map.md);
- [day9-example-contract-alignment.md](./artifacts/day9-example-contract-alignment.md);
- [day10-reference-documentation-alignment.md](./artifacts/day10-reference-documentation-alignment.md);
- [day11-mechanical-guard-implementation.md](./artifacts/day11-mechanical-guard-implementation.md);
- [day12-focused-validation-pass.md](./artifacts/day12-focused-validation-pass.md);
- [day13-full-validation-and-final-cleanup.md](./artifacts/day13-full-validation-and-final-cleanup.md);
- [day14-retrospective-ready-handoff.md](./artifacts/day14-retrospective-ready-handoff.md).

No broad QR correctness, global rank-policy, broad minimum-norm behavior,
external-library parity, package-manager support, shared-library ABI,
Windows report freshness, generated API publication, portable performance,
release readiness, or state-of-the-art claim was added.

## Sprint 185 Readiness

Sprint 185 should start from the next Epic 16 project-plan section. If it
continues public-header coherence work, use this Sprint 184 handoff:

| Future need | Sprint 184 handoff |
| --- | --- |
| Candidate selection | Start with a declaration baseline, docs surface map, and explicit claim-risk comparison before selecting one family. |
| Declaration preservation | Prefer comment-stripped declaration-set checks over line-number-sensitive checks for comment-heavy work. |
| Organization changes | Keep movement bounded, record before/after order, and require a declaration-set diff with no output. |
| Docs alignment | Audit README, tutorial, solver-selection, cookbook, API reference, and examples README before editing. |
| Guard pattern | Reuse the QR guard shape: section headings, declaration tokens, unsupported-claim absence, and required docs wording. |
| Validation | If any `.c` or `.h` file changes, run `make format && make lint && make test`; also run family guard, API docs validation, whitespace checks, and declaration-set comparison. |
| Non-claims | Continue rejecting broad correctness, global policy, external-parity, package/platform/ABI, Windows freshness, generated-publication, performance, release, and state-of-the-art claims unless a dedicated sprint proves them. |

Strong future public-header coherence candidates remain SVD/partial SVD and
LDLT. Each should be treated as a separate selected family with its own
baseline, contract map, guard coverage, documentation alignment, and full
validation pass.
