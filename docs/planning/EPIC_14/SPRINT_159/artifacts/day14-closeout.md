# Day 14 Closeout

## Scope

Day 14 closes Sprint 159 by recording the final validation set, promoted and
demoted row decisions, retrospective inputs, and Sprint 160 handoff readiness.

No C or public-header files were modified.

## Final Deliverable Summary

Sprint 159 promoted selected local generated report freshness checks into a
reviewed Linux hosted evidence path without broadening solver, platform,
package, performance, ABI, external-library parity, or state-of-the-art claims.

Final deliverables:

- `generated-report-freshness` job in `.github/workflows/ci.yml`;
- split hosted artifacts for selected oracle and QR minimum-norm comparison
  freshness;
- tightened selected-row normalizer semantics;
- focused selected comparison normalizer tests;
- aligned README, maintainer, corpus, and solver-selection documentation;
- complete Sprint 159 working notes and Day 1-14 artifacts.

## Promoted Row Closeout

| Surface | Command | Promoted evidence | Artifact group | Closeout status |
| --- | --- | --- | --- | --- |
| Selected QR oracle rows | `make report-index-oracle-freshness` | Reviewed Linux hosted execution, deterministic summary, split oracle artifact upload | `sprint159-oracle-freshness` | Promoted as fixture-local selected hosted freshness evidence. |
| Selected partial-SVD oracle rows | `make report-index-oracle-freshness` | Reviewed Linux hosted execution, deterministic summary, split oracle artifact upload | `sprint159-oracle-freshness` | Promoted as fixture-local selected hosted freshness evidence. |
| Selected QR minimum-norm comparison rows | `make report-index-comparison-freshness` | Reviewed Linux hosted execution, deterministic summary, split comparison artifact upload | `sprint159-comparison-qr-minnorm` | Promoted as fixture-local selected hosted comparison freshness evidence. |
| Oracle generated-reference rows | `make report-index-oracle-freshness` | Uploaded as context with selected oracle outputs | `sprint159-oracle-freshness` | Supplemental context only; not primary claim evidence. |

## Demoted Or Unpromoted Row Closeout

| Surface | Final Sprint 159 status | Reason |
| --- | --- | --- |
| Broad report-index freshness | Unpromoted advisory/local | Too broad; includes families outside selected hosted claim policy. |
| `build/report-index/normalized-index.tsv` | Not uploaded | Navigation output only; not selected evidence. |
| Benchmark, coverage, dead-code, sentinel, guardrail, package, CI metadata, documentation, and runtime-backend rows | Unpromoted | Not selected oracle/comparison freshness evidence. |
| Optional NumPy/SciPy comparison dependency rows | Context only | Deferred optional dependencies cannot create pass evidence. |
| macOS/Windows report-index parity | Unpromoted | Sprint 159 intentionally uses Linux reviewed hosted evidence only. |
| Package, ABI, shared-library, dynamic-loader, package-manager, and performance proof | Out of scope | Existing package/performance lanes remain separate evidence surfaces. |

## Final Validation Record

Final Day 14 validation passed:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
python3 tests/test_normalize_report_index.py
make docs-check
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml"); puts "ci.yml YAML parse ok"'
git diff --check -- .github/workflows/ci.yml README.md docs/maintainer_guide.md docs/solver_selection.md scripts/normalize_report_index.py tests/corpus/README.md tests/test_normalize_report_index.py docs/planning/EPIC_14/SPRINT_159
rg -n "[ \t]+$" .github/workflows/ci.yml README.md docs/maintainer_guide.md docs/solver_selection.md scripts/normalize_report_index.py tests/corpus/README.md tests/test_normalize_report_index.py docs/planning/EPIC_14/SPRINT_159
```

Observed final selected freshness status:

- oracle normalizer output: `54` rows, selected generated rows report `fresh`;
- comparison normalizer output: `7` rows, six selected comparison rows report
  `fresh`;
- focused normalizer tests report `test-normalize-report-index: ok`;
- `docs-check` reports API docs coverage `PASS`;
- workflow YAML parses;
- diff and trailing-whitespace hygiene pass.

## Quality-Gate Selection

Changed files are workflow, docs, Python script, and Python focused tests.
No `.c` or `.h` files are modified in the Sprint 159 working tree, so the
required C/header gate `make format && make lint && make test` is not required
for this day. Earlier Day 10 additionally ran `make lint` successfully after
the normalizer change.

## Claim Wording Review

Day 14 reviewed the final claim boundary:

- README identifies only the selected oracle and QR minimum-norm comparison
  gates as mirrored by reviewed Linux hosted CI;
- maintainer guide documents current selected-row `fresh` semantics and
  selected-row error states;
- corpus README keeps broad report-index and local-only families out of hosted
  claims;
- solver-selection docs mention reviewed Linux hosted freshness only for the
  selected evidence lane and continue rejecting broad parity/platform claims;
- Sprint artifacts preserve the distinction between generated row metadata and
  reviewed Linux hosted execution.

## Retrospective Inputs

Use these points when drafting `RETROSPECTIVE.md`:

### What Shipped

- A reviewed Linux hosted freshness job for selected oracle/comparison rows.
- Split oracle and comparison artifacts with 7-day retention and strict
  missing-file behavior.
- Deterministic hosted summaries for selected row counts, pass counts, commit,
  branch, support tier, fixture, and optional dependency context.
- Normalizer semantics that report selected current generated rows as `fresh`
  and fail missing/stale/invalid selected evidence clearly.
- Focused comparison selected-row tests.
- Public and maintainer documentation aligned to the selected hosted evidence
  boundary.

### What Stayed Out

- Broad report-index freshness.
- Broad QR or partial-SVD correctness.
- Broad external-library parity.
- macOS/Windows report-index parity.
- Package, shared-library, ABI, dynamic-loader, package-manager, performance,
  release, and state-of-the-art claims.

### Validation Highlights

- Selected oracle freshness passed locally.
- Selected comparison freshness passed locally.
- Normalizer tests passed locally.
- Documentation/API docs check passed locally.
- Workflow YAML and whitespace hygiene passed.

### Residual Risks

- Hosted Ubuntu may reveal environment-specific generator or summary behavior
  not seen locally; CI must pass before merge.
- Generated row metadata remains `local_only`; docs now explain that Sprint
  159 promotes reviewed Linux execution, not source metadata support-tier
  reclassification.
- Optional NumPy/SciPy comparison dependency defers remain context only.

### Sprint 160 Handoff

Start with one additional QR comparison fixture family. Prefer an
overdetermined compatible QR least-squares fixture with residual and solution
checks against the source-controlled dense helper. Require exact selected row
IDs, normalizer tests, runtime evidence, split artifacts, and non-claim wording
before hosted promotion.

## Completion Check

- Sprint 159 deliverables are complete and traceable.
- Validation status is recorded with exact commands.
- Promoted and unpromoted row decisions are explicit.
- Retrospective input set is ready.
- Sprint 160 QR comparison handoff is ready.
