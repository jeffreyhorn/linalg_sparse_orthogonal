# Day 3: Platform Gap Matrix

## Purpose

Classify selected generated report freshness paths across Linux, macOS, and
Windows; separate local generation from hosted publication; identify blocker
classes; and rank candidate promotion or deferral lanes before Sprint 175 makes
a Day 4 decision.

## Classification Legend

| Status | Meaning |
| --- | --- |
| reviewed hosted | CI runs the selected freshness lane on the platform and uploads selected artifacts with documented claim boundaries. |
| reviewed local/package | CI validates related package/build behavior on the platform, but not generated report freshness. |
| local-only | Maintained command exists for local use; generated output is ignored and not hosted evidence. |
| staged | Candidate lane exists but needs path, CI, artifact, or claim-boundary work before promotion. |
| blocked | Known assumptions prevent direct promotion without implementation or formal deferral. |
| unselected | Not a Sprint 175 candidate for report freshness promotion. |

## Cross-Platform Report Freshness Matrix

| Report freshness path | Linux | macOS | Windows | Current support-tier interpretation |
| --- | --- | --- | --- | --- |
| Selected oracle freshness: `make report-index-oracle-freshness` | reviewed hosted for selected QR/partial-SVD oracle artifacts | staged/local-only; no macOS report-freshness lane | blocked/staged; Make/POSIX shell assumptions and no Windows report-freshness lane | local generated rows plus reviewed Linux hosted mirror only |
| Selected comparison freshness: `make report-index-comparison-freshness` | reviewed hosted lane exists, but hosted summary/upload inventory is stale for Sprint 174 LU | staged/local-only; no macOS report-freshness lane | blocked/staged; Make/POSIX shell, temp probe, compiler invocation, and executable behavior need audit | selected local QR/partial-SVD/LU rows; hosted Linux claim currently needs LU artifact reconciliation |
| Selected canonical benchmark freshness: `make bench-canonical-report-freshness` | reviewed hosted selected-performance lane | staged/local-only; timing metadata semantics differ | blocked/staged; Make/POSIX shell and benchmark executable assumptions need audit | selected threshold-free benchmark methodology freshness, not portable performance proof |
| Generated API freshness: `make api-docs-freshness` | local-only generated API HTML | local-only candidate if Doxygen/Bash present | blocked/staged; Bash/Doxygen/path behavior needs audit | generated API docs local-only; not report freshness evidence |
| Coverage report: `make coverage` | local-only/advisory | local-only/advisory with backend differences | blocked/staged; coverage tooling differs | advisory coverage context, not platform freshness proof |
| Dead-code report: `make deadcode-report` | local-only/advisory | local-only/advisory | local-only/advisory | advisory maintainer report |
| Sentinel/guardrail reports | local-only/advisory or hard local guard by family | local-only/advisory | local-only/advisory | local performance/governance context, not cross-platform freshness proof |
| Normalized report index: `python3 scripts/normalize_report_index.py` | platform-neutral when generated artifacts exist | platform-neutral when generated artifacts exist | platform-neutral when generated artifacts exist | navigation/freshness aggregation, not pass evidence by itself |

## Local Generation Versus Hosted Publication

| Evidence type | Local generation support | Hosted publication support | Sprint 175 rule |
| --- | --- | --- | --- |
| Source-controlled commands and tests | Make targets, Python tests, normalizer checks, shell guards | not hosted evidence unless executed by a reviewed CI lane | cite as local proof only |
| Ignored generated artifacts | `build/`, `coverage/`, and `docs/api/` outputs regenerated locally | uploaded artifacts only when workflow explicitly includes paths | do not stage ignored artifacts |
| CI lane definition rows | `.github/workflows/*.yml` source-controlled comments and steps | reviewed hosted evidence only after the lane runs and uploads selected artifacts | keep claim boundary tied to selected job |
| Manifest rows | owner, command, support tier, artifact pattern, non-claims | not execution proof by itself | use as ownership record |
| Normalized report index | can read current generated rows and source-controlled rows | not release proof or broad platform proof | use after underlying generators run |

## Blocker Taxonomy

| Blocker class | Applies to | Concrete examples from Day 2/Day 3 |
| --- | --- | --- |
| shell | Windows, sometimes macOS | Make targets and helper scripts assume POSIX shell or Bash; Windows CI is CMake/PowerShell-first. |
| path | Windows, macOS | Generated paths under `build/` are portable in Python, but Make recipes and shell scripts may assume POSIX separators. |
| compiler | comparison, oracle, benchmark | Comparison probes and oracle generation build against the static library; Windows evidence is CMake-first rather than Make-first. |
| dependency | API docs, coverage, benchmarks | Doxygen, lcov/gcov/gcovr, benchmark binaries, and optional helper tools may not be installed on every hosted runner. |
| executable | Windows | Temporary probes and benchmark binaries may need `.exe` handling and CMake-produced output paths. |
| newline/encoding | report parsers | TSV/Markdown parsing uses Python `newline=""` in key places, but shell scripts and generated manifests need platform audit before promotion. |
| temp directory | comparison probes | `scripts/run_external_comparison.py` uses temporary project probes that need platform-safe compile/run behavior. |
| generated-output staging | all generated reports | `build/`, `coverage/`, and `docs/api/` outputs are ignored local artifacts unless explicitly uploaded. |
| CI permission/artifact | hosted promotion | Hosted support requires workflow steps, selected artifact upload paths, retention, and failure behavior. |
| claim wording | all promotions | Docs must not broaden Linux-only hosted evidence into macOS/Windows, package, ABI, performance, release, or state-of-the-art claims. |

## Candidate Promotion Or Deferral Ranking

| Rank | Candidate lane | Closure value | Risk | Reasoning |
| ---: | --- | --- | --- | --- |
| 1 | Linux hosted selected comparison reconciliation for Sprint 174 LU | high | low | Existing Linux job already runs `make report-index-comparison-freshness`, but summary/upload inventory omits `lu-nonsym-square-5`; correcting this closes a real freshness-publication mismatch. It is not cross-platform beyond Linux, but it is the clearest complete closure. |
| 2 | macOS selected comparison freshness lane | high | medium | macOS already has reviewed static-first package/install proof and likely supports Make/Python/C compiler flow; a supplemental or reviewed comparison freshness lane could be feasible after path/compiler audit. |
| 3 | macOS selected oracle freshness lane | medium | medium | Similar feasibility to comparison, but lower Sprint 175 value because comparison has a known stale hosted inventory after Sprint 174. |
| 4 | Windows selected comparison formal deferral with blockers | high | medium | Windows has reviewed CMake-first evidence but Make/POSIX/temp-probe assumptions make direct promotion risky; a formal blocker record may be the right complete closure if Day 4 rejects implementation. |
| 5 | Windows selected comparison CMake-first promotion | very high | high | Valuable but likely too broad for safe closure without designing CMake-native report freshness generation or portable probe execution. |
| 6 | selected canonical benchmark freshness on macOS/Windows | medium | high | Timing metadata and executable/tooling differences make this risky, and it can accidentally imply portable performance. |
| 7 | generated API freshness lane | low for Sprint 175 | medium | Useful docs freshness, but Sprint 175 is report freshness promotion; generated API HTML remains local-only under Sprint 173. |
| 8 | coverage/deadcode/sentinel/guardrail platform promotion | low | high | Advisory/local rows and tool differences make these poor candidates for complete Sprint 175 closure. |

## Day 3 Recommended Decision Inputs

Day 4 should choose one of these two complete-closure paths:

1. **Reconcile Linux hosted selected comparison freshness for Sprint 174 LU.**
   - This closes a concrete mismatch: the Make target and required selected
     rows include LU, but hosted summary/upload inventory still lists three
     comparison targets.
   - It should be framed as Linux hosted selected comparison reconciliation,
     not macOS or Windows promotion.

2. **Select macOS selected comparison freshness promotion.**
   - This better matches the Sprint 175 "beyond Linux" wording.
   - It requires more implementation risk: workflow lane, artifact inventory,
     runner tool assumptions, path behavior, and support-tier documentation.

If Day 4 rejects both, the most defensible formal deferral is Windows selected
comparison freshness with explicit blockers: Make/POSIX shell, CMake-first
Windows support model, temporary C probe executable handling, generated output
paths, and artifact upload policy.

## Day 3 Completion Record

- Selected report freshness paths are classified across Linux, macOS, and
  Windows.
- Local-only generation is separated from hosted publication.
- Blocker classes are explicit before implementation.
- Candidate promotion or deferral lanes are ranked by closure value and risk.
- No support tier has been promoted by Day 3.
