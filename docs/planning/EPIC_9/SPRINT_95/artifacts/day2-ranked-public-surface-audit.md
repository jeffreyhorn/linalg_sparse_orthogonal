# Sprint 95 Day 2: Ranked Public-Surface Audit

## Purpose

Day 2 turns the Day 1 inventory into a ranked cleanup queue. The ranking favors
reader impact first, then truth risk, implementation cost, and validation risk.
It also separates plain prose cleanup from proof-owner and target-name work that
can affect build or test orchestration.

## Ranking Rules

| Factor | Meaning |
|---|---|
| Reader impact | How likely the surface is to shape a public user's first understanding or daily workflow. |
| Truth risk | How likely stale chronology or duplicate wording is to misstate current maintained behavior. |
| Implementation cost | How much edit coordination is required to land the cleanup. |
| Validation risk | Whether the cleanup may require link checks, generated-doc awareness, build target parity, or full code quality checks. |

## Ranked Findings

| Rank | Finding | Surfaces | Problem class | Risk profile | Sprint 95 action |
|---:|---|---|---|---|---|
| 1 | README front-door overload | `README.md` | Sprint chronology, duplicated adoption path, repeated support and install story. | High reader impact, medium truth risk, low implementation risk if prose-only. | Fix now. Day 4 defines boundary; Day 5 lands cleanup. |
| 2 | Public API comments expose development history | `include/sparse_matrix.h`, `include/sparse_types.h`, `include/sparse_qr.h`, `include/sparse_svd.h`, `include/sparse_eigs.h` | Public comments explain sprint/day origin instead of stable API behavior. | High reader impact, medium truth risk, high validation requirement because headers are code. | Fix now, but only during header cleanup day with full quality checks. |
| 3 | Install/support workflow repeats across public docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `Makefile` | Same reviewed-platform, install, support, and quality-review story appears in several places with different levels of detail. | High reader impact, medium truth risk, low to medium implementation risk. | Fix now after Day 3 ownership model. |
| 4 | Algorithm reference reads like a sprint ledger | `docs/algorithm.md` | Dense sprint-by-sprint chronology in a permanent technical reference. | Medium reader impact, high truth/context risk, medium implementation cost. | Fix now for the worst public sections after audience rules define provenance policy. |
| 5 | Proof owners are named by sprint rather than product behavior | `tests/test_sprint*_integration.c`, Makefile, `CMakeLists.txt`, README test lists | Filenames, suite names, and build target names require sprint numbers to understand coverage ownership. | Medium reader impact, low truth risk, high validation risk. | Fix now only for highest-value cases on Day 10-11; keep broader renames residual. |
| 6 | Benchmark story mixes measurement ownership with sprint evidence | `benchmarks/README.md`, `benchmarks/*.c`, README benchmark sections, Makefile benchmark targets | Benchmark docs and drivers expose sprint slices, sprint-era provenance, and repeated proof-owner explanations. | Medium reader impact, medium truth risk, medium validation risk. | Fix now for public docs/help wording; defer target/option renames unless proven low risk. |
| 7 | Tutorial and examples duplicate workflow ownership | `docs/tutorial.md`, `examples/README.md`, `examples/*.c`, README quick-start sections | Repeated explanation of one-shot, repeated-run, benchmark, and install boundaries. | Medium reader impact, low truth risk, low implementation risk if prose-only. | Fix now as part of tutorial and example cleanup. |
| 8 | Maintainer guide contains historical context that may be correct but too broad | `docs/maintainer_guide.md` | Maintainer-only history and proof ownership are useful, but some sections may become dumping grounds for every moved public note. | Low public reader impact, medium maintainer truth risk, medium implementation cost. | Residual except for ownership links needed by README/install/benchmark cleanup. |
| 9 | Matrix Market docs show low immediate narrative pressure | `docs/matrix_market.md` | No high-pressure sprint chronology found in Day 1 and Day 2 scans. | Low reader impact for Sprint 95 goals, low implementation risk. | Residual unless later link checks reveal duplication. |

## Fix-Now Queue

### 1. README narrative boundary and cleanup

- Define README as the concise adoption front door.
- Remove sprint-labeled feature history from permanent README sections.
- Collapse repeated install, support, benchmark, and maintainer-policy details
  into short links to owner surfaces.
- Keep current capability claims, but require a stable proof or owner link for
  each retained claim.

### 2. Audience ownership model

- Assign one owner for each major narrative:
  - adoption and first solve
  - tutorial and repeated-run learning path
  - install and package workflow
  - benchmark interpretation
  - maintainer quality and proof policy
  - API contract language
- Use the ownership model before rewriting README, tutorial, examples, install,
  benchmark, or maintainer surfaces.

### 3. Header and generated-doc source cleanup

- Rewrite touched public comments to describe stable API behavior.
- Remove sprint/day provenance from user-visible comments unless it is needed to
  understand compatibility.
- Treat `docs/api/html/**` as generated output; change headers first.
- Because headers are `.h` files, run `make format && make lint && make test`
  after these edits.

### 4. Install and support consolidation

- Keep `INSTALL.md` focused on operational setup, installed package shape, and
  install validation.
- Keep `docs/maintainer_guide.md` focused on quality policy, proof ownership,
  and maintainer-only interpretation.
- Keep Makefile comments and target descriptions stable, concise, and aligned
  with the docs.

### 5. Tutorial and example cleanup

- Make `docs/tutorial.md` the fuller learning path after README.
- Keep `examples/README.md` focused on choosing example binaries and local
  example behavior.
- Remove repeated benchmark/support policy explanations where links are enough.

### 6. Benchmark narrative cleanup

- Keep `benchmarks/README.md` as the benchmark-local command and interpretation
  owner.
- Remove sprint-closeout phrasing from public benchmark descriptions where the
  current behavior can be described directly.
- Preserve reproducibility details and evidence links when they are still useful
  to benchmark readers.

### 7. Highest-value proof-owner naming cleanup

- Pick a small set of sprint-named proof owners whose names appear in public
  docs or build/test orchestration.
- Rename or regroup only where the product-oriented name is clearer and the
  Makefile/CMake/test impact is bounded.
- Validate Makefile and CMake parity after any target or filename move.

## Residual Queue

- Broad rewrite of every sprint reference in `docs/algorithm.md`.
- Full repo-wide removal of sprint/day comments from internal tests.
- Renaming every `tests/test_sprint*_integration.c` file.
- Renaming benchmark CLI options such as `--sprint86-slice` unless a product
  replacement name is clearly better and compatibility is addressed.
- Hand-editing generated API HTML.
- Cleaning historical planning docs under `docs/planning/**`.
- Moving every maintainer-history note into a new archive; Day 3 should first
  decide whether the maintainer guide is the right owner for some of that
  history.

## Candidate Type Split

| Type | Candidates | Validation expectation |
|---|---|---|
| Rewrite-only docs | README, tutorial, examples README, install prose, benchmark README prose, maintainer guide prose. | Link/prose review; no full code quality chain unless code files change. |
| Generated-doc source cleanup | Public header comments feeding API docs. | Full `make format && make lint && make test` because `.h` files change. |
| Build/workflow comments | Makefile and CMake comments or command descriptions. | At least command/target sanity checks; full tests if target behavior changes. |
| Proof-owner renames | Test filenames, suite names, Makefile test source lists, CMake test registrations, README test lists. | Full build/test validation and Makefile/CMake parity checks. |
| Benchmark CLI or target changes | Benchmark driver option names, Makefile benchmark targets, README benchmark command references. | Compile benchmark drivers, run targeted command-help checks where applicable, and preserve compatibility if public options change. |

## Proof-Risk Notes

- `tests/test_sprint*_integration.c` files are not just prose. They are wired
  through Makefile source lists, explicit build rules, CMake registrations, and
  sometimes README coverage descriptions.
- Suite names printed by tests are also proof-owner labels. Renaming files
  without updating suite names would leave inconsistent public evidence.
- Windows and CMake subset references in README and INSTALL mention excluded
  sprint-named tests. Those references must be updated if any proof-owner names
  change.
- Benchmark target and option names may be used by CI, docs, or local operator
  habits. Treat public option names as compatibility surfaces, not just wording.

## Validation-Risk Notes

- Header cleanup modifies `.h` files and must run the full required quality
  chain.
- Proof-owner cleanup can change `.c`, Makefile, and CMake files; it should run
  full quality checks and, where possible, CMake test discovery parity.
- Docs-only rewrite batches should still check links and obvious stale anchors,
  but do not require the code quality chain unless code files are touched.
- Generated API HTML should not be edited directly. Any API-doc cleanup should
  flow from headers and regeneration, if this repo's doc workflow requires it.

## Day 2 Result

The Sprint 95 cleanup problem is now a ranked queue. The highest-value first
steps are README boundary, audience ownership, public header wording, support
surface consolidation, tutorial/example cleanup, benchmark narrative cleanup,
and a small proof-owner naming pass. Riskier broad renames and exhaustive
chronology removal stay residual unless a later day selects them deliberately.
