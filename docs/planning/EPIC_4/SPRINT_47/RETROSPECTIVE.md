# Sprint 47 Retrospective

**Sprint:** 47 — Benchmark CLI Modernization, Auxiliary Surface Safety & Example/Tooling Cleanup  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 47 baseline and auxiliary-surface scope captured before implementation
- [x] benchmark/example/tooling seam inventory refreshed against live code
- [x] bounded shared CLI parsing-helper design completed
- [x] landing/validation strategy for peer auxiliary surfaces completed
- [x] shared internal benchmark CLI parsing-helper seam landed
- [x] `bench_main` modernization batch landed
- [x] post-`bench_main` audit completed
- [x] reorder-mode / emitted-label parity cleanup landed
- [x] example safety audit completed
- [x] bounded example cleanup batch landed
- [x] bounded auxiliary tooling cleanup batch landed
- [x] touched benchmark/example docs refresh completed
- [x] full validation sweep completed
- [x] Sprint 47 closeout and handoff completed from the measured baseline

## What Went Well

1. **Sprint 47 delivered a real auxiliary-surface package instead of drifting into generic cleanup.**
   The sprint landed one coherent group of changes across:
   - shared internal benchmark CLI parsing helpers
   - `bench_main` modernization
   - reorder-mode / emitted-label parity cleanup
   - bounded example safety/helper adoption
   - bounded dead-code tooling hardening
   - touched benchmark/example docs refresh
   That is a stronger handoff than a loose collection of minor cleanup commits.

2. **The internal-first parsing-helper boundary was correct.** Sprint 47 did
   not create a public CLI helper API in the core library. Instead it added one
   small internal helper seam for benchmark-side parsing and used `bench_main`
   as the first consumer. That improved malformed-input behavior without
   creating new public API obligations.

3. **`bench_main` usability improved materially without broad benchmark framework churn.**
   By the end of Day 8, the main benchmark CLI had:
   - real `--help`
   - explicit missing-value failures
   - explicit unknown-option failures
   - explicit conflicting-mode rejection
   - shared helper-backed checked parsing
   - aligned reorder-mode help/parser/output behavior
   That is meaningful user-facing cleanup while still respecting the sprint’s
   bounded scope.

4. **The reorder-mode ownership story is much clearer now.** Sprint 47 did not
   just change parser text. It removed real ownership drift:
   - `bench_main` clearly owns `none|rcm|amd|nd`
   - COLAMD comparisons are explicitly handed off to:
     - `bench_reorder`
     - `bench_colamd`
   That is a better long-term auxiliary surface than simply accepting more enum
   aliases and keeping the conceptual boundary muddy.

5. **The example and tooling batches stayed intentionally narrow.** Sprint 47
   chose the right bounded follow-ons:
   - `example_eigs.c` for example-side helper/safety cleanup
   - `deadcode_report.py` and `deadcode_workflow.sh` for tooling-side hardening
   It did not diffuse into every example file or into dead-code workflow
   redesign, which kept the sprint honest.

6. **The docs refresh was grounded in live runtime behavior.** The touched docs
   were updated only after the benchmark/example/tooling behavior was already
   clear. As a result:
   - `benchmarks/README.md` now matches live `bench_main` behavior
   - `examples/README.md` now matches the helper convention and the real
     `example_eigs` story
   That prevented the sprint from ending with doc/runtime drift.

7. **The sprint closed from a measured maintained baseline.** Day 13 validated
   both the normal code-change floor and the strongest local reviewed path:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   It also reran the touched benchmark/example/tooling surfaces directly.

## What Didn't Go Well

1. **The auxiliary queue is still broader than one sprint should absorb.**
   Sprint 47 narrowed the right seams, but it did not and should not finish all
   auxiliary cleanup in one pass. Real later surfaces remain:
   - peer benchmark alignment work
   - broader example cleanup
   - any larger benchmark framework unification
   That is acceptable, but it means the auxiliary cleanup story is clearer
   rather than fully complete.

2. **One non-gating reviewed-build warning remains outside the touched Sprint 47 files.**
   The Day 13 clean reviewed CMake rebuild emitted a `-Wdouble-promotion`
   warning in `bench_eigs_reuse.c`. It did not fail the maintained gate and it
   was outside the touched Sprint 47 surface, but it is still worth tracking as
   general auxiliary follow-on context.

3. **The sprint’s wins are mainly safety/usability improvements rather than
   large architectural reductions.** That is appropriate for Sprint 47, but it
   means the value is distributed across behavior clarity, parser hardening,
   example/tooling safety, and doc/runtime alignment rather than one large
   structural extraction.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 47 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Sprint 47 artifact package

| Metric | Sprint 47 close state |
|---|---:|
| total artifact files under `SPRINT_47/artifacts/` | `15` |
| implementation-focused artifacts (Days 5, 6, 8, 10, 11) | `5` |
| validation / closeout artifacts (Days 13-14) | `2` |

### Auxiliary surface outputs

| Metric | Sprint 47 close state |
|---|---:|
| new internal helper headers added | `1` |
| primary benchmark runtime surfaces directly modernized | `1` |
| bounded example source files directly cleaned up | `1` |
| bounded tooling support files directly hardened | `2` |
| touched public docs refreshed | `2` |
| targeted auxiliary follow-ons rerun in Day 13 | `9` |

Notes:

- new internal helper headers added:
  - `benchmarks/bench_cli_parse_internal.h`
- primary benchmark runtime surface directly modernized:
  - `benchmarks/bench_main.c`
- bounded example source file directly cleaned up:
  - `examples/example_eigs.c`
- bounded tooling support files directly hardened:
  - `scripts/deadcode_report.py`
  - `scripts/deadcode_workflow.sh`
- touched public docs refreshed:
  - `benchmarks/README.md`
  - `examples/README.md`
- targeted auxiliary follow-ons rerun in Day 13:
  - `make tooling-build`
  - `./build/bench_main --help`
  - `./build/bench_main --reorder nd --size 8 --repeat 1`
  - `./build/bench_main --reorder colamd`
  - `./build/example_eigs`
  - `python3 -m py_compile scripts/deadcode_report.py`
  - `bash -n scripts/deadcode_workflow.sh`
  - synthetic valid `deadcode_report.py` check
  - synthetic malformed `deadcode_report.py` rejection check

## Residual Deferred Debt

Sprint 47 was explicitly about benchmark CLI modernization and bounded
auxiliary cleanup. The main open work it intentionally hands forward is:

- peer benchmark alignment work that stayed outside the first landing:
  - `bench_eigs.c`
  - `bench_iterative_reuse.c`
  - `bench_eigs_reuse.c`
- broader example cleanup surfaces that were not the right first bounded batch:
  - `example_ic_minres.c`
  - `example_analysis.c`
  - `example_condition.c`
- any future broader benchmark framework or CLI unification only when a later
  sprint chooses that wider scope directly
- any future broader dead-code workflow redesign only when a later sprint
  chooses that wider support-tooling scope directly
- any broader README/tutorial restructuring only when later work takes that
  outward-facing scope on directly

Not carried forward as unresolved Sprint 47 debt:

- missing shared internal benchmark CLI parsing helper seam
- missing `bench_main` modernization
- missing reorder-mode / emitted-label parity cleanup
- missing bounded example helper/safety landing
- missing bounded dead-code tooling hardening
- missing touched benchmark/example docs refresh
- missing measured validation closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-shared-cli-parsing-helper-design.md](./artifacts/day3-shared-cli-parsing-helper-design.md)
- [day4-validation-and-peer-surface-landing-design.md](./artifacts/day4-validation-and-peer-surface-landing-design.md)
- [day5-shared-cli-parsing-helper-batch.md](./artifacts/day5-shared-cli-parsing-helper-batch.md)
- [day6-bench-main-parser-modernization-batch.md](./artifacts/day6-bench-main-parser-modernization-batch.md)
- [day8-reorder-mode-parity-batch.md](./artifacts/day8-reorder-mode-parity-batch.md)
- [day10-example-safety-cleanup-batch.md](./artifacts/day10-example-safety-cleanup-batch.md)
- [day11-auxiliary-tooling-safety-cleanup-batch.md](./artifacts/day11-auxiliary-tooling-safety-cleanup-batch.md)
- [day12-benchmark-example-docs-refresh.md](./artifacts/day12-benchmark-example-docs-refresh.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 47 achieved its goal:

- Epic 4 now has a cleaner internal benchmark CLI parsing-helper seam
- `bench_main` is materially safer and clearer to use
- reorder-mode ownership and emitted reporting are aligned
- the first bounded example/tooling safety follow-ons are landed
- the touched benchmark/example docs now match the live runtime surface
- the remaining auxiliary queue is narrower and more explicit
- the sprint closed from a measured maintained validation baseline

Later benchmark/example/tooling cleanup can now start from a clearer
auxiliary-surface model and validated runtime/doc alignment instead of
reopening basic CLI ownership and malformed-input behavior questions.
