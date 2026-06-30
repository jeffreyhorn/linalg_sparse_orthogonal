# Sprint 100 Day 10 Benchmark, Coverage, and Performance Templates

## Purpose

Day 10 creates reusable templates for benchmark interpretation, coverage
evidence, and bounded performance sentinels. These templates preserve the
project's existing distinction between:

- benchmark reports as local measurement artifacts;
- coverage as a supplemental, tree-mutating signal;
- performance sentinels as narrow regression gates;
- tests and external oracles as correctness owners.

## Files

| file | role |
|---|---|
| `templates/benchmark-interpretation-template.md` | reusable blank template for benchmark/report interpretation |
| `templates/coverage-evidence-template.md` | reusable blank template for coverage evidence and threshold interpretation |
| `templates/performance-sentinel-template.md` | reusable blank template for thresholded runtime sentinels |
| `day10-benchmark-template-pilot-canonical-report.md` | pilot-filled example using the current canonical benchmark report surface |

## Design Requirements

Future benchmark, coverage, and performance artifacts should include:

- exact command and owner;
- fixture or source scope;
- machine, compiler, backend, and thread context where relevant;
- emitted metrics and units;
- artifact paths and output format;
- threshold state, if any;
- reviewed vs supplemental status;
- reset requirements for tree-mutating modes;
- unsupported or skipped cases;
- explicit non-claims.

## Required Separations

| evidence type | must stay separate because |
|---|---|
| benchmark report | local timing artifacts are not portable performance claims |
| coverage report | line coverage is not correctness, oracle, or reviewed-quality parity |
| performance sentinel | thresholded regression detection is narrower than benchmark superiority |
| correctness context | residual fields in benchmarks do not replace test/oracle ownership |
| platform context | machine class, compiler, and backend state affect timing interpretation |

## Existing Surface Patterns Used

| pattern | current owner | template effect |
|---|---|---|
| threshold-free canonical report | `make bench-canonical-report`, `scripts/bench_canonical_report.sh` | benchmark template requires report-only vs thresholded mode |
| bounded runtime lane | `make bench-fast` | benchmark template records runtime budget and fixture scope |
| narrow performance sentinel | `make wall-check`, `scripts/wall_check.sh` | sentinel template requires baseline source and machine-class assumptions |
| supplemental coverage | `make coverage`, `make coverage-lcov`, `make coverage-gcovr` | coverage template requires reviewed status and reset requirement |
| benchmark category split | `benchmarks/README.md` | benchmark template requires category: canonical, runtime-sensitive, or exploratory |

## Usage Notes

1. Use the benchmark template for any future CSV/report surface that will be
   interpreted in planning, release notes, or public docs.
2. Use the coverage template when a sprint changes coverage targets,
   thresholds, exclusions, or reviewed/supplemental status.
3. Use the performance sentinel template only when a command has a thresholded
   pass/fail contract.
4. Keep benchmark-local residual columns as context unless a test or external
   oracle explicitly owns the correctness claim.
5. Record timing as local unless the sprint defines a machine class, fixture
   corpus, repeat policy, and statistical interpretation.

## Completion Rule

A future benchmark, coverage, or performance claim is not earned unless the
filled template names:

- the command;
- the fixture or source scope;
- the metrics and units;
- the threshold or report-only status;
- the reviewed or supplemental status;
- reset requirements, if any;
- the explicit non-claims that remain after the evidence passes.
