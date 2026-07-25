# Sprint 135 Day 12 - Link and Claim Validation

## Purpose

Run documentation hygiene, link/path checks, and claim-boundary scans across
the productized Sprint 135 adoption surface. Day 12 validates the algorithm
split, cookbook, report-index handoff, and navigation alignment before the
integrated reader review.

## Validation Scope

Checked these public adoption and owner surfaces:

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `docs/solver_selection.md`
- `docs/cookbook.md`
- `docs/algorithm.md`
- `docs/algorithm_history.md`
- `docs/matrix_market.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `examples/README.md`
- Sprint 135 planning artifacts

Also checked linked maintained example/header targets used by the cookbook:

- `examples/example_compressed_input.c`
- `examples/example_matrix_market.c`
- `examples/example_svd_lowrank.c`
- `examples/example_eigs.c`
- `examples/example_iterative.c`
- `examples/example_ic_minres.c`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `examples/example_matrix_free.c`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`

## Checks Run

```bash
git diff --check
```

Result: pass.

```bash
rg -n '[[:blank:]]$' README.md INSTALL.md docs/tutorial.md docs/solver_selection.md docs/cookbook.md docs/algorithm.md docs/algorithm_history.md docs/matrix_market.md docs/maintainer_guide.md benchmarks/README.md examples/README.md docs/planning/EPIC_11/SPRINT_135 || true
```

Result: pass; no trailing whitespace reported.

```bash
ruby -e 'files=%w[README.md INSTALL.md docs/tutorial.md docs/solver_selection.md docs/cookbook.md docs/algorithm.md docs/algorithm_history.md docs/matrix_market.md docs/maintainer_guide.md benchmarks/README.md examples/README.md]; missing=[]; files.each do |file|; text=File.read(file); text.scan(/\[[^\]]+\]\(([^)]+)\)/).flatten.each do |raw|; next if raw.start_with?("http://","https://","mailto:","#"); path=raw.split("#",2).first; next if path.empty?; path=path.sub(/^<(.+)>$/,"\\1"); path=File.expand_path(path, File.dirname(file)); missing << "#{file}: #{raw}" unless File.exist?(path); end; end; if missing.empty?; puts "local markdown link targets present"; else; puts missing; exit 1; end'
```

Result: pass; local markdown link targets are present.

```bash
git diff --name-only -- '*.c' '*.h'
```

Result: pass; no C/header files changed.

## Claim Boundary Scans

Package/platform scan covered:

- package-manager and package-manager non-claims
- shared-library and dynamic-ABI wording
- `BUILD_SHARED_LIBS` rejection wording
- Linux, macOS, and Windows support-tier wording
- `pkg-config`, `find_package`, install contract, and static archive wording

Result: pass. Matches Sprint 133-134 boundaries:

- static-first install contract remains unchanged
- Linux remains the strongest reviewed package-contract source of truth
- macOS package install/export confidence remains supplemental
- Windows install/downstream confidence remains supplemental
- no package-manager, shared-library, dynamic-ABI, runtime-loader, or expanded
  platform-tier claim was added

Performance/report scan covered:

- portable performance and timing guarantee wording
- pass/fail timing gate wording
- state-of-the-art, scalability, memory, and coverage-completeness wording
- benchmark rows, local evidence, local measurement, threshold-free reports,
  wall-check, performance sentinels, canonical reports, large-matrix guardrails,
  `index.tsv`, `sentinels.tsv`, and `manifest.txt`

Result: pass after small wording fixes. Benchmark/report language remains
bounded:

- benchmark rows are local evidence
- `make bench-canonical-report` is threshold-free
- `make performance-sentinels` only hard-gates through the existing
  `wall-check` lane
- `make large-matrix-guardrails` is a bounded structural/report guardrail, not
  broad scalability or performance proof
- generated indexes and manifests are artifact maps and freshness context

## Fixes Made During Validation

Day 12 made public-doc wording fixes:

1. Reframed README LOBPCG preconditioning text from a front-door
   "speedup" phrase to fixture-level local benchmark evidence.
2. Reframed a current-reference RCM fixture speedup phrase as historical local
   measurement context.
3. Removed remaining sprint-era phrasing from current eigensolver reference
   wording in `docs/algorithm.md`.
4. Replaced an inline historical LOBPCG comparison pointer with a link to
   `docs/algorithm_history.md`.

## Residual Validation Queue

No Day 12 validation failures remain open.

Day 13 should still perform an integrated reader walkthrough because link
targets and claim scans cannot prove that the simplified adoption path feels
coherent end to end.

## Completion Criteria

- touched adoption docs pass available hygiene and link/path checks
- support-tier and performance claims remain evidence-bounded
- unresolved validation gaps are named with owners and follow-up paths
