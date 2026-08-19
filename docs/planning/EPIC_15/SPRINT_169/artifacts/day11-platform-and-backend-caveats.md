# Sprint 169 Day 11: Platform And Backend Caveats

## Purpose

Day 11 documents the exact platform, runner, backend, build, and fixture
constraints for the selected Sprint 169 performance lane. The goal is to make
the selected hosted freshness lane and S6 local smoke ceiling explicit without
implying Windows/macOS parity, portable performance, OpenMP speedup, backend
superiority, package proof, ABI proof, or state-of-the-art evidence.

## Implemented Documentation Updates

| Area | Change |
| --- | --- |
| `benchmarks/README.md` canonical report section | Added selected lane platform and backend caveats immediately after the hosted selected-performance description. |
| `benchmarks/README.md` report-index handoff | Added a reminder to read hosted selected-performance rows and local S6 rows with recorded platform, compiler, build, thread, CPU, fixture, and command context. |
| `docs/maintainer_guide.md` performance section | Added maintainer-facing selected performance platform/build caveats. |

## Selected Hosted Lane Constraints

The reviewed hosted selected-performance freshness lane is currently scoped to
Linux GitHub Actions. Its workflow sets:

| Metadata | Hosted value |
| --- | --- |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | `github-actions-ubuntu-latest` |
| `SPARSE_CANONICAL_BUILD_FLAGS` | `default_make_flags` |
| `SPARSE_CANONICAL_BUILD_MODE` | `serial` |
| `SPARSE_CANONICAL_SUPPORT_TIER` | `hosted_selected` |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | `hosted_selected_threshold_free` |

The hosted workflow also records `SPARSE_CANONICAL_CPU_MODEL` from
`/proc/cpuinfo` when available. That field is context only: GitHub-hosted CPU
assignment can vary and the recorded string is not a stable machine-class
baseline.

## Local Comparison Constraints

Local canonical report rows may record different values for:

- `platform`;
- `compiler`;
- `runner_context`;
- `build_flags`;
- `cpu_model`;
- `build_mode`;
- `OMP_NUM_THREADS`.

Local generated rows should be compared only after checking those metadata
fields. They are branch-local evidence, not hosted proof or portable
performance evidence.

## Backend And OpenMP Constraints

The selected canonical row is the default SPD/Cholesky
`bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` lane. It
reports `backend_context=n/a`.

Interpretation:

- do not cite the selected canonical row as LDLT dense-helper evidence;
- do not cite it as optional-backend availability or backend-superiority
  evidence;
- do not cite it as runtime-loader, package, or ABI evidence;
- do not treat `build_mode=openmp` or `OMP_NUM_THREADS` context on local
  generated rows as OpenMP speedup evidence;
- `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` are runtime
  report context for sentinel rows that own those fields, not selected
  canonical publication controls.

S6 uses the same selected fixture and command as a local smoke ceiling, but it
also records backend fields as `n/a`. S6 is local regression governance, not
hosted selected-performance publication.

## Fixture And Matrix-Size Constraints

The selected fixture is:

```text
tests/data/suitesparse/nos4.mtx
```

The selected command is:

```text
bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1
```

The selected canonical row records:

```text
matrix_size=n=100
```

That value is the benchmark-emitted selected dimension. It must not be
reinterpreted as nonzero-count evidence, broad SuiteSparse corpus evidence, or
broad matrix-family coverage.

## Retained Non-Claims

The documentation now preserves these non-claims for the selected lane:

- no Windows/macOS performance parity;
- no portable performance;
- no OpenMP speedup evidence;
- no backend parity or backend-superiority evidence;
- no package, ABI, install, or runtime-loader proof;
- no external-library parity;
- no release benchmark proof;
- no state-of-the-art performance claim.

## Validation

Day 11 changed documentation and planning artifacts only. No `.c` or `.h`
files were modified, so the full C quality gate is not required for this day.

Validation run:

```sh
rg -n "github-actions-ubuntu-latest|default_make_flags|build_mode=serial|cpu_model|matrix_size=n=100|backend_context=n/a|Windows and macOS|platform parity|portable performance|OpenMP speedup|backend-superiority|backend superiority|runtime-loader" \
  README.md benchmarks/README.md docs/maintainer_guide.md \
  docs/planning/EPIC_15/SPRINT_169 -g '*.md'
git diff --check
```

Results:

- targeted scan found the new selected-lane caveats and existing non-claim
  wording;
- `git diff --check` passed.

## Day 11 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected lane constraints are explicit. | Complete | Benchmark docs and maintainer guide now list hosted runner, build flags, build mode, CPU variability, backend context, command, fixture, and matrix-size semantics. |
| Users cannot infer broad platform parity from the selected lane. | Complete | Docs explicitly separate Linux hosted selected freshness from Windows/macOS performance parity and local generated rows. |
| Docs remain consistent with CI metadata. | Complete | Documented values match the hosted workflow metadata: `github-actions-ubuntu-latest`, `default_make_flags`, and `serial`. |
