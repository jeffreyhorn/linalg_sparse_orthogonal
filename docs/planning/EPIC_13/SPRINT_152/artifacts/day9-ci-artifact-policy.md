# Sprint 152 Day 9 CI And Artifact Policy

## Purpose

Day 9 decides the hosted CI and artifact policy for the selected generated
freshness gate. The policy keeps Sprint 152's selected oracle freshness
evidence local-required and fixture-scoped, and avoids converting generated
`build/` rows into hosted release, package, platform, ABI, performance, or
state-of-the-art proof.

## Current CI Surface

| Workflow | Current Role | Generated Artifact Handling | Sprint 152 Policy Impact |
| --- | --- | --- | --- |
| `.github/workflows/ci.yml` `build-and-test` | Linux supplemental direct runtime, sanitize, ASan, bench-fast | No artifact upload | Do not add selected oracle output here; this lane is runtime/benchmark signal. |
| `.github/workflows/ci.yml` `cmake-build-and-test` | Linux enforced reviewed CMake parity | No artifact upload | No selected oracle output; CMake parity remains build/test proof. |
| `.github/workflows/ci.yml` `package-contract` | Linux reviewed static-first package contract | No artifact upload | No selected oracle output; package proof remains static install/export only. |
| `.github/workflows/ci.yml` `tsan` | Linux supplemental ThreadSanitizer coverage | No artifact upload | No selected oracle output; TSan remains race-signal only. |
| `.github/workflows/ci.yml` `lint` | Linux enforced Makefile compile-quality path | No artifact upload | No selected oracle output; lint remains source-quality proof. |
| `.github/workflows/ci.yml` `deadcode` | Linux enforced dead-code report/check path | Uploads `build/deadcode/*` report artifacts | Keep existing uploads; dead-code remains advisory/completeness context, not selected oracle proof. |
| `.github/workflows/ci.yml` `coverage` | Linux supplemental coverage report | Uploads `coverage/html/` | Keep existing upload; coverage remains supplemental/advisory. |
| `.github/workflows/macos-ci.yml` | macOS reviewed compile/package and supplemental compiler lanes | No generated report upload | Do not add selected oracle output; macOS lanes preserve package/platform non-claims. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake CTest and install/downstream lanes | No generated report upload | Do not add selected oracle output; Windows lanes preserve CMake/static-package scope. |

There is currently no hosted oracle freshness lane and no hosted upload of
`build/corpus/oracle/`, `build/corpus-reports/`, or `build/report-index/`.

## Selected Policy Decision

Sprint 152 keeps selected oracle freshness as a local-required gate, not a
hosted CI artifact publication lane.

The maintained local command sequence is:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Day 10 should add a named local Makefile target or script wrapper for this
sequence so maintainers and future CI policy can use one stable command name.
The target should regenerate selected oracle output and run the required
freshness check. It should not upload or commit generated files.

## Local/CI Matrix

| Family | Local Policy | Hosted CI Policy | Artifact Policy | Non-Claims |
| --- | --- | --- | --- | --- |
| `oracle/generated_reference` | Required selected local freshness gate | Not hosted in Sprint 152 | Generated under ignored `build/corpus/oracle/` and `build/corpus-reports/`; no upload | No hosted CI proof, broad corpus completeness, external-library parity, or platform portability claim. |
| `oracle/solver_backed` | Required selected local freshness gate | Not hosted in Sprint 152 | Generated under ignored `build/corpus/oracle/` and `build/corpus-reports/`; no upload | No broad QR correctness, broad partial-SVD correctness, package, ABI, performance, platform, or state-of-the-art claim. |
| `report_index/missing_generated` | Local normalized-index support | Not hosted in Sprint 152 | Optional generated index under ignored `build/report-index/`; no upload | No pass evidence, completeness proof, release proof, or generated freshness proof by itself. |
| `benchmark` | Advisory local reports | Existing `bench-fast` remains runtime signal only | No report-index upload | No portable performance or superiority claim. |
| `sentinel` | Existing hard-gate semantics remain separate | Existing CI wall/runtime lanes remain scoped | No selected oracle upload | No portable performance or backend portability claim. |
| `guardrail` | Local generated guardrail reports remain scoped | No new Sprint 152 hosted lane | No upload | No broad scalability or memory-footprint claim. |
| `deadcode` | Existing local report/check path | Existing Linux dead-code lane uploads reports | Existing upload retained | No zero-dead-code or semantic correctness claim. |
| `coverage` | Existing local coverage target | Existing Linux coverage lane uploads HTML | Existing upload retained | No coverage completeness or product-quality claim. |
| `package` | Source-controlled contract rows | Existing Linux/macOS/Windows install lanes retained | No selected oracle upload | No shared-library ABI, package-manager, or dynamic-loader claim. |
| `ci` | Hosted logs are external evidence | Existing workflows retained | Existing workflow logs only | No local generated freshness artifact proof. |
| `documentation` | Source-controlled guidance | Not a generated freshness lane | No upload | No executable proof. |
| `runtime_backend` | Source-controlled governance | Existing sentinel/runtime lanes remain scoped | No selected oracle upload | No backend portability or runtime performance guarantee. |

## Artifact Retention And Ignore Policy

Current ignore rules already exclude:

- `build/`, including `build/corpus/oracle/`, `build/corpus-reports/`,
  `build/report-index/`, `build/deadcode/`, and benchmark reports;
- `coverage/`;
- compiled object/archive/shared-library outputs.

Sprint 152 does not change `.gitignore`.

Selected oracle artifacts:

- are regenerated locally as needed;
- remain uncommitted;
- are not uploaded to hosted CI in Sprint 152;
- may be deleted by `make clean` or generator stale-output cleanup;
- must be regenerated rather than hand-edited when stale or mismatched.

Existing hosted uploads remain unchanged:

- dead-code artifacts: `build/deadcode/report.md`,
  `build/deadcode/report.tsv`, `build/deadcode/coverage-notes.txt`,
  `build/deadcode/cppcheck.txt`, and `build/deadcode/xunused.txt`;
- coverage HTML: `coverage/html/`.

Those existing uploads remain advisory/supporting context and are not selected
oracle freshness proof.

## Platform And Compiler Boundaries

The selected oracle gate records platform and compiler metadata because local
generated rows are interpreted in their exact generation context.

Sprint 152 does not infer:

- Linux portability from local generated oracle rows;
- macOS or Windows parity from local generated oracle rows;
- compiler portability from one generated oracle run;
- hosted CI proof from local `build/` artifacts;
- package, ABI, runtime-loader, or performance support from oracle freshness.

If a future sprint promotes oracle freshness to hosted CI, it must name the
runner, compiler, build prerequisite, artifact retention, and non-claims in
the workflow itself.

## Day 10 Implementation Checklist

- Add a named local command surface for selected oracle freshness, preferably a
  Makefile target such as `report-index-oracle-freshness`.
- The target should run:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`
  followed by
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Keep the target local and out of hosted workflows for Sprint 152.
- Do not add artifact uploads for `build/corpus/oracle/`,
  `build/corpus-reports/`, or `build/report-index/`.
- Do not change existing dead-code or coverage upload retention.
- Validate the new local command path and focused Python/report checks.
- Record that CI workflow files were intentionally left unchanged unless Day
  10 discovers an alignment issue.

## Non-Claims

This policy does not claim hosted CI oracle proof, release artifact proof,
package-manager availability, shared-library ABI support, dynamic-loader
support, broad platform support, compiler portability, broad QR correctness,
broad partial-SVD correctness, external-library parity, portable performance,
benchmark superiority, complete coverage, zero dead code, or state-of-the-art
sparse linear algebra status.
