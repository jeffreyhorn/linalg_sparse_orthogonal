# Sprint 149 Day 12 Local Validation And Syntax Review

## Scope

Day 12 validates the Sprint 149 Windows CMake install/downstream work that can
be checked locally before hosted Windows CI runs the reviewed CMake
install/downstream lane.

## Local Checks

| Check | Command | Result |
| --- | --- | --- |
| Git whitespace | `git diff --check` | PASS |
| Targeted trailing whitespace | `rg -n "[[:blank:]]$" .github/workflows/windows-ci.yml README.md INSTALL.md docs/maintainer_guide.md tests/corpus/manifests/report_families.tsv docs/planning/EPIC_13/SPRINT_149` | PASS, no matches |
| Windows workflow YAML parse | `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/windows-ci.yml"); puts "yaml ok"'` | PASS |
| Corpus schema | `python3 scripts/validate_corpus_schema.py` | PASS |
| CI report index normalization | `python3 scripts/normalize_report_index.py --family ci --check` | PASS, 1 row checked |
| Package report index normalization | `python3 scripts/normalize_report_index.py --family package --check` | PASS, 6 rows checked |
| CMake install/package proof | `bash tests/test_cmake_install.sh` | PASS, 26 passed, 0 failed, 0 skipped |
| Make/pkg-config install proof | `bash tests/test_install.sh` | PASS, 23 passed, 0 failed |
| Static package deferral gate | `bash scripts/static_package_deferral_check.sh` | PASS |

## Claim Search Results

Focused stale-label search:

```sh
rg -n "Windows (remains|does not currently|supplemental).*install|supplemental CMake install/downstream|separate reviewed install-validation lane|no separate reviewed install-validation lane|Windows install-validation parity" \
  README.md INSTALL.md docs/maintainer_guide.md tests/corpus/manifests/report_families.tsv .github/workflows/windows-ci.yml
```

Result: PASS, no public documentation or workflow matches for stale
supplemental/reviewed install-validation labels.

Focused unsupported-claim search:

```sh
rg -n "Windows.*(pkg-config|Makefile|package-manager|shared-library|dynamic ABI|runtime-loader|broad Windows parity)" \
  README.md INSTALL.md docs/maintainer_guide.md tests/corpus/manifests/report_families.tsv .github/workflows/windows-ci.yml
```

Result: PASS. Matches are explicit non-claims for Windows Makefile parity,
Windows `pkg-config` execution parity, package-manager support, shared-library
support, dynamic ABI support, runtime-loader behavior, and broad Windows
parity.

## Quality Gate Status

No repository `.c` or `.h` files changed during Sprint 149 through Day 12, so
the full `make format && make lint && make test` quality gate is not required
by the sprint review-comment rule. The affected local package/install gates
above were run instead.

## Hosted-Only Evidence Pending

The promoted Windows evidence still depends on hosted CI running
`.github/workflows/windows-ci.yml` job
`Windows reviewed CMake install/downstream validation path`. Day 13 should
inspect that hosted run and record the workflow result, because local macOS
cannot execute the MSVC-specific install/downstream proof.
