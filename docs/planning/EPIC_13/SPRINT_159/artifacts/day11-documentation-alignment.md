# Day 11 Documentation Alignment

## Scope

Day 11 aligns maintainer, corpus, README, and solver-selection wording with
the Sprint 159 hosted report-freshness surface implemented on Days 6-10.

No C or public-header files were modified.

## Changed Files

| File | Change |
| --- | --- |
| `README.md` | Added the selected oracle freshness command to the recommended Make command list, marked both selected freshness gates as mirrored by reviewed Linux hosted CI, and tightened QR evidence wording. |
| `docs/maintainer_guide.md` | Updated selected oracle/comparison gate sections and normalized report-index interpretation for Sprint 159 hosted execution and selected-row fresh/error semantics. |
| `tests/corpus/README.md` | Clarified that Sprint 159 hosted CI covers only selected required oracle and QR minimum-norm comparison freshness rows, not broad report-index freshness or every local-only family. |
| `docs/solver_selection.md` | Updated QR and partial-SVD evidence boundaries to mention the selected reviewed Linux hosted report-freshness lane without turning it into broad platform or parity proof. |
| `docs/planning/EPIC_13/SPRINT_159/WORKING_NOTES.md` | Recorded Day 11 notes and Day 12 handoff. |

## Documentation Semantics

The updated wording uses this distinction:

- generated oracle/comparison rows still carry fixture-local generated
  metadata and ignored `build/` artifact paths;
- Sprint 159 adds reviewed Linux hosted execution and split artifact upload
  for the selected gates only;
- hosted execution does not promote broad report-index freshness, unselected
  generated families, optional dependency defers, package evidence, ABI
  evidence, performance evidence, platform parity, external-library parity, or
  state-of-the-art claims.

## Selected Hosted Evidence Surface

The documentation now identifies only these promoted hosted checks:

| Surface | Command | Hosted interpretation |
| --- | --- | --- |
| Selected QR and partial-SVD oracle rows | `make report-index-oracle-freshness` | Reviewed Linux hosted execution and split oracle artifact upload for selected generated rows. |
| Selected QR minimum-norm comparison rows | `make report-index-comparison-freshness` | Reviewed Linux hosted execution and split QR minimum-norm comparison artifact upload for the six selected rows. |

The broad command below remains advisory/local and is not a hosted claim
surface:

```sh
python3 scripts/normalize_report_index.py --check-freshness
```

## Claim Boundaries Preserved

The aligned docs continue to reject these unsupported claims:

- broad QR correctness or broad QR parity;
- broad partial-SVD correctness;
- LAPACK, NumPy, SciPy, SuiteSparse, Eigen, or external-library parity;
- broad platform support or Windows/macOS report-index parity;
- package-manager, static package, shared-library, ABI, or dynamic-loader proof;
- performance superiority;
- release proof;
- state-of-the-art sparse linear algebra claims.

## Sprint 160 QR Comparison Handoff

Sprint 160 should treat Sprint 159 as a narrow hosted freshness foundation, not
as broad external comparison closure. A concrete QR comparison follow-up should:

1. choose one additional QR comparison family only after naming the exact
   fixture, baseline, generated rows, artifact paths, optional dependency
   behavior, and non-claims;
2. decide whether the new comparison can use the existing source-controlled
   dense helper or needs a new reviewed baseline helper;
3. add selected-row normalizer tests before promotion, matching the Sprint 159
   selected comparison tests for complete, missing, unexpected, duplicate,
   stale, failed, skipped, and deferred rows;
4. measure cold and warm runtime before editing hosted CI;
5. upload split artifacts under a name that identifies the exact comparison
   family;
6. keep optional NumPy/SciPy rows contextual unless a later sprint explicitly
   promotes and gates them;
7. update public docs only after the selected comparison passes locally and in
   hosted CI.

Potential next candidates should favor complete closure over breadth:

- another underdetermined minimum-norm QR fixture with exact dense-reference
  values;
- one overdetermined compatible least-squares fixture with residual and
  solution checks;
- one rank-deficient rectangular fixture with rank/nullity and residual checks.

Avoid starting with broad SuiteSparse/LAPACK/NumPy/SciPy parity language.

## Validation Plan

Day 12 should run the selected freshness commands and documentation checks
after this wording update:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
python3 tests/test_normalize_report_index.py
git diff --check -- README.md docs/maintainer_guide.md tests/corpus/README.md docs/solver_selection.md docs/planning/EPIC_13/SPRINT_159
```

## Completion Check

- Docs now match the selected hosted evidence surface.
- Unsupported claims remain explicit non-claims.
- Sprint 160 has a concrete QR comparison handoff.
