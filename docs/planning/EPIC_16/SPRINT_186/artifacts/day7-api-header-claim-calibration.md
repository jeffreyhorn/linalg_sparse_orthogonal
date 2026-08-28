# Sprint 186 Day 7: API and Header Claim Calibration

## Purpose

Calibrate generated API and public header-coherence claim surfaces so Sprint
186 closeout does not widen Epic 16 evidence beyond local generated-doc
freshness, source-controlled API references, and declaration-preserving QR
header cleanup.

## Scope

Day 7 addresses these Day 4 calibration items:

| ID | Surface | Day 7 status |
| --- | --- | --- |
| D4-CAL-006 | `docs/api_reference.md` and generated API docs guidance | Calibrated to keep generated API HTML local-only and to keep source-controlled docs plus public headers as the supported API reference path. |
| D4-CAL-007 | QR-facing docs and public header references | Verified as declaration-preserving and selected-fixture-bounded without changing the public header. |
| D4-CAL-010 | state-of-the-art/support-tier language | Reviewed for API/header surfaces; no package, ABI, platform parity, portable performance, release-readiness, or state-of-the-art claim was added. |

## Documentation Changes

| File | Change |
| --- | --- |
| `docs/api_reference.md` | Added Sprint 186 closeout wording that generated API checks prove the configured local Doxygen input/output contract and staging guard only. |
| `docs/maintainer_guide.md` | Added final closeout guidance that keeps residual `R186-HOSTED-API` open and keeps header-coherence claims declaration-preserving. |

## Verified Existing Surfaces

| Surface | Day 7 result |
| --- | --- |
| `include/sparse_qr.h` | Left unchanged. The header already limits itself to API-local declarations and lifecycle/cancellation contracts, and the QR guard rejects unsupported parity/platform/package/performance claim wording. |
| `docs/cookbook.md` | Already describes QR evidence as fixture-local and rejects broad QR or external-library parity. |
| `docs/solver_selection.md` | Already names selected QR comparison rows and rejects broad QR and external-library parity. |
| `docs/tutorial.md` | Already routes exact declarations to `docs/api_reference.md` and public headers, with general non-claims for parity, performance, package/platform/ABI support, and state-of-the-art behavior. |

## Earned Claims Preserved

| Claim family | Day 7 result |
| --- | --- |
| Generated API local freshness | Preserved as `make api-docs-freshness` evidence for local Doxygen generation, page coverage, and generated-output staging. |
| Source-controlled API reference | Preserved as `docs/api_reference.md` plus checked-in public headers under `include/`. |
| QR header coherence | Preserved as declaration-preserving header comment/docs cleanup guarded by `make qr-header-docs-guard`. |
| Selected QR comparison evidence | Preserved as selected fixture-local minimum-norm and compatible least-squares comparison evidence only. |

## Non-Claims Preserved

Day 7 preserves these non-claims:

- generated API HTML is not hosted documentation;
- generated API HTML is not a retained CI artifact;
- generated API HTML is not source-controlled release evidence;
- generated API checks do not prove dynamic ABI compatibility,
  shared-library support, package-manager distribution, broad Windows parity,
  external-library parity, portable runtime behavior, or completeness beyond
  the configured Doxygen input set;
- QR header coherence does not claim declaration-set changes, ABI support,
  package support, broad platform parity, external-library parity, portable
  performance, or new solver behavior;
- selected QR comparison rows do not imply broad QR behavior or
  state-of-the-art solver coverage.

## Residuals Carried Forward

| Residual | Day 7 handling |
| --- | --- |
| R186-HOSTED-API | Remains active. Hosted generated API HTML, retained generated-doc CI artifacts, or committed generated output require a future product decision and corresponding guards. |
| R186-BROAD-COMPARISON | Remains active. QR comparison evidence remains selected-fixture-only. |

## Generated-Doc Risk

Generated API HTML remains ignored local output under `docs/api/html/`. If
public header comments change after the latest local freshness run, maintainers
should rerun `make api-docs-freshness` or treat the generated tree as stale for
that checkout. Day 7 did not edit public headers, so no regenerated HTML is
intended to be committed.

## Validation

Day 7 changed documentation files only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

Required focused validation:

```sh
make api-docs-validate
make api-docs-freshness
make qr-header-docs-guard
git diff --check
```
