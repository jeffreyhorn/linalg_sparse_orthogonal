# Sprint 76 Day 7 Artifact: Post-Landing Audit and Rerank

Date: 2026-06-17
Branch: sprint-76

## Purpose

Re-audit the benchmark-governance surface after the Day 6 canonical reporting
landing so Sprint 76's next batch targets the strongest remaining
contradiction instead of reworking the same workflow seam.

## Main Result

The Day 6 landing closed the strongest pure reporting-workflow contradiction:

- `scripts/bench_canonical_report.sh` no longer reads like the strongest
  remaining Sprint 76 seam
- `Makefile` no longer reads like the strongest remaining Sprint 76 seam
- a second workflow-only script/Makefile batch is not the highest-value next
  move

The strongest remaining seam has now shifted to support-surface drift around
the landed stronger bundle contract:

- required next batch:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- support only if wording truly forces it:
  - `README.md`

## Why the Seam Shifted

The Day 6 landing already fixed the key workflow-side gap:

- the canonical report bundle now carries bounded longitudinal metadata
- the workflow now exposes a bounded report label
- the generated bundle now includes both:
  - `manifest.txt`
  - `index.tsv`

That means the strongest remaining contradiction is no longer in the code or
workflow surfaces themselves. It is in how the maintained human-facing
governance surfaces still describe the older bundle.

## Ranked Findings

### 1. `benchmarks/README.md` is now the strongest next target

This is the main user-facing benchmark-governance interpretation surface, and
it still describes the Day 5-era bundle too narrowly:

- one CSV per canonical maintained benchmark
- one `manifest.txt` with exact commands

That description is no longer wrong in a strict sense, but it is no longer the
best reading of the landed bundle either because it does not mention:

- `index.tsv`
- structured per-artifact bundle rows
- bounded report-label and git-metadata support

So it is now the strongest next contradiction center.

### 2. `docs/maintainer_guide.md` is the strongest second target

The maintainer guide remains the authoritative policy surface, but it still
describes the old report-bundle shape:

- `manifest.txt` with exact fixture/command mapping

That is still partially true, but it lags the landed Day 6 contract in the
same way the benchmark-local README does. Because it is the policy authority,
that drift matters even though the file remains otherwise coherent.

### 3. `README.md` is support-only, not the next batch center

The top-level README still stays broadly truthful:

- `make bench-canonical-report` remains threshold-free
- it still writes one bounded snapshot of the maintained canonical surface

It does not mention the new bundle metadata, but that omission is acceptable
at the compact front-door level unless the support-surface batch proves the
top-level wording actually became inaccurate.

### 4. Threshold-policy work remains deferred, not cancelled

The runtime-threshold lane remains real:

- `bench-fast`
- `wall-check`
- `bench_reorder`
- `bench_amd_qg`

But it is no longer the next batch center while the benchmark-local and
maintainer-policy surfaces still lag the landed Day 6 bundle contract.

## Day 8 Implication

The next design pass should therefore treat Sprint 76's follow-through batch
as:

- required center:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- support only if truly forced:
  - `README.md`
- explicitly not next:
  - another workflow-only script/Makefile batch
  - threshold-policy widening
  - canonical benchmark driver edits

## Exit State

Sprint 76 now has one explicit Day 7 rerank:

- the workflow lane landed successfully
- the strongest remaining seam is support-surface reconciliation
- threshold-policy work stays real, but not next
