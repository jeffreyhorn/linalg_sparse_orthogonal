# Day 13: Hosted Evidence Closeout and Claim Calibration

## Item 225.5 disposition

The planned 20 hours of hosted candidate work are accounted for (Day 11: 8; Day 12: 10; Day 13: 2), but the **measurement outcome is blocked**. On the latest pushed candidate source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`, the named `canonical-refactor-csc-timing` job had three [hosted attempts](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44403764960). Every attempt failed baseline preflight with `S225_BASELINE_MISSING` and uploaded a digest-verified receipt. No attempt reached the clean build, raw candidate trials, or numerical decision. Thus there is no successful named timing job, no candidate trial ZIP, no candidate median or dispersion, and no repeated timing verdict to compare. The [Day 12 audit](day12-hosted-repetition-blocked.md) records every artifact ID, ZIP digest, host identity, and report status; the [retained ZIP bytes](day12-blocked-hosted-attempts/) preserve the inspectable blocked receipts.

The attempted baseline group was [rejected on Day 5](day5-baseline-variance-decision.md). Its runner images and CPU models differed; one job exceeded MAD/median `0.10`, and the group's median spread exceeded `0.10`. No `s225-v1` active pointer or reviewed baseline version exists. In the three candidate attempts, attempt 2 again changed image version and CPU. These are reasons to keep the class and dispersion gates closed, not candidate timing observations. None of the rejected baseline values is a reference median.

## Calibrated documentation

The [benchmark guide](../../../../../benchmarks/README.md#sprint-225-one-cell-timing-decision) now specifies the one-cell command, `refactor_csc_ms`, exact runner/compiler/build identity, 5+20 process schedule, three-job reference, MAD/p90 and between-job limits, 25% regression limit, 30-day baseline age, 30-second process timeout, 20-minute job cap, artifact naming and retention, and rerun procedure. The [maintainer guide](../../../../../docs/maintainer_guide.md) provides the blocked status and operational preflight/report-verification route. Both separate selected threshold-free freshness from this blocked timing gate and rule out portable speed, platform parity and external-library ranking claims. The observed hosted compiler was `/usr/bin/cc`, Apple clang `21.0.0 (clang-2100.1.1.101)`; `macos-latest` was the requested label, while the resolved image/version and CPU differed across attempts.

## Residual risks and next action

1. Obtain a runner class that repeats the exact resolved image, OS/architecture, CPU, compiler and build environment. If this requires a changed policy, review and version it before collection.
2. Collect three compatible independent baseline jobs, retain every raw bundle before rerunning, verify member and ZIP hashes, dispersion and source/build provenance, then review a source-controlled baseline manifest and active pointer.
3. Re-run the separate candidate job on one exact commit, download each attempt before another rerun, verify raw trial bytes and recomputed reports, and compare repeated outcomes. A preflight-ready status alone is not a timing pass.

## Quality record

Day 13 changed documentation and planning notes only; no `.c` or `.h` files changed. `python3 tests/test_selected_performance_docs.py` passed. `python3 tests/test_check_refactor_csc_timing.py` passed 17 tests. The timing-workflow suite passed two tests using the pinned PyYAML virtual environment; system Python lacked `yaml`. `API_DOCS_PYTHON=/tmp/sprint225-day8-venv/bin/python bash scripts/check_api_docs_local_only.sh` passed, including 122 structured workflow records. Relative Markdown links and `git diff --check` passed. All three retained receipt ZIPs were rehashed and matched the Day 12 digest ledger. Real preflight returned exit 2 with `S225_BASELINE_MISSING`, as expected. The hosted gate remains red until the baseline and candidate evidence above exists.
