# Day 12: Repeated Hosted Attempts on One Commit

## Scope and result

Reran the [Sprint 225 timing workflow run 44403764960](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44403764960) twice after its first attempt. These are **three independent hosted job attempts under one workflow run ID**, all on exact source commit `38323f0e584e1a4d2aa8d2c92db5c18830366afc`, policy `s225-v1`, and the same candidate workload definition. Each job captured its runner context, then failed baseline preflight with exit 2 and `S225_BASELINE_MISSING`. Build, candidate collection, and timing decision were skipped. This is reproducible blocking behavior, not repeated candidate timing evidence.

Each attempt uploaded a receipt before the next rerun. GitHub's artifact listing for this run showed only the current attempt after a rerun, so the downloaded ZIP bytes are retained in [day12-blocked-hosted-attempts](day12-blocked-hosted-attempts/). The local SHA-256 values below match the GitHub artifact digests observed at download time. Each ZIP contains exactly `sprint225-host-provenance.json` and `sprint225-timing-decision.json`; none contains candidate raw streams.

| Attempt | Job ID | Artifact ID | ZIP SHA-256 | Resolved image/version | CPU | Decision |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | `135153401892` | `14473492579` | `6ee6c5306ff2d55f9fd0748ba66ffa96ef0df7619183b356837b9a00f8725f2e` | `macos26` / `20261102.0473.1` | Apple M1 (Virtual) | `S225_BASELINE_MISSING` |
| 2 | `135155131051` | `14474442513` | `0329c9741c1a1a4ce764602beb8dfc6221202380eea7b6d79b77e3cba5eb8108` | `macos26` / `20261112.0504.1` | Apple M2 Pro (Virtual) | `S225_BASELINE_MISSING` |
| 3 | `135155713728` | `14473847941` | `9807dc7059ee4171f8f7cc4353a0523b81fc1adcc8cbc3c45a17cb7344ea8a60` | `macos26` / `20261102.0473.1` | Apple M1 (Virtual) | `S225_BASELINE_MISSING` |

All three host records report `macOS-26.7.2-arm64-arm-64bit-Mach-O`, `/usr/bin/cc`, and Apple clang `21.0.0 (clang-2100.1.1.101)`. The resolved image version and CPU differ on attempt 2, so these attempts would not form one `s225-v1` runner class even if they had candidate trials. The changing ephemeral runner label is recorded but is not a class equality key. The observed variation is consistent with the Day 5 baseline rejection; it does not justify loosening the class rule using these attempts.

## Independent audit

Downloaded each ZIP through the GitHub artifact API and recomputed its SHA-256. A local audit opened the three archives, checked exact two-file membership, matched each host record's workflow run ID, attempt, policy and source commit, and read each decision report. Every report has schema 2, `status=blocked`, `primary_rule=S225_BASELINE_MISSING`, and null baseline reference and candidate median. No candidate raw trial, measured statistic, MAD, p90 or threshold verdict exists to recompute. A blank numeric comparison is the required outcome; treating these failures as favorable samples would violate the locked method.

## Disposition

The Day 12 completion path is **blocked with reproducible evidence**. Three complete preflight-and-upload attempts fail for the same missing accepted baseline, while host context shows runner-class drift. The reviewed numerical limits, baseline lifetime, and exact identity requirement remain unchanged. Before a timing verdict can be evaluated, a stable runner/compiler class and three compatible, independently reviewed baseline jobs must be established under a reviewed policy. Candidate collection can then be repeated on one exact commit, retaining each attempt before another rerun.
