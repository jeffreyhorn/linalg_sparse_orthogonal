# Platform Tier Template

## Summary

| field | value |
|---|---|
| platform | Linux / macOS / Windows / other |
| compiler or generator | TODO |
| workflow or local command owner | TODO |
| current tier | strongest reviewed / reviewed narrower / reviewed subset / supplemental / staged |
| reviewed status | reviewed / supplemental / staged / non-goal |
| expected test count | TODO or `n/a` |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Tier Claim

Supported claim:

> TODO: write the exact platform support claim.

Excluded or staged claim:

> TODO: write the platform claim that remains excluded or staged.

## Reviewed Commands

| command or workflow step | owner | expected result | notes |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Supplemental Commands

| command or workflow step | owner | expected result | notes |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Expected Counts and Exclusions

| field | value |
|---|---|
| expected CTest count | TODO or `n/a` |
| expected Make test count | TODO or `n/a` |
| count owner | TODO |
| staged test exclusions | TODO |
| staged workflow exclusions | TODO |
| package/install exclusions | TODO |

## Platform Artifacts

| artifact | owner | interpretation |
|---|---|---|
| TODO | TODO | TODO |

## Change Rules

Before changing this platform tier, confirm:

- [ ] reviewed and supplemental lanes are named separately
- [ ] expected test counts are updated with the workflow or command that
      enforces them
- [ ] exclusions are printed or documented where users will see them
- [ ] install/package claims are not widened without package proof
- [ ] performance claims are not widened without benchmark or sentinel proof
- [ ] the maintainer guide and user-facing docs use the same support tier

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| reviewed build/configure | TODO | TODO |
| reviewed test discovery | TODO | TODO |
| reviewed execution | TODO | TODO |
| supplemental confidence | TODO | TODO |
| staged exclusions | TODO | TODO |

## Non-Claims

This platform tier does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |
