# ABI and Shared-Library Decision Template

## Summary

| field | value |
|---|---|
| decision owner | TODO |
| decision date or sprint | TODO |
| package shape before decision | static-first / shared / mixed |
| package shape after decision | static-first / shared / mixed / deferred |
| ABI promise | none / exact-version package only / limited / stable |
| affected platforms | TODO |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Decision

Decision:

> TODO: keep static-first, add shared-library support, or defer.

Reason:

> TODO: summarize the evidence and tradeoff.

## Current Contract Before Decision

| field | current value |
|---|---|
| library artifact | TODO |
| package version behavior | TODO |
| shared-library output | TODO |
| runtime-loader proof | TODO |
| symbol/export policy | TODO |
| cross-platform install proof | TODO |

## Required Proof If Shared-Library or ABI Claims Are Added

| proof area | required evidence | owner |
|---|---|---|
| shared artifact generation | TODO | TODO |
| install/export metadata | TODO | TODO |
| runtime loader behavior | TODO | TODO |
| downstream consumer compile/link/run | TODO | TODO |
| symbol visibility/export policy | TODO | TODO |
| version compatibility policy | TODO | TODO |
| Linux validation | TODO | TODO |
| macOS validation | TODO | TODO |
| Windows validation | TODO | TODO |
| uninstall/cleanup behavior | TODO | TODO |

## Required Proof If Static-First Remains

| proof area | required evidence | owner |
|---|---|---|
| static archive install | TODO | TODO |
| no unexpected shared artifacts | TODO | TODO |
| exact-version CMake package behavior | TODO | TODO |
| pkg-config metadata | TODO | TODO |
| consumer compile/link/run | TODO | TODO |
| documentation non-claims | TODO | TODO |

## Risk Assessment

| risk | impact | mitigation |
|---|---|---|
| TODO | TODO | TODO |

## Documentation Updates

| doc surface | required wording |
|---|---|
| README / install docs | TODO |
| maintainer guide | TODO |
| package templates | TODO |
| release notes or planning artifacts | TODO |

## Non-Claims After Decision

After this decision, the project still does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |
