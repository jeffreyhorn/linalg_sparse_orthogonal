# Package Proof Template

## Summary

| field | value |
|---|---|
| package surface | Make install / CMake install / pkg-config / CMake package / mixed |
| command | TODO |
| artifact owner | TODO |
| proof owner | TODO |
| consumer owner | TODO |
| platform | TODO |
| reviewed status | reviewed / supplemental / local proof / staged |
| package shape | static-first / shared / mixed / undecided |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Claim Boundary

Bounded claim:

> TODO: write the exact package claim this proof supports.

Disallowed broader claim:

> TODO: write the package, platform, or ABI claim this proof does not support.

## Commands

| phase | command | expected status |
|---|---|---|
| configure or clean | TODO | pass / skip / n/a |
| build | TODO | pass / skip / n/a |
| install | TODO | pass / skip / n/a |
| downstream configure | TODO | pass / skip / n/a |
| downstream build/link | TODO | pass / skip / n/a |
| downstream run | TODO | pass / skip / n/a |
| uninstall | TODO | pass / skip / n/a |

## Installed Artifact Contract

| artifact | expected? | proof method | notes |
|---|---|---|---|
| static archive | yes / no | TODO | TODO |
| shared library | yes / no | TODO | TODO |
| public headers | yes / no | TODO | TODO |
| generated version header | yes / no | TODO | TODO |
| `sparse.pc` | yes / no | TODO | TODO |
| `SparseConfig.cmake` | yes / no | TODO | TODO |
| `SparseConfigVersion.cmake` | yes / no | TODO | TODO |
| `SparseTargets.cmake` | yes / no | TODO | TODO |

## Version and Metadata Proof

| field | value |
|---|---|
| version source | TODO |
| generated header version checked | yes / no / n/a |
| `pkg-config --modversion` checked | yes / no / n/a |
| CMake exact-version behavior checked | yes / no / n/a |
| mismatched-version behavior checked | yes / no / n/a |
| metadata non-claim | TODO |

## Consumer Validation

| consumer path | command or owner | expected behavior | claim impact |
|---|---|---|---|
| basic pkg-config consumer | TODO | TODO | TODO |
| maintained example via pkg-config | TODO | TODO | TODO |
| CMake `find_package(Sparse)` consumer | TODO | TODO | TODO |
| exact-version CMake consumer | TODO | TODO | TODO |
| mismatch-version CMake consumer | TODO | TODO | TODO |

## Platform and Exclusion Notes

| field | value |
|---|---|
| platform tier | TODO |
| reviewed lane? | TODO |
| supplemental lane? | TODO |
| staged exclusions | TODO |
| unsupported or not claimed | TODO |

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| install artifacts | TODO | TODO |
| metadata/version | TODO | TODO |
| downstream consumer compile/link/run | TODO | TODO |
| uninstall or cleanup | TODO | TODO |
| platform interpretation | TODO | TODO |

## Non-Claims

This package proof does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |
