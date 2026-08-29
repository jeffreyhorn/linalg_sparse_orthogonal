#!/usr/bin/env bash
# package_manager_deferral_check.sh - package-manager provider claim guard.
#
# This script preserves the Sprint 171 package-manager non-claim baseline and
# the Sprint 180 selected local Homebrew proof boundary. The selected Homebrew
# proof artifacts are allowed, but public Homebrew support remains unclaimed
# while the local proof exits at the missing standalone-license gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() {
    echo "package-manager-deferral-check: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "package-manager-deferral-check: $1 ok"
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local message="$3"

    if ! grep -Eq "$pattern" "$file"; then
        fail "$message"
    fi
}

require_absent_grep() {
    local pattern="$1"
    local path="$2"
    local message="$3"
    local matches

    matches="$(grep -REn "$pattern" "$path" 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "$message"
    fi
}

check_deferral_record() {
    local deferral_record="$ROOT_DIR/docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md"

    if [ ! -f "$deferral_record" ]; then
        fail "Sprint 171 package-manager deferral record is missing"
    fi

    require_grep \
        'Package-manager support is formally deferred' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer states package-manager support is deferred"
    require_grep \
        'No vcpkg' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer keeps vcpkg unsupported"
    require_grep \
        'Homebrew' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer names Homebrew as unsupported"
    require_grep \
        'Conan' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer names Conan as unsupported"
    require_grep \
        'pkgsrc' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer names pkgsrc as unsupported"
    require_grep \
        'provider registry readiness' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer keeps provider registry readiness unsupported"
    require_grep \
        'Evidence Needed To Revisit' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer lists evidence needed to revisit support"
    require_grep \
        'Downstream consumer proof' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer requires downstream consumer proof"
    require_grep \
        'Guard coverage' \
        "$deferral_record" \
        "Sprint 171 deferral record no longer requires guard coverage"

    pass "deferral record"
}

check_provider_recipe_absence() {
    local matches

    matches="$(
        find "$ROOT_DIR" \
            \( -path "$ROOT_DIR/.git" \
            -o -path "$ROOT_DIR/docs/planning" \
            -o -path "$ROOT_DIR/build" \
            -o -path "$ROOT_DIR/build-*" \
            -o -path "$ROOT_DIR/archive" \) -prune \
            -o \( -name 'vcpkg.json' \
            -o -name 'vcpkg-configuration.json' \
            -o -name 'portfile.cmake' \
            -o -name 'conanfile.py' \
            -o -name 'conanfile.txt' \
            -o -path '*/ports/*' \
            -o -path '*/Formula/*' \
            -o -path '*/pkgsrc/*' \
            -o -path '*/debian/control' \
            -o -path '*/debian/rules' \
            -o -path '*/debian/changelog' \
            -o -name '*.spec' \) -print
    )"

    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "unselected package-manager provider recipe artifacts appeared"
    fi

    pass "provider recipe absence"
}

check_selected_homebrew_local_proof() {
    local template="$ROOT_DIR/packaging/homebrew/sparse-lu-ortho.rb.in"
    local notes="$ROOT_DIR/packaging/homebrew/README.md"
    local proof="$ROOT_DIR/scripts/homebrew_local_formula_proof.sh"
    local output
    local status
    local generated
    local root_license_metadata

    [ -f "$template" ] || fail "selected Homebrew local formula template is missing"
    [ -f "$notes" ] || fail "selected Homebrew local proof notes are missing"
    [ -x "$proof" ] || fail "selected Homebrew local proof script is missing or not executable"

    require_grep \
        'temporary local formula' \
        "$template" \
        "Homebrew formula template no longer states rendered formula is temporary"
    require_grep \
        'Homebrew/core, bottles, Linuxbrew' \
        "$notes" \
        "Homebrew provider notes no longer reject broader Homebrew claims"
    require_grep \
        'local Homebrew formula proof only' \
        "$proof" \
        "Homebrew proof script no longer states local-only scope"
    require_grep \
        'no standalone LICENSE, COPYING, or NOTICE' \
        "$proof" \
        "Homebrew proof script no longer has the missing-license stop condition"
    require_grep \
        'not placeholder metadata' \
        "$proof" \
        "Homebrew proof script no longer rejects placeholder license metadata"
    require_grep \
        'source archive is missing required entry' \
        "$proof" \
        "Homebrew proof script no longer verifies required source archive entries"
    require_grep \
        'installed package metadata gained unsupported provider, shared-library, selector, or ABI wording' \
        "$proof" \
        "Homebrew proof script no longer rejects unsupported installed package metadata"
    require_grep \
        'formula test do block no longer requires exact-version find_package\(Sparse\)' \
        "$proof" \
        "Homebrew proof script no longer guards exact-version downstream CMake test"
    require_grep \
        'formula test do block no longer links Sparse::sparse_lu_ortho' \
        "$proof" \
        "Homebrew proof script no longer guards downstream imported target link"
    require_grep \
        'formula test do block no longer rejects shared-library artifacts' \
        "$proof" \
        "Homebrew proof script no longer guards downstream shared-artifact rejection"

    generated="$(
        find "$ROOT_DIR/packaging/homebrew" \
            \( -name '*.tar.gz' \
            -o -name '*.tgz' \
            -o -name '*.zip' \
            -o -name '*.log' \
            -o -name '*.rb' \
            -o -name '*.bottle.*' \
            -o -path '*/Formula/*' \) -print
    )"
    if [ -n "$generated" ]; then
        echo "$generated" >&2
        fail "generated Homebrew proof output appeared in source-controlled packaging path"
    fi

    root_license_metadata="$(
        find "$ROOT_DIR" -maxdepth 1 \
            \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) \
            -print -quit
    )"

    set +e
    output="$("$proof" 2>&1)"
    status=$?
    set -e

    case "$status" in
        0)
            if ! printf '%s\n' "$output" | grep -Fq 'local Homebrew formula proof'; then
                printf '%s\n' "$output" >&2
                fail "Homebrew proof success path no longer emits local proof scope"
            fi
            ;;
        2)
            if ! printf '%s\n' "$output" | grep -Fq 'local Homebrew proof remains unclaimed'; then
                printf '%s\n' "$output" >&2
                fail "Homebrew proof unavailable path no longer keeps support unclaimed"
            fi
            if [ -z "$root_license_metadata" ]; then
                if ! printf '%s\n' "$output" | grep -Fq 'no standalone LICENSE, COPYING, or NOTICE'; then
                    printf '%s\n' "$output" >&2
                    fail "Homebrew proof missing-license blocker no longer names standalone license metadata"
                fi
                if printf '%s\n' "$output" | grep -Eq 'temp root:|creating local source archive|archive sha256:|rendering temporary formula|installing local formula|running brew test'; then
                    printf '%s\n' "$output" >&2
                    fail "Homebrew proof should stop before archive/render/install/test work when root license metadata is absent"
                fi
            fi
            ;;
        *)
            printf '%s\n' "$output" >&2
            fail "Homebrew local proof script exited with unexpected status $status"
            ;;
    esac

    pass "selected Homebrew local proof boundary"
}

check_package_metadata_neutrality() {
    require_absent_grep \
        'vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|registry-ready|binary package|package-manager support' \
        "$ROOT_DIR/sparse.pc.in" \
        "pkg-config metadata gained package-manager provider wording"
    require_absent_grep \
        'vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|registry-ready|binary package|package-manager support' \
        "$ROOT_DIR/cmake/SparseConfig.cmake.in" \
        "CMake package config template gained package-manager provider wording"

    pass "package metadata neutrality"
}

check_public_nonclaims() {
    require_grep \
        'package-manager support' \
        "$ROOT_DIR/README.md" \
        "README no longer keeps package-manager support scoped as a non-claim"
    require_grep \
        'local Homebrew formula proof' \
        "$ROOT_DIR/README.md" \
        "README no longer records current local Homebrew proof status"
    require_grep \
        'package-manager distribution' \
        "$ROOT_DIR/README.md" \
        "README no longer separates package-manager distribution from source install evidence"
    require_grep \
        '^[[:space:]]*-[[:space:]]+package-manager deferral:' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer has the package-manager deferral support-split entry"
    require_grep \
        'Homebrew local formula proof artifacts exist' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer records the current Homebrew local proof blocker"
    require_grep \
        'not a user-facing Homebrew installation path' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer keeps the missing-license blocker out of user-facing Homebrew support"
    require_grep \
        'distribution, static/shared selectors' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer keeps package-manager distribution out of scope"
    require_grep \
        'do not present this template as an available Homebrew install method' \
        "$ROOT_DIR/packaging/homebrew/README.md" \
        "Homebrew README no longer keeps the local template unclaimed while proof is blocked"
    require_grep \
        'package-manager support' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer keeps package-manager support scoped as a non-claim"
    require_grep \
        'scripts/homebrew_local_formula_proof\.sh' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer documents the Homebrew local proof script"
    require_grep \
        'do not infer shared-library support, ABI stability, package-manager support' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer separates package-manager support from package evidence"

    pass "package-manager public non-claims"
}

check_deferral_record
check_provider_recipe_absence
check_selected_homebrew_local_proof
check_package_metadata_neutrality
check_public_nonclaims

echo "package-manager-deferral-check: passed"
