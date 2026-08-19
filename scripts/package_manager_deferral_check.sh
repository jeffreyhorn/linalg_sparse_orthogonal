#!/usr/bin/env bash
# package_manager_deferral_check.sh - package-manager non-claim guard.
#
# This script proves that package-manager support remains formally deferred.
# It does not invoke provider tooling because Sprint 171 selected deferral,
# not a provider recipe or package-manager proof.

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
        'package-manager distribution' \
        "$ROOT_DIR/README.md" \
        "README no longer separates package-manager distribution from source install evidence"
    require_grep \
        'package-manager[[:space:]]*$' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer starts the package-manager distribution non-claim"
    require_grep \
        'distribution, static/shared selectors' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer keeps package-manager distribution out of scope"
    require_grep \
        'package-manager support' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer keeps package-manager support scoped as a non-claim"
    require_grep \
        'do not infer shared-library support, ABI stability, package-manager support' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer separates package-manager support from package evidence"

    pass "package-manager public non-claims"
}

check_deferral_record
check_provider_recipe_absence
check_package_metadata_neutrality
check_public_nonclaims

echo "package-manager-deferral-check: passed"
