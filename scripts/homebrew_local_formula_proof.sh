#!/usr/bin/env bash
# homebrew_local_formula_proof.sh - local Homebrew formula proof.
#
# This script proves only the Sprint 180 local Homebrew formula path. It does
# not prove Homebrew/core, bottles, Linuxbrew, hosted binaries, or broad
# package-manager support.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$ROOT_DIR/packaging/homebrew/sparse-lu-ortho.rb.in"
VERSION_FILE="$ROOT_DIR/VERSION"
FORMULA_NAME="sparse-lu-ortho-local"
KEEP_TEMP=0
TMPROOT=""
UNINSTALL_ON_EXIT=0
LICENSE_METADATA_ENTRIES=()

usage() {
    cat <<'EOF'
Usage: scripts/homebrew_local_formula_proof.sh [--keep-temp]

Runs the Sprint 180 local Homebrew formula proof. The proof is local-only and
does not claim Homebrew/core, bottle, Linuxbrew, or broad package-manager
support.

Options:
  --keep-temp   Preserve generated archive/formula/log files for debugging.
  -h, --help    Show this help.
EOF
}

info() {
    echo "homebrew-local-formula-proof: $*"
}

fail() {
    echo "homebrew-local-formula-proof: FAIL: $*" >&2
    exit 1
}

unavailable() {
    echo "homebrew-local-formula-proof: UNAVAILABLE: $*" >&2
    echo "homebrew-local-formula-proof: local Homebrew proof remains unclaimed" >&2
    exit 2
}

cleanup() {
    local status=$?

    if [ "$UNINSTALL_ON_EXIT" -eq 1 ] && command -v brew >/dev/null 2>&1; then
        brew uninstall --force "$FORMULA_NAME" >/dev/null 2>&1 || {
            echo "homebrew-local-formula-proof: WARN: cleanup could not uninstall $FORMULA_NAME" >&2
        }
    fi

    if [ -n "$TMPROOT" ] && [ -d "$TMPROOT" ] && [ "$KEEP_TEMP" -eq 0 ]; then
        rm -rf "$TMPROOT"
    elif [ -n "$TMPROOT" ] && [ -d "$TMPROOT" ]; then
        echo "homebrew-local-formula-proof: kept temp root: $TMPROOT" >&2
    fi

    exit "$status"
}

trap cleanup EXIT

export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --keep-temp)
            KEEP_TEMP=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown option: $1"
            ;;
    esac
    shift
done

require_tool() {
    local tool="$1"
    local message="$2"

    if ! command -v "$tool" >/dev/null 2>&1; then
        unavailable "$message"
    fi
}

require_command() {
    local command_text="$1"
    local message="$2"
    local executable

    read -r executable _ <<EOF
$command_text
EOF
    if [ -z "$executable" ] || ! command -v "$executable" >/dev/null 2>&1; then
        unavailable "$message"
    fi
}

checksum_file() {
    local path="$1"

    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        unavailable "no SHA-256 tool found; install shasum or sha256sum"
    fi
}

require_placeholder() {
    local placeholder="$1"

    if ! grep -Fq "$placeholder" "$TEMPLATE"; then
        fail "template is missing required placeholder $placeholder"
    fi
}

require_template_text() {
    local text="$1"
    local message="$2"

    if ! grep -Fq "$text" "$TEMPLATE"; then
        fail "$message"
    fi
}

verify_formula_test_contract() {
    require_template_text \
        'find_package(Sparse #{expected_version} EXACT REQUIRED)' \
        "formula test do no longer requires exact-version find_package(Sparse)"
    require_template_text \
        'target_link_libraries(homebrew_local_formula_test PRIVATE Sparse::sparse_lu_ortho)' \
        "formula test do no longer links Sparse::sparse_lu_ortho"
    require_template_text \
        '#include <sparse/sparse_matrix.h>' \
        "formula test do no longer compiles against installed sparse_matrix header"
    require_template_text \
        '#include <sparse/sparse_types.h>' \
        "formula test do no longer compiles against installed sparse_types header"
    require_template_text \
        'assert_match "OK", output' \
        "formula test do no longer asserts successful downstream executable output"
    require_template_text \
        'raise "static archive missing after install"' \
        "formula test do no longer checks installed static archive"
    require_template_text \
        'raise "CMake package config missing after install"' \
        "formula test do no longer checks installed CMake package config"
    require_template_text \
        'raise "pkg-config metadata missing after install"' \
        "formula test do no longer checks installed pkg-config metadata"
    require_template_text \
        'shared-library artifacts are outside the local proof boundary' \
        "formula test do no longer rejects shared-library artifacts"
}

render_formula() {
    local output="$1"

    SPARSE_HOMEBREW_HOMEPAGE="$HOMEPAGE" \
    SPARSE_FORMULA_URL="$FORMULA_URL" \
    SPARSE_FORMULA_SHA256="$ARCHIVE_SHA256" \
    SPARSE_VERSION="$EXPECTED_VERSION" \
    SPARSE_HOMEBREW_LICENSE="$HOMEBREW_LICENSE" \
    ruby - "$TEMPLATE" "$output" <<'RUBY'
template = ARGV.fetch(0)
output = ARGV.fetch(1)
text = File.read(template)
replacements = {
  "__SPARSE_HOMEBREW_HOMEPAGE__" => ENV.fetch("SPARSE_HOMEBREW_HOMEPAGE"),
  "__SPARSE_FORMULA_URL__" => ENV.fetch("SPARSE_FORMULA_URL"),
  "__SPARSE_FORMULA_SHA256__" => ENV.fetch("SPARSE_FORMULA_SHA256"),
  "__SPARSE_VERSION__" => ENV.fetch("SPARSE_VERSION"),
  "__SPARSE_HOMEBREW_LICENSE__" => ENV.fetch("SPARSE_HOMEBREW_LICENSE")
}
replacements.each do |placeholder, value|
  abort("empty replacement for #{placeholder}") if value.empty?
  text = text.gsub(placeholder, value)
end
unresolved = text.scan(/__SPARSE_[A-Z0-9_]+__/).uniq
abort("unresolved placeholders: #{unresolved.join(", ")}") unless unresolved.empty?
File.write(output, text)
RUBY
}

detect_license_metadata() {
    local license_file_found=0
    local license_path

    LICENSE_METADATA_ENTRIES=()

    while IFS= read -r license_path; do
        LICENSE_METADATA_ENTRIES+=("$(basename "$license_path")")
        license_file_found=1
    done < <(find "$ROOT_DIR" -maxdepth 1 \
        \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) \
        -print | sort)

    if [ "$license_file_found" -ne 1 ]; then
        unavailable "formula rendering blocked: no standalone LICENSE, COPYING, or NOTICE file exists for provider metadata"
    fi

    if [ -z "${SPARSE_HOMEBREW_LICENSE:-}" ]; then
        unavailable "formula rendering blocked: SPARSE_HOMEBREW_LICENSE is not set to accurate local-proof license metadata"
    fi

    case "$SPARSE_HOMEBREW_LICENSE" in
        NOASSERTION|UNKNOWN|TBD|TODO|FIXME|PLACEHOLDER|__SPARSE_HOMEBREW_LICENSE__|*placeholder*|*Placeholder*|*PLACEHOLDER*)
            unavailable "formula rendering blocked: SPARSE_HOMEBREW_LICENSE must be an accurate Homebrew license identifier, not placeholder metadata"
            ;;
    esac

    HOMEBREW_LICENSE="$SPARSE_HOMEBREW_LICENSE"
}

make_source_archive() {
    local archive="$1"
    local entries=(
        CMakeLists.txt
        Makefile
        VERSION
        sparse.pc.in
        cmake
        include
        src
        examples
    )
    local entry

    for entry in "${LICENSE_METADATA_ENTRIES[@]}"; do
        entries+=("$entry")
    done

    if ! tar -czf "$archive" -C "$ROOT_DIR" "${entries[@]}"; then
        fail "could not create local Homebrew proof source archive: $archive"
    fi
}

archive_contains() {
    local archive="$1"
    local entry="$2"

    tar -tzf "$archive" | grep -Fxq "$entry" ||
        tar -tzf "$archive" | grep -Fxq "$entry/"
}

verify_source_archive() {
    local archive="$1"
    local required_entry
    local required_entries=(
        CMakeLists.txt
        Makefile
        VERSION
        sparse.pc.in
        cmake
        include
        src
        examples
    )

    for required_entry in "${required_entries[@]}" "${LICENSE_METADATA_ENTRIES[@]}"; do
        if ! archive_contains "$archive" "$required_entry"; then
            fail "source archive is missing required entry: $required_entry"
        fi
    done
}

check_installed_static_surface() {
    local cmake_package_dir
    local config_file
    local metadata_file
    local prefix
    local shared_artifacts
    local pc_file
    local targets_file
    local targets_noconfig_file
    local version_file

    prefix="$(brew --prefix "$FORMULA_NAME")"
    cmake_package_dir="$prefix/lib/cmake/Sparse"
    config_file="$cmake_package_dir/SparseConfig.cmake"
    pc_file="$prefix/lib/pkgconfig/sparse.pc"
    targets_file="$cmake_package_dir/SparseTargets.cmake"
    targets_noconfig_file="$cmake_package_dir/SparseTargets-noconfig.cmake"
    version_file="$cmake_package_dir/SparseConfigVersion.cmake"

    [ -d "$prefix" ] || fail "Homebrew prefix missing after local formula install"
    [ -d "$prefix/lib" ] || fail "installed lib directory missing after Homebrew local formula install"
    [ -f "$prefix/lib/libsparse_lu_ortho.a" ] || fail "static archive missing after Homebrew local formula install"
    [ -d "$prefix/include/sparse" ] || fail "installed sparse headers missing after Homebrew local formula install"
    [ -f "$config_file" ] || fail "SparseConfig.cmake missing after Homebrew local formula install"
    [ -f "$version_file" ] || fail "SparseConfigVersion.cmake missing after Homebrew local formula install"
    [ -f "$targets_file" ] || fail "SparseTargets.cmake missing after Homebrew local formula install"
    [ -f "$targets_noconfig_file" ] || fail "SparseTargets-noconfig.cmake missing after Homebrew local formula install"
    [ -f "$pc_file" ] || fail "sparse.pc missing after Homebrew local formula install"

    if ! grep -Fq "Sparse::sparse_lu_ortho STATIC IMPORTED" "$targets_file"; then
        fail "installed CMake target metadata no longer proves static imported target"
    fi

    if ! grep -Fq '${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a' "$targets_noconfig_file"; then
        fail "installed CMake target metadata no longer points at static archive"
    fi

    if grep -Eiv '^(prefix|exec_prefix|libdir|includedir)=' "$pc_file" | \
        grep -Eiq '^Libs\.private:|shared|soname|dylib|dll|abi|homebrew|apt|dnf|pacman|vcpkg|conan'; then
        fail "installed sparse.pc gained unsupported provider, shared-library, or ABI wording"
    fi

    for metadata_file in "$config_file" "$version_file" "$targets_file" "$targets_noconfig_file" "$pc_file"; do
        if grep -Eiq 'Homebrew|vcpkg|Conan|pkgsrc|apt|dnf|pacman|registry-ready|binary package|package-manager support|dynamic ABI|SOVERSION|SONAME|DLL|dylib|BUILD_SHARED_LIBS|SPARSE_(ABI|SHARED|STATIC)|Libs\.private|Sparse::.*shared|shared[_ -]?library' "$metadata_file"; then
            fail "installed package metadata gained unsupported provider, shared-library, selector, or ABI wording: $metadata_file"
        fi
    done

    shared_artifacts="$(find "$prefix/lib" "$prefix/bin" \
        \( -name '*.dylib' -o -name '*.so' -o -name '*.so.*' -o -name '*.dll' \) \
        2>/dev/null || true)"
    if [ -n "$shared_artifacts" ]; then
        echo "$shared_artifacts" >&2
        fail "shared-library artifacts appeared in static-only local Homebrew proof"
    fi
}

info "scope: local Homebrew formula proof only; no Homebrew/core, bottles, Linuxbrew, or broad package-manager support"

[ -f "$TEMPLATE" ] || fail "missing formula template: $TEMPLATE"
[ -f "$VERSION_FILE" ] || fail "missing VERSION file: $VERSION_FILE"
EXPECTED_VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"
[ -n "$EXPECTED_VERSION" ] || fail "VERSION is empty"

require_tool brew "brew not found; local Homebrew proof cannot run on this host"
require_tool cmake "cmake not found; local Homebrew proof prerequisites are unavailable"
require_tool ruby "ruby not found; local Homebrew proof prerequisites are unavailable"
require_tool tar "tar not found; local Homebrew proof prerequisites are unavailable"
require_command "${CC:-cc}" "C compiler '${CC:-cc}' not found; local Homebrew proof prerequisites are unavailable"
checksum_file "$VERSION_FILE" >/dev/null

require_placeholder "__SPARSE_HOMEBREW_HOMEPAGE__"
require_placeholder "__SPARSE_FORMULA_URL__"
require_placeholder "__SPARSE_FORMULA_SHA256__"
require_placeholder "__SPARSE_VERSION__"
require_placeholder "__SPARSE_HOMEBREW_LICENSE__"
ruby -c "$TEMPLATE" >/dev/null
verify_formula_test_contract
detect_license_metadata

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-homebrew-proof.XXXXXX")"
ARCHIVE="$TMPROOT/sparse-lu-ortho-$EXPECTED_VERSION.tar.gz"
FORMULA_DIR="$TMPROOT/tap/Formula"
FORMULA_FILE="$FORMULA_DIR/$FORMULA_NAME.rb"
INSTALL_LOG="$TMPROOT/brew-install.log"
TEST_LOG="$TMPROOT/brew-test.log"
HOMEPAGE="${SPARSE_HOMEBREW_HOMEPAGE:-https://github.com/local/sparse-lu-ortho-local-proof}"

mkdir -p "$FORMULA_DIR"

info "temp root: $TMPROOT"
info "creating local source archive"
make_source_archive "$ARCHIVE"
verify_source_archive "$ARCHIVE"
ARCHIVE_SHA256="$(checksum_file "$ARCHIVE")"
FORMULA_URL="file://$ARCHIVE"
info "archive sha256: $ARCHIVE_SHA256"

info "rendering temporary formula: $FORMULA_FILE"
render_formula "$FORMULA_FILE"
ruby -c "$FORMULA_FILE" >/dev/null

info "installing local formula from source"
UNINSTALL_ON_EXIT=1
if ! brew install --build-from-source "$FORMULA_FILE" >"$INSTALL_LOG" 2>&1; then
    cat "$INSTALL_LOG" >&2
    fail "local Homebrew formula install proof failed; see $INSTALL_LOG"
fi

info "checking static installed package surface"
check_installed_static_surface

info "running brew test for downstream CMake consumer"
if ! brew test "$FORMULA_NAME" >"$TEST_LOG" 2>&1; then
    cat "$TEST_LOG" >&2
    fail "local Homebrew formula downstream consumer proof failed; see $TEST_LOG"
fi

info "uninstalling local formula"
brew uninstall --force "$FORMULA_NAME" >/dev/null
UNINSTALL_ON_EXIT=0

info "passed: local Homebrew formula proof completed for static source formula scope only"
