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
FORMULA_CLASS="SparseLuOrthoLocal"
KEEP_TEMP=0
TMPROOT=""
INSTALLED=0

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

    if [ "$INSTALLED" -eq 1 ] && command -v brew >/dev/null 2>&1; then
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

    if find "$ROOT_DIR" -maxdepth 1 \
        \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) \
        -print -quit | grep -q .; then
        license_file_found=1
    fi

    if [ "$license_file_found" -ne 1 ]; then
        unavailable "formula rendering blocked: no standalone LICENSE, COPYING, or NOTICE file exists for provider metadata"
    fi

    if [ -z "${SPARSE_HOMEBREW_LICENSE:-}" ]; then
        unavailable "formula rendering blocked: SPARSE_HOMEBREW_LICENSE is not set to accurate local-proof license metadata"
    fi

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
    local license_path

    while IFS= read -r license_path; do
        entries+=("$(basename "$license_path")")
    done < <(find "$ROOT_DIR" -maxdepth 1 \
        \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) \
        -print | sort)

    tar -czf "$archive" -C "$ROOT_DIR" "${entries[@]}"
}

check_installed_static_surface() {
    local prefix
    local shared_artifacts
    local pc_file
    local targets_file

    prefix="$(brew --prefix "$FORMULA_NAME")"
    pc_file="$prefix/lib/pkgconfig/sparse.pc"
    targets_file="$prefix/lib/cmake/Sparse/SparseTargets.cmake"

    [ -f "$prefix/lib/libsparse_lu_ortho.a" ] || fail "static archive missing after Homebrew local formula install"
    [ -d "$prefix/include/sparse" ] || fail "installed sparse headers missing after Homebrew local formula install"
    [ -f "$prefix/lib/cmake/Sparse/SparseConfig.cmake" ] || fail "SparseConfig.cmake missing after Homebrew local formula install"
    [ -f "$pc_file" ] || fail "sparse.pc missing after Homebrew local formula install"

    if [ -f "$targets_file" ] && ! grep -Fq "Sparse::sparse_lu_ortho STATIC IMPORTED" "$targets_file"; then
        fail "installed CMake target metadata no longer proves static imported target"
    fi

    if grep -Eiq '^Libs\.private:|shared|soname|dylib|dll|abi|homebrew|apt|dnf|pacman|vcpkg|conan' "$pc_file"; then
        fail "installed sparse.pc gained unsupported provider, shared-library, or ABI wording"
    fi

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
require_tool "${CC:-cc}" "C compiler '${CC:-cc}' not found; local Homebrew proof prerequisites are unavailable"
checksum_file "$VERSION_FILE" >/dev/null

require_placeholder "__SPARSE_HOMEBREW_HOMEPAGE__"
require_placeholder "__SPARSE_FORMULA_URL__"
require_placeholder "__SPARSE_FORMULA_SHA256__"
require_placeholder "__SPARSE_VERSION__"
require_placeholder "__SPARSE_HOMEBREW_LICENSE__"
ruby -c "$TEMPLATE" >/dev/null

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
ARCHIVE_SHA256="$(checksum_file "$ARCHIVE")"
FORMULA_URL="file://$ARCHIVE"
info "archive sha256: $ARCHIVE_SHA256"

detect_license_metadata

info "rendering temporary formula: $FORMULA_FILE"
render_formula "$FORMULA_FILE"
ruby -c "$FORMULA_FILE" >/dev/null

info "installing local formula from source"
if ! brew install --build-from-source "$FORMULA_FILE" >"$INSTALL_LOG" 2>&1; then
    cat "$INSTALL_LOG" >&2
    fail "local Homebrew formula install proof failed; see $INSTALL_LOG"
fi
INSTALLED=1

info "checking static installed package surface"
check_installed_static_surface

info "running brew test for downstream CMake consumer"
if ! brew test "$FORMULA_NAME" >"$TEST_LOG" 2>&1; then
    cat "$TEST_LOG" >&2
    fail "local Homebrew formula downstream consumer proof failed; see $TEST_LOG"
fi

info "uninstalling local formula"
brew uninstall --force "$FORMULA_NAME" >/dev/null
INSTALLED=0

info "passed: local Homebrew formula proof completed for static source formula scope only"
