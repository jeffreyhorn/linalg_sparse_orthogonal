#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

/*
 * Minimal unit test framework for C.
 *
 * Usage:
 *   void test_something(void) {
 *       ASSERT_TRUE(1 == 1);
 *       ASSERT_EQ(2, 2);
 *       ASSERT_NEAR(3.14, 3.14159, 0.01);
 *       ASSERT_ERR(some_func(NULL), SPARSE_ERR_NULL);
 *   }
 *
 *   int main(void) {
 *       TEST_SUITE_BEGIN("My Tests");
 *       RUN_TEST(test_something);
 *       TEST_SUITE_END();
 *   }
 */

#include <math.h>
#include <stdio.h>
#include <time.h>

/* ─── Global state ───────────────────────────────────────────────────── */

static int tf_tests_run = 0;
static int tf_tests_failed = 0;
static int tf_asserts = 0;
static int tf_current_failed = 0;
static const char *tf_current_name = NULL;
static clock_t tf_suite_start;

/* ─── Suite begin / end ──────────────────────────────────────────────── */

#define TEST_SUITE_BEGIN(name)                                                                     \
    do {                                                                                           \
        printf("=== %s ===\n\n", (name));                                                          \
        tf_tests_run = tf_tests_failed = tf_asserts = 0;                                           \
        tf_suite_start = clock();                                                                  \
    } while (0)

#define TEST_SUITE_END()                                                                           \
    do {                                                                                           \
        double elapsed = (double)(clock() - tf_suite_start) / CLOCKS_PER_SEC;                      \
        printf("\n--- Summary ---\n");                                                             \
        printf("Tests run:    %d\n", tf_tests_run);                                                \
        printf("Tests failed: %d\n", tf_tests_failed);                                             \
        printf("Assertions:   %d\n", tf_asserts);                                                  \
        printf("Time:         %.3f s\n", elapsed);                                                 \
        if (tf_tests_failed == 0)                                                                  \
            printf("ALL TESTS PASSED\n");                                                          \
        else                                                                                       \
            printf("SOME TESTS FAILED\n");                                                         \
        return tf_tests_failed > 0 ? 1 : 0;                                                        \
    } while (0)

/* ─── Running a test ─────────────────────────────────────────────────── */

#define RUN_TEST(fn)                                                                               \
    do {                                                                                           \
        tf_current_name = #fn;                                                                     \
        tf_current_failed = 0;                                                                     \
        tf_tests_run++;                                                                            \
        fn();                                                                                      \
        if (tf_current_failed == 0)                                                                \
            printf("  [PASS] %s\n", #fn);                                                          \
    } while (0)

/* ─── Assertion helpers ──────────────────────────────────────────────── */

#define TF_FAIL_(fmt, ...)                                                                         \
    do {                                                                                           \
        if (tf_current_failed == 0) {                                                              \
            printf("  [FAIL] %s\n", tf_current_name);                                              \
            tf_tests_failed++;                                                                     \
        }                                                                                          \
        tf_current_failed++;                                                                       \
        printf("         %s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__);                      \
    } while (0)

#define ASSERT_TRUE(cond)                                                                          \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        if (!(cond))                                                                               \
            TF_FAIL_("ASSERT_TRUE(%s) failed", #cond);                                             \
    } while (0)

#define ASSERT_FALSE(cond)                                                                         \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        if ((cond))                                                                                \
            TF_FAIL_("ASSERT_FALSE(%s) failed", #cond);                                            \
    } while (0)

#define ASSERT_EQ(a, b)                                                                            \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        long long va_ = (long long)(a);                                                            \
        long long vb_ = (long long)(b);                                                            \
        if (va_ != vb_)                                                                            \
            TF_FAIL_("ASSERT_EQ(%s, %s): got %lld, expected %lld", #a, #b, va_, vb_);              \
    } while (0)

#define ASSERT_NEQ(a, b)                                                                           \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        long long va_ = (long long)(a);                                                            \
        long long vb_ = (long long)(b);                                                            \
        if (va_ == vb_)                                                                            \
            TF_FAIL_("ASSERT_NEQ(%s, %s): both are %lld", #a, #b, va_);                            \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                                     \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        double va_ = (double)(a);                                                                  \
        double vb_ = (double)(b);                                                                  \
        double vt_ = (double)(tol);                                                                \
        if (fabs(va_ - vb_) > vt_)                                                                 \
            TF_FAIL_("ASSERT_NEAR(%s, %s, %s): |%.15g - %.15g| = "                                 \
                     "%.3e > %.3e",                                                                \
                     #a, #b, #tol, va_, vb_, fabs(va_ - vb_), vt_);                                \
    } while (0)

#define ASSERT_ERR(expr, expected)                                                                 \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        sparse_err_t got_ = (expr);                                                                \
        sparse_err_t exp_ = (expected);                                                            \
        if (got_ != exp_)                                                                          \
            TF_FAIL_("ASSERT_ERR(%s, %s): got %d (%s), expected "                                  \
                     "%d (%s)",                                                                    \
                     #expr, #expected, (int)got_, sparse_strerror(got_), (int)exp_,                \
                     sparse_strerror(exp_));                                                       \
    } while (0)

/* Fatal assertion: asserts SPARSE_OK and returns from the test on failure.
 * Use when subsequent code would dereference pointers that are only valid
 * on success (e.g., ldlt.D after sparse_ldlt_factor). */
#define REQUIRE_OK(expr)                                                                           \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        sparse_err_t got_ = (expr);                                                                \
        if (got_ != SPARSE_OK) {                                                                   \
            TF_FAIL_("REQUIRE_OK(%s): got %d (%s), expected 0 (success)", #expr, (int)got_,        \
                     sparse_strerror(got_));                                                       \
            return;                                                                                \
        }                                                                                          \
    } while (0)

#define ASSERT_NULL(ptr)                                                                           \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        if ((ptr) != NULL)                                                                         \
            TF_FAIL_("ASSERT_NULL(%s) failed: got %p", #ptr, (void *)(ptr));                       \
    } while (0)

#define ASSERT_NOT_NULL(ptr)                                                                       \
    do {                                                                                           \
        tf_asserts++;                                                                              \
        if ((ptr) == NULL)                                                                         \
            TF_FAIL_("ASSERT_NOT_NULL(%s) failed", #ptr);                                          \
    } while (0)

/* ─── Cross-platform env-var helpers ─────────────────────────────────
 *
 * setenv() / unsetenv() are POSIX.1-2001 extensions and are NOT
 * available in MSVC's <stdlib.h>; the documented Windows / CMake /
 * ctest workflow needs `_putenv_s` instead.  Tests that mutate env
 * vars should call `tf_setenv` / `tf_unsetenv` rather than the
 * underlying functions directly so the test binaries compile across
 * platforms.
 *
 * Implemented as macros (not `static inline` functions) so the
 * underlying setenv/unsetenv references only appear at use sites.
 * That way only the .c files that actually CALL `tf_setenv` /
 * `tf_unsetenv` need to define `_POSIX_C_SOURCE >= 200809L` at the
 * top of the file (before any system header) — files that include
 * `test_framework.h` but don't touch env vars don't pay the
 * feature-test cost (and tsan CI builds against test_eigs.c etc.
 * don't fail with implicit-declaration errors).
 *
 * Both macros return 0 on success, non-zero on failure (matching
 * setenv()'s convention).  `tf_unsetenv` calls `_putenv_s(name, "")`
 * on Windows — note that empty-string-as-unset is the documented
 * MSVC convention (per `_putenv_s` MSDN); on POSIX we use unsetenv()
 * directly.  Caller-visible behaviour: `getenv(name)` returns NULL
 * after `tf_unsetenv(name)` on POSIX; on Windows `getenv(name)`
 * returns NULL after `tf_unsetenv(name)` because empty-string
 * environment entries are removed by `_putenv_s`. */
#ifdef _WIN32
#define tf_setenv(name, value) _putenv_s((name), (value))
#define tf_unsetenv(name) _putenv_s((name), "")
#else
/* On POSIX (glibc / Apple libc): caller's .c file must `#define
 * _POSIX_C_SOURCE 200809L` BEFORE any include so `setenv` /
 * `unsetenv` get declared by `<stdlib.h>`. */
#define tf_setenv(name, value) setenv((name), (value), /*overwrite=*/1)
#define tf_unsetenv(name) unsetenv((name))
#endif

/* ─── Cross-platform temp-file path helper ──────────────────────────
 *
 * Tests that round-trip Matrix Market files write to a temp directory.
 * POSIX has /tmp; Windows has $TEMP / $TMP and no /tmp at all
 * (`sparse_save_mm("/tmp/foo.mtx")` returns SPARSE_ERR_IO on Windows).
 * `tf_tmp(name)` returns a portable temp path:
 *
 *   POSIX  : $TMPDIR/<name>, falling back to "/tmp/<name>"
 *   Windows: $TEMP/<name>, falling back to $TMP/<name>, falling back to
 *            ".\<name>"
 *
 * Sprint 29 Day 14 added this when the Windows CMake CI matrix lit up
 * for the first time (Sprint 29 Days 7-8) and surfaced the hardcoded
 * "/tmp/..." literals in test_sparse_io.c + test_integration.c.
 *
 * Caveat: uses a function-static buffer for inline-call ergonomics
 * (`sparse_save_mm(A, tf_tmp("foo.mtx"))`).  Two consecutive calls
 * overwrite each other — callers that need to hold two paths
 * simultaneously must copy the returned string into a local buffer. */
#include <stdlib.h>
#include <string.h>
/* Manual concatenation (memcpy + strlen, no snprintf) so the helper
 * compiles in TUs that set `_POSIX_C_SOURCE 199309L` BEFORE any include
 * — at that POSIX level Apple's `<stdio.h>` hides snprintf behind
 * `__DARWIN_C_LEVEL >= __DARWIN_C_FULL` and Xcode 16's clang escalates
 * the implicit declaration to a hard error.  memcpy/strlen are not
 * feature-gated. */
static inline const char *tf_tmp(const char *name) {
    static char buf[260];
    const char *dir;
    char sep;
#ifdef _WIN32
    dir = getenv("TEMP");
    if (!dir)
        dir = getenv("TMP");
    if (!dir)
        dir = ".";
    sep = '\\';
#else
    dir = getenv("TMPDIR");
    if (!dir)
        dir = "/tmp";
    sep = '/';
#endif
    size_t len_d = strlen(dir);
    size_t len_n = strlen(name);
    /* Truncate cleanly if the composed path would overflow the
     * fixed buffer (260 bytes accommodates POSIX PATH_MAX + Windows
     * MAX_PATH; truncation is signalling that the test fixture name
     * is unreasonably long and we'd want to know). */
    if (len_d >= sizeof buf - 2)
        len_d = sizeof buf - 2;
    if (len_n >= sizeof buf - len_d - 1)
        len_n = sizeof buf - len_d - 2;
    memcpy(buf, dir, len_d);
    buf[len_d] = sep;
    memcpy(buf + len_d + 1, name, len_n);
    buf[len_d + 1 + len_n] = '\0';
    return buf;
}

#endif /* TEST_FRAMEWORK_H */
