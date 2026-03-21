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

#include <stdio.h>
#include <math.h>
#include <time.h>

/* ─── Global state ───────────────────────────────────────────────────── */

static int tf_tests_run    = 0;
static int tf_tests_failed = 0;
static int tf_asserts      = 0;
static int tf_current_failed = 0;
static const char *tf_current_name = NULL;
static clock_t tf_suite_start;

/* ─── Suite begin / end ──────────────────────────────────────────────── */

#define TEST_SUITE_BEGIN(name)                                        \
    do {                                                              \
        printf("=== %s ===\n\n", (name));                             \
        tf_tests_run = tf_tests_failed = tf_asserts = 0;              \
        tf_suite_start = clock();                                     \
    } while (0)

#define TEST_SUITE_END()                                              \
    do {                                                              \
        double elapsed = (double)(clock() - tf_suite_start)           \
                         / CLOCKS_PER_SEC;                            \
        printf("\n--- Summary ---\n");                                \
        printf("Tests run:    %d\n", tf_tests_run);                   \
        printf("Tests failed: %d\n", tf_tests_failed);               \
        printf("Assertions:   %d\n", tf_asserts);                    \
        printf("Time:         %.3f s\n", elapsed);                   \
        if (tf_tests_failed == 0)                                     \
            printf("ALL TESTS PASSED\n");                             \
        else                                                          \
            printf("SOME TESTS FAILED\n");                            \
        return tf_tests_failed > 0 ? 1 : 0;                          \
    } while (0)

/* ─── Running a test ─────────────────────────────────────────────────── */

#define RUN_TEST(fn)                                                  \
    do {                                                              \
        tf_current_name = #fn;                                        \
        tf_current_failed = 0;                                        \
        tf_tests_run++;                                               \
        fn();                                                         \
        if (tf_current_failed == 0)                                   \
            printf("  [PASS] %s\n", #fn);                             \
    } while (0)

/* ─── Assertion helpers ──────────────────────────────────────────────── */

#define TF_FAIL_(fmt, ...)                                            \
    do {                                                              \
        if (tf_current_failed == 0) {                                 \
            printf("  [FAIL] %s\n", tf_current_name);                 \
            tf_tests_failed++;                                        \
        }                                                             \
        tf_current_failed++;                                          \
        printf("         %s:%d: " fmt "\n",                           \
               __FILE__, __LINE__, __VA_ARGS__);                      \
    } while (0)

#define ASSERT_TRUE(cond)                                             \
    do {                                                              \
        tf_asserts++;                                                 \
        if (!(cond))                                                  \
            TF_FAIL_("ASSERT_TRUE(%s) failed", #cond);               \
    } while (0)

#define ASSERT_FALSE(cond)                                            \
    do {                                                              \
        tf_asserts++;                                                 \
        if ((cond))                                                   \
            TF_FAIL_("ASSERT_FALSE(%s) failed", #cond);              \
    } while (0)

#define ASSERT_EQ(a, b)                                               \
    do {                                                              \
        tf_asserts++;                                                 \
        long long va_ = (long long)(a);                               \
        long long vb_ = (long long)(b);                               \
        if (va_ != vb_)                                               \
            TF_FAIL_("ASSERT_EQ(%s, %s): got %lld, expected %lld",   \
                     #a, #b, va_, vb_);                               \
    } while (0)

#define ASSERT_NEQ(a, b)                                              \
    do {                                                              \
        tf_asserts++;                                                 \
        long long va_ = (long long)(a);                               \
        long long vb_ = (long long)(b);                               \
        if (va_ == vb_)                                               \
            TF_FAIL_("ASSERT_NEQ(%s, %s): both are %lld",            \
                     #a, #b, va_);                                    \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                        \
    do {                                                              \
        tf_asserts++;                                                 \
        double va_ = (double)(a);                                     \
        double vb_ = (double)(b);                                     \
        double vt_ = (double)(tol);                                   \
        if (fabs(va_ - vb_) > vt_)                                    \
            TF_FAIL_("ASSERT_NEAR(%s, %s, %s): |%.15g - %.15g| = "   \
                     "%.3e > %.3e",                                   \
                     #a, #b, #tol, va_, vb_,                          \
                     fabs(va_ - vb_), vt_);                           \
    } while (0)

#define ASSERT_ERR(expr, expected)                                    \
    do {                                                              \
        tf_asserts++;                                                 \
        sparse_err_t got_ = (expr);                                   \
        sparse_err_t exp_ = (expected);                               \
        if (got_ != exp_)                                             \
            TF_FAIL_("ASSERT_ERR(%s, %s): got %d (%s), expected "    \
                     "%d (%s)",                                       \
                     #expr, #expected,                                \
                     (int)got_, sparse_strerror(got_),                \
                     (int)exp_, sparse_strerror(exp_));               \
    } while (0)

#define ASSERT_NULL(ptr)                                              \
    do {                                                              \
        tf_asserts++;                                                 \
        if ((ptr) != NULL)                                            \
            TF_FAIL_("ASSERT_NULL(%s) failed: got %p", #ptr,         \
                     (void *)(ptr));                                  \
    } while (0)

#define ASSERT_NOT_NULL(ptr)                                          \
    do {                                                              \
        tf_asserts++;                                                 \
        if ((ptr) == NULL)                                            \
            TF_FAIL_("ASSERT_NOT_NULL(%s) failed", #ptr);            \
    } while (0)

#endif /* TEST_FRAMEWORK_H */
