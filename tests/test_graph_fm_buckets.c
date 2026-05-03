/*
 * Sprint 23 Day 9 — gain-bucket data structure unit tests.
 *
 * Pins the API contract for `src/sparse_graph_fm_buckets.h`:
 *
 *   - 100-vertex insert / pop_max-all sweep: popped sequence is
 *     non-increasing in gain.  Proves the cursor-based max-find
 *     returns the highest-gain vertex on every call.
 *   - 1000-operation random insert / remove / pop_max cycle: drives
 *     the cursor up + down across the full bucket range.  Run
 *     under ASan + UBSan (the suite-level `make sanitize` target)
 *     to catch any double-free / OOB / undefined-linkage bug the
 *     pop sequence alone wouldn't surface.
 *
 * Day 10 wires the same API into `graph_refine_fm`'s hot loop; this
 * test is the one that gates the swap.
 */
#include "sparse_graph_fm_buckets.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* splitmix64 PRNG — same seed pattern as `sparse_graph.c`'s coarsening
 * RNG so the fixture is reproducible across compilers / architectures. */
static uint64_t splitmix64_test(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Map a uint64_t to a signed gain in `[-max_gain, +max_gain]` inclusive. */
static idx_t random_gain(uint64_t r, idx_t max_gain) {
    idx_t span = 2 * max_gain + 1;
    return (idx_t)(r % (uint64_t)span) - max_gain;
}

/* ─── 100-vertex pop-all sweep: pop sequence is non-increasing ─────── */

static void test_fm_buckets_pop_all_non_increasing(void) {
    const idx_t n = 100;
    const idx_t max_gain = 50;
    fm_bucket_array_t arr = {0};
    REQUIRE_OK(fm_bucket_array_init(&arr, n, max_gain));

    /* Insert 100 vertices with random gains in [-50, +50]. */
    idx_t inserted_gain[100];
    uint64_t state = 0xFEEDC0DECAFEBABEULL;
    for (idx_t v = 0; v < n; v++) {
        idx_t g = random_gain(splitmix64_test(&state), max_gain);
        inserted_gain[v] = g;
        fm_bucket_insert(&arr, v, g);
    }

    /* Pop all 100 out via pop_max; assert non-increasing in gain. */
    idx_t prev_gain = max_gain + 1; /* one above the maximum legal gain */
    int seen[100] = {0};
    for (idx_t step = 0; step < n; step++) {
        idx_t v = -1;
        idx_t g = 0;
        REQUIRE_OK(fm_bucket_pop_max(&arr, &v, &g));
        ASSERT_TRUE(v >= 0 && v < n);
        ASSERT_TRUE(g <= prev_gain);
        ASSERT_EQ(g, inserted_gain[v]); /* gain matches what we inserted with */
        ASSERT_FALSE(seen[v]);          /* no double-pop */
        seen[v] = 1;
        prev_gain = g;
    }

    /* Bucket is now empty: pop_max returns SPARSE_ERR_BOUNDS. */
    idx_t v_dummy = -2;
    idx_t g_dummy = -2;
    ASSERT_ERR(fm_bucket_pop_max(&arr, &v_dummy, &g_dummy), SPARSE_ERR_BOUNDS);
    /* Outputs untouched on the empty path. */
    ASSERT_EQ(v_dummy, -2);
    ASSERT_EQ(g_dummy, -2);

    fm_bucket_array_free(&arr);
}

/* ─── 1000-op insert / remove / pop_max stress + cursor invariant ──── */

/* Recompute the expected cursor from `counts[]` and assert it matches
 * the structure's internal cursor.  Linear in `num_buckets` — fine
 * for a unit test, would defeat the O(1) contract in production use. */
static void assert_cursor_invariant(const fm_bucket_array_t *arr) {
    idx_t expected = -1;
    for (idx_t b = arr->num_buckets - 1; b >= 0; b--) {
        if (arr->counts[b] > 0) {
            expected = b;
            break;
        }
    }
    ASSERT_EQ(arr->cursor, expected);
}

static void test_fm_buckets_random_stress(void) {
    const idx_t n = 200;
    const idx_t max_gain = 30;
    fm_bucket_array_t arr = {0};
    REQUIRE_OK(fm_bucket_array_init(&arr, n, max_gain));

    /* Track per-vertex membership state externally so the test can
     * legally call insert/remove without violating the bucket-array's
     * "vertex appears in at most one bucket" contract. */
    idx_t cur_gain[200];
    int in_bucket[200];
    for (idx_t v = 0; v < n; v++) {
        cur_gain[v] = 0;
        in_bucket[v] = 0;
    }

    uint64_t state = 0xDEADBEEFCAFEBABEULL;
    int n_inserts = 0;
    int n_removes = 0;
    int n_pops = 0;

    for (int op = 0; op < 1000; op++) {
        uint64_t r = splitmix64_test(&state);
        int kind = (int)(r % 3); /* 0=insert, 1=remove, 2=pop_max */
        idx_t v = (idx_t)((r >> 8) % (uint64_t)n);
        idx_t g = random_gain(splitmix64_test(&state), max_gain);

        if (kind == 0) {
            /* Insert: only if vertex isn't already in a bucket. */
            if (!in_bucket[v]) {
                fm_bucket_insert(&arr, v, g);
                in_bucket[v] = 1;
                cur_gain[v] = g;
                n_inserts++;
            }
        } else if (kind == 1) {
            /* Remove: only if vertex is currently in a bucket. */
            if (in_bucket[v]) {
                fm_bucket_remove(&arr, v, cur_gain[v]);
                in_bucket[v] = 0;
                n_removes++;
            }
        } else {
            /* pop_max: pulls highest-gain vertex out, regardless of which. */
            idx_t pv = -1;
            idx_t pg = 0;
            sparse_err_t rc = fm_bucket_pop_max(&arr, &pv, &pg);
            if (rc == SPARSE_OK) {
                ASSERT_TRUE(pv >= 0 && pv < n);
                ASSERT_TRUE(in_bucket[pv]);
                ASSERT_EQ(pg, cur_gain[pv]);
                in_bucket[pv] = 0;
                n_pops++;
            } else {
                ASSERT_ERR(rc, SPARSE_ERR_BOUNDS);
            }
        }

        assert_cursor_invariant(&arr);
    }

    /* Drain the rest with pop_max; each pop returns the highest
     * remaining gain. */
    idx_t prev_gain = max_gain + 1;
    while (1) {
        idx_t pv = -1;
        idx_t pg = 0;
        sparse_err_t rc = fm_bucket_pop_max(&arr, &pv, &pg);
        if (rc != SPARSE_OK) {
            ASSERT_ERR(rc, SPARSE_ERR_BOUNDS);
            break;
        }
        ASSERT_TRUE(pg <= prev_gain);
        ASSERT_TRUE(in_bucket[pv]);
        in_bucket[pv] = 0;
        prev_gain = pg;
    }

    /* Every vertex is now out of any bucket. */
    for (idx_t v = 0; v < n; v++)
        ASSERT_FALSE(in_bucket[v]);

    printf("    fm_buckets stress: %d inserts, %d removes, %d pops over 1000 ops\n", n_inserts,
           n_removes, n_pops);

    fm_bucket_array_free(&arr);
}

/* ─── Argument validation + free idempotency ──────────────────────── */

static void test_fm_buckets_init_arg_validation(void) {
    fm_bucket_array_t arr = {0};
    ASSERT_ERR(fm_bucket_array_init(NULL, 100, 50), SPARSE_ERR_NULL);
    ASSERT_ERR(fm_bucket_array_init(&arr, -1, 50), SPARSE_ERR_BADARG);
    ASSERT_ERR(fm_bucket_array_init(&arr, 100, -1), SPARSE_ERR_BADARG);
}

static void test_fm_buckets_free_idempotent(void) {
    fm_bucket_array_t arr = {0};
    /* Free of a zero-init struct is a no-op (no spurious frees). */
    fm_bucket_array_free(&arr);
    /* NULL is also tolerated. */
    fm_bucket_array_free(NULL);
    /* Init + free + free (second free should observe NULL pointers
     * post-first-free and no-op cleanly). */
    REQUIRE_OK(fm_bucket_array_init(&arr, 10, 5));
    fm_bucket_array_free(&arr);
    fm_bucket_array_free(&arr);
}

/* ─── Empty bucket array: pop_max on n=0 returns BOUNDS ──────────── */

static void test_fm_buckets_empty(void) {
    fm_bucket_array_t arr = {0};
    REQUIRE_OK(fm_bucket_array_init(&arr, 0, 5));
    idx_t v = -2;
    idx_t g = -2;
    ASSERT_ERR(fm_bucket_pop_max(&arr, &v, &g), SPARSE_ERR_BOUNDS);
    fm_bucket_array_free(&arr);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 23 Day 9: gain-bucket data structure for FM refinement");
    RUN_TEST(test_fm_buckets_init_arg_validation);
    RUN_TEST(test_fm_buckets_free_idempotent);
    RUN_TEST(test_fm_buckets_empty);
    RUN_TEST(test_fm_buckets_pop_all_non_increasing);
    RUN_TEST(test_fm_buckets_random_stress);
    TEST_SUITE_END();
}
