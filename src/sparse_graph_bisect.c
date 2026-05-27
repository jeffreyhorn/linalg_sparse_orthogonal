/*
 * sparse_graph_bisect.c — coarse-level bisection slice extracted from
 * Sprint 43 Day 9's first graph-bisection decomposition batch.
 *
 * This file owns:
 *   - brute-force coarse bisection
 *   - GGGP coarse bisection
 *   - spectral coarse bisection helpers
 *   - coarsest-bisection strategy parsing and dispatch
 *
 * It intentionally stays narrow: no hierarchy/coarsening ownership,
 * no FM refinement, no uncoarsening, and no top-level partition
 * orchestration.
 */

#include "sparse_eigs.h"
#include "sparse_graph_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Brute-force minimum-cut bisection for n ≤ 20.  Vertex 0 is fixed
 * to side 0 (the side-swapped mirror has identical cut, so this
 * halves the search), then 2^(n-1) ≤ 524288 patterns are scanned.
 * The lowest-cut pattern that satisfies vertex-weight balance
 * |w0 - w1| ≤ max_vwgt wins. */
static sparse_err_t bisect_brute_force(const sparse_graph_t *G, idx_t *part_out) {
    idx_t n = G->n;
    if (n == 1) {
        part_out[0] = 0;
        return SPARSE_OK;
    }

    idx_t max_vwgt = 1;
    for (idx_t i = 0; i < n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (w > max_vwgt)
            max_vwgt = w;
    }
    /* Tolerance: max_vwgt allows a single-vertex move to balance. */
    idx_t tolerance = max_vwgt;

    int have_best = 0;
    idx_t best_cut = 0;
    uint32_t best_pat = 0;
    /* `mid_pat` is a fallback for the (rare) case where no balanced
     * partition exists within tolerance — pick the most-balanced
     * pattern at the lowest imbalance seen so the routine never
     * returns garbage. */
    int have_mid = 0;
    idx_t best_imbal = 0;
    uint32_t mid_pat = 0;
    idx_t mid_cut = 0;

    uint32_t total_pats = 1U << (uint32_t)(n - 1);
    for (uint32_t p = 0; p < total_pats; p++) {
        uint32_t pattern = p << 1; /* bit 0 = vertex 0's side = 0 */

        idx_t w0 = 0;
        idx_t w1 = 0;
        for (idx_t i = 0; i < n; i++) {
            idx_t w = G->vwgt ? G->vwgt[i] : 1;
            if ((pattern >> (uint32_t)i) & 1U)
                w1 += w;
            else
                w0 += w;
        }
        idx_t imbal = w0 > w1 ? w0 - w1 : w1 - w0;
        idx_t cut = 0;
        for (idx_t i = 0; i < n; i++) {
            uint32_t side_i = (pattern >> (uint32_t)i) & 1U;
            for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
                idx_t j = G->adjncy[k];
                if (j <= i)
                    continue;
                uint32_t side_j = (pattern >> (uint32_t)j) & 1U;
                if (side_i != side_j)
                    cut += G->ewgt ? G->ewgt[k] : 1;
            }
        }

        if (imbal <= tolerance) {
            if (!have_best || cut < best_cut) {
                have_best = 1;
                best_cut = cut;
                best_pat = pattern;
            }
        }
        if (!have_mid || imbal < best_imbal || (imbal == best_imbal && cut < mid_cut)) {
            have_mid = 1;
            best_imbal = imbal;
            mid_pat = pattern;
            mid_cut = cut;
        }
    }

    uint32_t winner = have_best ? best_pat : mid_pat;
    for (idx_t i = 0; i < n; i++)
        part_out[i] = (winner >> (uint32_t)i) & 1U;
    return SPARSE_OK;
}

/* BFS from `start` filling `dist[v]` (-1 if unreachable).  Caller
 * provides scratch queue of length ≥ G->n. */
static void bfs_distances(const sparse_graph_t *G, idx_t start, idx_t *dist, idx_t *queue) {
    for (idx_t i = 0; i < G->n; i++)
        dist[i] = -1;
    dist[start] = 0;
    idx_t head = 0;
    idx_t tail = 0;
    queue[tail++] = start;
    while (head < tail) {
        idx_t v = queue[head++];
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                /* `tail < G->n` is invariant: each vertex enters the
                 * queue at most once (gated by the `dist[u] == -1`
                 * check above), so over the lifetime of the BFS at
                 * most G->n entries get appended.  clang-analyzer
                 * doesn't track the dist[]-vs-queue invariant; this
                 * suppression matches the existing pattern at
                 * sparse_graph.c:269/301/622/624/656 + Sprint 22's
                 * sparse_etree.c. */
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                queue[tail++] = u;
            }
        }
    }
}

/* Greedy Graph-Growing Partition (METIS §3) for n in [21, 40]:
 * find a peripheral vertex via two BFS passes, BFS-grow side 0 from
 * it until half the vertex weight is consumed, leave the rest on
 * side 1.  The resulting partition is often coarsely balanced —
 * Day 4's per-level FM refinement is what actually polishes the
 * cut. */
static sparse_err_t bisect_gggp(const sparse_graph_t *G, idx_t *part_out) {
    idx_t n = G->n;
    idx_t *dist = malloc((size_t)n * sizeof(idx_t));
    idx_t *queue = malloc((size_t)n * sizeof(idx_t));
    int *visited = calloc((size_t)n, sizeof(int));
    if (!dist || !queue || !visited) {
        free(dist);
        free(queue);
        free(visited);
        return SPARSE_ERR_ALLOC;
    }

    /* Two-BFS peripheral-vertex finder. */
    bfs_distances(G, 0, dist, queue);
    idx_t v0 = 0;
    idx_t best_d = 0;
    for (idx_t i = 0; i < n; i++) {
        if (dist[i] > best_d) {
            best_d = dist[i];
            v0 = i;
        }
    }
    bfs_distances(G, v0, dist, queue);
    idx_t v_periph = v0;
    best_d = 0;
    for (idx_t i = 0; i < n; i++) {
        if (dist[i] > best_d) {
            best_d = dist[i];
            v_periph = i;
        }
    }

    idx_t total_vwgt = 0;
    for (idx_t i = 0; i < n; i++)
        total_vwgt += G->vwgt ? G->vwgt[i] : 1;
    idx_t target = total_vwgt / 2;

    for (idx_t i = 0; i < n; i++)
        part_out[i] = 1;

    /* BFS from peripheral; stop assigning to side 0 once target is
     * reached (or surpassed by the most recent push).  Disconnected
     * components beyond the periphery's cluster stay on side 1. */
    idx_t head = 0;
    idx_t tail = 0;
    queue[tail++] = v_periph;
    visited[v_periph] = 1;
    idx_t consumed = 0;
    {
        idx_t w = G->vwgt ? G->vwgt[v_periph] : 1;
        part_out[v_periph] = 0;
        consumed += w;
    }
    while (head < tail && consumed < target) {
        idx_t v = queue[head++];
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            if (visited[u])
                continue;
            visited[u] = 1;
            queue[tail++] = u;
            idx_t w = G->vwgt ? G->vwgt[u] : 1;
            if (consumed + w > target + 1 && consumed > 0)
                continue; /* would overshoot; leave on side 1 */
            part_out[u] = 0;
            consumed += w;
        }
    }

    free(visited);
    free(queue);
    free(dist);
    return SPARSE_OK;
}

/* Sprint 25 Day 6: Laplacian builder for spectral bisection.
 *
 * L = D - A where D is the diagonal degree matrix and A is the
 * adjacency matrix.  Symmetric, positive semi-definite (smallest
 * eigenvalue λ_0 = 0); for connected graphs the next eigenvalue
 * λ_1 > 0 and its eigenvector v_1 (the Fiedler vector) is what
 * Day 7-8's spectral bisection uses for partition selection.
 *
 * For unit-weighted graphs (G->ewgt == NULL), edge weight = 1, so
 * L[i][i] = degree(i) and L[i][j] = -1.  See
 * docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md. */
sparse_err_t graph_build_laplacian(const sparse_graph_t *G, SparseMatrix **L_out) {
    if (!G || !L_out)
        return SPARSE_ERR_NULL;
    *L_out = NULL;

    SparseMatrix *L = sparse_create(G->n, G->n);
    if (!L)
        return SPARSE_ERR_ALLOC;

    if (G->n == 0) {
        *L_out = L;
        return SPARSE_OK;
    }

    /* For each vertex i: emit -weight(i, j) for every j adjacent to
     * i (off-diagonals); accumulate the row's weight sum for the
     * diagonal entry.  The graph adjacency is symmetric, so the
     * resulting matrix is symmetric too. */
    for (idx_t i = 0; i < G->n; i++) {
        idx_t row_sum = 0;
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            idx_t w = G->ewgt ? G->ewgt[k] : 1;
            sparse_err_t rc = sparse_insert(L, i, j, -(double)w);
            if (rc != SPARSE_OK) {
                sparse_free(L);
                return rc;
            }
            row_sum += w;
        }
        /* Diagonal = sum of incident edge weights (= weighted degree).
         * For an isolated vertex this stays 0, matching the Laplacian
         * definition for disconnected components. */
        sparse_err_t rc = sparse_insert(L, i, i, (double)row_sum);
        if (rc != SPARSE_OK) {
            sparse_free(L);
            return rc;
        }
    }

    *L_out = L;
    return SPARSE_OK;
}

/* qsort comparator: strictly-ascending double values. */
static int cmp_double_asc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

/* Sprint 25 Day 7: spectral bisection at the coarsest level.
 *
 * Implements the algorithm specified in
 * docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md:
 *   1. Build Laplacian L = D - A via graph_build_laplacian.
 *   2. Compute the smallest two eigenpairs of L via sparse_eigs_sym
 *      (which = SPARSE_EIGS_SMALLEST, k=2, compute_vectors=1,
 *      reorthogonalize=1, tol=1e-8).
 *   3. Extract the Fiedler vector v_1 (column 1 of result.eigenvectors).
 *   4. Detect disconnected graphs via λ_1 ≈ 0; fall back to GGGP.
 *   5. Compute median(v_1) and assign part[i] = 0 if v_1[i] < median
 *      else 1.
 *   6. Check the 60/40 balance contract; on imbalance, fall back to
 *      GGGP.
 *   7. On any sparse_eigs_sym failure (allocation, non-convergence),
 *      fall back to GGGP.
 *
 * Return contract: ALWAYS produces a valid {0, 1} partition in
 * part_out on SPARSE_OK return.  GGGP is the universal fallback —
 * the spectral path is opt-in via SPARSE_ND_COARSEST_BISECTION=spectral
 * but never breaks the basic {valid partition produced} contract.
 * Trivial sizes (n ≤ 2) skip Lanczos entirely. */
sparse_err_t graph_bisect_coarsest_spectral(const sparse_graph_t *G, idx_t *part_out) {
    if (!G || !part_out)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    /* Trivial sizes: no point invoking Lanczos.  n=1 produces a
     * degenerate single-vertex partition; n=2 produces the unique
     * 2-way split. */
    if (G->n == 1) {
        part_out[0] = 0;
        return SPARSE_OK;
    }
    if (G->n == 2) {
        part_out[0] = 0;
        part_out[1] = 1;
        return SPARSE_OK;
    }

    /* Build Laplacian. */
    SparseMatrix *L = NULL;
    sparse_err_t rc = graph_build_laplacian(G, &L);
    if (rc != SPARSE_OK)
        return rc;

    /* Allocate eigenvalue + eigenvector buffers (k=2; column-major
     * eigenvectors stored as [n_components × k]).  On any allocation
     * failure, free the Laplacian + fall back to GGGP. */
    idx_t n = G->n;
    double *eigvals = malloc(2 * sizeof(double));
    double *eigvecs = malloc((size_t)n * 2 * sizeof(double));
    if (!eigvals || !eigvecs) {
        free(eigvals);
        free(eigvecs);
        sparse_free(L);
        return bisect_gggp(G, part_out);
    }

    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_SMALLEST,
        .sigma = 0.0,
        .max_iterations = 0, /* library default */
        .tol = 1e-8,
        .reorthogonalize = 1,
        .compute_vectors = 1,
        .backend = SPARSE_EIGS_BACKEND_AUTO,
        .block_size = 0,
    };
    sparse_eigs_t result = {
        .eigenvalues = eigvals,
        .eigenvectors = eigvecs,
    };
    sparse_err_t eigs_rc = sparse_eigs_sym(L, /*k=*/2, &opts, &result);
    sparse_free(L);

    /* On Lanczos failure or insufficient convergence, fall back to
     * GGGP.  Both eigenpairs must converge for the Fiedler vector
     * to be meaningful. */
    if (eigs_rc != SPARSE_OK || result.n_converged < 2) {
        free(eigvals);
        free(eigvecs);
        return bisect_gggp(G, part_out);
    }

    /* Disconnected graph detection: a Laplacian's algebraic
     * connectivity is λ_1 > 0 for connected graphs.  When the graph
     * has multiple components, λ_1 ≈ 0 (within numerical tolerance),
     * and v_1 is degenerate (lives in the span of the components'
     * indicator vectors).  Threshold: λ_1 > 1e-6 to distinguish from
     * the trivial λ_0 = 0. */
    double lambda_0 = eigvals[0];
    double lambda_1 = eigvals[1];
    if (lambda_1 - lambda_0 < 1e-6) {
        free(eigvals);
        free(eigvecs);
        return bisect_gggp(G, part_out);
    }

    /* Compute median of the Fiedler vector v_1.  Column-major layout:
     * v_1 is stored at eigvecs[n..2n-1].  eigvecs was allocated with
     * size n*2 doubles (line above) and we returned early for n <= 2,
     * so eigvecs[n] is in-bounds when n >= 3.  clang-analyzer doesn't
     * track this allocation/branch invariant under the
     * sparse_graph_partition → ... → spectral call chain. */
    // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
    double *v1 = &eigvecs[n];
    double *sorted = malloc((size_t)n * sizeof(double));
    if (!sorted) {
        free(eigvals);
        free(eigvecs);
        return bisect_gggp(G, part_out);
    }
    memcpy(sorted, v1, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), cmp_double_asc);
    double median = sorted[n / 2];
    free(sorted);

    /* Median partition: part[i] = 0 if v_1[i] < median else 1.
     * Vertices with v_1[i] == median go to side 1 (deterministic).
     * The PLAN's "lower-id tie-break" is implicit here: vertex-id
     * ordering doesn't affect partition assignment because the
     * median value is computed deterministically and comparisons
     * are stable; what matters is that the same input produces
     * the same output. */
    for (idx_t i = 0; i < n; i++) {
        part_out[i] = (v1[i] < median) ? 0 : 1;
    }

    free(eigvals);
    free(eigvecs);

    /* 60/40 balance check: if min(side_0, side_1) / max < 0.4 the
     * Fiedler cut is too skewed for ND's recursion-balance contract
     * (depth would blow past level_cap on Pres_Poisson-class
     * fixtures).  Common trigger: star-graph fixtures, where the
     * Fiedler cut puts the hub on one side and all leaves on the
     * other (1/(n-1) imbalance).  Falls back to bisect_gggp, which
     * produces a vertex-weighted balanced cut. */
    idx_t n0 = 0;
    idx_t n1 = 0;
    for (idx_t i = 0; i < n; i++) {
        if (part_out[i] == 0)
            n0++;
        else
            n1++;
    }
    idx_t lo = (n0 < n1) ? n0 : n1;
    idx_t hi = (n0 < n1) ? n1 : n0;
    /* `10 * lo < 4 * hi` ⇔ lo/hi < 0.4 — integer arithmetic
     * avoids floating-point.  Cast to int64_t to avoid idx_t
     * overflow on large graphs (matches Sprint 24 Day 11's
     * pattern in graph_edge_separator_to_vertex_separator). */
    if ((int64_t)10 * (int64_t)lo < (int64_t)4 * (int64_t)hi) {
        return bisect_gggp(G, part_out);
    }

    return SPARSE_OK;
}

/* Sprint 25 Day 6: coarsest-bisection strategy enum + env-var
 * parser.  Mirrors Sprint 25 Day 1's `coarsening_strategy_t` /
 * `parse_coarsening_strategy` pattern for SPARSE_ND_COARSENING. */
typedef enum {
    COARSEST_BISECT_DEFAULT = 0, /* Sprint 22 routing: brute @ n≤20, GGGP otherwise */
    COARSEST_BISECT_SPECTRAL = 1,
    COARSEST_BISECT_GGGP = 2,
    COARSEST_BISECT_BRUTE = 3,
} coarsest_bisect_strategy_t;

static coarsest_bisect_strategy_t parse_coarsest_bisect_strategy(void) {
    const char *env = getenv("SPARSE_ND_COARSEST_BISECTION");
    if (!env)
        return COARSEST_BISECT_DEFAULT;
    if (strcmp(env, "spectral") == 0)
        return COARSEST_BISECT_SPECTRAL;
    if (strcmp(env, "gggp") == 0)
        return COARSEST_BISECT_GGGP;
    if (strcmp(env, "brute") == 0)
        return COARSEST_BISECT_BRUTE;
    /* Silent fallback to default routing on unrecognized input,
     * matching Sprint 24 Day 5 / Sprint 25 Day 1 patterns. */
    return COARSEST_BISECT_DEFAULT;
}

sparse_err_t graph_bisect_coarsest(const sparse_graph_t *G, idx_t *part_out) {
    if (!G || !part_out)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    /* Sprint 25 Day 6: SPARSE_ND_COARSEST_BISECTION env-var gate.
     *   - default: Sprint 22 routing — brute @ n≤20, GGGP otherwise.
     *   - spectral: Day 7-8's Fiedler-vector bisection (Day 6 stub
     *     falls through to GGGP after exercising the Laplacian
     *     builder; Day 7 lights up the Lanczos call).
     *   - gggp: force GGGP regardless of n.
     *   - brute: force brute @ n≤20; n>20 falls back to GGGP
     *     (brute on n>20 is intractable: 2^(n-1) patterns).
     * See docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md. */
    coarsest_bisect_strategy_t strategy = parse_coarsest_bisect_strategy();

    switch (strategy) {
    case COARSEST_BISECT_SPECTRAL:
        return graph_bisect_coarsest_spectral(G, part_out);
    case COARSEST_BISECT_GGGP:
        return bisect_gggp(G, part_out);
    case COARSEST_BISECT_BRUTE:
        if (G->n <= 20)
            return bisect_brute_force(G, part_out);
        return bisect_gggp(G, part_out);
    case COARSEST_BISECT_DEFAULT:
    default:
        break;
    }

    /* n ≤ 20: brute-force enumeration is tractable (≤ 524 288 patterns).
     * n > 20: GGGP runs in O(n + |E|) regardless of size — it's the
     * fallback bisection when the multilevel hierarchy can't drive
     * the coarsest level below the brute-force threshold (e.g. when
     * heavy-edge matching saturates on a structurally regular input
     * like bcsstk14).  Day 4's per-level FM uncoarsening polishes
     * whatever GGGP produces. */
    if (G->n <= 20)
        return bisect_brute_force(G, part_out);
    return bisect_gggp(G, part_out);
}
