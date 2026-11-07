#ifndef CHOLMOD_PILOT_CSC_H
#define CHOLMOD_PILOT_CSC_H

#include "sparse_matrix.h"
#include <math.h>
#include <stdint.h>

static int cholmod_pilot_csc_matches_input(const SparseMatrix *a, const int32_t p[5],
                                           const int32_t i[7], const double values[7]) {
    double reconstructed[4][4] = {{0.0}};
    if (!a || !p || !i || !values || p[0] != 0 || p[4] != 7) {
        return 0;
    }
    for (int32_t col = 0; col < 4; ++col) {
        if (p[col] < 0 || p[col] >= p[col + 1] || p[col + 1] > 7) {
            return 0;
        }
        int32_t previous = col - 1;
        for (int32_t entry = p[col]; entry < p[col + 1]; ++entry) {
            int32_t row = i[entry];
            double value = values[entry];
            if (row <= previous || row < col || row >= 4 || !isfinite(value) || value == 0.0 ||
                value != sparse_get(a, row, col)) {
                return 0;
            }
            reconstructed[row][col] = value;
            reconstructed[col][row] = value;
            previous = row;
        }
    }
    for (idx_t row = 0; row < 4; ++row) {
        for (idx_t col = 0; col < 4; ++col) {
            if (reconstructed[row][col] != sparse_get(a, row, col)) {
                return 0;
            }
        }
    }
    return 1;
}

#endif
