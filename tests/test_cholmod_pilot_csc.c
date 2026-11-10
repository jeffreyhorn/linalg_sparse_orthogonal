#include "../scripts/cholmod_pilot_csc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    SparseMatrix *a = NULL;
    if (sparse_load_mm(&a, "tests/data/symmetric_4.mtx") != SPARSE_OK || !a) {
        fprintf(stderr, "could not load CSC pilot fixture\n");
        return 1;
    }
    int32_t p[5] = {0, 3, 5, 6, 7};
    int32_t i[7] = {0, 1, 2, 1, 2, 2, 3};
    double values[7] = {10.0, 1.0, 2.0, 12.0, 1.0, 14.0, 16.0};
    int failed = !cholmod_pilot_csc_matches_input(a, p, i, values);

    p[2] = 4;
    failed |= cholmod_pilot_csc_matches_input(a, p, i, values);
    p[2] = 5;
    i[2] = 1;
    failed |= cholmod_pilot_csc_matches_input(a, p, i, values);
    i[2] = 2;
    values[2] = 3.0;
    failed |= cholmod_pilot_csc_matches_input(a, p, i, values);
    values[2] = nan("");
    failed |= cholmod_pilot_csc_matches_input(a, p, i, values);
    values[2] = 2.0;
    i[3] = 0;
    failed |= cholmod_pilot_csc_matches_input(a, p, i, values);
    i[3] = 1;

    sparse_free(a);
    if (failed) {
        fprintf(stderr, "CSC pilot validation regression\n");
        return 1;
    }
    puts("CSC pilot validation: PASS");
    return 0;
}
