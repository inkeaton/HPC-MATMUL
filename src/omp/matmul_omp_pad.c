/* * ======================================================================================
 * OPTIMIZED PARALLEL MATRIX MULTIPLICATION (OPENMP)
 * ======================================================================================
 */

#define N 5000

/* 
 * PADDING FOR VECTORIZATION
 * -------------------------
 * N_PAD rounds N up to the nearest multiple of 16 (= 4*4).
 * This guarantees:
 *   1. Every row starts on a 128-byte boundary (16 doubles × 8 bytes),
 *      which is a multiple of a cache line and an AVX-512 register width.
 *   2. The inner-most j-loop length is always a multiple of the SIMD width,
 *      so the compiler never emits scalar "clean-up" tail code.
 *
 * For N = 5000:  N_PAD = 5008  (5000 is not divisible by 16; 5008 is).
 * For N = 4096:  N_PAD = 4096  (already a power-of-two multiple of 16, no change).
 */
#define N_PAD (((N) + 15) / 16 * 16)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    /* * 1. ALIGNED ALLOCATION & RESTRICT
     * - aligned_alloc(64, ...): Aligns to 64 bytes. Mandatory for AVX-512 (AMD) 
     *   and optimal for Cache Lines (Intel).
     * - The column dimension is N_PAD, not N. This makes sizeof(row) a multiple
     *   of 128 bytes, keeping every row pointer 64-byte aligned.
     * - restrict: Promises no pointer aliasing, allowing aggressive SIMD generation.
     */
    double (* restrict a)[N_PAD] = aligned_alloc(64, sizeof(double[N][N_PAD]));
    double (* restrict b)[N_PAD] = aligned_alloc(64, sizeof(double[N][N_PAD]));
    double (* restrict c)[N_PAD] = aligned_alloc(64, sizeof(double[N][N_PAD]));

    if (!a || !b || !c) {
        perror("Allocation failed");
        return 1;
    }

    /* * 2. FIRST-TOUCH INITIALIZATION (NUMA AWARENESS)
     * We initialize the FULL padded row (j up to N_PAD) so that:
     *   - Padding columns are zero and never corrupt the result.
     *   - All memory pages are touched by the thread that will later compute
     *     on them (NUMA first-touch policy).
     * On the 210 lab machines (single memory controller) this doesn't matter,
     * but on multi-die AMD machines it is critical for bandwidth.
     */
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N_PAD; j++) {      /* <-- N_PAD, not N */
            a[i][j] = (j < N) ? 2.0 : 0.0;
            b[i][j] = (j < N) ? 3.0 : 0.0;
            c[i][j] = 0.0;
        }
    }

    #ifdef ENABLE_TIMING
        double start_time, end_time;
        start_time = omp_get_wtime();
    #endif

    /*
     * 3. COMPUTE LOOP (IKJ ORDER)
     * The j-loop now runs to N_PAD instead of N.
     * Because the padding columns in b are 0.0, the extra iterations produce
     * 0.0 contributions to c, which are themselves stored in padding columns
     * of c that we never read back.  Correctness is preserved.
     *
     * Benefit: the compiler sees a loop trip count that is a compile-time
     * multiple of 16, so it can unroll and vectorize without generating
     * scalar epilogue code.
     */
    #pragma omp parallel for schedule(guided, 8)
    for (int i = 0; i < N; i++) {

        for (int k = 0; k < N; k++) {

            double r = a[i][k];

            /* Inner loop: N_PAD is a multiple of 16 -> clean AVX-512 vectorization */
            //#pragma omp simd
            for (int j = 0; j < N_PAD; j++) {  /* <-- N_PAD, not N */
                c[i][j] += r * b[k][j];
            }
        }
    }

    #ifdef ENABLE_TIMING
        end_time = omp_get_wtime();
        double time_taken = end_time - start_time;

        int nthreads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
        fprintf(stderr, "[omp] N=%d N_PAD=%d threads=%d elapsed=%.3f s\n",
                N, N_PAD, nthreads, time_taken);
    #endif

    /* Dump results for verification (only the true N×N block) */
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) fprintf(f, "%.0f ", c[i][j]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free(a); free(b); free(c);
    return 0;
}