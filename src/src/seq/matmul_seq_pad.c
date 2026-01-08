/* * ======================================================================================
 * TILED MATRIX MULTIPLICATION WITH PADDING
 * ======================================================================================
 */

#define N 10000
#define BLOCK_SIZE 64 

// Calculate the next multiple of BLOCK_SIZE.
// (15000 + 63) / 64 * 64 = 15040
#define PADDED_N (((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> // Required for memset

int main(int argc, char **argv)
{
    printf("Real N: %d, Padded N: %d\n", N, PADDED_N);

    /* * 1. ALLOCATION USING PADDED_N
     * We allocate slightly more memory than needed to ensure the dimensions 
     * are perfectly divisible by BLOCK_SIZE (64).
     */
    double (* restrict a)[PADDED_N] = aligned_alloc(64, sizeof(double[PADDED_N][PADDED_N]));
    double (* restrict b)[PADDED_N] = aligned_alloc(64, sizeof(double[PADDED_N][PADDED_N]));
    double (* restrict c)[PADDED_N] = aligned_alloc(64, sizeof(double[PADDED_N][PADDED_N]));

    if (!a || !b || !c) {
        perror("Memory allocation failed");
        return 1;
    }

    /* * 2. INITIALIZATION
     * First, zero out EVERYTHING (including the padding zone).
     * This is crucial so that the extra padded calculations add +0.0 
     * and do not affect the final result.
     */
    memset(a, 0, sizeof(double[PADDED_N][PADDED_N]));
    memset(b, 0, sizeof(double[PADDED_N][PADDED_N]));
    memset(c, 0, sizeof(double[PADDED_N][PADDED_N]));

    /* Initialize only the "Real" N x N part with data */
    srand((unsigned int)time(NULL)); // Seed random number generator with current time

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Generate random double between 0.0 and 1.0
            a[i][j] = (double)rand() / RAND_MAX; 
            
            // Generate random double between 0.0 and 10.0 (example)
            b[i][j] = ((double)rand() / RAND_MAX) * 10.0;
            
            c[i][j] = 0.0;
        }
    }

    #ifdef ENABLE_TIMING
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* * 3. TILED LOOP (NO MIN CHECKS)
     * We iterate up to PADDED_N.
     * Because PADDED_N is a perfect multiple of BLOCK_SIZE, we never have partial blocks.
     */
    for (int ii = 0; ii < PADDED_N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < PADDED_N; jj += BLOCK_SIZE) { // Order: ii-jj-kk (Correct for Cache)
            for (int kk = 0; kk < PADDED_N; kk += BLOCK_SIZE) {
                
                /* * INNER LOOPS: CONSTANT TRIP COUNT
                 * The compiler sees: "i < ii + 64". 
                 * It knows this loop runs EXACTLY 64 times.
                 * This enables aggressive unrolling and vectorization.
                 */
                for (int i = ii; i < ii + BLOCK_SIZE; i++) {
                    for (int k = kk; k < kk + BLOCK_SIZE; k++) {
                        
                        double r = a[i][k];
                        
                        #pragma omp simd
                        for (int j = jj; j < jj + BLOCK_SIZE; j++) {
                            c[i][j] += r * b[k][j];
                        }
                    }
                }
            }
        }
    }

    #ifdef ENABLE_TIMING
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        fprintf(stderr, "[seq-padded] N=%d, Padded=%d, elapsed=%.3f s\n", 
                N, PADDED_N, time_taken);
    #endif

    /* * 4. OUTPUT
     * When writing the result, we only care about the original N x N area.
     * We ignore the padded rows/cols.
     */
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        // Only dump the top-left 1000x1000, which is well within the real data
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) {
                fprintf(f, "%.0f ", c[i][j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free(a);
    free(b);
    free(c);
    return 0;
}