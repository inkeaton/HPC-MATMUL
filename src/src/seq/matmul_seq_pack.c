#define N 10000
#define BLOCK_SIZE 64 

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> // For memset

inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    /* 1. Allocation */
    double (* restrict a)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict b)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict c)[N] = aligned_alloc(64, sizeof(double[N][N]));
    
    // Allocate a small buffer for Packing B. 
    // We align it to 64 bytes for AVX-512 efficiency.
    double *packed_B = aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    if (!a || !b || !c || !packed_B) {
        perror("Memory allocation failed");
        return 1;
    }

    /* Initialize */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    #ifdef ENABLE_TIMING
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* * 2. TILED & PACKED MATRIX MULTIPLICATION
     * Order: ii -> jj -> kk
     * This ensures we load a block of C, finish it completely, and write it back once.
     */
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            
            // Bounds for the current blocks
            int i_max = min(ii + BLOCK_SIZE, N);
            int j_max = min(jj + BLOCK_SIZE, N);

            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                int k_max = min(kk + BLOCK_SIZE, N);

                /* * STEP A: PACKING
                 * Copy the current tile of B into a contiguous buffer.
                 * This eliminates the large stride jumps (N) in the main memory.
                 */
                int packed_idx = 0;
                for (int k = kk; k < k_max; k++) {
                    for (int j = jj; j < j_max; j++) {
                        packed_B[packed_idx++] = b[k][j];
                    }
                }

                /* * STEP B: COMPUTATION
                 * Standard IKJ loop, but reading from 'packed_B' instead of 'b'.
                 */
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        
                        double r = a[i][k];
                        
                        // Pointer to the start of the current row in packed_B
                        // Since packed_B is sequential, row k starts at (k - kk) * width
                        double *b_ptr = &packed_B[(k - kk) * (j_max - jj)];

                        // Use vectorization on the dense, contiguous packed buffer
                        #pragma omp simd
                        for (int j = jj; j < j_max; j++) {
                            // Note: b_ptr[j - jj] aligns valid access to 0..BLOCK_SIZE
                            c[i][j] += r * b_ptr[j - jj];
                        }
                    }
                }
            }
        }
    }

    #ifdef ENABLE_TIMING
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        fprintf(stderr, "[seq-packed] N=%d, block=%d, elapsed=%.3f s\n", N, BLOCK_SIZE, time_taken);
    #endif

    // File Output logic (same as before) ...
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) fprintf(f, "%.0f ", c[i][j]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free(a); free(b); free(c); free(packed_B);
    return 0;
}