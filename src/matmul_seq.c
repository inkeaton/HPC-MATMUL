/* * ======================================================================================
 * OPTIMIZED SEQUENTIAL MATRIX MULTIPLICATION (GEMM)
 * ======================================================================================
 * * COMPILATION INSTRUCTIONS:
 * -------------------------
 * * MACHINE 1: INTEL i9-12900K ("Machine 210")
 * Flags: -O3 -march=native -funroll-loops -DENABLE_TIMING
 * Command: 
 * gcc -O3 -march=native -funroll-loops -DENABLE_TIMING matmul_opt_seq.c -o matmul_seq
 * or icx -O3 -xHost -qopt-zmm-usage=high -funroll-loops -DENABLE_TIMING matmul_opt_seq.c -o matmul_seq
 * * MACHINE 2: AMD RYZEN 9 7900X ("Machine AMD")
 * Flags: -O3 -march=native -mprefer-vector-width=512 -funroll-loops -DENABLE_TIMING
 * Command:
 * gcc -O3 -march=native -mprefer-vector-width=512 -funroll-loops -DENABLE_TIMING matmul_opt_seq.c -o matmul_seq
 * ---------------------------------------------
 * * EXECUTION INSTRUCTIONS:
 * ---------------------------------------------
 * * MACHINE 1 (Intel Hybrid Architecture):
 * Must pin to a P-Core (Cores 0-15) to avoid slow E-Cores.
 * Command: taskset -c 0 ./matmul_seq
 * * MACHINE 2 (AMD Chiplet Architecture):
 * Pin to a single Core Complex (CCX) to avoid L3 cache thrashing across dies.
 * Command: taskset -c 0 ./matmul_seq
 * ---------------------------------------------
 * * TESTING TO DO:
 * - Compare performance with different BLOCK_SIZE values (e.g., 32, 64, 128).
 * - Compare performance with different compiler flags (e.g., -O2, -O3, etc).
 * - compare performance with differentt alignment values in aligned_alloc (e.g., 32, 64, 128).
 * * ======================================================================================
 */

#define N 5000
// BLOCK_SIZE 64 is tuned for 32KB-48KB L1 and 1MB+ L2 caches.
// 3 arrays * 64^2 * 8 bytes = ~98 KB, fitting easily in L2.
#define BLOCK_SIZE 64 

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Helper to calculate min of two numbers for boundary checks
inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    /* * 1. MEMORY ALLOCATION & ALIGNMENT
     * aligned_alloc(64, ...): Aligns data to 64-byte boundaries.
     * - Required for AVX-512 (AMD) to avoid segfaults or penalties.
     * - Optimal for Cache Lines (64 bytes) on Intel/AMD to prevent split-loads.
     * * 'restrict': Promises the compiler that pointers a, b, and c do not overlap.
     * - Allows generation of efficient vector instructions without runtime safety checks.
     */
    double (* restrict a)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict b)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict c)[N] = aligned_alloc(64, sizeof(double[N][N]));

    if (!a || !b || !c) {
        perror("Memory allocation failed");
        return 1;
    }

    /* Initialize A and B with constants; zero C. */
    // Note: Initialization is O(N^2) and less critical, but good to keep simple.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    /* Timing the start */
    #ifdef ENABLE_TIMING
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* * 2. BLOCKED MATRIX MULTIPLICATION (Tiling)
     * Instead of iterating over the full N (5000), we iterate over small blocks.
     * This keeps the active data (the working set) inside the fast L1/L2 cache.
     */
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Handle edge cases where N is not a multiple of BLOCK_SIZE
                int i_max = min(ii + BLOCK_SIZE, N);
                int k_max = min(kk + BLOCK_SIZE, N);
                int j_max = min(jj + BLOCK_SIZE, N);

                /* * Standard IKJ loop inside the block.
                 * The compiler can auto-vectorize the inner 'j' loop.
                 */
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        
                        // Load A[i][k] into a register once, reuse it for the whole J loop.
                        double r = a[i][k]; 
                        
                        /* * 3. VECTORIZATION
                         * #pragma omp simd: Explicitly hints to vectorise this loop.
                         * With -march=native:
                         * - Intel Machine 210 uses AVX2 (ymm registers, 4 doubles at once).
                         * - AMD Machine uses AVX-512 (zmm registers, 8 doubles at once).
                         */
                        #pragma omp simd
                        for (int j = jj; j < j_max; j++) {
                            c[i][j] += r * b[k][j];
                        }
                    }
                }
            }
        }
    }

    /* Timing the end and reporting */
    #ifdef ENABLE_TIMING
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        fprintf(stderr, "[seq] N=%d, block=%d, elapsed=%.3f s\n", 
                N, BLOCK_SIZE, time_taken);
    #endif

    /* Dump a 1000x1000 top-left block to file for inspection. */
    FILE *f = fopen("mat-res.txt", "w");
    if (!f)
    {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", N);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    /* Free resources before exit. */
    free(a);
    free(b);
    free(c);
    return 0;
}