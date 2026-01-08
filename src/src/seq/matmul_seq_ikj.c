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
 * - Compare performance with different compiler flags (e.g., -O2, -O3, etc).
 * - compare performance with different alignment values in aligned_alloc (e.g., 32, 64, 128).
 * * ======================================================================================
 */

#define N 10000

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    /* Timing the start */
    #ifdef ENABLE_TIMING
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* Naive matrix multiplication (i-k-j order): C = A * B. */
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; k++) {
            double r = a[i][k];
            for (int j = 0; j < N; ++j)
                c[i][j] += r * b[k][j];
        }


    /* Timing the end and reporting */
    #ifdef ENABLE_TIMING
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        fprintf(stderr, "[seq-ikj] N=%d, elapsed=%.3f s\n", 
                N, time_taken);
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