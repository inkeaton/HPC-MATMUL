/* * ======================================================================================
 * OPTIMIZED PARALLEL MATRIX MULTIPLICATION (OPENMP)
 * ======================================================================================
 */

#define N 10000

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // Required for OpenMP functions

inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    /* * 1. ALIGNED ALLOCATION & RESTRICT
     * - aligned_alloc(64, ...): Aligns to 64 bytes. Mandatory for AVX-512 (AMD) 
     * and optimal for Cache Lines (Intel).
     * - restrict: Promises no pointer aliasing, allowing aggressive SIMD generation.
     */
    double (* restrict a)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict b)[N] = aligned_alloc(64, sizeof(double[N][N]));
    double (* restrict c)[N] = aligned_alloc(64, sizeof(double[N][N]));

    if (!a || !b || !c) {
        perror("Allocation failed");
        return 1;
    }

    /* * 2. FIRST-TOUCH INITIALIZATION (NUMA AWARENESS)
     * We parallelize the initialization loops.
     * In Linux, memory is physically allocated only when first written to.
     * By having the same threads write to the data now as will compute on it later,
     * we ensure the data sits in the RAM bank closest to the core using it.
     * This should not be a problem on 210 machine as all cores share the same memory controller.
     * However, on AMD machine with multiple dies, this is crucial for performance.
     */
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    #ifdef ENABLE_TIMING
        double start_time, end_time;
        start_time = omp_get_wtime();
    #endif

    #pragma omp parallel for schedule(guided, 8)
    for (int i = 0; i < N; i++) {
        
        // Inside this loop, 'i' is private to the thread.
        // We iterate through all K columns for this specific row 'i'.
        for (int k = 0; k < N; k++) {
            
            // Load A[i][k] once per inner loop
            double r = a[i][k];

            // Inner loop: Vectorized by AVX2
            #pragma omp simd
            for (int j = 0; j < N; j++) {
                c[i][j] += r * b[k][j];
            }
        }
    }

    // TMP: Change this logic to unify correctly with previous parallel region
    #ifdef ENABLE_TIMING
        end_time = omp_get_wtime();
        double time_taken = end_time - start_time; // Simple subtraction
      
        int nthreads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
        fprintf(stderr, "[omp] N=%d threads=%d elapsed=%.3f s\n", N, nthreads, time_taken);
    #endif

    /* Dump results for verification */
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