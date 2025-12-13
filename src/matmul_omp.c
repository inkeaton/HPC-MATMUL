/*  
/   # COMPILE OPTIONS:
/   ## Use the following for OMP Places and Bindings (especially for Hybrid CPUs):
/       export OMP_PLACES=threads
/       export OMP_PROC_BIND=spread
/   ## Or use only performance cores:
/       export OMP_PLACES="{0}:16" // with the previous
/   ## GCC:
/       gcc -O3 -march=native -fopenmp -ffast-math -funroll-loops -std=c11 matmul_omp.c -o matmul_omp
/   ## ICX (Intel oneAPI):
/       icx -O3 -xHost -qopenmp -fp-model fast -std=c11 matmul_omp.c -o matmul_omp
/   ## RUNNING (Important for Hybrid CPU):
/       OMP_NUM_THREADS=24 ./matmul_omp
*/

#define n 5000
#define BLOCK_SIZE 64 // Fits well in L1/L2 cache

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // Required for OpenMP

// Helper min function
static inline int min(int a, int b)
{
    return (a < b) ? a : b;
}

int main(int argc, char **argv)
{

    /* 1. Memory Alignment
    /  Using aligned_alloc ensures 64-byte alignment for AVX2 instructions.
    */
    double (*restrict a)[n] = aligned_alloc(64, sizeof(double[n][n]));
    double (*restrict b)[n] = aligned_alloc(64, sizeof(double[n][n]));
    double (*restrict c)[n] = aligned_alloc(64, sizeof(double[n][n]));

    if (!a || !b || !c)
    {
        perror("Memory allocation failed");
        return 1;
    }

    /* Initialize A and B.*/
    // Should we parallelize this too?
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    #ifdef ENABLE_TIMING
        double start_time, end_time;
        start_time = omp_get_wtime();
    #endif

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // HOTSPOT BEGIN
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /* 2. OpenMP Parallel Tiled Multiplication
    /  collapse(2): Merges loops 'ii' and 'jj' into one large loop of tasks.
    /  schedule(dynamic): ESSENTIAL for Hybrid P-core/E-core CPUs.
    /  It allows fast cores to grab more blocks than slow cores.
    */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE)
    {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE)
        {

            // For each block of C (C[ii..ii+BLOCK][jj..jj+BLOCK])
            // We iterate over the 'k' dimension (kk)
            for (int kk = 0; kk < n; kk += BLOCK_SIZE)
            {

                // --- Micro-Kernel (Sequential AVX2 optimized via flags) ---
                for (int i = ii; i < min(ii + BLOCK_SIZE, n); i++)
                {
                    for (int k = kk; k < min(kk + BLOCK_SIZE, n); k++)
                    {

                        // '#pragma omp simd' helps if -O3 doesn't auto-vectorize,
                        // but standard -O3 usually handles this innermost loop well.
                        // #pragma omp simd
                        for (int j = jj; j < min(jj + BLOCK_SIZE, n); j++)
                        {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // HOTSPOT END
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        fprintf(stderr, "[omp-opt] n=%d threads=%d elapsed=%.3f s\n", n, nthreads, time_taken);
    #endif

    /* Dump results */
    FILE *f = fopen("mat-res.txt", "w");
    if (!f)
    {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    free(a);
    free(b);
    free(c);
    return 0;
}