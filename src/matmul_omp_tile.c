/* * ======================================================================================
 * OPTIMIZED PARALLEL MATRIX MULTIPLICATION (OPENMP)
 * ======================================================================================
 * * COMPILATION INSTRUCTIONS:
 * -------------------------
 * * MACHINE 1: INTEL i9-12900K ("Machine 210") - Hybrid Architecture
 * Compiler: Intel oneAPI (icx) is recommended for best hybrid scheduling.
 * Command: 
 * icx -O3 -xHost -qopenmp -funroll-loops -DENABLE_TIMING matmul_opt_omp.c -o matmul_omp
 * Alternative (GCC):
 * gcc -O3 -march=native -fopenmp -funroll-loops -DENABLE_TIMING matmul_opt_omp.c -o matmul_omp
 *
 * * MACHINE 2: AMD RYZEN 9 7900X ("Machine AMD") - Zen 4 Architecture
 * Compiler: GCC is highly effective here.
 * Command:
 * gcc -O3 -march=native -fopenmp -mprefer-vector-width=512 -funroll-loops -DENABLE_TIMING matmul_opt_omp.c -o matmul_omp
 *
 * * EXECUTION & TUNING (ENVIRONMENT VARIABLES):
 * ---------------------------------------------
 * * MACHINE 1 (Intel Hybrid):
 * We need to balance load across fast P-cores and slow E-cores.
 * Use 'spread' to use all cores, but 'guided' schedule (in code) handles the speed difference.
 * export OMP_NUM_THREADS=24   (Use all logical threads)
 * export OMP_PLACES=cores
 * export OMP_PROC_BIND=spread
 * ./matmul_omp
 *
 * * MACHINE 2 (AMD Chiplet):
 * We want to spread threads across both CCX dies to maximize memory bandwidth.
 * export OMP_NUM_THREADS=24
 * export OMP_PLACES=threads
 * export OMP_PROC_BIND=spread
 * ./matmul_omp
 * ---------------------------------------------
 * * TESTING TO DO:
 * - Compare performance with different number of threads (e.g., 8,16,24).
 * - Compare performance with different scheduling strategies (static, dynamic, guided).
 * - Compare performance with different scheduling chunk sizes (e.g., 1, 8, 32).
 * - Compare performance without first-touch initialization.
 * * ======================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* 1. CONFIGURATION FLAGS */

// Default matrix size
#ifndef N
    #define N 1000 
#endif

// Default tile/block size for cache optimization
#ifndef TILE_SIZE
    #define TILE_SIZE 64
#endif

// Utility macro for the tiling bounds
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Handle the restrict keyword macro
#ifdef RESTRICT
    #define PTR_RESTRICT restrict
#else
    #define PTR_RESTRICT
#endif

int main() {
    /* 2. TIMING SETUP */
    #ifdef TIME
        struct timespec start, end;
    #endif

    /* 3. POINTER DECLARATION WITH OPTIONAL RESTRICT */
    double (* PTR_RESTRICT a)[N] = NULL;
    double (* PTR_RESTRICT b)[N] = NULL;
    double (* PTR_RESTRICT c)[N] = NULL;

    /* 4. MEMORY ALLOCATION (ALIGNED VS STANDARD) */
    #ifdef ALIGN
        int err_a = posix_memalign((void **)&a, 64, sizeof(double[N][N]));
        int err_b = posix_memalign((void **)&b, 64, sizeof(double[N][N]));
        int err_c = posix_memalign((void **)&c, 64, sizeof(double[N][N]));

        if (err_a != 0 || err_b != 0 || err_c != 0) {
            fprintf(stderr, "Aligned memory allocation failed.\n");
            free(a); free(b); free(c);
            return 1;
        }
    #else
        a = malloc(sizeof(double[N][N]));
        b = malloc(sizeof(double[N][N]));
        c = malloc(sizeof(double[N][N]));

        if (!a || !b || !c) {
            fprintf(stderr, "Standard memory allocation failed.\n");
            free(a); free(b); free(c);
            return 1;
        }
    #endif

    /* 5. INITIALIZATION */
    // Using OpenMP here speeds up the allocation of huge matrices
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    /* 6. START TIMER */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 7. CORE COMPUTATION (Multithreaded Tiled i-k-j) */
    // collapse(2) merges the outer two tile loops so OpenMP can distribute 
    // a larger pool of distinct chunks perfectly across your CPU cores.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < N; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
                
                // Inner loops processing the current tile
                for (int i = i0; i < MIN(i0 + TILE_SIZE, N); ++i) {
                    for (int k = k0; k < MIN(k0 + TILE_SIZE, N); ++k) {
                        double r = a[i][k];
                        
                        #ifdef ALIGN
                            #pragma vector aligned
                        #endif

                        for (int j = j0; j < MIN(j0 + TILE_SIZE, N); ++j) {
                            c[i][j] += r * b[k][j];
                        }
                    }
                }
            }
        }
    }

    /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // GFLOPS calculation: 2 * N^3 operations
        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops = total_flops / (time_taken * 1e9);
        
        printf("[omp-tile] N=%d, TILE_SIZE=%d | elapsed=%.3f s, GFLOPS=%.2f\n", 
                N, TILE_SIZE, time_taken, gflops);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}