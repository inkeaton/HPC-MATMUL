/* * ======================================================================================
 * OPTIMIZED PARALLEL MATRIX MULTIPLICATION (OPENMP)
 * ======================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // Required for OpenMP functions

/* 1. CONFIGURATION FLAGS */

// Default matrix size if not specified at compile time
#ifndef N
    #define N 5000 
#endif

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
        double start_time, end_time;
        start_time = omp_get_wtime();
    #endif
    
    /* 7. CORE COMPUTATION (i-k-j) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; k++) {
            double r = a[i][k];
            for (int j = 0; j < N; ++j) {
                c[i][j] += r * b[k][j];
            }
        }
    }

        /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        end_time = omp_get_wtime();
        double time_taken = end_time - start_time; // Simple subtraction
      
        int nthreads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
        fprintf(stderr, "[omp-ikj] N=%d threads=%d | elapsed=%.3f s\n", N, nthreads, time_taken);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}