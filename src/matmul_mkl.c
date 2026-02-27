// icx -O3 -xHost -DTIME -DALIGN -DRESTRICT -DN=10000 -qmkl=sequential src/matmul_mkl.c -o bin/matmul_mkl_seq

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- PREPROCESSOR FIX --- */
// Temporarily save our 'N' macro and undefine it so it doesn't 
// destroy Intel's internal function parameter names.
#pragma push_macro("N")
#undef N
#include <mkl.h> 
// Restore our 'N' macro from the command line
#pragma pop_macro("N")

/* 1. CONFIGURATION FLAGS */

#ifndef N
#define N 1000 
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
    // MKL absolutely loves 64-byte aligned memory for AVX instructions
    int err_a = posix_memalign((void **)&a, 64, sizeof(double[N][N]));
    int err_b = posix_memalign((void **)&b, 64, sizeof(double[N][N]));
    int err_c = posix_memalign((void **)&c, 64, sizeof(double[N][N]));

        if (err_a != 0 || err_b != 0 || err_c != 0) {
            fprintf(stderr, "Aligned memory allocation failed.\n");
            return 1;
        }
    #else
        a = malloc(sizeof(double[N][N]));
        b = malloc(sizeof(double[N][N]));
        c = malloc(sizeof(double[N][N]));
    #endif

    /* 5. INITIALIZATION */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = (double)rand() / RAND_MAX;
            b[i][j] = ((double)rand() / RAND_MAX) * 10.0;
            c[i][j] = 0.0;
        }
    }

    /* 6. START TIMER */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 7. CORE COMPUTATION: MKL DGEMM */
    // cblas_dgemm computes: C = alpha * A * B + beta * C
    cblas_dgemm(CblasRowMajor,   // Our C arrays are row-major
                CblasNoTrans,    // Do not transpose A
                CblasNoTrans,    // Do not transpose B
                N, N, N,         // M, N, K (Matrix dimensions)
                1.0,             // Alpha multiplier
                (double *)a, N,  // Matrix A and its leading dimension
                (double *)b, N,  // Matrix B and its leading dimension
                0.0,             // Beta multiplier
                (double *)c, N); // Matrix C and its leading dimension

    /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Formatted exactly to match your benchmark.sh script output scraper
        printf("[mkl-dgemm] N=%d | elapsed=%.3f s\n", 
                N, time_taken);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}