// icx -O3 -xHost -DTIME -DALIGN -DRESTRICT -DN=10000 -qmkl=sequential src/matmul_mkl.c -o bin/matmul_mkl_seq_float

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
    float (* PTR_RESTRICT a)[N] = NULL;
    float (* PTR_RESTRICT b)[N] = NULL;
    float (* PTR_RESTRICT c)[N] = NULL;

   /* 4. MEMORY ALLOCATION (ALIGNED VS STANDARD) */
#ifdef ALIGN
    // 64-byte alignment perfectly matches x86 cache lines and AVX-512 registers
    int err_a = posix_memalign((void **)&a, 64, sizeof(float[N][N]));
    int err_b = posix_memalign((void **)&b, 64, sizeof(float[N][N]));
    int err_c = posix_memalign((void **)&c, 64, sizeof(float[N][N]));

        if (err_a != 0 || err_b != 0 || err_c != 0) {
            fprintf(stderr, "Aligned memory allocation failed.\n");
            return 1;
        }
    #else
        a = malloc(sizeof(float[N][N]));
        b = malloc(sizeof(float[N][N]));
        c = malloc(sizeof(float[N][N]));
    #endif

    /* 5. INITIALIZATION */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 2.0f;
            b[i][j] = 3.0f;
            c[i][j] = 0.0f;
        }
    }

    /* 6. START TIMER */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 7. CORE COMPUTATION: MKL SGEMM */
    // cblas_sgemm computes: C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor,   // Our C arrays are row-major
                CblasNoTrans,    // Do not transpose A
                CblasNoTrans,    // Do not transpose B
                N, N, N,         // M, N, K (Matrix dimensions)
                1.0f,            // Alpha multiplier
                (float *)a, N,   // Matrix A and its leading dimension
                (float *)b, N,   // Matrix B and its leading dimension
                0.0f,            // Beta multiplier
                (float *)c, N);  // Matrix C and its leading dimension

    /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Output label updated to mkl-sgemm
        printf("[mkl-sgemm] N=%d | elapsed=%.3f s\n", 
                N, time_taken);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}