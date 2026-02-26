/* * ======================================================================================
 * OPTIMIZED PARALLEL MATRIX MULTIPLICATION (OPENMP)
 * ======================================================================================
 */

/* 
 * PADDING FOR VECTORIZATION
 * -------------------------
 * N_PAD rounds N up to the nearest multiple of 16 (= 4*4).
 * This guarantees:
 *   1. Every row starts on a 128-byte boundary (16 doubles × 8 bytes),
 *      which is a multiple of a cache line and an AVX-512 register width.
 *   2. The inner-most j-loop length is always a multiple of the SIMD width,
 *      so the compiler never emits scalar "clean-up" tail code.
 *
 * For N = 5000:  N_PAD = 5008  (5000 is not divisible by 16; 5008 is).
 * For N = 4096:  N_PAD = 4096  (already a power-of-two multiple of 16, no change).
 */
// The alignment step needed to satisfy both the 64-element cache block 
// and the compiler's 4-way loop unrolling (64 * 4 = 256).
/* 1. CONFIGURATION FLAGS */

// Default matrix size if not specified at compile time
#ifndef N
    #define N 5000 
#endif

// The alignment step needed to satisfy both the 64-element cache block 
// and the compiler's 4-way loop unrolling (64 * 4 = 256).
#define ALIGN_STEP 256 
#define N_PAD (((N + ALIGN_STEP - 1) / ALIGN_STEP) * ALIGN_STEP)

// Handle the restrict keyword macro
#ifdef RESTRICT
    #define PTR_RESTRICT restrict
#else
    #define PTR_RESTRICT
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    /* 2. TIMING SETUP */
    #ifdef TIME
        struct timespec start, end;
    #endif

    /* 3. POINTER DECLARATION WITH OPTIONAL RESTRICT */
    // Note: Pointers are sized to N_PAD, not N
    double (* PTR_RESTRICT a)[N_PAD] = NULL;
    double (* PTR_RESTRICT b)[N_PAD] = NULL;
    double (* PTR_RESTRICT c)[N_PAD] = NULL;

    /* 4. MEMORY ALLOCATION (ALIGNED VS STANDARD) */
    #ifdef ALIGN
        int err_a = posix_memalign((void **)&a, 64, sizeof(double[N_PAD][N_PAD]));
        int err_b = posix_memalign((void **)&b, 64, sizeof(double[N_PAD][N_PAD]));
        int err_c = posix_memalign((void **)&c, 64, sizeof(double[N_PAD][N_PAD]));

        if (err_a != 0 || err_b != 0 || err_c != 0) {
            fprintf(stderr, "Aligned memory allocation failed.\n");
            free(a); free(b); free(c);
            return 1;
        }
    #else
        a = malloc(sizeof(double[N_PAD][N_PAD]));
        b = malloc(sizeof(double[N_PAD][N_PAD]));
        c = malloc(sizeof(double[N_PAD][N_PAD]));

        if (!a || !b || !c) {
            fprintf(stderr, "Standard memory allocation failed.\n");
            free(a); free(b); free(c);
            return 1;
        }
    #endif

    /* 5. INITIALIZATION */
    for (int i = 0; i < N_PAD; i++) {
        for (int j = 0; j < N_PAD; j++) {
            // Only fill the "real" matrix with data
            if (i < N && j < N) {
                a[i][j] = 2.0;
                b[i][j] = 3.0;
            } else {
                // Fill the padded edges with zeroes
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
            c[i][j] = 0.0;
        }
    }

    /* 6. START TIMER */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 7. CORE COMPUTATION (i-k-j) */
    for (int i = 0; i < N_PAD; ++i) {
        for (int k = 0; k < N_PAD; k++) {
            double r = a[i][k];
            
            // Only enforce vector alignment if we actually aligned the memory
            #ifdef ALIGN
                #pragma vector aligned
            #endif
            for (int j = 0; j < N_PAD; ++j) {
                c[i][j] += r * b[k][j];
            }
        }
    }

    /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Calculate GFLOPS based on the *useful* algorithmic operations (N), not N_PAD.
        // This provides an apples-to-apples comparison with the unpadded version.
        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops = total_flops / (time_taken * 1e9);
        
        printf("[seq-pad] N=%d, N_PAD=%d,| elapsed=%.3f s, GFLOPS=%.2f\n", N, N_PAD, time_taken, gflops);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}