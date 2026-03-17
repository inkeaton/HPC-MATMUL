
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 1. CONFIGURATION FLAGS */

// Default matrix size if not specified at compile time
#ifndef N
    #define N 5000 
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

    /* 7. CORE COMPUTATION (Tiled i-k-j) */
    for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < N; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
                
                // Inner loops processing the current tile
                for (int i = i0; i < MIN(i0 + TILE_SIZE, N); ++i) {
                    for (int k = k0; k < MIN(k0 + TILE_SIZE, N); ++k) {
                        double r = a[i][k];
                        
                        // Enforce vector alignment if memory is aligned
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
        
        printf("[seq-tile] N=%d, TILE_SIZE=%d | elapsed=%.3f s, GFLOPS=%.2f\n", 
                N, TILE_SIZE,time_taken, gflops);
    #endif

    /* 9. CLEANUP */
    free(a);
    free(b);
    free(c);

    return 0;
}