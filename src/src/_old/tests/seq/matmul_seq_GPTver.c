// matmul_vec.c
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define N 5000

int main(void) {
    int i, j, k;

    // Use 32-byte alignment so we can use aligned AVX loads/stores
    double (*a)[N];
    double (*b)[N];
    double (*c)[N];

    if (posix_memalign((void **)&a, 32, sizeof(double[N][N])) != 0 ||
        posix_memalign((void **)&b, 32, sizeof(double[N][N])) != 0 ||
        posix_memalign((void **)&c, 32, sizeof(double[N][N])) != 0) {
        perror("posix_memalign");
        return 1;
    }

    // Initialize matrices
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }

    // C = A * B, same loop order as your code: i, k, j
    for (i = 0; i < N; ++i) {
        for (k = 0; k < N; ++k) {

            // Broadcast a[i][k] into an AVX register
            __m256d aik_vec = _mm256_set1_pd(a[i][k]);

            // Process 4 columns of B/C at once
            j = 0;
            for (; j <= N - 4; j += 4) {
                __m256d b_vec = _mm256_load_pd(&b[k][j]);   // aligned load
                __m256d c_vec = _mm256_load_pd(&c[i][j]);   // aligned load

                // c[i][j..j+3] += a[i][k] * b[k][j..j+3]
                c_vec = _mm256_fmadd_pd(aik_vec, b_vec, c_vec);

                _mm256_store_pd(&c[i][j], c_vec);           // aligned store
            }

            // Remainder loop (for general N, though 5000 is divisible by 4)
            for (; j < N; ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    // Output top-left 1000x1000 block as in your original code
    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    fprintf(f, "%d\n\n", N);
    int max = 1000;
    if (max > N) max = N;

    for (i = 0; i < max; ++i) {
        for (j = 0; j < max; ++j) {
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


// gcc -O3 -march=native -mfma -mavx2 -ffast-math matmul_vec.c -o matmul_vec