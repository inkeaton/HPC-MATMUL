#define n 5000
#define BS 64  // Block Size: 64x64 doubles = 32KB (Fits easily in L2 cache)

#ifndef ENABLE_TIMING
   #define ENABLE_TIMING 1
#endif

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for tiling bounds
#define min(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char **argv) {
  
  /* * MEMORY ALIGNMENT OPTIMIZATION
   * We use posix_memalign instead of malloc to align memory to 64-byte boundaries.
   * This ensures data aligns perfectly with cache lines and AVX registers.
   */
  double (*a)[n];
  double (*b)[n];
  double (*c)[n];

  if (posix_memalign((void**)&a, 64, sizeof(double[n][n])) != 0) { perror("align a"); return 1; }
  if (posix_memalign((void**)&b, 64, sizeof(double[n][n])) != 0) { perror("align b"); return 1; }
  if (posix_memalign((void**)&c, 64, sizeof(double[n][n])) != 0) { perror("align c"); return 1; }

  /* Initialize */
  // Parallel initialization helps touch pages in parallel (First Touch Policy)
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i][j] = 2.0;
      b[i][j] = 3.0;
      c[i][j] = 0.0;
    }
  }

  double t0 = 0.0;
  if (ENABLE_TIMING == 1) {
    t0 = omp_get_wtime();
  }

  /* * OPTIMIZED MATRIX MULTIPLICATION
   * 1. Tiling: Loops are broken into blocks (BS) to fit in L2 Cache.
   * 2. Collapse: We parallelize the two outer loops (i and j) to create ~6000 tasks.
   * 3. Dynamic: Tasks are distributed dynamically to balance P-cores and E-cores.
   */
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += BS) {
    for (int j = 0; j < n; j += BS) {
      for (int k = 0; k < n; k += BS) {
        
        /* Boundaries for the current block (handles edge cases) */
        int i_end = min(i + BS, n);
        int j_end = min(j + BS, n);
        int k_end = min(k + BS, n);

        /* MICRO KERNEL: The actual computation on small blocks */
        for (int ii = i; ii < i_end; ii++) {
          for (int kk = k; kk < k_end; kk++) {
            
            double A_val = a[ii][kk]; // Load A once into register
            
            /* SIMD Vectorization:
               The compiler will use AVX2 instructions here because:
               1. The loop is simple.
               2. Memory is contiguous.
               3. We explicitly requested it with pragma omp simd.
            */
            #pragma omp simd
            for (int jj = j; jj < j_end; jj++) {
              c[ii][jj] += A_val * b[kk][jj];
            }
          }
        }
      }
    }
  }

  if (ENABLE_TIMING == 1) {
    double t1 = omp_get_wtime();
    fprintf(stderr, "[optimized] n=%d elapsed=%.3f s\n", n, t1 - t0);
  }

  /* Verification Output */
  FILE *f = fopen("mat-res.txt", "w");
  if (!f) {
    perror("fopen");
    return 1;
  }

  fprintf(f, "%d\n\n", n);
  for (int ii = 0; ii < 1000; ii++) {
    for (int jj = 0; jj < 1000; jj++) {
      fprintf(f, "%.0f ", c[ii][jj]);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  free(a);
  free(b);
  free(c);
  return 0;
}