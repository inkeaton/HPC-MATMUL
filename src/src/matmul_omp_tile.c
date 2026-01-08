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

#define N 10000
// BLOCK_SIZE 64 fits well in L1/L2 caches and aligns with 64-byte cache lines.
#define BLOCK_SIZE 64 

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

    /* * 3. PARALLEL BLOCKED MATRIX MULTIPLICATION
     * Strategy:
     * - Loop Permutation: We use ii-jj-kk order.
     * The outer loops (ii, jj) select a Block C[ii][jj].
     * Since every block C[ii][jj] is distinct, different threads can compute 
     * different blocks simultaneously without race conditions (no locks needed).
     */

    /* * 4. OPENMP DIRECTIVES EXPLAINED
     * - collapse(2): Merges 'ii' and 'jj' into one single loop of ~6000 tasks.
     * This creates a large pool of work, allowing better load balancing.
     * * - schedule(guided, 8):
     * Starts with large chunks (low overhead) and shrinks to size 8.
     * - Intel: Prevents fast P-cores from waiting on slow E-cores (better than static).
     * - General: Reduces scheduler locking overhead (better than dynamic, 1).
     * - Minimum size 8: Ensures we don't process partial cache lines (avoids false sharing).
     */
    #pragma omp parallel for collapse(2) schedule(guided, 8)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            
            // The 'kk' loop iterates over input data. It is run serially by the thread
            // that owns the current C[ii][jj] block.
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                // Boundary checks for edge blocks
                int i_max = min(ii + BLOCK_SIZE, N);
                int k_max = min(kk + BLOCK_SIZE, N);
                int j_max = min(jj + BLOCK_SIZE, N);

                // Standard IKJ micro-kernel
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        
                        double r = a[i][k];
                        
                        /* * 5. SIMD VECTORIZATION
                         * #pragma omp simd: Forces vector instructions.
                         * - Machine 210: Uses AVX2/FMA (4 doubles/cycle).
                         * - Machine AMD: Uses AVX-512 (8 doubles/cycle).
                         */
                        #pragma omp simd
                        for (int j = jj; j < j_max; j++) {
                            c[i][j] += r * b[k][j];
                        }
                    }
                }
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