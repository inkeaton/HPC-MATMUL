/* * ======================================================================================
 * OPTIMIZED MPI MATRIX MULTIPLICATION (1D SLAB DECOMPOSITION)
 * ======================================================================================
 * * ALGORITHM EXPLAINED:
 * 1. Decomposition: We slice Matrix A and C by rows (Horizontal Slabs).
 * - Why? C stores arrays in row-major order. Sending rows is a single continuous
 * memory copy, which maximizes bandwidth and cache efficiency.
 * 2. Broadcast: Matrix B is sent to ALL nodes.
 * - Why? B is small enough (~200MB) to fit in RAM on both machines.
 * - Avoiding complex 2D algorithms saves significant synchronization overhead.
 * 3. Local Compute: Each node calculates a slice of C using a Tiled, Vectorized kernel.
 * * * COMPILATION INSTRUCTIONS:
 * -------------------------
 * * MACHINE 1: INTEL i9-12900K ("Machine 210")
 * mpicc -O3 -march=native -funroll-loops -DENABLE_TIMING matmul_mpi_opt.c -o matmul_mpi
 * (Ensure mpicc wraps 'icx' or 'gcc' with AVX2 support)
 * * * MACHINE 2: AMD RYZEN 9 7900X ("Machine AMD")
 * mpicc -O3 -march=native -mprefer-vector-width=512 -funroll-loops -DENABLE_TIMING matmul_mpi_opt.c -o matmul_mpi
 * (Ensure mpicc wraps 'gcc' or 'aocc' with AVX-512 support)
 * * * EXECUTION INSTRUCTIONS (CRITICAL FOR PERFORMANCE):
 * --------------------------------------------------
 * * MACHINE 1 (Intel Hybrid Architecture):
 * The i9-12900K has 8 fast P-cores (16 threads) and 8 slow E-cores.
 * MPI is synchronous; if one rank lands on an E-core, ALL ranks wait for it.
 * DO NOT use -np 24. Use only the P-cores.
 * * Command: mpirun -np 16 --bind-to core ./matmul_mpi
 * * * MACHINE 2 (AMD Chiplet Architecture):
 * The Ryzen 7900X has 12 powerful cores (24 threads) across 2 dies.
 * We want to use all cores and lock them to preserve L3 cache locality.
 * * Command: mpirun -np 12 --bind-to core ./matmul_mpi
 * or mpirun -np 24 --use-hwthread-cpus --bind-to hwthread ./bin/matmul_mpi
 * --------------------------------------------------
 * * TESTING TO DO:
 * - Compare performance with different number of processes (e.g., np=8,16,24) and running on cores or hwthreads
 * - Compare performance when using both E and P cores on Intel (np=24).
 * - Compare performance with and without core affinity
 * --------------------------------------------------
 * * TO BE DONE:
 * - This solution is good, but cannot scale too well on N. Test implementation with 2D decomposition. (Cannon's or SUMMA)
 * - Test overlapping communication and computation using MPI_Ibcast and MPI_Iscatterv.
 * - Test using MPI derived datatypes to send rows instead of 1D flat arrays.
 * - Test using MPI_Reduce_scatter to combine communication and reduction of C.
 * - Test using MPI 3 shared memory windows for B to avoid explicit broadcast.
 * * ======================================================================================
 */

#define N 5000
// BLOCK_SIZE 64 fits well in L1/L2 caches and matches the 64-byte cache line size.
#define BLOCK_SIZE 64 

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper to calculate min of two numbers
inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    int rank, size;
    
    // Initialize MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* * 1. DECOMPOSITION STRATEGY (Load Balancing)
     * We calculate how many rows each process gets.
     * Since N=5000 is not always divisible by size (e.g., 24), we handle the remainder.
     */
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    int rows_per_proc = N / size;
    int remainder = N % size;
    int current_displ = 0;

    for (int i = 0; i < size; i++) {
        // Distribute remainder rows among the first few ranks
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        
        // sendcounts takes the total number of DOUBLES (elements), not just rows
        sendcounts[i] = rows * N; 
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }
    
    // Calculate local dimensions for THIS process
    int local_rows = sendcounts[rank] / N;
    int local_elements = local_rows * N;

    /* * 2. MEMORY ALLOCATION (Aligned)
     * We use aligned_alloc(64, ...) for all buffers.
     * This is mandatory for AVX-512 (AMD) and optimal for AVX2 (Intel).
     */
    
    // Arrays for the full matrices (Only Rank 0 needs A and C fully)
    double (*a)[N] = NULL;
    double (*c)[N] = NULL;
    
    // Everyone needs the full matrix B
    double (* restrict b)[N] = aligned_alloc(64, sizeof(double[N][N]));

    // Buffers for the "Slab" of A and the partial result "Slab" of C
    // We allocate them as 1D arrays for easier MPI transfer, but access logic is 2D
    double * restrict local_a_flat = aligned_alloc(64, local_elements * sizeof(double));
    double * restrict local_c_flat = aligned_alloc(64, local_elements * sizeof(double));

    if (!b || !local_a_flat || !local_c_flat) {
        fprintf(stderr, "Rank %d failed to allocate memory.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rank 0 initializes the data
    if (rank == 0) {
        a = aligned_alloc(64, sizeof(double[N][N]));
        c = aligned_alloc(64, sizeof(double[N][N]));
        
        if (!a || !c) MPI_Abort(MPI_COMM_WORLD, 1);

        // Simple Initialization
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                a[i][j] = 2.0;
                b[i][j] = 3.0;
                c[i][j] = 0.0;
            }
    }

    // Barrier to ensure Rank 0 is done initializing before timing starts
    MPI_Barrier(MPI_COMM_WORLD);

    // Timing start
    #ifdef ENABLE_TIMING
        double start_time = 0.0;
        if (rank == 0) start_time = MPI_Wtime();
    #endif

    /* * 3. COMMUNICATION PHASE
     * Move data to the worker nodes.
     */
    
    // Step A: Broadcast B to everyone.
    // Tree-based broadcast is extremely efficient on shared memory.
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step B: Scatter rows of A.
    // We use Scatterv (Vector Scatter) to handle the uneven row counts calculated earlier.
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, 
                 local_a_flat, local_elements, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    /* * 4. COMPUTATION PHASE (Optimized Micro-Kernel)
     * Each rank computes C_local = A_local * B.
     */
    
    // Initialize local result to 0
    for(int i=0; i < local_elements; i++) local_c_flat[i] = 0.0;

    // Cache-Blocked Loop (Tiling)
    // We iterate through blocks of the local rows (ii) and full columns (jj, kk)
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                int i_max = min(ii + BLOCK_SIZE, local_rows);
                int k_max = min(kk + BLOCK_SIZE, N);
                int j_max = min(jj + BLOCK_SIZE, N);

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        
                        // Load A from the 1D flat buffer
                        // Logic: row 'i' in local_a corresponds to row 'displs[rank]/N + i' in global A
                        double r = local_a_flat[i * N + k];
                        
                        /* * VECTORIZATION
                         * #pragma omp simd: Hints the compiler to generate vector instructions.
                         * - Intel: Generates vfmadd...ymm (AVX2)
                         * - AMD: Generates vfmadd...zmm (AVX-512)
                         * 'restrict' pointers ensure this is safe.
                         */
                        #pragma omp simd
                        for (int j = jj; j < j_max; j++) {
                             local_c_flat[i * N + j] += r * b[k][j];
                        }
                    }
                }
            }
        }
    }

    /* * 5. GATHER PHASE
     * Collect the partial C slabs back to Rank 0.
     */
    MPI_Gatherv(local_c_flat, local_elements, MPI_DOUBLE,
                c, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Timing end
    #ifdef ENABLE_TIMING
        double end_time = 0.0;
        if (rank == 0) end_time = MPI_Wtime();
    #endif


    /* * 6. REPORTING & CLEANUP */
    if (rank == 0) {
        #ifdef ENABLE_TIMING
        double time_taken = end_time - start_time;
        // Print to stderr to separate metrics from data output
        fprintf(stderr, "[mpi] N=%d, processes=%d, elapsed=%.3f s\n", 
                N, size, time_taken);
        #endif
        

        // Write result to file
        FILE *f = fopen("mat-res.txt", "w");
        if (f) {
            fprintf(f, "%d\n\n", N);
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) fprintf(f, "%.0f ", c[i][j]);
                fprintf(f, "\n");
            }
            fclose(f);
        } else {
            perror("Failed to write output file");
        }
        
        free(a); free(c);
    }

    // Free resources
    free(b);
    free(local_a_flat);
    free(local_c_flat);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}