/* COMPILE:
 * mpicc -O3 -march=native -funroll-loops -std=c11 matmul_mpi_opt.c -o matmul_mpi
 * * RUN (Using all 24 logical cores):
 * mpirun -np 24 --bind-to core ./matmul_mpi
 * * RUN (P-Cores only - Recommended for consistency):
 * mpirun -np 16 --bind-to core ./matmul_mpi
 */

#define n 5000
#define BLOCK_SIZE 64 // Local cache tiling size

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper for tiling
static inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Calculate Row Distribution (Domain Decomposition)
    // We use Scatterv/Gatherv to handle cases where n is not divisible by world_size
    int *sendcounts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));

    int rem = n % world_size;
    int sum = 0;
    for (int i = 0; i < world_size; i++)
    {
        sendcounts[i] = (n / world_size) * n; // Base number of elements
        if (i < rem)
            sendcounts[i] += n; // Add remainder rows
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Calculate local rows for this specific rank
    int local_rows = sendcounts[world_rank] / n;

    // 2. Memory Allocation
    double (*a)[n] = NULL;
    double (*b)[n] = malloc(sizeof(double[n][n])); // Everyone needs full B
    double (*c)[n] = NULL;                         // Rank 0 holds full result

    // Local buffers for slices
    double *local_a = malloc(local_rows * n * sizeof(double));
    double *local_c = calloc(local_rows * n, sizeof(double)); // Use calloc to zero init

    if (world_rank == 0)
    {
        // Only Rank 0 allocates the full masters
        a = malloc(sizeof(double[n][n]));
        c = malloc(sizeof(double[n][n]));

        // Initialize A and B
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i][j] = 2.0;
                b[i][j] = 3.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef ENABLE_TIMING
        double t0 = 0.0;
        if (world_rank == 0) t0 = MPI_Wtime();
    #endif

    // 3. Communication Phase
    // Broadcast B to everyone (Replication)
    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A to workers (Decomposition)
    // Note: We cast a to double* because Scatterv expects a flat buffer
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                 local_a, local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 4. Computation Phase (Local Tiled Matrix Multiplication)
    // We calculate local_c = local_a * b
    // local_a is (local_rows x n), b is (n x n)

    // Outer loops iterate over blocks
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE)
    {
        for (int kk = 0; kk < n; kk += BLOCK_SIZE)
        {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE)
            {

                // Inner loops iterate inside the block
                for (int i = ii; i < min(ii + BLOCK_SIZE, local_rows); i++)
                {
                    for (int k = kk; k < min(kk + BLOCK_SIZE, n); k++)
                    {

                        double temp_a = local_a[i * n + k]; // Access local_a as flat or 2D

                        // Auto-vectorizable inner loop
                        for (int j = jj; j < min(jj + BLOCK_SIZE, n); j++)
                        {
                            local_c[i * n + j] += temp_a * b[k][j];
                        }
                    }
                }
            }
        }
    }

    // 5. Gather Results
    MPI_Gatherv(local_c, local_rows * n, MPI_DOUBLE,
                c, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    #ifdef ENABLE_TIMING
        if (world_rank == 0)
        {
            double t1 = MPI_Wtime();
            fprintf(stderr, "[mpi-opt] ranks=%d n=%d elapsed=%.3f s\n", world_size, n, t1 - t0);
        }
    #endif

    // Verification IO
    if (world_rank == 0)
    {
        FILE *f = fopen("mat-res.txt", "w");
        if (!f)
        {
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f, "%d\n\n", n);
        for (int ii = 0; ii < 1000; ii++)
        {
            for (int jj = 0; jj < 1000; jj++)
            {
                fprintf(f, "%.0f ", c[ii][jj]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        free(a);
        free(c);
    }

    // Cleanup
    free(b);
    free(local_a);
    free(local_c);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}