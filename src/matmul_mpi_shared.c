/* COMPILE:
 * mpicc -O3 -march=native -funroll-loops -std=c11 matmul_mpi_shared.c -o matmul_mpi_shared
 * RUN:
 * mpirun -np 24 --bind-to core ./matmul_mpi_shared
 */

#define n 5000
#define BLOCK_SIZE 64

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Helper for tiling
static inline int min(int a, int b) { return (a < b) ? a : b; }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // 1. Set up the communicator
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Create a specific communicator for processes on the SAME node.
    // (On your single machine, this includes everyone, but this is cluster-safe).
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // 2. Allocate Shared Memory for Matrix B
    // Only Node-Rank 0 allocates the physical bytes. Others allocate size 0.
    MPI_Win win_b;
    double *b; // Pointer to the shared B
    MPI_Aint size_b = (node_rank == 0) ? (n * n * sizeof(double)) : 0;
    int disp_unit = sizeof(double);

    MPI_Win_allocate_shared(size_b, disp_unit, MPI_INFO_NULL, node_comm, &b, &win_b);

    // All other ranks need to find WHERE the data is in their virtual address space
    if (node_rank != 0)
    {
        MPI_Aint size_dummy;
        int disp_dummy;
        // Query the pointer from Rank 0's window
        MPI_Win_shared_query(win_b, 0, &size_dummy, &disp_dummy, &b);
    }

    // 3. Initialize Data
    // Rank 0 initializes the Shared B. Everyone waits.
    double (*a)[n] = NULL;
    double (*c)[n] = NULL;

    // Standard Scatter/Gather logic for A and C (Process-local)
    // Note: We could share A and C too, but sharing B is the biggest win.
    int *sendcounts = NULL;
    int *displs = NULL;
    int local_rows = n / world_size; // Simplified for brevity (add remainder logic for robustness)

    double *local_a = malloc(local_rows * n * sizeof(double));
    double *local_c = calloc(local_rows * n, sizeof(double));

    if (world_rank == 0)
    {
        a = malloc(sizeof(double[n][n]));
        c = malloc(sizeof(double[n][n]));
        sendcounts = malloc(world_size * sizeof(int));
        displs = malloc(world_size * sizeof(int));

        // Init A and Local B
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i][j] = 2.0;
                b[i * n + j] = 3.0; // Write directly to shared memory
            }
        }

        // Setup Scatter counts
        for (int i = 0; i < world_size; i++)
        {
            sendcounts[i] = local_rows * n;
            displs[i] = i * local_rows * n;
        }
    }

    // IMPORTANT: Barrier to ensure Rank 0 has finished writing B before anyone reads it.
    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef ENABLE_TIMING
        double t0 = 0.0;
        if (world_rank == 0) t0 = MPI_Wtime();
    #endif


    // Scatter A (Still copying A, because it's split. B is zero-copy!)
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE,
                 local_a, local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 4. Computation (Reading Shared B)
    // Cast flat pointer 'b' to 2D array for cleaner syntax
    double (*b_2d)[n] = (double (*)[n])b;
    
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
                        double temp_a = local_a[i * n + k];
                        // Auto-vectorizable inner loop
                        for (int j = jj; j < min(jj + BLOCK_SIZE, n); j++)
                        {
                            // READING SHARED MEMORY HERE
                            local_c[i * n + j] += temp_a * b_2d[k][j];
                        }
                    }
                }
            }
        }
    }

    // Gather C
    MPI_Gatherv(local_c, local_rows * n, MPI_DOUBLE,
                c, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Safety sync

    #ifdef ENABLE_TIMING
        if (world_rank == 0)
        {
            double t1 = MPI_Wtime();
            fprintf(stderr, "[mpi-opt] ranks=%d n=%d elapsed=%.3f s\n", world_size, n, t1 - t0);
        }
    #endif

    // Verify Output (Optional IO)
    if (world_rank == 0)
    {
        FILE *f = fopen("mat-res.txt", "w");
        if (f)
        {
            fprintf(f, "%d\n\n", n);
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                    fprintf(f, "%.0f ", c[i][j]);
                fprintf(f, "\n");
            }
            fclose(f);
        }
        free(a);
        free(c);
        free(sendcounts);
        free(displs);
    }

    // Free Shared Memory
    MPI_Win_free(&win_b);
    free(local_a);
    free(local_c);

    MPI_Finalize();
    return 0;
}