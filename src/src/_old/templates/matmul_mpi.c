#define n 5000

// guard
#ifndef ENABLE_TIMING
#define ENABLE_TIMING 1
#endif

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* Discover MPI world size and this rank. */
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* For now only rank 0 holds the matrices; distribution will be added later. */
    double (*a)[n] = NULL;
    double (*b)[n] = NULL;
    double (*c)[n] = NULL;

    if (world_rank == 0)
    {
        a = malloc(sizeof(double[n][n]));
        b = malloc(sizeof(double[n][n]));
        c = malloc(sizeof(double[n][n]));

        /* Initialize A and B with constants; zero C. */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                a[i][j] = 2.0;
                b[i][j] = 3.0;
                c[i][j] = 0.0;
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = 0.0;
    if (ENABLE_TIMING == 1 && world_rank == 0)
    {
        /* MPI_Wtime is the portable wall-clock timer for MPI codes. */
        t0 = MPI_Wtime();
    }

    /* Naive multiplication on rank 0 only (distribution to come later). */
    if (world_rank == 0)
    {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; ++j)
                    c[i][j] += a[i][k] * b[k][j];
    }

    if (ENABLE_TIMING == 1 && world_rank == 0)
    {
        double t1 = MPI_Wtime();
        fprintf(stderr, "[mpi] ranks=%d n=%d elapsed=%.3f s\n", world_size, n, t1 - t0);
    }

    if (world_rank == 0)
    {
        /* Dump a 1000x1000 top-left block to file for inspection. */
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
    }

    /* Free resources before exit. */
    free(a);
    free(b);
    free(c);

    MPI_Finalize();
    return 0;
}
