#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT 0

/* Index helper for row-major storage */
#define IDX(i, j, n) ((i) * (n) + (j))

/* Initialize matrix with simple pattern: M[i,j] = i*n + j + 1 */
void init_matrix(double *M, int n, int seed)
{
    (void)seed;  // seed unused; here just a simple pattern
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[IDX(i, j, n)] = (double)(i * n + j + 1);
        }
    }
}

/* Print n x n matrix (only for small matrices) */
void print_matrix(const char *name, double *M, int n)
{
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.1f ", M[IDX(i, j, n)]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;
    int N;              /* Matrix size N x N */
    int printMatrices = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* --- Parse command line --- */
    if (rank == ROOT) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s N [print]\n", argv[0]);
            fprintf(stderr, "  N      : matrix dimension (NxN)\n");
            fprintf(stderr, "  print  : 1 to print matrices (only reasonable for small N)\n");
        }
    }

    if (argc < 2) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    if (argc >= 3) {
        printMatrices = atoi(argv[2]);
    }

    if (N <= 0) {
        if (rank == ROOT) {
            fprintf(stderr, "Error: N must be positive.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* For simplicity, require N divisible by number of processes */
    if (N % size != 0) {
        if (rank == ROOT) {
            fprintf(stderr,
                    "Error: N (%d) must be divisible by number of processes P (%d).\n",
                    N, size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int localRows = N / size;               /* each process handles localRows rows */
    int localSize = localRows * N;          /* number of elements per local block */

    /* --- Allocate matrices --- */

    double *A = NULL;   /* full A on ROOT only */
    double *B = NULL;   /* full B on ROOT only */
    double *C = NULL;   /* full C on ROOT only */

    if (rank == ROOT) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        if (!A || !B || !C) {
            fprintf(stderr, "Error: not enough memory for full matrices on ROOT.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        init_matrix(A, N, 1);
        init_matrix(B, N, 2);

        if (printMatrices && N <= 16) {
            print_matrix("A", A, N);
            print_matrix("B", B, N);
        }
    }

    /* Every process needs:
       - localA: its chunk of rows of A
       - localC: its chunk of rows of C
       - fullB : entire B (broadcasted once)
    */
    double *localA = (double *)malloc(localSize * sizeof(double));
    double *localC = (double *)malloc(localSize * sizeof(double));
    double *fullB  = (double *)malloc(N * N * sizeof(double));
    if (!localA || !localC || !fullB) {
        fprintf(stderr, "Rank %d: Error allocating local buffers.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* --- Distribute A and B --- */

    /* Scatter rows of A: ROOT sends, everyone receives localRows*N elements */
    MPI_Scatter(A, localSize, MPI_DOUBLE,
                localA, localSize, MPI_DOUBLE,
                ROOT, MPI_COMM_WORLD);

    /* ROOT copies B into fullB, others don't care initial contents */
    if (rank == ROOT) {
        for (int i = 0; i < N * N; i++) {
            fullB[i] = B[i];
        }
    }

    /* Broadcast full B to all processes */
    MPI_Bcast(fullB, N * N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    /* --- Parallel matrix multiplication: C = A * B --- */

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += localA[IDX(i, k, N)] * fullB[IDX(k, j, N)];
            }
            localC[IDX(i, j, N)] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double localTime = t1 - t0;

    /* Get the maximum time across all processes (the slowest one) */
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);

    /* Gather localC blocks into C on ROOT */
    MPI_Gather(localC, localSize, MPI_DOUBLE,
               C,       localSize, MPI_DOUBLE,
               ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        if (printMatrices && N <= 16) {
            print_matrix("C = A * B", C, N);
        }

        /* Estimate performance: about 2*N^3 floating-point ops */
        double flops   = 2.0 * (double)N * (double)N * (double)N;
        double gflops  = flops / (maxTime * 1.0e9);

        printf("N = %d, processes = %d\n", N, size);
        printf("Parallel time: %.6f s (max over ranks)\n", maxTime);
        printf("Approx performance: %.3f GFLOP/s\n", gflops);
    }

    /* --- Cleanup --- */
    free(localA);
    free(localC);
    free(fullB);

    if (rank == ROOT) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
/*
# Small debug run (prints matrices)
mpicc -o mpi_matmul mpi_matmul.c
mpirun -np 4 ./mpi_matmul 8 1

# Performance runs (no printing)
mpirun -np 4  ./mpi_matmul 1024
mpirun -np 8  ./mpi_matmul 1024
mpirun -np 12 ./mpi_matmul 1024
mpirun -np 16 ./mpi_matmul 1024
mpirun -np 24 ./mpi_matmul 1024

! Try larger N too: N = 1024, 2048, maybe 4096 if you have RAM (and patience).
*/