#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* Change this as needed, but must satisfy:
 * - N % q == 0, where q = sqrt(P)
 */
#define N 8

/* Helper to print a matrix (only used on rank 0) */
void print_matrix(const char *name, double *M, int n)
{
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", M[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Initialize matrix with a simple pattern: A[i,j] = i*N + j + 1 */
void init_matrix(double *M, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i * n + j] = (double)(i * n + j + 1);
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* We want P = q^2 processes in a 2D grid */
    int q = (int)round(sqrt((double)size));
    if (q * q != size) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: number of processes (%d) must be a perfect square (q^2).\n",
                    size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* N must be divisible by q so each block is an integer size */
    if (N % q != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: N (%d) must be divisible by sqrt(P) = %d.\n", N, q);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int blockSize = N / q;                 /* local block is blockSize x blockSize */
    int localElems = blockSize * blockSize;

    /* Rank coordinates in the process grid: (row, col) = (pr, pc) */
    int pr = rank / q;   /* process row */
    int pc = rank % q;   /* process column */

    /* Full matrices on rank 0 only */
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    /* Buffers to pack blocks when scattering/gathering (rank 0 only) */
    double *sendBlocksA = NULL;
    double *sendBlocksB = NULL;
    double *recvBlocksC = NULL;

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)calloc(N * N, sizeof(double));  /* initialize C to 0 */

        /* Fill A and B with known values */
        init_matrix(A, N);
        init_matrix(B, N);

        print_matrix("A", A, N);
        print_matrix("B", B, N);

        /* Prepare packed blocks: one contiguous block per process */
        sendBlocksA = (double *)malloc(size * localElems * sizeof(double));
        sendBlocksB = (double *)malloc(size * localElems * sizeof(double));

        for (int proc = 0; proc < size; proc++) {
            int r = proc / q;   /* block row in process grid */
            int c = proc % q;   /* block column in process grid */

            double *subA = &sendBlocksA[proc * localElems];
            double *subB = &sendBlocksB[proc * localElems];

            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int globalRow = r * blockSize + i;
                    int globalCol = c * blockSize + j;

                    subA[i * blockSize + j] = A[globalRow * N + globalCol];
                    subB[i * blockSize + j] = B[globalRow * N + globalCol];
                }
            }
        }
    }

    /* Local blocks: each process owns one block of A, B and C */
    double *localA = (double *)malloc(localElems * sizeof(double));
    double *localB = (double *)malloc(localElems * sizeof(double));
    double *localC = (double *)calloc(localElems, sizeof(double)); /* start with 0 */

    /* Scatter blocks of A and B from rank 0 to all processes */
    MPI_Scatter(sendBlocksA, localElems, MPI_DOUBLE,
                localA,      localElems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(sendBlocksB, localElems, MPI_DOUBLE,
                localB,      localElems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* We don't need sendBlocksA/B anymore on rank 0 */
    if (rank == 0) {
        free(sendBlocksA);
        free(sendBlocksB);
    }

    /* Create row communicators (each row of the grid) for broadcasts */
    MPI_Comm rowComm;
    MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &rowComm);

    /* Buffer for broadcasted A block in Fox algorithm */
    double *tempA = (double *)malloc(localElems * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Fox algorithm: q steps */
    for (int step = 0; step < q; step++) {

        /* Which column will broadcast its A-block in this row? */
        int rootCol = (pr + step) % q;

        if (pc == rootCol) {
            /* This process is the root in this row: copy its A into tempA */
            for (int i = 0; i < localElems; i++) {
                tempA[i] = localA[i];
            }
        }

        /* Broadcast tempA within the row */
        MPI_Bcast(tempA, localElems, MPI_DOUBLE, rootCol, rowComm);

        /* Multiply tempA (A-block) with our localB (B-block) and accumulate into localC */
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                double sum = 0.0;
                for (int k = 0; k < blockSize; k++) {
                    double a_ik = tempA[i * blockSize + k];
                    double b_kj = localB[k * blockSize + j];
                    sum += a_ik * b_kj;
                }
                localC[i * blockSize + j] += sum;
            }
        }

        /* Now rotate B blocks upwards in each column (cyclic shift) */
        int upPr   = (pr - 1 + q) % q;   /* row above (with wrap-around) */
        int downPr = (pr + 1) % q;       /* row below */

        int dest = upPr * q + pc;
        int src  = downPr * q + pc;

        MPI_Status status;
        MPI_Sendrecv_replace(localB, localElems, MPI_DOUBLE,
                             dest, 0,
                             src,  0,
                             MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Gather all localC blocks back to rank 0 */
    if (rank == 0) {
        recvBlocksC = (double *)malloc(size * localElems * sizeof(double));
    }

    MPI_Gather(localC,     localElems, MPI_DOUBLE,
               recvBlocksC, localElems, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Unpack blocks into global C */
        for (int proc = 0; proc < size; proc++) {
            int r = proc / q;
            int c = proc % q;

            double *subC = &recvBlocksC[proc * localElems];

            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int globalRow = r * blockSize + i;
                    int globalCol = c * blockSize + j;
                    C[globalRow * N + globalCol] = subC[i * blockSize + j];
                }
            }
        }

        print_matrix("C = A * B", C, N);
        printf("Total time (Fox, 2D block): %f seconds\n", t1 - t0);
    }

    /* Cleanup */
    free(localA);
    free(localB);
    free(localC);
    free(tempA);
    MPI_Comm_free(&rowComm);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
        free(recvBlocksC);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
