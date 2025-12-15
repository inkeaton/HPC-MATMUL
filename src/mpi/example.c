/*
Matrix multiplication: example.c
For simplicity, we suppose that N is a multiple of P.
Process 0 initializes matrix and vector, then print both
Process 0 scatters matrix to all processes
Process 0 broadcasts vectors to all processes
Each process computes matrix multiplication, then stores the results in a local vector
Process 0 gathers all local vectors, getting the final result
Process 0 visualizes the final result
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 8   /* size of matrix and vector; must be divisible by #processes */

/* function prototypes */
void initializeMatrix(int rows, int a[][N]);
void initializeVector(int n, int v[]);
void printMatrix(int rows, int a[][N]);
void printVector(int n, int v[]);
void mult(int rows, int a[][N], int v[], int out[]);

int main(int argc, char *argv[])
{
    int myrank, P;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* We assume N is a multiple of P */
    if (N % P != 0) {
        if (myrank == 0) {
            fprintf(stderr,
                    "Error: N (%d) must be divisible by number of processes P (%d)\n",
                    N, P);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Each process will hold N/P rows of the matrix */
    int sendMatrix[N][N];       /* full matrix – used only on rank 0 */
    int recvMatrix[N / P][N];   /* block of rows for this process    */
    int vector[N];              /* full vector                       */
    int localResult[N / P];     /* partial result for this process   */
    int result[N];              /* final result – used only on rank 0 */

    if (myrank == 0) {
        /* initialize matrix and vector only on master */
        initializeMatrix(N, sendMatrix);
        initializeVector(N, vector);

        printf("Matrix:\n");
        printMatrix(N, sendMatrix);
        printf("\nVector:\n");
        printVector(N, vector);
        printf("\n");
    }

    /* Scatter rows of the matrix to all processes */
    MPI_Scatter(sendMatrix,          /* send buffer (root only)          */
                N * N / P, MPI_INT,  /* elements per process             */
                recvMatrix,          /* receive buffer (each process)    */
                N * N / P, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Broadcast the vector to all processes */
    MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each process multiplies its own block of rows by the vector */
    mult(N / P, recvMatrix, vector, localResult);

    /* Gather all partial results into rank 0 */
    MPI_Gather(localResult, N / P, MPI_INT,
               result,      N / P, MPI_INT,
               0, MPI_COMM_WORLD);

    if (myrank == 0) {
        printf("Result (A * x):\n");
        printVector(N, result);
        printf("\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/* Fill matrix with simple values: a[i][j] = i*N + j + 1 */
void initializeMatrix(int rows, int a[][N])
{
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = i * N + j + 1;
        }
    }
}

/* Fill vector with 1,2,3,... */
void initializeVector(int n, int v[])
{
    int i;
    for (i = 0; i < n; i++) {
        v[i] = i + 1;
    }
}

void printMatrix(int rows, int a[][N])
{
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < N; j++) {
            printf("%4d ", a[i][j]);
        }
        printf("\n");
    }
}

void printVector(int n, int v[])
{
    int i;
    for (i = 0; i < n; i++) {
        printf("%4d ", v[i]);
    }
    printf("\n");
}

/* rows = number of rows owned by this process (N/P) */
void mult(int rows, int a[][N], int v[], int out[])
{
    int i, j;
    for (i = 0; i < rows; i++) {
        out[i] = 0;
        for (j = 0; j < N; j++) {
            out[i] += a[i][j] * v[j];
        }
    }
}
