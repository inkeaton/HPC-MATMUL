// mpiicx -O3 -DTIME -DALIGN -DRESTRICT -DN=10000 -DBLOCK_SIZE=64 -qmkl=cluster src/matmul_mpi_scala.c -o bin/mpi_scala

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* --- PREPROCESSOR FIX --- */
#pragma push_macro("N")
#undef N
#include <mkl.h> 
#pragma pop_macro("N")
/* ------------------------ */

/* 1. CONFIGURATION FLAGS */
#ifndef N
    #define N 1000 
#endif

// ScaLAPACK block size (MB = NB = BLOCK_SIZE)
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 64 
#endif

#ifdef RESTRICT
    #define PTR_RESTRICT restrict
#else
    #define PTR_RESTRICT
#endif

/* 2. FORTRAN SCALAPACK PROTOTYPES */
// Fortran expects everything to be passed by reference (pointers)
extern void Cblacs_pinfo(int* mypnum, int* nprocs);
extern void Cblacs_get(int context, int request, int* value);
extern void Cblacs_gridinit(int* context, const char * order, int nprow, int npcol);
extern void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);
extern int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info);
extern void pdgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, 
                    double *a, int *ia, int *ja, int *desca, 
                    double *b, int *ib, int *jb, int *descb, 
                    double *beta, 
                    double *c, int *ic, int *jc, int *descc);

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 3. SET UP 2D BLACS GRID */
    int ictxt, nprow, npcol, myrow, mycol;
    int iam, nprocs;

    Cblacs_pinfo(&iam, &nprocs);
    Cblacs_get(-1, 0, &ictxt);

    // Calculate a roughly square grid for the processes (e.g., 4 procs -> 2x2)
    nprow = (int)sqrt((double)size);
    while (size % nprow != 0) {
        nprow--;
    }
    npcol = size / nprow;

    // Initialize the grid in Row-major order
    Cblacs_gridinit(&ictxt, "Row", nprow, npcol);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // If a process is not part of the active grid, let it exit gracefully
    if (myrow < 0 || mycol < 0) {
        Cblacs_exit(0);
        MPI_Finalize();
        return 0;
    }

    /* 4. CALCULATE LOCAL MATRIX SIZES */
    // Fortran pass-by-reference constants
    int ZERO = 0, ONE = 1, N_VAL = N, B_VAL = BLOCK_SIZE;
    
    // numroc_ tells each process exactly how many rows and columns it owns
    int local_rows = numroc_(&N_VAL, &B_VAL, &myrow, &ZERO, &nprow);
    int local_cols = numroc_(&N_VAL, &B_VAL, &mycol, &ZERO, &npcol);

    // Leading dimension (must be at least 1)
    int lld = (local_rows > 1) ? local_rows : 1;

    /* 5. MEMORY ALLOCATION */
    // ScaLAPACK uses 1D arrays to represent local 2D data in Column-Major format
    size_t local_elements = (size_t)local_rows * (size_t)local_cols;
    double * PTR_RESTRICT local_A;
    double * PTR_RESTRICT local_B;
    double * PTR_RESTRICT local_C;

    #ifdef ALIGN
        posix_memalign((void **)&local_A, 64, local_elements * sizeof(double));
        posix_memalign((void **)&local_B, 64, local_elements * sizeof(double));
        posix_memalign((void **)&local_C, 64, local_elements * sizeof(double));
    #else
        local_A = malloc(local_elements * sizeof(double));
        local_B = malloc(local_elements * sizeof(double));
        local_C = malloc(local_elements * sizeof(double));
    #endif

    // Initialization
    for (size_t i = 0; i < local_elements; i++) {
        local_A[i] = (double)rand() / RAND_MAX;
        local_B[i] = ((double)rand() / RAND_MAX) * 10.0;
        local_C[i] = 0.0;
    }

    /* 6. CREATE DESCRIPTORS */
    int descA[9], descB[9], descC[9], info;
    descinit_(descA, &N_VAL, &N_VAL, &B_VAL, &B_VAL, &ZERO, &ZERO, &ictxt, &lld, &info);
    descinit_(descB, &N_VAL, &N_VAL, &B_VAL, &B_VAL, &ZERO, &ZERO, &ictxt, &lld, &info);
    descinit_(descC, &N_VAL, &N_VAL, &B_VAL, &B_VAL, &ZERO, &ZERO, &ictxt, &lld, &info);

    /* 7. TIMING & SYNCHRONIZATION */
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef TIME
        struct timespec start, end;
        if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 8. CORE COMPUTATION: ScaLAPACK PDGEMM */
    char trans = 'N';
    double alpha = 1.0, beta = 0.0;

    pdgemm_(&trans, &trans, 
            &N_VAL, &N_VAL, &N_VAL, 
            &alpha, 
            local_A, &ONE, &ONE, descA, 
            local_B, &ONE, &ONE, descB, 
            &beta, 
            local_C, &ONE, &ONE, descC);

    /* 9. STOP TIMER & REPORT */
    #ifdef TIME
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            
            printf("[scalapack-pdgemm] N=%d | elapsed=%.3f s\n", 
                    N, time_taken);
        }
    #endif

    /* 10. CLEANUP */
    free(local_A);
    free(local_B);
    free(local_C);

    Cblacs_gridexit(ictxt);
    MPI_Finalize();
    return 0;
}