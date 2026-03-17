#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 1. CONFIGURATION FLAGS */
#ifndef N
#define N 1000 
#endif

#ifdef RESTRICT
#define PTR_RESTRICT restrict
#else
#define PTR_RESTRICT
#endif

int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 2. DYNAMIC ROW DISTRIBUTION */
    int base_rows = N / size;
    int remainder = N % size;
    
    // If rank is less than the remainder, it gets one extra row
    int local_rows = base_rows + (rank < remainder ? 1 : 0);

    // Arrays to hold the variable send counts and memory offsets for Master
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            // We multiply by N because MPI sends elements, not just rows
            sendcounts[i] = rows_for_i * N; 
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    #ifdef TIME
        struct timespec start, end;
    #endif

    /* 3. POINTER DECLARATION */
    double (* PTR_RESTRICT a)[N] = NULL;
    double (* PTR_RESTRICT b)[N] = NULL;
    double (* PTR_RESTRICT c)[N] = NULL;
    
    // Local arrays dynamically sized to the specific rank's local_rows
    double (* PTR_RESTRICT local_a)[N] = NULL;
    double (* PTR_RESTRICT local_c)[N] = NULL;

    /* 4. MEMORY ALLOCATION */
    #ifdef ALIGN
        if (rank == 0) {
            posix_memalign((void **)&a, 64, sizeof(double[N][N]));
            posix_memalign((void **)&c, 64, sizeof(double[N][N]));
        }
        posix_memalign((void **)&b, 64, sizeof(double[N][N]));
        posix_memalign((void **)&local_a, 64, sizeof(double[local_rows][N]));
        posix_memalign((void **)&local_c, 64, sizeof(double[local_rows][N]));
    #else
        if (rank == 0) {
            a = malloc(sizeof(double[N][N]));
            c = malloc(sizeof(double[N][N]));
        }
        b = malloc(sizeof(double[N][N]));
        local_a = malloc(sizeof(double[local_rows][N]));
        local_c = malloc(sizeof(double[local_rows][N]));
    #endif

    /* 5. INITIALIZATION (Master Only) */
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = 2.0;
                b[i][j] = 3.0;
                c[i][j] = 0.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef TIME
        if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 6. COMMUNICATION: VARIABLE SCATTER */
    // Master scatters uneven blocks of A to all processes
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, 
                 local_a, local_rows * N, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // B is still completely broadcasted to everyone
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 7. CORE COMPUTATION (Local i-k-j) */
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_c[i][j] = 0.0;
        }
    }

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < N; k++) {
            double r = local_a[i][k];
            
            #ifdef ALIGN
                #pragma vector aligned
            #endif
            for (int j = 0; j < N; ++j) {
                local_c[i][j] += r * b[k][j];
            }
        }
    }

    /* 8. COMMUNICATION: VARIABLE GATHER */
    // Master collects all uneven local_c chunks back into the full C matrix
    MPI_Gatherv(local_c, local_rows * N, MPI_DOUBLE, 
                c, sendcounts, displs, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    /* 9. STOP TIMER & REPORT (Master Only) */
    #ifdef TIME
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            
            double total_flops = 2.0 * (double)N * (double)N * (double)N;
            double gflops = total_flops / (time_taken * 1e9);
            
            printf("[mpi-ikj] N=%d, PROCESSES=%d | elapsed=%.3f s, GFLOPS=%.2f\n", 
                    N, size, time_taken, gflops);
    }
    #endif

    /* 10. CLEANUP */
    if (rank == 0) {
        free(a);
        free(c);
        free(sendcounts);
        free(displs);
    }
    free(b);
    free(local_a);
    free(local_c);

    MPI_Finalize();
    return 0;
}