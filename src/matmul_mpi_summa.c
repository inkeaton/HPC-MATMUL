#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

    /* 2. GRID CONFIGURATION */
    // SUMMA requires a 2D Cartesian grid. For simplicity, we enforce a perfect square.
    int p = (int)sqrt((double)size);
    if (p * p != size) {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] SUMMA requires a perfect square number of processes (e.g., 4, 9, 16). You provided %d.\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    if (N % p != 0) {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] N (%d) must be evenly divisible by the grid dimension p (%d).\n", N, p);
        }
        MPI_Finalize();
        return 1;
    }

    int b = N / p; // The local block size (b x b)

    /* 3. CREATE COMMUNICATORS */
    MPI_Comm cart_comm, row_comm, col_comm;
    int dims[2] = {p, p};
    int periods[2] = {0, 0}; // No wrap-around needed
    
    // Create the main 2D grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, my_coords);
    int myrow = my_coords[0];
    int mycol = my_coords[1];

    // Split the grid into Row and Column sub-communicators
    int remain_dims[2];
    
    remain_dims[0] = 0; remain_dims[1] = 1; // Vary columns -> Row Communicator
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);
    
    remain_dims[0] = 1; remain_dims[1] = 0; // Vary rows -> Column Communicator
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    /* 4. MEMORY ALLOCATION */
    // We allocate 1D contiguous arrays to represent our 2D blocks
    size_t elements = (size_t)b * (size_t)b;
    double * PTR_RESTRICT local_a;
    double * PTR_RESTRICT local_b;
    double * PTR_RESTRICT local_c;
    
    // Buffers to receive the broadcasted blocks
    double * PTR_RESTRICT temp_a; 
    double * PTR_RESTRICT temp_b;

#ifdef ALIGN
    posix_memalign((void **)&local_a, 64, elements * sizeof(double));
    posix_memalign((void **)&local_b, 64, elements * sizeof(double));
    posix_memalign((void **)&local_c, 64, elements * sizeof(double));
    posix_memalign((void **)&temp_a,  64, elements * sizeof(double));
    posix_memalign((void **)&temp_b,  64, elements * sizeof(double));
#else
    local_a = malloc(elements * sizeof(double));
    local_b = malloc(elements * sizeof(double));
    local_c = malloc(elements * sizeof(double));
    temp_a  = malloc(elements * sizeof(double));
    temp_b  = malloc(elements * sizeof(double));
#endif

    /* 5. LOCAL INITIALIZATION */
    // Instead of scattering from Master, each node generates its own chunk of data
    for (size_t i = 0; i < elements; i++) {
        local_a[i] = 2.0;
        local_b[i] = 3.0;
        local_c[i] = 0.0;
    }

    MPI_Barrier(cart_comm); // Sync before starting the clock
#ifdef TIME
    struct timespec start, end;
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
#endif

    /* 6. CORE COMPUTATION: SUMMA LOOP */
    // Iterate through the grid dimension
    for (int k = 0; k < p; k++) {
        
        // --- STEP A: Broadcast block of A horizontally ---
        if (mycol == k) {
            // I am the root for this row's broadcast. Copy my A block into the temp buffer.
            for (size_t i = 0; i < elements; i++) temp_a[i] = local_a[i];
        }
        // Every process in the row receives the block from column k
        MPI_Bcast(temp_a, elements, MPI_DOUBLE, k, row_comm);

        // --- STEP B: Broadcast block of B vertically ---
        if (myrow == k) {
            // I am the root for this col's broadcast. Copy my B block into the temp buffer.
            for (size_t i = 0; i < elements; i++) temp_b[i] = local_b[i];
        }
        // Every process in the column receives the block from row k
        MPI_Bcast(temp_b, elements, MPI_DOUBLE, k, col_comm);

        // --- STEP C: Local Computation (temp_a * temp_b) ---
        // Using the optimized i-k-j loop structure from your original code
        for (int i = 0; i < b; i++) {
            for (int k_idx = 0; k_idx < b; k_idx++) {
                double r = temp_a[i * b + k_idx];
                
#ifdef ALIGN
#pragma vector aligned
#endif
                for (int j = 0; j < b; j++) {
                    local_c[i * b + j] += r * temp_b[k_idx * b + j];
                }
            }
        }
    }

    /* 7. STOP TIMER & REPORT */
#ifdef TIME
    MPI_Barrier(cart_comm); // Ensure all ranks finish computing before stopping the timer
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        printf("[mpi-summa] N=%d, ALIGN=%d, RESTRICT=%d | elapsed=%.3f s\n", 
                N, 
#ifdef ALIGN
                1, 
#else
                0, 
#endif
#ifdef RESTRICT
                1, 
#else
                0,
#endif
                time_taken);
    }
#endif

    /* 8. CLEANUP */
    free(local_a); free(local_b); free(local_c);
    free(temp_a);  free(temp_b);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}