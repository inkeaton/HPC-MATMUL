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
    int p = (int)sqrt((double)size);
    if (p * p != size) {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] Cannon's algorithm requires a perfect square number of processes (e.g., 4, 9, 16). You provided %d.\n", size);
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

    /* 3. CREATE COMMUNICATORS WITH PERIODIC BOUNDARIES */
    MPI_Comm cart_comm;
    int dims[2] = {p, p};
    int periods[2] = {1, 1}; 
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // Get the reordered rank in the new Cartesian communicator
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int my_coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, my_coords);
    int myrow = my_coords[0];
    int mycol = my_coords[1];

    /* 4. MEMORY ALLOCATION */
    size_t elements = (size_t)b * (size_t)b;
    double * PTR_RESTRICT local_a;
    double * PTR_RESTRICT local_b;
    double * PTR_RESTRICT local_c;

    #ifdef ALIGN
        posix_memalign((void **)&local_a, 64, elements * sizeof(double));
        posix_memalign((void **)&local_b, 64, elements * sizeof(double));
        posix_memalign((void **)&local_c, 64, elements * sizeof(double));
    #else
        local_a = malloc(elements * sizeof(double));
        local_b = malloc(elements * sizeof(double));
        local_c = malloc(elements * sizeof(double));
    #endif

    /* 5. LOCAL INITIALIZATION */
    for (size_t i = 0; i < elements; i++) {
        local_a[i] = 2.0;
        local_b[i] = 3.0;
        local_c[i] = 0.0;
    }

    MPI_Barrier(cart_comm); 
    #ifdef TIME
        struct timespec start, end;
        if (cart_rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 6. INITIAL SKEWING PHASE */
    int left, right, up, down;

    // Skew Matrix A: Shift row 'i' left by 'i' positions
    // Direction 1 = Columns. Negative displacement = shift left.
    MPI_Cart_shift(cart_comm, 1, -myrow, &right, &left); 
    MPI_Sendrecv_replace(local_a, elements, MPI_DOUBLE, left, 1, right, 1, cart_comm, MPI_STATUS_IGNORE);

    // Skew Matrix B: Shift column 'j' up by 'j' positions
    // Direction 0 = Rows. Negative displacement = shift up.
    MPI_Cart_shift(cart_comm, 0, -mycol, &down, &up);
    MPI_Sendrecv_replace(local_b, elements, MPI_DOUBLE, up, 2, down, 2, cart_comm, MPI_STATUS_IGNORE);

    /* 7. CORE COMPUTATION: CANNON'S LOOP */
    for (int step = 0; step < p; step++) {
        
        // --- A. Local Computation (local_c += local_a * local_b) ---
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < b; k++) {
                double r = local_a[i * b + k];
                
                #ifdef ALIGN
                    #pragma vector aligned
                #endif
                for (int j = 0; j < b; j++) {
                    local_c[i * b + j] += r * local_b[k * b + j];
                }
            }
        }

        // --- B. Ring Shift ---
        // Shift Matrix A left by 1 position
        MPI_Cart_shift(cart_comm, 1, -1, &right, &left);
        MPI_Sendrecv_replace(local_a, elements, MPI_DOUBLE, left, 1, right, 1, cart_comm, MPI_STATUS_IGNORE);

        // Shift Matrix B up by 1 position
        MPI_Cart_shift(cart_comm, 0, -1, &down, &up);
        MPI_Sendrecv_replace(local_b, elements, MPI_DOUBLE, up, 2, down, 2, cart_comm, MPI_STATUS_IGNORE);
    }

    /* 8. STOP TIMER & REPORT */
    #ifdef TIME
        MPI_Barrier(cart_comm); 
        if (cart_rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            
            printf("[mpi-cannon] N=%d | elapsed=%.3f s\n", N, time_taken);
        }
    #endif

    /* 9. CLEANUP */
    free(local_a); free(local_b); free(local_c);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}