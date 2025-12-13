#define n 5000

// Heuristics for Heterogeneous Cores
// P-cores (Ranks 0-15) are treated as 2x faster than E-cores (Ranks 16-23)
#define P_CORE_WEIGHT 2.0
#define E_CORE_WEIGHT 1.0
#define P_CORE_LIMIT 16 // Ranks 0 to 15 are P-cores

#ifndef ENABLE_TIMING
   #define ENABLE_TIMING 1
#endif

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* -----------------------------------------------------------------------
   * 1. SHARED MEMORY WINDOW FOR MATRIX B
   * Instead of malloc, we allocate B in a shared memory window.
   * This allows all 24 processes to read the SAME physical 200MB of RAM.
   * ----------------------------------------------------------------------- */
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  
  int shm_rank;
  MPI_Comm_rank(shmcomm, &shm_rank);

  MPI_Win win_b;
  double *b = NULL;
  MPI_Aint size_b = 0;

  // Only Rank 0 of the shared group allocates the memory bytes
  if (shm_rank == 0) {
      size_b = (MPI_Aint)n * n * sizeof(double);
  }

  // Create the window
  MPI_Win_allocate_shared(size_b, sizeof(double), MPI_INFO_NULL, shmcomm, &b, &win_b);

  // All other ranks query the address of Rank 0's memory
  if (shm_rank != 0) {
      int disp_unit;
      MPI_Aint stride;
      MPI_Win_shared_query(win_b, 0, &size_b, &disp_unit, &b);
  }

  /* -----------------------------------------------------------------------
   * 2. HETEROGENEOUS LOAD BALANCING
   * We calculate how many rows each rank gets based on core type.
   * ----------------------------------------------------------------------- */
  int *sendcounts = malloc(world_size * sizeof(int));
  int *displs = malloc(world_size * sizeof(int));
  int *rows_per_rank = malloc(world_size * sizeof(int));

  if (world_rank == 0) {
      double total_weight = 0.0;
      double *weights = malloc(world_size * sizeof(double));

      // Assign weights based on rank index (P vs E core mapping)
      for (int i = 0; i < world_size; i++) {
          if (i < P_CORE_LIMIT) weights[i] = P_CORE_WEIGHT;
          else weights[i] = E_CORE_WEIGHT;
          total_weight += weights[i];
      }

      // Distribute rows proportional to weight
      int assigned_rows = 0;
      int current_displ = 0;

      for (int i = 0; i < world_size; i++) {
          // Calculate target rows for this rank
          int target = (int)(n * (weights[i] / total_weight));
          
          // Adjust last rank to pick up any rounding slack
          if (i == world_size - 1) {
              target = n - assigned_rows;
          }

          rows_per_rank[i] = target;
          sendcounts[i] = target * n; // Sendcounts is in doubles (rows * cols)
          displs[i] = current_displ;
          
          assigned_rows += target;
          current_displ += sendcounts[i];
      }
      free(weights);
  }

  // Broadcast the computed plan to everyone so they know how much to receive
  MPI_Bcast(sendcounts, world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs, world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(rows_per_rank, world_size, MPI_INT, 0, MPI_COMM_WORLD);

  int my_rows = rows_per_rank[world_rank];


  /* -----------------------------------------------------------------------
   * 3. LOCAL MEMORY & INITIALIZATION
   * ----------------------------------------------------------------------- */
  double *a_full = NULL; // Only Rank 0 holds full A
  double *c_full = NULL; // Only Rank 0 holds full C result
  
  double *a_local = malloc(my_rows * n * sizeof(double));
  double *c_local = malloc(my_rows * n * sizeof(double));

  if (world_rank == 0) {
      a_full = malloc(n * n * sizeof(double));
      c_full = malloc(n * n * sizeof(double)); // Final Result

      // Initialize A
      for (int i = 0; i < n * n; i++) a_full[i] = 2.0;

      // Initialize B (directly into shared memory)
      for (int i = 0; i < n * n; i++) b[i] = 3.0;
  }

  // Mandatory Sync: Wait for Rank 0 to fill Matrix B in shared memory
  MPI_Barrier(MPI_COMM_WORLD);

  /* -----------------------------------------------------------------------
   * 4. EXECUTION
   * ----------------------------------------------------------------------- */
  double t0 = 0.0;
  if (ENABLE_TIMING == 1 && world_rank == 0) {
      t0 = MPI_Wtime();
  }

  // Scatter A (Rank 0 sends pieces of A to everyone)
  MPI_Scatterv(a_full, sendcounts, displs, MPI_DOUBLE, 
               a_local, sendcounts[world_rank], MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

  // Compute: C_local = A_local * B_shared
  // Initialize local C to 0
  for (int i = 0; i < my_rows * n; i++) c_local[i] = 0.0;

  for (int i = 0; i < my_rows; i++) {
      for (int k = 0; k < n; k++) {
          double a_val = a_local[i * n + k];
          // Pointer arithmetic for B since it's a 1D shared array
          double *b_row_k = &b[k * n]; 
          
          for (int j = 0; j < n; j++) {
              c_local[i * n + j] += a_val * b_row_k[j];
          }
      }
  }

  // Gather C (Everyone sends results back to Rank 0)
  MPI_Gatherv(c_local, sendcounts[world_rank], MPI_DOUBLE,
              c_full, sendcounts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (ENABLE_TIMING == 1 && world_rank == 0) {
      double t1 = MPI_Wtime();
      fprintf(stderr, "[mpi-opt] ranks=%d n=%d elapsed=%.3f s\n", world_size, n, t1 - t0);
  }

  /* -----------------------------------------------------------------------
   * 5. OUTPUT & CLEANUP
   * ----------------------------------------------------------------------- */
  if (world_rank == 0) {
      FILE *f = fopen("mat-res.txt", "w");
      if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

      fprintf(f, "%d\n\n", n);
      // Accessing c_full as 1D array
      for (int ii = 0; ii < 1000; ii++) {
          for (int jj = 0; jj < 1000; jj++) {
              fprintf(f, "%.0f ", c_full[ii * n + jj]);
          }
          fprintf(f, "\n");
      }
      fclose(f);
      
      free(a_full);
      free(c_full);
  }

  // Free shared window handles (B is freed by MPI_Win_Free)
  MPI_Win_free(&win_b); 
  
  free(a_local);
  free(c_local);
  free(sendcounts);
  free(displs);
  free(rows_per_rank);

  MPI_Finalize();
  return 0;
}

// compile
// mpicc -O3 matmul_mpi_gem.c -o matmul_gem
// mpirun -np 24 --bind-to core ./matmul_gem