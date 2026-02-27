#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 1. CONFIGURATION FLAGS */

#ifndef N
    #define N 5000 
#endif

// Warning: TILE_SIZE must be <= 32 for CUDA (32x32 = 1024 max threads per block)
// Tesla T4 has 64KB Shared Memory per SM. 32x32 doubles = 8KB per tile.
#ifndef TILE_SIZE
    #define TILE_SIZE 32 
#endif

// CUDA uses __restrict__ natively
#ifdef RESTRICT
    #define PTR_RESTRICT __restrict__
#else
    #define PTR_RESTRICT
#endif

// Helper macro to catch CUDA errors instantly
#define cudaCheck(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/* 2. GPU SHARED MEMORY COMPUTATION KERNEL */
__global__ void matmul_kernel_shared(const double * PTR_RESTRICT a, 
                                     const double * PTR_RESTRICT b, 
                                     double * PTR_RESTRICT c, 
                                     int n) {
    // Allocate Shared Memory for the tiles
    __shared__ double ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ double ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    // Loop over the tiles of the input matrices required to compute the C element
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int p = 0; p < numTiles; ++p) {
        
        // Collaborative loading of A and B tiles into shared memory
        // We include boundary checks in case N is not perfectly divisible by TILE_SIZE
        if (row < n && p * TILE_SIZE + tx < n) {
            ds_A[ty][tx] = a[row * n + (p * TILE_SIZE + tx)];
        } else {
            ds_A[ty][tx] = 0.0;
        }

        if (p * TILE_SIZE + ty < n && col < n) {
            ds_B[ty][tx] = b[(p * TILE_SIZE + ty) * n + col];
        } else {
            ds_B[ty][tx] = 0.0;
        }

        // Wait for all threads to finish loading the tile
        __syncthreads();

        // Compute partial sum for the current tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        // Wait for all threads to finish using the tile before loading the next one
        __syncthreads();
    }

    // Write the computed sum to global memory
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(double);

    /* 3. TIMING SETUP (Using accurate CUDA Events) */
    #ifdef TIME
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
    #endif

    /* 4. HOST POINTERS & ALLOCATION */
    double *h_a, *h_b, *h_c;

    #ifdef ALIGN
        int err_a = posix_memalign((void **)&h_a, 64, bytes);
        int err_b = posix_memalign((void **)&h_b, 64, bytes);
        int err_c = posix_memalign((void **)&h_c, 64, bytes);
        if (err_a != 0 || err_b != 0 || err_c != 0) {
            fprintf(stderr, "Aligned memory allocation failed.\n");
            return 1;
        }
    #else
        h_a = (double *)malloc(bytes);
        h_b = (double *)malloc(bytes);
        h_c = (double *)malloc(bytes);
    #endif

    /* 5. INITIALIZATION (On Host) */
    for (int i = 0; i < N * N; i++) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = ((double)rand() / RAND_MAX) * 10.0;
        h_c[i] = 0.0;
    }

    /* 6. DEVICE POINTERS & ALLOCATION */
    double *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_b, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));

    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* 7. CONFIGURE CUDA GRID */
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    /* 8. START TIMER */
    #ifdef TIME
        cudaCheck(cudaDeviceSynchronize()); 
        cudaCheck(cudaEventRecord(start));
    #endif

    /* 9. LAUNCH SHARED MEMORY KERNEL */
    matmul_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    #ifdef TIME
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));

        /* 10. STOP TIMER & REPORT */
        float milliseconds = 0;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        // Formatted so our benchmark.sh script perfectly captures 'elapsed=X.XXX s'
        printf("[cuda-shared] N=%d, TILE_SIZE=%d | elapsed=%.3f s\n", 
                N, TILE_SIZE, seconds); 
                
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
    #else
        // If not timing, still sync to catch async execution errors
        cudaCheck(cudaDeviceSynchronize());
    #endif

    /* 11. CLEANUP */
    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}