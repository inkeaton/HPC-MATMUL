#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 1. CONFIGURATION FLAGS */

#ifndef N
    #define N 5000 
#endif

// Define independent dimensions for rectangular blocks
#ifndef TILE_X
    #define TILE_X 32 // Threads along the X axis (Columns of C)
#endif

#ifndef TILE_Y
    #define TILE_Y 16 // Threads along the Y axis (Rows of C)
#endif

#ifndef TILE_K
    #define TILE_K 16 // Inner dimension for the dot product tile
#endif

#ifdef RESTRICT
    #define PTR_RESTRICT __restrict__
#else
    #define PTR_RESTRICT
#endif

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
    // Allocate Shared Memory for the independent tile shapes
    __shared__ double ds_A[TILE_Y][TILE_K];
    __shared__ double ds_B[TILE_K][TILE_X];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_Y + ty;
    int col = bx * TILE_X + tx;

    // Flatten thread ID to handle non-square memory loading
    int tid = ty * TILE_X + tx;
    int num_threads = TILE_Y * TILE_X;

    double sum = 0.0;

    int numTiles = (n + TILE_K - 1) / TILE_K;
    for (int p = 0; p < numTiles; ++p) {
        
        // 1D Cooperative load for A (Shape: TILE_Y * TILE_K)
        for (int i = tid; i < TILE_Y * TILE_K; i += num_threads) {
            int r = i / TILE_K;
            int c_idx = i % TILE_K;
            int global_row = by * TILE_Y + r;
            int global_col = p * TILE_K + c_idx;
            
            if (global_row < n && global_col < n) {
                ds_A[r][c_idx] = a[global_row * n + global_col];
            } else {
                ds_A[r][c_idx] = 0.0;
            }
        }

        // 1D Cooperative load for B (Shape: TILE_K * TILE_X)
        for (int i = tid; i < TILE_K * TILE_X; i += num_threads) {
            int r = i / TILE_X;
            int c_idx = i % TILE_X;
            int global_row = p * TILE_K + r;
            int global_col = bx * TILE_X + c_idx;
            
            if (global_row < n && global_col < n) {
                ds_B[r][c_idx] = b[global_row * n + global_col];
            } else {
                ds_B[r][c_idx] = 0.0;
            }
        }

        __syncthreads();

        // Compute partial sum using the inner dimension TILE_K
        for (int k = 0; k < TILE_K; ++k) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write the computed sum to global memory
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(double);

    #ifdef TIME
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
    #endif

    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_a[i] = 2.0;
        h_b[i] = 3.0;
        h_c[i] = 0.0;
    }

    double *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_b, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));

    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Configure the Grid using the independent X and Y blocks
    dim3 threadsPerBlock(TILE_X, TILE_Y);
    dim3 blocksPerGrid((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);

    #ifdef TIME
        cudaCheck(cudaDeviceSynchronize()); 
        cudaCheck(cudaEventRecord(start));
    #endif

    matmul_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    #ifdef TIME
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        printf("[cuda-shared] N=%d, TILE_X=%d, TILE_Y=%d, TILE_K=%d | elapsed=%.3f s\n", 
                N, TILE_X, TILE_Y, TILE_K, seconds); 
                
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
    #else
        cudaCheck(cudaDeviceSynchronize());
    #endif

    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}