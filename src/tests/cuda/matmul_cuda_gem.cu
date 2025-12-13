/* -----------------------------------------------------------
   VERSION 1: OPTIMIZED FOR NVIDIA L4 / TESLA T4
   Technique: Shared Memory Tiling (Blocking)
   ----------------------------------------------------------- */
#define n 5000
#define TILE_WIDTH 32  // 32x32 doubles = 8KB (Fits easily in Shared Mem)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking wrapper
#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__global__ void matmul_tiled_L4(const double *A, const double *B, double *C, int width) {
    // 1. Allocate Shared Memory tiles
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    // 2. Identify thread coordinates
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify global row/col for this thread
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double acc = 0.0;

    // 3. Loop over tiles across the K dimension
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        
        // --- Collaborative Loading ---
        // Each thread loads one element of the tile into shared memory
        
        // Load A element (handle boundary checks)
        if (row < width && (m * TILE_WIDTH + tx) < width)
            As[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
        else
            As[ty][tx] = 0.0;

        // Load B element (handle boundary checks)
        if (col < width && (m * TILE_WIDTH + ty) < width)
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        else
            Bs[ty][tx] = 0.0;

        // Wait for all threads to finish loading
        __syncthreads();

        // --- Computation ---
        // Perform dot product on the tiles in fast shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Wait for computation to finish before loading next tile
        __syncthreads();
    }

    // 4. Write result
    if (row < width && col < width) {
        C[row * width + col] = acc;
    }
}

int main() {
    size_t bytes = n * n * sizeof(double);
    
    // Host allocation
    double *h_a = (double*)malloc(bytes);
    double *h_b = (double*)malloc(bytes);
    double *h_c = (double*)malloc(bytes);

    // Initialization
    for(int i=0; i<n*n; i++) { h_a[i] = 2.0; h_b[i] = 3.0; }

    // Device allocation
    double *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_b, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));

    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // L4 Kernel Launch Configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Launching L4 Tiled Kernel...\n");
    matmul_tiled_L4<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    cudaCheck(cudaDeviceSynchronize());
    printf("Done.\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}