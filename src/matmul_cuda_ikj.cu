#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/* 1. CONFIGURATION FLAGS */

#ifndef N
    #define N 5000 
#endif

// Warning: TILE_SIZE must be <= 32 for CUDA (32x32 = 1024 max threads per block)
#ifndef TILE_SIZE
    #define TILE_SIZE 16 
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

/* 2. GPU COMPUTATION KERNEL */
__global__ void matmul_kernel(const double * PTR_RESTRICT a, 
                              const double * PTR_RESTRICT b, 
                              double * PTR_RESTRICT c, 
                              int n) {
    // Calculate global row and column for this specific thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (crucial if N is not a perfect multiple of TILE_SIZE)
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    /* 3. TIMING SETUP */
    #ifdef TIME
        struct timespec start, end;
    #endif

    size_t bytes = N * N * sizeof(double);

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

    // Copy data from Host (CPU) to Device (GPU)
    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* 7. CONFIGURE CUDA GRID */
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // Ceiling division to ensure we launch enough blocks to cover the matrix
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    /* 8. START TIMER */
    // Synchronize first to ensure no background GPU tasks bleed into our timer
    cudaDeviceSynchronize(); 
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    /* 9. LAUNCH KERNEL */
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // CRITICAL: Wait for GPU to finish before stopping the clock
    cudaCheck(cudaDeviceSynchronize()); 

    /* 10. STOP TIMER & REPORT */
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        printf("[cuda-matmul] N=%d, TILE_SIZE=%d, ALIGN=%d, RESTRICT=%d | elapsed=%.3f s\n", 
                N, TILE_SIZE, time_taken);
    #endif

    /* 11. CLEANUP */
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); // Bring results back
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}