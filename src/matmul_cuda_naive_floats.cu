#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef N
    #define N 5000 
#endif

#ifndef TILE_X
    #define TILE_X 16 
#endif

#ifndef TILE_Y
    #define TILE_Y 16 
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

__global__ void matmul_kernel_naive(const float * PTR_RESTRICT a, 
                                    const float * PTR_RESTRICT b, 
                                    float * PTR_RESTRICT c, 
                                    int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    #ifdef TIME
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
    #endif

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = ((float)rand() / RAND_MAX) * 10.0f;
        h_c[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_b, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));

    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_X, TILE_Y);
    dim3 blocksPerGrid((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);

    #ifdef TIME
        cudaCheck(cudaDeviceSynchronize()); 
        cudaCheck(cudaEventRecord(start));
    #endif

    matmul_kernel_naive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    #ifdef TIME
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        printf("[cuda-naive-float] N=%d | elapsed=%.3f s\n", N, seconds);
    #else
        cudaCheck(cudaDeviceSynchronize());
    #endif

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}