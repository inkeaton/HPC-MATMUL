/*
 * HIP MATRIX MULTIPLICATION (FIXED)
 * Compile with: hipcc -O3 -std=c++14 matmul_hip.cpp -o matmul_hip
 */

// CHANGED: n -> N to avoid conflicting with internal HIP variables
#define N 5000
#define TILE_WIDTH 32 

#ifndef ENABLE_TIMING
   #define ENABLE_TIMING 1
#endif

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void hip_check(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

__global__ void matmul_tiled_kernel(const double *a, const double *b, double *c, int width) {
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double val = 0.0;

    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        
        if (row < width && (m * TILE_WIDTH + tx) < width)
            As[ty][tx] = a[row * width + (m * TILE_WIDTH + tx)];
        else
            As[ty][tx] = 0.0;

        if (col < width && (m * TILE_WIDTH + ty) < width)
            Bs[ty][tx] = b[(m * TILE_WIDTH + ty) * width + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        c[row * width + col] = val;
    }
}

int main(int argc, char **argv) {
    // CHANGED: n -> N
    size_t bytes = sizeof(double) * N * N;

    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = 2.0;
            h_b[i * N + j] = 3.0;
            h_c[i * N + j] = 0.0;
        }

    double *d_a = nullptr;
    double *d_b = nullptr;
    double *d_c = nullptr;
    hip_check(hipMalloc((void **)&d_a, bytes), "hipMalloc a");
    hip_check(hipMalloc((void **)&d_b, bytes), "hipMalloc b");
    hip_check(hipMalloc((void **)&d_c, bytes), "hipMalloc c");

    hip_check(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice), "H2D a");
    hip_check(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice), "H2D b");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    // CHANGED: n -> N
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    hipEvent_t start, stop;
    if (ENABLE_TIMING == 1) {
        hip_check(hipEventCreate(&start), "event create start");
        hip_check(hipEventCreate(&stop), "event create stop");
        hip_check(hipEventRecord(start), "event record start");
    }

    // CHANGED: n -> N
    hipLaunchKernelGGL(matmul_tiled_kernel, grid, block, 0, 0, d_a, d_b, d_c, N);
    hip_check(hipGetLastError(), "kernel launch");

    if (ENABLE_TIMING == 1) {
        hip_check(hipEventRecord(stop), "event record stop");
        hip_check(hipEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        hip_check(hipEventElapsedTime(&ms, start, stop), "event elapsed");
        fprintf(stderr, "[hip-opt] n=%d elapsed=%.3f ms\n", N, ms);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    hip_check(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost), "D2H c");

    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) fprintf(f, "%.0f ", h_c[i * N + j]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}