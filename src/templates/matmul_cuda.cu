#define n 5000

// guard
#ifndef ENABLE_TIMING
#define ENABLE_TIMING 1
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* error wrapper */
static void cuda_check(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

/* Naive GPU kernel: computes one C element per thread. */
__global__ void matmul_kernel(const double *a, const double *b, double *c, int ld)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ld || col >= ld)
    {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < ld; ++k)
    {
        acc += a[row * ld + k] * b[k * ld + col];
    }
    c[row * ld + col] = acc;
}

int main(int argc, char **argv)
{
    size_t bytes = sizeof(double) * n * n;

    /* Allocate and initialize host matrices. */
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = 2.0;
            h_b[i * n + j] = 3.0;
            h_c[i * n + j] = 0.0;
        }

    /* Allocate device buffers and copy A, B. */
    double *d_a = nullptr;
    double *d_b = nullptr;
    double *d_c = nullptr;
    cuda_check(cudaMalloc((void **)&d_a, bytes), "cudaMalloc a");
    cuda_check(cudaMalloc((void **)&d_b, bytes), "cudaMalloc b");
    cuda_check(cudaMalloc((void **)&d_c, bytes), "cudaMalloc c");

    cuda_check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D a");
    cuda_check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "H2D b");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    if (ENABLE_TIMING == 1)
    {
        cuda_check(cudaEventCreate(&start), "event create start");
        cuda_check(cudaEventCreate(&stop), "event create stop");
        cuda_check(cudaEventRecord(start), "event record start");
    }

    /* Launch naive kernel. */
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError(), "kernel launch");

    if (ENABLE_TIMING == 1)
    {
        cuda_check(cudaEventRecord(stop), "event record stop");
        cuda_check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        fprintf(stderr, "[cuda] n=%d elapsed=%.3f ms\n", n, ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /* Copy result back to host. */
    cuda_check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");

    /* Dump a 1000x1000 top-left block to file for inspection. */
    FILE *f = fopen("mat-res.txt", "w");
    if (!f)
    {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            fprintf(f, "%.0f ", h_c[i * n + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    /* Free resources before exit. */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
