/* * COMPILE OPTIONS:
 * nvcc -O3 -arch=sm_70 -std=c++17 matmul_cuda_floats.cu -o matmul_cuda_floats
 */

#define n 5000
#define TILE_WIDTH 32 // Block size (32x32 threads)

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

/* * Tiled Matrix Multiplication Kernel
 * Uses Shared Memory to reduce Global Memory access.
 */
__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c, int width)
{
    // Shared memory for the sub-blocks (tiles) of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Row and Column index for the result element
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global row and col indices
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0f;

    // Loop over the matrix tiles
    // 'm' is the index of the tile (0, 1, 2... width/TILE_WIDTH)
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m)
    {

        // 1. Load Data into Shared Memory
        // Check bounds to handle matrices that aren't multiples of TILE_WIDTH
        if (row < width && (m * TILE_WIDTH + tx) < width)
            As[ty][tx] = a[row * width + (m * TILE_WIDTH + tx)];
        else
            As[ty][tx] = 0.0f;

        if (col < width && (m * TILE_WIDTH + ty) < width)
            Bs[ty][tx] = b[(m * TILE_WIDTH + ty) * width + col];
        else
            Bs[ty][tx] = 0.0f;

        // 2. Synchronize to ensure the tile is loaded
        __syncthreads();

        // 3. Compute partial dot product using Shared Memory
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            val += As[ty][k] * Bs[k][tx];
        }

        // 4. Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < width && col < width)
    {
        c[row * width + col] = val;
    }
}

int main(int argc, char **argv)
{
    // CHANGED: Using float instead of double
    size_t bytes = sizeof(float) * n * n;

    /* Allocate and initialize host matrices. */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = 2.0f;
            h_b[i * n + j] = 3.0f;
            h_c[i * n + j] = 0.0f;
        }

    /* Allocate device buffers */
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    cuda_check(cudaMalloc((void **)&d_a, bytes), "cudaMalloc a");
    cuda_check(cudaMalloc((void **)&d_b, bytes), "cudaMalloc b");
    cuda_check(cudaMalloc((void **)&d_c, bytes), "cudaMalloc c");

    cuda_check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D a");
    cuda_check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "H2D b");

    // Define dimensions
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    printf("Grid size: %d x %d\n", grid.x, grid.y);
    printf("Block size: %d x %d\n", block.x, block.y);
    
    #ifdef ENABLE_TIMING
        cudaEvent_t start, stop;
        cuda_check(cudaEventCreate(&start), "event create start");
        cuda_check(cudaEventCreate(&stop), "event create stop");
        cuda_check(cudaEventRecord(start), "event record start");
    #endif

    /* Launch Tiled kernel */
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError(), "kernel launch");

    #ifdef ENABLE_TIMING
        cuda_check(cudaEventRecord(stop), "event record stop");
        cuda_check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        fprintf(stderr, "[cuda-opt] n=%d elapsed=%.3f ms\n", n, ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif

    /* Copy result back to host. */
    cuda_check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");

    /* Dump a small block to file for inspection. */
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

    /* Free resources */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}