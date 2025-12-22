/* * ======================================================================================
 * OPTIMIZED HIP MATRIX MULTIPLICATION (AMD RDNA 3)
 * ======================================================================================
 * * SYSTEM: AMD Ryzen 9 7900X + Radeon RX 7700 XT ("Machine AMD")
 * * TARGET ARCH: gfx1101 (RDNA 3) [cite: 124, 131]
 *
 * * COMPILATION:
 * The AMD compiler is 'hipcc'.
 * Flag '--offload-arch=gfx1101' targets your specific RX 7700 XT GPU.
 *
 * Command:
 * hipcc -O3 --offload-arch=gfx1101 -DENABLE_TIMING matmul_hip.cpp -o matmul_hip
 *
 * * EXECUTION:
 * ./matmul_hip
 * ======================================================================================
 */

#define N 5000
// Wavefront Size is 32 on gfx1101, so TILE_WIDTH 32 remains optimal 
#define TILE_WIDTH 32

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>


/* * ERROR WRAPPER (HIP Version) */
static void hipCheck(hipError_t err, const char *msg)
{
    if (err != hipSuccess)
    {
        fprintf(stderr, "%s failed: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

/* * HIP KERNEL: Identical to CUDA */
__global__ void MatrixMulHIP(double *C, const double *A, const double *B, int w)
{
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    double Cvalue = 0.0;

    for (int m = 0; m < (w + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        
        if (Row < w && (m * TILE_WIDTH + tx) < w)
            As[ty][tx] = A[Row * w + (m * TILE_WIDTH + tx)];
        else
            As[ty][tx] = 0.0;

        if (Col < w && (m * TILE_WIDTH + ty) < w)
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * w + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < w && Col < w) {
        C[Row * w + Col] = Cvalue;
    }
}

int main()
{
    // Force selection of the discrete GPU (Agent 2) instead of Integrated (Agent 3)
    // Usually Device 0 is the discrete card, but we can check properties if needed.
    hipCheck(hipSetDevice(0), "hipSetDevice failed");
    
    printf("[Setup] Initializing Matrix size N=%d on AMD GPU...\n", N);
    size_t size = N * N * sizeof(double);

    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 2.0;
            h_B[i * N + j] = 3.0;
            h_C[i * N + j] = 0.0;
        }
    }

    double *d_A, *d_B, *d_C;
    hipCheck(hipMalloc((void **)&d_A, size), "hipMalloc A");
    hipCheck(hipMalloc((void **)&d_B, size), "hipMalloc B");
    hipCheck(hipMalloc((void **)&d_C, size), "hipMalloc C");

    hipCheck(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice), "hipMemcpy A");
    hipCheck(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice), "hipMemcpy B");

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    #ifdef ENABLE_TIMING
        hipEvent_t start, stop;
        hipCheck(hipEventCreate(&start), "hipEventCreate start");
        hipCheck(hipEventCreate(&stop), "hipEventCreate stop");
        hipCheck(hipEventRecord(start, NULL), "hipEventRecord start");
    #endif

    printf("[Compute] Launching HIP Kernel on RX 7700 XT...\n");
    MatrixMulHIP<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);
    
    hipCheck(hipGetLastError(), "Kernel launch failed");

    #ifdef ENABLE_TIMING
        hipCheck(hipEventRecord(stop, NULL), "hipEventRecord stop");
        hipCheck(hipEventSynchronize(stop), "hipEventSynchronize stop");

        float milliseconds = 0;
        hipCheck(hipEventElapsedTime(&milliseconds, start, stop), "hipEventElapsedTime");
        double seconds = milliseconds / 1000.0;
        double gflops = (2.0 * N * N * N) / seconds / 1e9;

        fprintf(stderr, "[optimized-hip] n=%d, time=%.3f s, GFLOPS=%.2f\n", N, seconds, gflops);
        
        hipCheck(hipEventDestroy(start), "hipEventDestroy start");
        hipCheck(hipEventDestroy(stop), "hipEventDestroy stop");
    #else
        hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
    #endif

    hipCheck(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost), "hipMemcpy C");

    // Free resources
    hipCheck(hipFree(d_A), "hipFree A");
    hipCheck(hipFree(d_B), "hipFree B");
    hipCheck(hipFree(d_C), "hipFree C");
    free(h_A); free(h_B); free(h_C);

    return 0;
}