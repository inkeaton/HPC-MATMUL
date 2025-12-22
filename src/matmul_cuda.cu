/* * ======================================================================================
 * OPTIMIZED CUDA MATRIX MULTIPLICATION (TILED + ERROR CHECKING)
 * ======================================================================================
 * SYSTEM: NVIDIA Tesla T4 (Colab)
 * ARCHITECTURE: Turing (Compute Capability 7.5)
 *
 * * COMPILATION INSTRUCTIONS:
 * -------------------------
 * We use 'nvcc' (NVIDIA CUDA Compiler).
 * Flag '-arch=sm_75' targets the Turing architecture specifically.
 * Flag '-DENABLE_TIMING' enables the timing logic.
 *
 * * Command (With Timing):
 * nvcc -O3 -arch=sm_75 -DENABLE_TIMING matmul_cuda_final.cu -o matmul_cuda
 *
 * * Command (Without Timing):
 * nvcc -O3 -arch=sm_75 matmul_cuda_final.cu -o matmul_cuda
 *
 * * EXECUTION:
 * ./matmul_cuda
 * ======================================================================================
 */

#define N 5000
// TILE_WIDTH 32 is chosen because:
// 1. It matches the GPU Warp Size (32), ensuring Coalesced Memory Access.
// 2. 32x32 doubles = 8KB per array. 2 arrays = 16KB total Shared Memory per block.
//    The Tesla T4 has 64KB Shared Memory per SM, so this fits comfortably.
#define TILE_WIDTH 32

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* * ERROR WRAPPER
 * Checks the return code of CUDA functions and exits if an error occurs.
 */
static void cudaCheck(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

/* * CUDA KERNEL: Tiled Matrix Multiplication 
 * Calculates C = A * B using Shared Memory Tiling.
 */
__global__ void MatrixMulCUDA(double *C, const double *A, const double *B, int w)
{
    // 1. SHARED MEMORY ALLOCATION
    // This memory is visible to all threads in a block and acts as a manual L1 cache.
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    // 2. THREAD IDENTIFICATION
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Calculate the global Row and Column indices of the element C this thread computes
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Accumulator for the dot product
    double Cvalue = 0.0;

    // 3. MAIN LOOP: ITERATE OVER TILES
    // Instead of reading the full Row A and Col B, we step through them in chunks of TILE_WIDTH.
    for (int m = 0; m < (w + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        /* * PHASE A: COLLABORATIVE LOADING
         * Each thread loads ONE element from A and ONE element from B into Shared Memory.
         * The threads work together to fill the tile.
         */
         
        // Load A[Row][m*TILE + tx] into As[ty][tx] with boundary check
        if (Row < w && (m * TILE_WIDTH + tx) < w)
            As[ty][tx] = A[Row * w + (m * TILE_WIDTH + tx)];
        else
            As[ty][tx] = 0.0; // Padding with zero

        // Load B[m*TILE + ty][Col] into Bs[ty][tx] with boundary check
        if (Col < w && (m * TILE_WIDTH + ty) < w)
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * w + Col];
        else
            Bs[ty][tx] = 0.0; // Padding with zero

        /* * SYNC 1: WAIT FOR LOAD
         * Ensure the entire tile is loaded before any thread starts computing.
         */
        __syncthreads();

        /* * PHASE B: COMPUTATION (DOT PRODUCT)
         * Calculate partial dot product using FAST Shared Memory data.
         * Note: We access As[ty][k] and Bs[k][tx]. This loop has ZERO global memory access.
         */
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        /* * SYNC 2: WAIT FOR COMPUTE
         * Ensure all threads are done using the current tile before overwriting it.
         */
        __syncthreads();
    }

    // 4. WRITE RESULT
    // Write the final accumulated value back to Slow Global Memory.
    if (Row < w && Col < w) {
        C[Row * w + Col] = Cvalue;
    }
}

int main()
{
    printf("[Setup] Initializing Matrix size N=%d...\n", N);
    size_t size = N * N * sizeof(double);

    // Host Memory Allocation (RAM)
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        perror("Host allocation failed");
        return 1;
    }

    // Initialization (Row-Major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 2.0;
            h_B[i * N + j] = 3.0;
            h_C[i * N + j] = 0.0;
        }
    }

    // Device Memory Allocation (VRAM)
    double *d_A, *d_B, *d_C;
    
    // Applying cudaCheck wrapper
    cudaCheck(cudaMalloc((void **)&d_A, size), "cudaMalloc A");
    cudaCheck(cudaMalloc((void **)&d_B, size), "cudaMalloc B");
    cudaCheck(cudaMalloc((void **)&d_C, size), "cudaMalloc C");

    printf("[Transfer] Copying input data to GPU...\n");
    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A HostToDevice");
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B HostToDevice");

    // Grid Configuration
    // Block: 32x32 threads
    // Grid: Enough blocks to cover the N=5000 matrix
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // --- TIMING START SECTION ---
    #ifdef ENABLE_TIMING
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start), "cudaEventCreate start");
        cudaCheck(cudaEventCreate(&stop), "cudaEventCreate stop");
        cudaCheck(cudaEventRecord(start), "cudaEventRecord start");
    #endif
    // ----------------------------

    printf("[Compute] Launching CUDA Kernel...\n");
    
    // LAUNCH KERNEL
    MatrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);
    
    // Check for synchronous errors (e.g., invalid configuration) immediately
    cudaCheck(cudaGetLastError(), "Kernel launch failed");

    // --- TIMING END SECTION ---
    #ifdef ENABLE_TIMING
        cudaCheck(cudaEventRecord(stop), "cudaEventRecord stop");
        cudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float milliseconds = 0;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        double seconds = milliseconds / 1000.0;

        fprintf(stderr, "[optimized-cuda] n=%d, time=%.3f sn", N, seconds);
        
        cudaCheck(cudaEventDestroy(start), "cudaEventDestroy start");
        cudaCheck(cudaEventDestroy(stop), "cudaEventDestroy stop");
    #else
        // If timing is off, explicitly sync to catch asynchronous kernel execution errors
        cudaCheck(cudaDeviceSynchronize(), "Kernel Execution Sync");
    #endif
    // --------------------------

    printf("[Transfer] Copying results back to CPU...\n");
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C DeviceToHost");

    // Verification Output
    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        // Dump top-left 1000x1000 block
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) {
                fprintf(f, "%.0f ", h_C[i * N + j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        printf("[Output] Results written to mat-res.txt\n");
    }

    // Cleanup
    cudaCheck(cudaFree(d_A), "cudaFree A");
    cudaCheck(cudaFree(d_B), "cudaFree B");
    cudaCheck(cudaFree(d_C), "cudaFree C");
    free(h_A); free(h_B); free(h_C);

    return 0;
}