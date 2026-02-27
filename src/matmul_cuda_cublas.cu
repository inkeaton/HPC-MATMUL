// nvcc -O3 -DTIME -DALIGN -DRESTRICT -DN=10000 src/matmul_cublas.cu -o bin/cuda_cublas -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* 1. CONFIGURATION FLAGS */
#ifndef N
    #define N 5000 
#endif

#ifdef RESTRICT
    #define PTR_RESTRICT __restrict__
#else
    #define PTR_RESTRICT
#endif

// CUDA Error Checker
#define cudaCheck(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// cuBLAS Error Checker
#define cublasCheck(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d (line %d)\n", stat, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    size_t bytes = N * N * sizeof(double);

    /* 2. TIMING SETUP (Using accurate CUDA Events) */
    #ifdef TIME
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
    #endif

    /* 3. HOST POINTERS & ALLOCATION */
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

    /* 4. INITIALIZATION (On Host) */
    for (int i = 0; i < N * N; i++) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = ((double)rand() / RAND_MAX) * 10.0;
        h_c[i] = 0.0; // Automatically overwritten by cuBLAS since beta = 0.0
    }

    /* 5. DEVICE POINTERS & ALLOCATION */
    double *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_b, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));

    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* 6. CUBLAS INITIALIZATION */
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    double alpha = 1.0;
    double beta = 0.0;

    /* 7. START TIMER */
    #ifdef TIME
        cudaCheck(cudaDeviceSynchronize()); 
        cudaCheck(cudaEventRecord(start));
    #endif

    /* 8. LAUNCH CUBLAS DGEMM */
    // Note: To handle C's Row-Major layout in Column-Major cuBLAS, 
    // we compute C = B * A instead of C = A * B.
    cublasCheck(cublasDgemm(handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N, 
                            N, N, N, 
                            &alpha, 
                            d_b, N,  // Matrix B passed first
                            d_a, N,  // Matrix A passed second
                            &beta, 
                            d_c, N));

    /* 9. STOP TIMER & REPORT */
    #ifdef TIME
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        // Formatted for seamless benchmark.sh extraction
        printf("[cublas-dgemm] N=%d | elapsed=%.3f s\n", 
                N, seconds);
            
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    #else
        cudaCheck(cudaDeviceSynchronize());
    #endif

    /* 10. CLEANUP */
    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    cublasCheck(cublasDestroy(handle));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}