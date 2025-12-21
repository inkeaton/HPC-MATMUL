/*
 * OPENCL MATRIX MULTIPLICATION (FIXED)
 * TILE_WIDTH reduced to 16 to fix Error -54 (Register Pressure)
 * Compile: gcc -O3 matmul_opencl.c -o matmul_opencl -lOpenCL
 */

#define N 5000
// CHANGE: Reduced from 32 to 16 to fit in register budget
#define TILE_WIDTH 16 

#ifndef ENABLE_TIMING
   #define ENABLE_TIMING 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

void check_err(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error %d: %s\n", err, msg);
        exit(1);
    }
}

const char *kernel_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                 \n"
// CHANGE: Hardcoded macro inside string must match host
"#define TILE_WIDTH 16                                                         \n"
"                                                                              \n"
"__kernel void matmul_tiled(const int width,                                   \n"
"                           __global const double* a,                          \n"
"                           __global const double* b,                          \n"
"                           __global double* c) {                              \n"
"    __local double As[TILE_WIDTH][TILE_WIDTH];                                \n"
"    __local double Bs[TILE_WIDTH][TILE_WIDTH];                                \n"
"                                                                              \n"
"    int bx = get_group_id(0);    int by = get_group_id(1);                    \n"
"    int tx = get_local_id(0);    int ty = get_local_id(1);                    \n"
"                                                                              \n"
"    int row = by * TILE_WIDTH + ty;                                           \n"
"    int col = bx * TILE_WIDTH + tx;                                           \n"
"                                                                              \n"
"    double val = 0.0;                                                         \n"
"                                                                              \n"
"    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {         \n"
"        if (row < width && (m * TILE_WIDTH + tx) < width)                     \n"
"            As[ty][tx] = a[row * width + (m * TILE_WIDTH + tx)];              \n"
"        else                                                                  \n"
"            As[ty][tx] = 0.0;                                                 \n"
"                                                                              \n"
"        if (col < width && (m * TILE_WIDTH + ty) < width)                     \n"
"            Bs[ty][tx] = b[(m * TILE_WIDTH + ty) * width + col];              \n"
"        else                                                                  \n"
"            Bs[ty][tx] = 0.0;                                                 \n"
"                                                                              \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                         \n"
"                                                                              \n"
"        for (int k = 0; k < TILE_WIDTH; ++k) {                                \n"
"            val += As[ty][k] * Bs[k][tx];                                     \n"
"        }                                                                     \n"
"        barrier(CLK_LOCAL_MEM_FENCE);                                         \n"
"    }                                                                         \n"
"                                                                              \n"
"    if (row < width && col < width) {                                         \n"
"        c[row * width + col] = val;                                           \n"
"    }                                                                         \n"
"}                                                                             \n";

int main() {
    cl_int err;
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

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_err(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_err(err, "clCreateCommandQueue");

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, &err);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build Log:\n%s\n", buffer);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "matmul_tiled", &err);
    check_err(err, "clCreateKernel");

    int width = N;
    clSetKernelArg(kernel, 0, sizeof(int), &width);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);

    size_t local_size[2] = {TILE_WIDTH, TILE_WIDTH};
    size_t global_size[2];
    global_size[0] = ((N + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    global_size[1] = ((N + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
    check_err(err, "clEnqueueNDRangeKernel");

    clWaitForEvents(1, &event);

    if (ENABLE_TIMING == 1) {
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        double ms = (double)(end - start) / 1000000.0;
        fprintf(stderr, "[opencl] n=%d elapsed=%.3f ms\n", N, ms);
    }

    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    FILE *f = fopen("mat-res.txt", "w");
    if (f) {
        fprintf(f, "%d\n\n", N);
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) fprintf(f, "%.0f ", h_c[i * N + j]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    clReleaseMemObject(d_a); clReleaseMemObject(d_b); clReleaseMemObject(d_c);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_a); free(h_b); free(h_c);
    return 0;
}