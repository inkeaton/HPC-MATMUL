/*
/   # COMPILE OPTIONS:
/   ## GCC:
/   	gcc -O3 -march=native -ffast-math -funroll-loops -std=c11 matmul_opt.c -o matmul_opt
/   ## ICX (Intel oneAPI):
/   	icx -O3 -xHost -fp-model fast -std=c11 matmul_opt.c -o matmul_opt
*/

#define n 5000
#define BLOCK_SIZE 64 // 64 doubles * 8 bytes = 512 bytes stride. 64x64 block = 32KB (Fits in 48KB L1 Cache)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Helper min function for blocking logic
// TMP: maybe not using a function is more efficient
static inline int min(int a, int b)
{
	return (a < b) ? a : b;
}

int main(int argc, char **argv)
{
	/*
	 # OPTIMIZATION 1: Memory Alignment
	 * Align memory to 64 bytes (cache line size) for optimal vectorization.
	 * We use pointer-to-array syntax to keep a[i][j] notation valid.
	 */
	double (*restrict a)[n] = aligned_alloc(64, sizeof(double[n][n]));
	double (*restrict b)[n] = aligned_alloc(64, sizeof(double[n][n]));
	double (*restrict c)[n] = aligned_alloc(64, sizeof(double[n][n]));

	if (!a || !b || !c)
	{
		perror("Memory allocation failed");
		return 1;
	}

	/* Initialize A and B with constants; zero C. */
	// The compiler can easily vectorize this simple initialization
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j] = 2.0;
			b[i][j] = 3.0;
			c[i][j] = 0.0;
		}
	}

	/* Timing the start */
    #ifdef ENABLE_TIMING
        struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);
	#endif

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// HOTSPOT BEGIN
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	/*
	 # OPTIMIZATION 2: Cache Blocking (Tiling)
	 * Instead of iterating 0..N, we iterate block by block.
	 * This keeps active data inside the L1/L2 cache.
	 */
	for (int ii = 0; ii < n; ii += BLOCK_SIZE)
	{
		for (int kk = 0; kk < n; kk += BLOCK_SIZE)
		{
			for (int jj = 0; jj < n; jj += BLOCK_SIZE)
			{

				// Standard loop, but constrained within the current blocks
				// The compiler will unroll and vectorize the innermost 'j' loop
				for (int i = ii; i < min(ii + BLOCK_SIZE, n); i++)
				{
					for (int k = kk; k < min(kk + BLOCK_SIZE, n); k++)
					{

						// This loop should automatically use AVX2 ymm registers
						// #pragma omp simd (optional hint)
						for (int j = jj; j < min(jj + BLOCK_SIZE, n); j++)
						{
							c[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			}
		}
	}

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// HOTSPOT END
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	/* Timing the end and reporting */
    #ifdef ENABLE_TIMING
		clock_gettime(CLOCK_MONOTONIC, &end);
		double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
		fprintf(stderr, "[optimized-seq] n=%d elapsed=%.3f s\n", n, time_taken);
    #endif

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
			fprintf(f, "%.0f ", c[i][j]);
		}
		fprintf(f, "\n");
	}

	fclose(f);

	/* Free aligned resources. */
	free(a);
	free(b);
	free(c);
	return 0;
}