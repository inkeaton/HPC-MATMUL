#define n 5000

// guard
#ifndef ENABLE_TIMING
#define ENABLE_TIMING 1
#endif

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    /* Allocate dense square matrices (row-major). */
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    /* Initialize A and B with constants; zero C. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    double t0 = 0.0;
    if (ENABLE_TIMING == 1)
    {
        /* omp_get_wtime is portable and thread-safe. */
        t0 = omp_get_wtime();
    }

    /* OpenMP parallelization will be added later; currently the naive triple loop. */
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; ++j)
                c[i][j] += a[i][k] * b[k][j];

    if (ENABLE_TIMING == 1)
    {
        double t1 = omp_get_wtime();
        fprintf(stderr, "[openmp] n=%d elapsed=%.3f s\n", n, t1 - t0);
    }

    /* Dump a 1000x1000 top-left block to file for inspection. */
    FILE *f = fopen("mat-res.txt", "w");
    if (!f)
    {
        perror("fopen");
        return 1;
    }

    fprintf(f, "%d\n\n", n);
    for (int ii = 0; ii < 1000; ii++)
    {
        for (int jj = 0; jj < 1000; jj++)
        {
            fprintf(f, "%.0f ", c[ii][jj]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    /* Free resources before exit. */
    free(a);
    free(b);
    free(c);
    return 0;
}
