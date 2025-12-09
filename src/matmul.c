#define n 2000

// guard
#ifndef ENABLE_TIMING
   #define ENABLE_TIMING 1
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
   /* Allocate dense square matrices (row-major). */
   double (*a)[n] = malloc(sizeof(double[n][n]));
   double (*b)[n] = malloc(sizeof(double[n][n]));
   double (*c)[n] = malloc(sizeof(double[n][n]));

   /* Initialize A and B with constants; zero C. */
   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
         a[i][j] = 2.0;
         b[i][j] = 3.0;
         c[i][j] = 0.0;
      }

   /* Timing the start */
   clock_t time;
   if (ENABLE_TIMING==1) {
      time = clock();
   }

   /* Naive matrix multiplication: C = A * B. */
   for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; k++)
         for (int j = 0; j < n; ++j)
            c[i][j] += a[i][k] * b[k][j];

   /* Timing the end and reporting */
   if (ENABLE_TIMING==1) {
      time = clock() - time;
      double time_taken = ((double)time)/CLOCKS_PER_SEC;
      fprintf(stderr, "[seq] n=%d elapsed=%.3f s\n", n, time_taken);
   }

   /* Dump a 1000x1000 top-left block to file for inspection. */
   FILE *f = fopen("mat-res.txt", "w");
   if (!f) {
      perror("fopen");
      return 1;
   }

   fprintf(f, "%d\n\n", n);
   for (int i = 0; i < 1000; i++) {
      for (int j = 0; j < 1000; j++) {
         fprintf(f, "%.0f ", c[i][j]);
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
