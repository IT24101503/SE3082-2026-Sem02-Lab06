#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define MAX_ITERATIONS 2000

int main(int argc, char **argv) {
    int width = argc > 1 ? atoi(argv[1]) : DEFAULT_WIDTH;
    int height = argc > 2 ? atoi(argv[2]) : DEFAULT_HEIGHT;
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "width and height must be positive\n");
        return EXIT_FAILURE;
    }

    long long inside = 0;
    double start = omp_get_wtime();
#pragma omp parallel for schedule(static) reduction(+:inside)
    for (int row = 0; row < height; ++row) {
        double imaginary = -1.25 + 2.5 * row / (height - 1.0);
        for (int column = 0; column < width; ++column) {
            double real = -2.0 + 3.0 * column / (width - 1.0);
            double z_real = 0.0;
            double z_imaginary = 0.0;
            int iteration;
            for (iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
                double next_real = z_real * z_real - z_imaginary * z_imaginary + real;
                double next_imaginary = 2.0 * z_real * z_imaginary + imaginary;
                z_real = next_real;
                z_imaginary = next_imaginary;
                if (z_real * z_real + z_imaginary * z_imaginary > 4.0) {
                    break;
                }
            }
            if (iteration == MAX_ITERATIONS) {
                ++inside;
            }
        }
    }
    double elapsed = omp_get_wtime() - start;

    double box_area = 3.0 * 2.5;
    double estimated_area = box_area * (double)inside / ((double)width * height);
    printf("grid = %d x %d, inside = %lld, area = %.10f\n",
           width, height, inside, estimated_area);
    printf("threads = %d, time = %.6f seconds\n", omp_get_max_threads(), elapsed);
    return EXIT_SUCCESS;
}
