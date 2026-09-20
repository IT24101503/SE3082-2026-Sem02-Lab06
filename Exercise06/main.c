#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000
#define STRIP_SIZE 16

int main() {
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    #pragma omp parallel for
    for (int base = 0; base < N; base += STRIP_SIZE) {
        int limit = base + STRIP_SIZE;
        if (limit > N) limit = N;

        #pragma omp simd
        for (int i = base; i < limit; i++) {
            C[i] = A[i] * B[i];
        }
    }

    printf("C[0] = %f, C[N-1] = %f\n", C[0], C[N-1]);

    free(A); free(B); free(C);
    return 0;
}