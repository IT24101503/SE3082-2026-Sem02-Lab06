#include <omp.h>
#include <stdio.h>

int main(void) {
#pragma omp parallel
    {
        printf("thread # %d\n", omp_get_thread_num());

#pragma omp single
        {
            printf("Inside Single # %d\n", omp_get_thread_num());
            printf("Exiting Single\n");
        }

        printf("Hi again from thread # %d\n", omp_get_thread_num());
    }

    return 0;
}
