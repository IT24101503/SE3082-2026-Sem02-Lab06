#include <omp.h>
#include <stdio.h>

int main(void) {
#pragma omp parallel
    {
        printf("thread # %d\n", omp_get_thread_num());

#pragma omp master
        {
            printf("Inside Master # %d\n", omp_get_thread_num());
            printf("Exiting Master\n");
        }

        printf("Hi again from thread # %d\n", omp_get_thread_num());
    }

    return 0;
}
