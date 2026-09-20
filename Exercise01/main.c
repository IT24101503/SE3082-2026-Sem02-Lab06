#include <omp.h>
#include <stdio.h>

int main(void) {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("thread # %d\n", thread_id);

#pragma omp barrier
        printf("Outside Barrier # %d\n", omp_get_thread_num());
        printf("Hi again from thread # %d\n", omp_get_thread_num());
    }

    return 0;
}
