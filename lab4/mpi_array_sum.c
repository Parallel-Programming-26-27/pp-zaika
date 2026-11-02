#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 0;

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "usage: %s N\n", argv[0]);
            N = -1;
        } else {
            N = atoll(argv[1]);
        }
        if (N <= 0 || N % size != 0) {
            fprintf(stderr, "Error: N must be > 0 and divisible by P=%d\n", size);
            N = -1;
        }
    }

    // все процессы узнают N (или -1 при ошибке)
    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (N <= 0) {
        MPI_Finalize();
        return 1;
    }

    long long local_n = N / size;

    long long *array = NULL;
    if (rank == 0) {
        array = (long long *)malloc(N * sizeof(long long));
        for (long long i = 0; i < N; i++) {
            array[i] = i + 1;
        }
    }

    long long *chunk = (long long *)malloc(local_n * sizeof(long long));
    MPI_Scatter(array, (int)local_n, MPI_LONG_LONG_INT,
                chunk, (int)local_n, MPI_LONG_LONG_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (long long i = 0; i < local_n; i++) {
        local_sum += chunk[i];
    }

    // --- Gather: корень собирает все локальные суммы и складывает вручную ---
    long long *all_sums = NULL;
    if (rank == 0) {
        all_sums = (long long *)malloc(size * sizeof(long long));
    }
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT,
               all_sums, 1, MPI_LONG_LONG_INT,
               0, MPI_COMM_WORLD);

    long long gather_total = 0;
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            gather_total += all_sums[r];
        }
    }

    // --- Reduce: корень получает свёрнутую сумму ---
    long long reduce_total = 0;
    MPI_Reduce(&local_sum, &reduce_total, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    // --- Allreduce: сумму получают все ---
    long long allreduce_total = 0;
    MPI_Allreduce(&local_sum, &allreduce_total, 1, MPI_LONG_LONG_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        long long expected = N * (N + 1) / 2;
        printf("N = %lld, P = %d, expected = %lld\n", N, size, expected);
        printf("Gather   result: %lld  [%s]\n", gather_total,
               gather_total == expected ? "OK" : "FAIL");
        printf("Reduce   result: %lld  [%s]\n", reduce_total,
               reduce_total == expected ? "OK" : "FAIL");
        printf("Allreduce result: %lld  [%s]\n", allreduce_total,
               allreduce_total == expected ? "OK" : "FAIL");
        free(all_sums);
        free(array);
    } else {
        // на не-корневых процессах проверяем, что Allreduce дал тот же итог
        long long expected = N * (N + 1) / 2;
        printf("rank %d: Allreduce result: %lld  [%s]\n", rank, allreduce_total,
               allreduce_total == expected ? "OK" : "FAIL");
    }

    free(chunk);
    MPI_Finalize();
    return 0;
}