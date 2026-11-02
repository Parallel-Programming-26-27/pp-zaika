#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int pid, total_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    int arr_size = -1;
    int num_runs = 5;
    int num_reps = 10000;

    // Чтение аргументов командной строки на корневом процессе
    if (pid == 0) {
        if (argc >= 2) arr_size = atoi(argv[1]);
        if (argc >= 3) num_runs = atoi(argv[2]);
        if (argc >= 4) num_reps = atoi(argv[3]);

        if (arr_size <= 0 || num_runs <= 0 || num_reps <= 0) {
            printf("Usage: mpiexec -n P %s N [RUNS] [REPS], N > 0\n", argv[0]);
        }
    }

    // Рассылка параметров всем процессам
    MPI_Bcast(&arr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_runs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_reps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (arr_size <= 0 || num_runs <= 0 || num_reps <= 0) {
        MPI_Finalize();
        return 1;
    }

    // Расчет смещений и количеств элементов для каждого процесса
    int base_chunk = arr_size / total_procs;
    int remainder = arr_size % total_procs;

    int my_len = base_chunk + (pid < remainder ? 1 : 0);
    int *elem_counts = (int*)malloc(total_procs * sizeof(int));
    int *elem_displs = (int*)malloc(total_procs * sizeof(int));

    for (int i = 0, offset = 0; i < total_procs; i++) {
        elem_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        elem_displs[i] = offset;
        offset += elem_counts[i];
    }

    // Выделение памяти
    int *full_array = NULL;
    if (pid == 0) {
        full_array = (int*)malloc((size_t)arr_size * sizeof(int));
    }

    int *local_chunk = (int*)malloc((size_t)my_len * sizeof(int));
    long long *gathered_sums = (pid == 0) ? (long long*)malloc(total_procs * sizeof(long long)) : NULL;

    if ((pid == 0 && full_array == NULL) || local_chunk == NULL) {
        fprintf(stderr, "[Process %d] Error: Memory allocation failed\n", pid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long expected_total = (long long)arr_size * (arr_size + 1) / 2;
    long long local_sum = 0;
    long long total_gather = 0;
    long long total_reduce = 0;
    long long total_allreduce = 0;
    double cumulative_time = 0.0;

    // Основной цикл замеров (включая распределение и агрегацию)
    for (int run = 0; run < num_runs; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        if (pid == 0) {
            for (int i = 0; i < arr_size; i++) {
                full_array[i] = i + 1;
            }
        }

        // Распределение данных
        if (remainder == 0) {
            MPI_Scatter(full_array, my_len, MPI_INT, local_chunk, my_len, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(full_array, elem_counts, elem_displs, MPI_INT, local_chunk, my_len, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Локальная сумма
        local_sum = 0;
        for (int i = 0; i < my_len; i++) {
            local_sum += local_chunk[i];
        }

        // Сбор результатов тремя способами
        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, gathered_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (pid == 0) {
            total_gather = 0;
            for (int i = 0; i < total_procs; i++) {
                total_gather += gathered_sums[i];
            }
        }

        MPI_Reduce(&local_sum, &total_reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum, &total_allreduce, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

        // Замер времени (берем максимальное по всем процессам)
        double t_end = MPI_Wtime() - t_start;
        double t_max;
        MPI_Reduce(&t_end, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        cumulative_time += t_max;
    }

    printf("[Process %d] chunk: %d elements, local_sum = %lld, Allreduce = %lld\n",
           pid, my_len, local_sum, total_allreduce);

    // Проверка корректности
    int local_ok = (total_allreduce == expected_total);
    int global_ok;
    MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        int final_ok = (total_gather == expected_total) && (total_reduce == expected_total) && global_ok;
        printf("\nN = %d, P = %d%s\n", arr_size, total_procs, remainder ? " (N not divisible by P, used MPI_Scatterv)" : "");
        printf("Expected sum N(N+1)/2 = %lld\n", expected_total);
        printf("Gather    (manual sum on root): %lld\n", total_gather);
        printf("Reduce    (MPI_SUM, root only): %lld\n", total_reduce);
        printf("Allreduce (MPI_SUM, all procs): %lld\n", total_allreduce);
        printf("Correctness: %s\n", final_ok ? "YES" : "FAILED");
        printf("Average execution time (runs: %d): %.6f s\n", num_runs, cumulative_time / num_runs);
    }

    // Замер чистой агрегации (без Scatter)
    long long temp_g = 0, temp_r = 0;
    for (int i = 0; i < 100; i++) {
        MPI_Reduce(&local_sum, &temp_r, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_g_start = MPI_Wtime();
    for (int i = 0; i < num_reps; i++) {
        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, gathered_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (pid == 0) {
            temp_g = 0;
            for (int k = 0; k < total_procs; k++) {
                temp_g += gathered_sums[k];
            }
        }
    }
    double time_gather = (MPI_Wtime() - t_g_start) / num_reps;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_r_start = MPI_Wtime();
    for (int i = 0; i < num_reps; i++) {
        MPI_Reduce(&local_sum, &temp_r, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    double time_reduce = (MPI_Wtime() - t_r_start) / num_reps;

    if (pid == 0) {
        printf("\nAggregation benchmark (P = %d, reps: %d), time on root:\n", total_procs, num_reps);
        printf("Gather + manual sum: %.3e s, result is %s\n", time_gather, temp_g == expected_total ? "correct" : "INVALID");
        printf("Reduce:              %.3e s, result is %s\n", time_reduce, temp_r == expected_total ? "correct" : "INVALID");
    }

    free(elem_counts);
    free(elem_displs);
    free(local_chunk);
    free(full_array);
    free(gathered_sums);

    MPI_Finalize();
    return 0;
}