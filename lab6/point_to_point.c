#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MSG_TAG 100
#define ACK_TAG 200
#define BUF_SIZE 100

#define WARMUP_ITERS 10
#define BENCH_ITERS 1000

static char bench_buffer[1048576]; // 1 MB

// Функция пинг-понга для замера времени
void do_benchmark(int pid, int payload_size) {
    if (pid == 0) {
        MPI_Send(bench_buffer, payload_size, MPI_CHAR, 1, MSG_TAG, MPI_COMM_WORLD);
        MPI_Recv(bench_buffer, payload_size, MPI_CHAR, 1, ACK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (pid == 1) {
        MPI_Recv(bench_buffer, payload_size, MPI_CHAR, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(bench_buffer, payload_size, MPI_CHAR, 0, ACK_TAG, MPI_COMM_WORLD);
    }
}

// Замер задержек для разных размеров пакетов
void run_network_test(int pid) {
    int test_sizes[] = {1, 1024, 65536, 1048576};

    if (pid == 0) {
        printf("Размер (байт) | Итераций | Среднее время (сек)\n");
        printf("---------------------------------------------\n");
    }

    for (int i = 0; i < 4; i++) {
        int sz = test_sizes[i];

        // Прогрев
        for (int j = 0; j < WARMUP_ITERS; j++) {
            do_benchmark(pid, sz);
        }

        double t_start = MPI_Wtime();

        // Основной замер
        for (int j = 0; j < BENCH_ITERS; j++) {
            do_benchmark(pid, sz);
        }

        double t_total = MPI_Wtime() - t_start;

        if (pid == 0) {
            printf("%13d | %8d | %.3e\n", sz, BENCH_ITERS, t_total / BENCH_ITERS);
            fflush(stdout);
        }
    }
}

int main(int argc, char** argv) {
    int rank, comm_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Проверка минимального количества процессов
    if (comm_size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Ошибка: для работы программы нужно минимум 2 процесса.\n");
            fprintf(stderr, "Запуск: mpiexec -n 2 ./point_to_point\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Ветка замера производительности
    if (argc > 1 && strcmp(argv[1], "time") == 0) {
        run_network_test(rank);
        MPI_Finalize();
        return 0;
    }

    // Ветка обычного обмена сообщениями
    char tx_buf[BUF_SIZE];
    char rx_buf[BUF_SIZE];

    if (rank == 0) {
        // Корневой процесс отправляет сообщения
        for (int dest = 1; dest < comm_size; dest++) {
            snprintf(tx_buf, BUF_SIZE, "Hello, rank %d, from rank 0", dest);
            printf("[Процесс 0] -> [Процесс %d]: \"%s\"\n", dest, tx_buf);
            fflush(stdout);

            MPI_Send(tx_buf, strlen(tx_buf) + 1, MPI_CHAR, dest, MSG_TAG, MPI_COMM_WORLD);
        }

        // Корневой процесс принимает ответы
        int replies_count = 0;
        for (int src = 1; src < comm_size; src++) {
            MPI_Recv(rx_buf, BUF_SIZE, MPI_CHAR, src, ACK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[Процесс 0] <- [Процесс %d]: \"%s\"\n", src, rx_buf);
            fflush(stdout);
            replies_count++;
        }

        printf("[Процесс 0] Итого: отправлено %d, получено ответов %d\n", comm_size - 1, replies_count);
        fflush(stdout);

    } else {
        // Рабочие процессы принимают и отвечают
        MPI_Recv(rx_buf, BUF_SIZE, MPI_CHAR, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Процесс %d] <- [Процесс 0]: \"%s\"\n", rank, rx_buf);
        fflush(stdout);

        snprintf(tx_buf, BUF_SIZE, "Message received by rank %d", rank);
        printf("[Процесс %d] -> [Процесс 0]: \"%s\"\n", rank, tx_buf);
        fflush(stdout);

        MPI_Send(tx_buf, strlen(tx_buf) + 1, MPI_CHAR, 0, ACK_TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}