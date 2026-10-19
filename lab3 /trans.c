

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void transpose_by_rows(const float *src, float *dst, int n)
{
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            dst[col * n + row] = src[row * n + col];
        }
    }
}

static void transpose_by_columns(const float *src, float *dst, int n)
{
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            dst[col * n + row] = src[row * n + col];
        }
    }
}

static void transpose_tiles(const float *src, float *dst, int n, int block_size)
{
    for (int row0 = 0; row0 < n; row0 += block_size) {
        int row_end = row0 + block_size;
        if (row_end > n) row_end = n;

        for (int col0 = 0; col0 < n; col0 += block_size) {
            int col_end = col0 + block_size;
            if (col_end > n) col_end = n;

            for (int row = row0; row < row_end; ++row) {
                for (int col = col0; col < col_end; ++col) {
                    dst[col * n + row] = src[row * n + col];
                }
            }
        }
    }
}

static void transpose_rows_parallel(const float *src, float *dst,
                                    int n, int thread_count)
{
    omp_set_num_threads(thread_count);

    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            dst[col * n + row] = src[row * n + col];
        }
    }
}

static void transpose_tiles_parallel(const float *src, float *dst,
                                     int n, int block_size, int thread_count)
{
    omp_set_num_threads(thread_count);

    #pragma omp parallel for
    for (int row0 = 0; row0 < n; row0 += block_size) {
        int row_end = row0 + block_size;
        if (row_end > n) row_end = n;

        for (int col0 = 0; col0 < n; col0 += block_size) {
            int col_end = col0 + block_size;
            if (col_end > n) col_end = n;

            for (int row = row0; row < row_end; ++row) {
                for (int col = col0; col < col_end; ++col) {
                    dst[col * n + row] = src[row * n + col];
                }
            }
        }
    }
}

static void transpose_in_place(float *matrix, int n)
{
    for (int row = 0; row < n; ++row) {
        for (int col = row + 1; col < n; ++col) {
            float value = matrix[row * n + col];
            matrix[row * n + col] = matrix[col * n + row];
            matrix[col * n + row] = value;
        }
    }
}

static void transpose_in_place_parallel(float *matrix, int n, int thread_count)
{
    omp_set_num_threads(thread_count);

    #pragma omp parallel for
    for (int row = 0; row < n; ++row) {
        for (int col = row + 1; col < n; ++col) {
            float value = matrix[row * n + col];
            matrix[row * n + col] = matrix[col * n + row];
            matrix[col * n + row] = value;
        }
    }
}

static double median_of_five(double values[5])
{
    for (int pass = 0; pass < 4; ++pass) {
        for (int pos = 0; pos < 4 - pass; ++pos) {
            if (values[pos] > values[pos + 1]) {
                double temp = values[pos];
                values[pos] = values[pos + 1];
                values[pos + 1] = temp;
            }
        }
    }
    return values[2];
}

int main(int argc, char **argv)
{
    int n = 5000;
    int block_size = 32;
    int thread_count = 8;
    char method[32] = "rows";

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) block_size = atoi(argv[2]);
    if (argc > 3) thread_count = atoi(argv[3]);
    if (argc > 4) {
        strncpy(method, argv[4], sizeof(method) - 1);
        method[sizeof(method) - 1] = '\0';
    }

    if (n <= 0 || block_size <= 0 || thread_count <= 0) {
        fprintf(stderr, "N, block size and thread count must be positive\n");
        return 1;
    }

    long long element_count = (long long)n * n;
    size_t bytes = (size_t)element_count * sizeof(float);

    float *matrix = malloc(bytes);
    float *reference = malloc(bytes);
    float *output = malloc(bytes);
    float *working_copy = malloc(bytes);

    if (matrix == NULL || reference == NULL || output == NULL || working_copy == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(matrix);
        free(reference);
        free(output);
        free(working_copy);
        return 1;
    }

    for (long long index = 0; index < element_count; ++index) {
        matrix[index] = (float)index;
    }

    transpose_by_rows(matrix, reference, n);

    int in_place = (strcmp(method, "inplace") == 0 ||
                    strcmp(method, "inplace_omp") == 0);
    double samples[5];

    /* Первый запуск прогревает программу; затем сохраняются пять замеров. */
    for (int attempt = 0; attempt < 6; ++attempt) {
        double start, finish;

        if (in_place) {
            memcpy(working_copy, matrix, bytes);
            start = omp_get_wtime();

            if (strcmp(method, "inplace") == 0) {
                transpose_in_place(working_copy, n);
            } else {
                transpose_in_place_parallel(working_copy, n, thread_count);
            }

            finish = omp_get_wtime();
        } else {
            start = omp_get_wtime();

            if (strcmp(method, "rows") == 0) {
                transpose_by_rows(matrix, output, n);
            } else if (strcmp(method, "cols") == 0) {
                transpose_by_columns(matrix, output, n);
            } else if (strcmp(method, "blocked") == 0) {
                transpose_tiles(matrix, output, n, block_size);
            } else if (strcmp(method, "rows_omp") == 0) {
                transpose_rows_parallel(matrix, output, n, thread_count);
            } else if (strcmp(method, "blocked_omp") == 0) {
                transpose_tiles_parallel(matrix, output, n, block_size, thread_count);
            } else {
                fprintf(stderr, "Unknown method: %s\n", method);
                free(matrix);
                free(reference);
                free(output);
                free(working_copy);
                return 1;
            }

            finish = omp_get_wtime();
        }

        if (attempt > 0) {
            samples[attempt - 1] = finish - start;
        }
    }

    double median_time = median_of_five(samples);
    float *actual = in_place ? working_copy : output;
    long long errors = 0;

    for (long long index = 0; index < element_count; ++index) {
        if (actual[index] != reference[index]) {
            ++errors;
        }
    }

    double bandwidth = (2.0 * (double)element_count * sizeof(float)) /
                       median_time / 1e9;

    printf("N=%d block=%d threads=%d method=%s time=%.4f s "
           "BW=%.2f GB/s check=%s\n",
           n, block_size, thread_count, method, median_time, bandwidth,
           errors == 0 ? "OK" : "FAIL");

    free(matrix);
    free(reference);
    free(output);
    free(working_copy);
    return errors == 0 ? 0 : 2;
}
