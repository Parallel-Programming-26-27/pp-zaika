#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define RUNS 5
#define TOL 1e-9

void generate(double *X, double *y, int N, int d) {
    srand(42);
    for (int i = 0; i < N; i++) {
        y[i] = 0.5;
        for (int j = 0; j < d; j++) {
            X[i * d + j] = 2.0 * rand() / RAND_MAX - 1.0;
            if (j % 4 == 0) {
                y[i] += 2.0 * X[i * d + j];
            }
        }
        y[i] += 0.1 * (2.0 * rand() / RAND_MAX - 1.0);
    }
}

double soft(double z, double a) {
    if (z > a) return z - a;
    if (z < -a) return z + a;
    return 0.0;
}

double loss(double *X, double *y, int N, int d, double *w, double b, double lambda) {
    double sq = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = b;
        for (int j = 0; j < d; j++) {
            pred += X[i * d + j] * w[j];
        }
        sq += (pred - y[i]) * (pred - y[i]);
    }
    double l1 = 0.0;
    for (int j = 0; j < d; j++) {
        l1 += fabs(w[j]);
    }
    return sq / (2.0 * N) + lambda * l1;
}

double train_serial(double *X, double *y, int N, int d, int K, double lambda, double eta, double *w) {
    double b = 0.0;
    double *g = calloc(d, sizeof(double));
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        for (int i = 0; i < N; i++) {
            double pred = b;
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            double e = pred - y[i];
            for (int j = 0; j < d; j++) g[j] += e * X[i * d + j];
            gb += e;
        }
        for (int j = 0; j < d; j++) w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
        b -= eta * gb / N;
    }
    free(g);
    return b;
}

double train_omp(double *X, double *y, int N, int d, int K, double lambda, double eta, double *w) {
    double b = 0.0;
    double *g = calloc(d, sizeof(double));
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        #pragma omp parallel for reduction(+: g[:d], gb)
        for (int i = 0; i < N; i++) {
            double pred = b;
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            double e = pred - y[i];
            for (int j = 0; j < d; j++) g[j] += e * X[i * d + j];
            gb += e;
        }
        for (int j = 0; j < d; j++) w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
        b -= eta * gb / N;
    }
    free(g);
    return b;
}

double train_simd(double *X, double *y, int N, int d, int K, double lambda, double eta, double *w) {
    double b = 0.0;
    double *g = calloc(d, sizeof(double));
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        for (int i = 0; i < N; i++) {
            double pred = b;
            #pragma omp simd reduction(+: pred)
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++) g[j] += e * X[i * d + j];
            gb += e;
        }
        for (int j = 0; j < d; j++) w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
        b -= eta * gb / N;
    }
    free(g);
    return b;
}

double train_omp_simd(double *X, double *y, int N, int d, int K, double lambda, double eta, double *w) {
    double b = 0.0;
    double *g = calloc(d, sizeof(double));
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        #pragma omp parallel for reduction(+: g[:d], gb)
        for (int i = 0; i < N; i++) {
            double pred = b;
            #pragma omp simd reduction(+: pred)
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++) g[j] += e * X[i * d + j];
            gb += e;
        }
        for (int j = 0; j < d; j++) w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
        b -= eta * gb / N;
    }
    free(g);
    return b;
}

double train(int v, double *X, double *y, int N, int d, int K, double lambda, double eta, double *w) {
    if (v == 0) return train_serial(X, y, N, d, K, lambda, eta, w);
    if (v == 1) return train_omp(X, y, N, d, K, lambda, eta, w);
    if (v == 2) return train_simd(X, y, N, d, K, lambda, eta, w);
    return train_omp_simd(X, y, N, d, K, lambda, eta, w);
}

double median(double *t) {
    for (int i = 0; i < RUNS; i++) {
        for (int j = 0; j < RUNS - 1 - i; j++) {
            if (t[j] > t[j + 1]) {
                double tmp = t[j]; t[j] = t[j + 1]; t[j + 1] = tmp;
            }
        }
    }
    return t[RUNS / 2];
}

int main(int argc, char *argv[]) {
    int N = 100000, d = 32, K = 100, threads = omp_get_max_threads();
    double lambda = 0.001, eta = 0.5;

    if (argc > 6) {
        N = atoi(argv[1]); d = atoi(argv[2]); K = atoi(argv[3]);
        lambda = atof(argv[4]); eta = atof(argv[5]); threads = atoi(argv[6]);
    }
    omp_set_num_threads(threads);

    double *X = malloc((long long)N * d * sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *w = malloc(d * sizeof(double));
    double *w_serial = malloc(d * sizeof(double));

    if (!X || !y || !w || !w_serial) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    generate(X, y, N, d);
    printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d runs=%d\n", N, d, K, lambda, eta, threads, RUNS);

    const char *names[] = {"serial", "omp", "simd", "omp_simd"};
    double F_serial = 0.0;

    for (int v = 0; v < 4; v++) {
        double t[RUNS];
        double b = 0.0;

        for (int r = 0; r < RUNS; r++) {
            for (int j = 0; j < d; j++) w[j] = 0.0;
            double start = omp_get_wtime();
            b = train(v, X, y, N, d, K, lambda, eta, w);
            t[r] = omp_get_wtime() - start;
        }

        double F = loss(X, y, N, d, w, b, lambda);
        if (v == 0) {
            F_serial = F;
            for (int j = 0; j < d; j++) w_serial[j] = w[j];
        }

        double dw = 0.0;
        int zeros = 0;
        for (int j = 0; j < d; j++) {
            if (fabs(w[j] - w_serial[j]) > dw) dw = fabs(w[j] - w_serial[j]);
            if (fabs(w[j]) < 1e-6) zeros++;
        }

        char *check = "OK";
        if (dw > TOL || fabs(F - F_serial) > TOL) check = "FAIL";

        printf("%-9s F=%.10f  time=%.4f s  dw=%.1e  zeros=%d  check=%s\n",
               names[v], F, median(t), dw, zeros, check);
    }

    free(X); free(y); free(w); free(w_serial);
    return 0;
}