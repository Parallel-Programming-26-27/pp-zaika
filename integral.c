#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double f(double x)
{
    return exp(-x * x);
}

#define A_DEFAULT 0.0
#define B_DEFAULT 1.0

static double integrate_serial(double a, double b, long long n)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }
    return h * sum;
}

static double integrate_parallel(double a, double b, long long n, int threads)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;
    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; ++i) {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }
    return h * sum;
}

static void integrate_eps(double a, double b, double eps, int threads)
{
    long long n = 1024;
    double prev = integrate_serial(a, b, n);
    double cur = prev;
    long long n_final = n;

    double t0 = omp_get_wtime();
    while (1) {
        n *= 2;
        cur = integrate_serial(a, b, n);
        if (fabs(cur - prev) < eps) { n_final = n; prev = cur; break; }
        prev = cur;
    }
    double t_s = omp_get_wtime() - t0;
    printf("[eps, serial]    n = %lld, I = %.15f, time = %.6f s\n", n_final, prev, t_s);

    t0 = omp_get_wtime();
    double par = integrate_parallel(a, b, n_final, threads);
    double t_p = omp_get_wtime() - t0;
    printf("[eps, parallel]  n = %lld, I = %.15f, time = %.6f s, threads = %d\n",
           n_final, par, t_p, threads);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && (argv[1][0] == 'e' || argv[1][0] == 'E')) {
        double eps = (argc >= 3) ? atof(argv[2]) : 1e-6;
        int threads = (argc >= 4) ? atoi(argv[3]) : 8;
        printf("=== Режим точности: eps = %.3e ===\n", eps);
        integrate_eps(A_DEFAULT, B_DEFAULT, eps, threads);
        return 0;
    }

    double a = A_DEFAULT;
    double b = B_DEFAULT;
    long long n = 100000000LL;
    int threads = 8;

    if (argc > 1) a = atof(argv[1]);
    if (argc > 2) b = atof(argv[2]);
    if (argc > 3) n = atoll(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);

    printf("f(x) = exp(-x^2), [a;b] = [%g; %g], n = %lld, threads = %d\n",
           a, b, n, threads);
    printfn");

    double t0 = omp_get_wtime();
    double I_s = integrate_serial(a, b, n);
    double t_s = omp_get_wtime() - t0;
    printf("serial    | I = %.15f | time = %.6f s\n", I_s, t_s);

    t0 = omp_get_wtime();
    double I_p = integrate_parallel(a, b, n, threads);
    double t_p = omp_get_wtime() - t0;
    printf("parallel  | I = %.15f | time = %.6f s | threads = %d\n",
           I_p, t_p, threads);

    printfn");
    if (t_p > 0.0) {
        printf("Speedup = %.3f\n", t_s / t_p);
        printf("Efficiency = %.3f\n", (t_s / t_p) / threads);
    }
    printf("|I_par - I_ser| = %.3e\n", fabs(I_p - I_s));

    return 0;
}