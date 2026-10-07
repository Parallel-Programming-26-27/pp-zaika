#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double pi_serial(long long n)
{
    double sum = 0.0;
    double sign = 1.0;
    for (long long i = 0; i < n; ++i) {
        sum += sign / (double)(2LL * i + 1LL);
        sign = -sign;
    }
    return 4.0 * sum;
}

static double pi_reduction(long long n, int threads)
{
    double sum = 0.0;
    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; ++i) {
        double term = ((i & 1LL) ? -1.0 : 1.0) / (double)(2LL * i + 1LL);
        sum += term;
    }
    return 4.0 * sum;
}

static double pi_atomic(long long n, int threads)
{
    double sum = 0.0;
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        double term = ((i & 1LL) ? -1.0 : 1.0) / (double)(2LL * i + 1LL);
        #pragma omp atomic
        sum += term;
    }
    return 4.0 * sum;
}

static double pi_critical(long long n, int threads)
{
    double sum = 0.0;
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        double term = ((i & 1LL) ? -1.0 : 1.0) / (double)(2LL * i + 1LL);
        #pragma omp critical
        {
            sum += term;
        }
    }
    return 4.0 * sum;
}

static void print_result(const char *name, double pi, double t, int threads)
{
    double err = fabs(pi - M_PI);
    printf("%-12s | pi = %.15f | err = %.3e | time = %.6f s | threads = %d\n",
           name, pi, err, t, threads);
}

int main(int argc, char **argv)
{
    long long n = 1000000000LL;
    int threads = 8;

    if (argc > 1) n = atoll(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    if (n <= 0) { fprintf(stderr, "n must be > 0\n"); return 1; }
    if (threads <= 0) { fprintf(stderr, "threads must be > 0\n"); return 1; }

    printf("n = %lld, threads = %d\n", n, threads);

    double t0 = omp_get_wtime();
    double pi_s = pi_serial(n);
    double t_s = omp_get_wtime() - t0;
    print_result("serial", pi_s, t_s, 1);

    t0 = omp_get_wtime();
    double pi_r = pi_reduction(n, threads);
    double t_r = omp_get_wtime() - t0;
    print_result("reduction", pi_r, t_r, threads);

    t0 = omp_get_wtime();
    double pi_a = pi_atomic(n, threads);
    double t_a = omp_get_wtime() - t0;
    print_result("atomic", pi_a, t_a, threads);

    t0 = omp_get_wtime();
    double pi_c = pi_critical(n, threads);
    double t_c = omp_get_wtime() - t0;
    print_result("critical", pi_c, t_c, threads);

    printfn");
    if (t_r > 0.0) printf("Speedup (reduction vs serial): %.3f\n", t_s / t_r);
    if (t_a > 0.0) printf("Speedup (atomic    vs serial): %.3f\n", t_s / t_a);
    if (t_c > 0.0) printf("Speedup (critical  vs serial): %.3f\n", t_s / t_c);
    printf("Efficiency (reduction, %d threads): %.3f\n",
           threads, t_r > 0.0 ? (t_s / t_r) / threads : 0.0);

    return 0;
}