/* Minimal MPI demo — proves multi-node MPI over the cluster0 TCP fabric and prints
 * a rough performance number. Just a sanity/connectivity smoke test, not a
 * serious performance measurement.
 *
 * Each rank reports its host; then all ranks repeatedly MPI_Allreduce a buffer
 * and rank 0 reports the aggregate data volume, wall time, and effective rate.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256];
    int hlen; MPI_Get_processor_name(host, &hlen);

    /* Ordered roll-call so we can see every node participate. */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank) printf("  rank %2d/%d on %s\n", rank, size, host), fflush(stdout);
    }

    const int N = 1 << 20;        /* 1M doubles = 8 MiB per allreduce */
    const int ITERS = 200;
    double *sbuf = malloc(sizeof(double) * N);
    double *rbuf = malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) sbuf[i] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int it = 0; it < ITERS; it++)
        MPI_Allreduce(sbuf, rbuf, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        double secs = t1 - t0;
        double bytes = (double)N * sizeof(double) * ITERS;   /* moved per rank */
        double gib = bytes / (1024.0 * 1024.0 * 1024.0);
        printf("\n  allreduce: %d ranks, %d iters x 8 MiB\n", size, ITERS);
        printf("  wall time    : %.3f s\n", secs);
        printf("  per-iter     : %.3f ms\n", 1000.0 * secs / ITERS);
        printf("  rate/rank    : %.2f GiB/s\n", gib / secs);
        printf("  reduced sum  : %.0f (expect %d)\n", rbuf[0], size);
    }
    free(sbuf); free(rbuf);
    MPI_Finalize();
    return 0;
}
