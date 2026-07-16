/* Hybrid MPI + CUDA smoke test — proves the full data path a hybrid job exists
 * to exercise: a VM CPU rank generates data, hands it to the host GPU rank over
 * MPI (the cluster0 TCP fabric), and the host copies it to the real GPU, runs
 * y = a*x + b in a CUDA kernel, copies it back, and validates.
 *
 * Topology (matches job.hybrid.cuda.example.yaml, node_count: 2; generalizes to
 * more VM ranks):
 *   rank 0   -> the host GPU node. Owns the GPU. Receives each worker's x,
 *               cudaMemcpy H2D, launches axpb, cudaMemcpy D2H, checks the result.
 *   rank > 0 -> VM CPU node(s). Generate their slice of x on the CPU and
 *               MPI_Send it to rank 0. They touch NO CUDA API whatsoever (they
 *               have no GPU and run without --nv), so the driver is never even
 *               dlopen'd there — only libcudart (present in the image) is linked.
 *
 * The single binary runs on every rank; behavior is chosen purely by rank, so
 * the GPU code path only ever executes where the GPU (and --nv) actually is.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

/* y = a*x + b, elementwise — the "Ax+B" the demo advertises. */
__global__ void axpb(float *y, const float *x, float a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}

/* Deterministic per-rank input so rank 0 can validate without a second channel:
 * worker `src` produces x[i] = (i % 100) + src. */
static float make_x(int src, int i) { return (float)(i % 100) + (float)src; }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hlen;
    MPI_Get_processor_name(host, &hlen);

    /* Ordered roll-call so every node is visible in the log. */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank) {
            printf("  rank %2d/%d on %s%s\n", rank, size, host,
                   rank == 0 ? "  [GPU host]" : "  [VM worker]");
            fflush(stdout);
        }
    }

    const int N = 1 << 20;               /* 1M floats = 4 MiB per worker */
    const float a = 2.0f, b = 3.0f;

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "need >= 2 ranks (1 GPU host + >=1 VM worker)\n");
        MPI_Finalize();
        return 2;
    }

    if (rank != 0) {
        /* VM worker: generate x on the CPU, ship it to the GPU host. No CUDA. */
        float *x = (float *)malloc(sizeof(float) * N);
        for (int i = 0; i < N; i++) x[i] = make_x(rank, i);
        MPI_Send(x, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        free(x);
    } else {
        /* GPU host: report the device, then process each worker's chunk. */
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, dev));
        printf("\n  GPU host device : %s (compute %d.%d, %.1f GB)\n",
               prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  kernel          : y = %.1f*x + %.1f over %d floats/worker\n",
               a, b, N);

        float *x = (float *)malloc(sizeof(float) * N);
        float *y = (float *)malloc(sizeof(float) * N);
        float *dx = NULL, *dy = NULL;
        CUDA_OK(cudaMalloc(&dx, sizeof(float) * N));
        CUDA_OK(cudaMalloc(&dy, sizeof(float) * N));

        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        double worst = 0.0;
        int workers = size - 1;

        double t0 = MPI_Wtime();
        for (int src = 1; src < size; src++) {
            MPI_Status st;
            MPI_Recv(x, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &st);
            CUDA_OK(cudaMemcpy(dx, x, sizeof(float) * N, cudaMemcpyHostToDevice));
            axpb<<<blocks, threads>>>(dy, dx, a, b, N);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaMemcpy(y, dy, sizeof(float) * N, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaDeviceSynchronize());
            for (int i = 0; i < N; i++) {
                float expect = a * make_x(src, i) + b;
                double err = fabs((double)y[i] - (double)expect);
                if (err > worst) worst = err;
            }
        }
        double t1 = MPI_Wtime();

        double moved_gib = (double)workers * N * sizeof(float) * 2 /* H2D+D2H */
                           / (1024.0 * 1024.0 * 1024.0);
        printf("\n  workers processed : %d\n", workers);
        printf("  bytes H2D+D2H     : %.2f GiB\n", moved_gib);
        printf("  wall time         : %.3f s\n", t1 - t0);
        printf("  max |error|       : %.3g  (%s)\n", worst,
               worst == 0.0 ? "EXACT" : (worst < 1e-3 ? "PASS" : "FAIL"));

        cudaFree(dx);
        cudaFree(dy);
        free(x);
        free(y);
    }

    MPI_Finalize();
    return 0;
}
