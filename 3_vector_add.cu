#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
typedef  float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(const FLOAT *x, const FLOAT *y, FLOAT *z, int N) {
    /* 2D grid */
    unsigned int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    if (idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(const FLOAT *x, const FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++) z[i] = y[i] + x[i];
}

int main() {
    int N = 10000;
    unsigned int nBytes = N * sizeof(FLOAT);
    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid(s, s);

    FLOAT *dx, *hx;
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nBytes);
    cudaMalloc((void **)&dy, nBytes);
    cudaMalloc((void **)&dz, nBytes);

    /* init time */
    float milliseconds = 0;

    /* allocate CPU mem */
    hx = (FLOAT *) malloc(nBytes);
    hy = (FLOAT *) malloc(nBytes);
    hz = (FLOAT *) malloc(nBytes);

    /* init */
    for (int i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* launch GPU kernel */
    vec_add<<<grid, bs>>>(dx, dy, dz, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* copy GPU result to CPU */
    cudaMemcpy(hz, dz, nBytes, cudaMemcpyDeviceToHost);

    /* CPU compute */
    auto* hz_cpu_res = (FLOAT *) malloc(nBytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i) {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}