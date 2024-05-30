#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum(float *x)
{
    // 泛指当前block在所有block范围内的id
    unsigned int block_id = blockIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 泛指当前线程在其block内的id
    unsigned int local_tid = threadIdx.x;
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}

int main() {
    int N = 32;
    unsigned int nBytes = N * sizeof(float);
    float *dx, *hx;
    /* 分配GPU mem */
    cudaMalloc((void **)&dx, nBytes);
    /* 分配CPU mem */
    hx = (float*) malloc(nBytes);
    /* 初始化 host data */
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = (float)i;
        printf("%g\n", hx[i]);
    }
    /* copy data to GPU */
    cudaMemcpy(dx, hx, nBytes, cudaMemcpyHostToDevice);
    /* launch GPU kernel */
    sum<<<1, N>>>(dx);
    /* copy data from GPU */
    cudaMemcpy(hx, dx, nBytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for (int i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }
    cudaFree(dx);
    free(hx);
    return 0;
}