#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>

__global__ void reduce_baseline(const int* input, int* output, size_t n) {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
        *output = sum;
    }
}

bool CheckResult(const int *out, int ground_truth, int n){
    if (*out != ground_truth) {
        return false;
    }
    return true;
}

int main() {
    float millie_seconds = 0;
    const int N = 32 * 1024 * 1024;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    const int BlockSize = 1;
    const int GridSize = 1;
    int *a = (int *) malloc(N * sizeof (int));
    int *d_a;
    cudaMalloc((void **) &d_a, N * sizeof(int));

    int *out = (int *) malloc(GridSize * sizeof (int));
    int *d_out;
    cudaMalloc((void **) &d_out, GridSize * sizeof (int));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }

    int ground_truth = N * 1;
    // 将初始化后的数据拷贝到GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    // 定义分配的block数量和threads数量
    dim3 Grid(GridSize);
    dim3 Block(BlockSize);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<1,1>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millie_seconds, start, stop);
    // 将结果拷回CPU并check正确性
    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, ground_truth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("res per block : %d ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %d \n", ground_truth);
    }
    printf("reduce_baseline latency = %f ms\n", millie_seconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}