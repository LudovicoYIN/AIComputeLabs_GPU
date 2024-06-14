#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
/**
 * 总体思路：
 * 通过全局id将输入数据读取到shared_memory,再通过归并思想，每个block结果被加到0号线程位置，再写回d_out
 * @tparam blockSize
 * @param d_in
 * @param d_out
 * time latency: 1.14ms
 */
template<int blockSize>
__global__ void reduce_v0(const float *d_in, float *d_out) {
    __shared__ float shared_memory[blockSize];
    unsigned int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    shared_memory[tid] = d_in[global_tid];
    // 涉及到shared memory的读写最好都加上__sync threads
    __syncthreads();
    for (int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) {
            shared_memory[tid] += shared_memory[tid + index];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = shared_memory[0];
    }
}

bool CheckResult(const float *out, float ground_truth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != ground_truth) {
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
    const int BlockSize = 256;
    const int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    auto *a = (float *) malloc(N * sizeof (float));
    float *d_a;
    cudaMalloc((void **) &d_a, N * sizeof(float));

    auto *out = (float *) malloc(GridSize * sizeof (float));
    float *d_out;
    cudaMalloc((void **) &d_out, GridSize * sizeof (float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    int ground_truth = N * 1.0f;
    // 将初始化后的数据拷贝到GPU
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    // 定义分配的block数量和threads数量
    dim3 Grid(GridSize);
    dim3 Block(BlockSize);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<BlockSize><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millie_seconds, start, stop);
    // 将结果拷回CPU并check正确性
    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, ground_truth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("groudtruth is: %f \n", ground_truth);
    }
    printf("reduce_v0 latency = %f ms\n", millie_seconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}