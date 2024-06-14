#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
/**
 * 总体思路：和v1一致，只不过减少了bank conflict
 * 也就是说不是相邻两位去加了，而是隔了一半的距离
 * @tparam blockSize                                                                                                                    
 * @param d_in
 * @param d_out
 * time latency: 0.79ms
 */
template<int blockSize>
__global__ void reduce_v2(float *d_in, float *d_out){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // 基于v1作出改进: 从之前的当前线程ID加2*线程ID位置然后不断加上*2位置上的数据，改成不断地对半相加，以消除bank conflict
    // 此时一个block对d_in这块数据的reduce sum结果保存在id为0的线程上面
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: 哪里来回哪里去，把reduce结果写回显存
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
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
    reduce_v2<BlockSize><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v2 latency = %f ms\n", millie_seconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}