#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
/**
 * 总体思路：和v3思想一致，就是最后一个warp的线程单独拎出来计算，减少了一次__syncthreads，这个操作较为耗时
 * @tparam blockSize                                                                                                                    
 * @param d_in
 * @param d_out
 * time latency: 0.33ms
 */

//v4: 最后一个warp不用参与__syncthreads
//latency: 0.694ms
__device__ void WarpSharedMemReduce(volatile float* smem, int tid){
    // CUDA不保证所有的shared memory读操作都能在写操作之前完成，因此存在竞争关系，可能导致结果错误
    // 比如smem[tid] += smem[tid + 16] => smem[0] += smem[16], smem[16] += smem[32]
    // 此时L9中smem[16]的读和写到底谁在前谁在后，这是不确定的，所以在Volta架构后最后加入中间寄存器(L11)配合syncwarp和volatile(使得不会看见其他线程更新smem上的结果)保证读写依赖
    float x = smem[tid];
    if (blockDim.x >= 64) {
        x += smem[tid + 32]; __syncwarp();
        smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}

template<int blockSize>
__global__ void reduce_v4(float *d_in, float *d_out){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
    __syncthreads();

    // 基于v3改进：把最后一个warp抽离出来reduce，避免多做一次sync threads
    // 此时一个block对d_in这块数据的reduce sum结果保存在id为0的线程上面
    for (unsigned int index = blockDim.x / 2; index > 32; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // last warp拎出来单独做reduce
    if (tid < 32) {
        WarpSharedMemReduce(smem, tid);
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
    dim3 Block(BlockSize / 2);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<BlockSize / 2><<<Grid,Block>>>(d_a, d_out);
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