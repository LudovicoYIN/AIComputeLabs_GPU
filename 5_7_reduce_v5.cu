#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
/**
 * 总体思路：和v4思想一致，就是把循环给打开了，然后，最后一个warp的线程单独拎出来计算。
 * @tparam blockSize                                                                                                                    
 * @param d_in
 * @param d_out
 * time latency: 0.32ms
 */

#define THREAD_PER_BLOCK 256
// v5：循环展开
template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {
    //对v4 L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            smem[threadIdx.x] += smem[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }
        __syncthreads();
    }
    // the final warp
    if (threadIdx.x < 32) {
        volatile float* vshm = smem;
        if (blockDim.x >= 64) {
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }
}

template <int blockSize>
__global__ void reduce_v5(float *d_in, float *d_out){
    __shared__ float smem[THREAD_PER_BLOCK];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id, *2代表当前block要处理2*blocksize的数据
    // ep. blocksize = 2, blockIdx.x = 1, when tid = 0, gtid = 4, gtid + blockSize = 6; when tid = 1, gtid = 5, gtid + blockSize = 7
    // ep. blocksize = 2, blockIdx.x = 0, when tid = 0, gtid = 0, gtid + blockSize = 2; when tid = 1, gtid = 1, gtid + blockSize = 3
    // so, we can understand L59, one thread handle data located in tid and tid + blockSize
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
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
    reduce_v5<BlockSize / 2><<<Grid,Block>>>(d_a, d_out);
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