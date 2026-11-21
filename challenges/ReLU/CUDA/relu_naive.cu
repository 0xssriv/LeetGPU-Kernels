#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /*Better solution*/
    if(idx < N)
    {
        float x = input[idx];
        unsigned int t1 = *((unsigned int*)(&x));
        unsigned int mask = ~(-(t1 >> 31));
        t1 = t1 & mask;
        output[idx] = *((float*)(&t1));
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

