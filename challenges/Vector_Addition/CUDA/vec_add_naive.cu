#include <cuda_runtime.h>

__global__ void vector_add(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
    {
	    C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int N_THREADS = 256;
    const int N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

    vector_add<<<N_BLOCKS, N_THREADS>>>(A , B , C , N);
    //cudaDeviceSynchronize();
}

