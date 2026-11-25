#include <cuda_runtime.h>

/* Matrix addition at its core is just vector addition.
   Matrices are stored row wise in memory so you can consider an NxN matrix
   as a flat N^2 element vector. So, an optimal vector addition kernel
   using vectorization solves this. 
 */

__global__ void matrix_add(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N_4packed ,int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N_4packed)
    {
        float4 op1 = A[idx];
        float4 op2 = B[idx];
        op1.x += op2.x;
        op1.y += op2.y;
        op1.z += op2.z;
        op1.w += op2.w;
        C[idx] = op1;
        return;
    }
    int scalar_offset = ((N_4packed << 2) - N_4packed) + idx;
    if(scalar_offset < N)
    {
        const float* a = (const float*)A; 
        const float* b = (const float*)B; 
        float* c = (float*)C;
        c[scalar_offset] = a[scalar_offset] + b[scalar_offset];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int N_THREADS = 256;
    const int N_linear = N*N;
    const int N_4packed = N_linear >> 2;
    const int remaining_scalars = N_linear & 3;
    const int thread_count = N_4packed + remaining_scalars;
    const int N_BLOCKS = (thread_count + N_THREADS - 1) / N_THREADS;

    matrix_add<<<N_BLOCKS, N_THREADS>>>((const float4*)A, (const float4*)B, (float4*)C, N_4packed ,N_linear);
    //cudaDeviceSynchronize();
}
