#include <cuda_runtime.h>

/*
 Vector addition has 2 memory loads(A[idx] and B[idx]) , 1 floating point addition , and 1 memory store(C[idx]) so 1 FLOP per 12 bytes moved, meaning that the task is memory bound.
 
 Therefore, improvement over the naive solution is done by using vectorization - loading in float4's (16 bytes) as opposed to scalar float (4 byte) loads
 Vectorized loads/stores move 4x more bytes with 4x fewer load/store instructions. Since the block is 1D and adjacent threads in the warp access adjacent float4s/floats, 
 the warp accesses are coalesced - each warp loads in 512 bytes which corresponds to 16 contiguous HBM sectors and so overall, the memory reaches closer to its peak bandwidth.
 The naive scalar addition also has coalesced access, but each warp only moves 128 bytes which is 4 contiguous HBM sectors and so the bandwidth utilization is 4x less which is why
 vectorization helps here in order to increase memory bandwidth utilization and push the workload towards being more compute bound.
 
 When N is not divisible by 4, the remainder floats are treated as standard scalars.
 */

__global__ void vector_add(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N_4packed ,int N) 
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
    const int N_4packed = N >> 2;
    const int remaining_scalars = N & 3;
    const int thread_count = N_4packed + remaining_scalars;
    const int N_BLOCKS = (thread_count + N_THREADS - 1) / N_THREADS;

    vector_add<<<N_BLOCKS, N_THREADS>>>((const float4*)A, (const float4*)B, (float4*)C, N_4packed ,N);
    //cudaDeviceSynchronize();
}

