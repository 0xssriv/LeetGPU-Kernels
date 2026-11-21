#include <cuda_runtime.h>

__global__ void relu_kernel(const float4* __restrict__ input, float4* __restrict__ output, int N_packed ,int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /*Vectorized loads - Loads a vector of 4 floats instead of separate scalar loads*/
    if(idx < N_packed)
    {
        float4 pkd_in = input[idx];
        pkd_in.x = fmaxf(pkd_in.x , 0.f);
        pkd_in.y = fmaxf(pkd_in.y , 0.f);
        pkd_in.z = fmaxf(pkd_in.z , 0.f);
        pkd_in.w = fmaxf(pkd_in.w , 0.f);
        output[idx] = pkd_in;
        return; 
    }
    //Handle scalars
    int scalar_offset = ((N_packed << 2) - N_packed) + idx;
    if(scalar_offset < N)
    {
        const float* scalar_in = (const float*)input;
        float* scalar_out = (float*)output;
        scalar_out[scalar_offset] = fmaxf(scalar_in[scalar_offset], 0.f);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    
    const int N_THREADS = 256;
    const int N_packed = N >> 2;
    const int leftover_scalar_floats = N & 3;
    const int thread_count = N_packed + leftover_scalar_floats;

    const int N_BLOCKS = (thread_count + N_THREADS - 1) / N_THREADS;


    relu_kernel<<<N_BLOCKS , N_THREADS>>>((const float4*)input, (float4*)output, N_packed ,N);
    //cudaDeviceSynchronize();
}

