#include <cuda_runtime.h>

//Simple solution
__global__ void matrix_transpose_kernel(const float* __restrict__ input, float* __restrict__ output, int height, int width) 
{
    int x_src = blockDim.x * blockIdx.x + threadIdx.x;
    int y_src = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x_src < width && y_src < height)
    {
        output[x_src * height + y_src] = input[y_src * width + x_src];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    
    dim3 block(16,16); // 256 threads per block, logically arranged in a 16x16 square matrix
    dim3 grid((cols + block.x - 1)/block.x , (rows + block.y - 1)/block.y );
    matrix_transpose_kernel<<<grid , block>>>(input, output, rows, cols);
    //cudaDeviceSynchronize();
}

