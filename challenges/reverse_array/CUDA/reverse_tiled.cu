#include <cuda_runtime.h>

#define THREAD_COLS 32
#define THREAD_ROWS 8
const int dataPerBlock = 256;

__global__ void reverse_array(float* input, int N_half ,int N) 
{
    //Locate data block
    int data_blk = dataPerBlock * blockIdx.x;
    //Locate thread within data block
    int thread_idx = data_blk + (THREAD_COLS * threadIdx.y + threadIdx.x);
    __shared__ float tile[THREAD_ROWS * 2][THREAD_COLS];

    //Load the 32x16 tile. The upper half of tile contains elements from left half and lower half contains elements from right half
    // Both these loads are coalesced - 4 contiguous HBM sectors from each half loaded back to back.
    if(thread_idx < N_half)
    {
        tile[threadIdx.y][threadIdx.x] = input[thread_idx];
        tile[threadIdx.y + THREAD_ROWS][threadIdx.x] = input[N - thread_idx - 1];
    }
    __syncthreads();

    //Now perform a row by row read of shared memory and write the reversed elements at the swapped positions. Once again, this is coalesced. 4 contiguous HBM sectors written to each
    //half back to back.
    if(thread_idx < N_half)
    {   
        input[thread_idx] = tile[threadIdx.y + THREAD_ROWS][threadIdx.x]; //Lower half
        input[N - thread_idx - 1] = tile[threadIdx.y][threadIdx.x];   //Upper half 
    }
}
// input is device pointer
extern "C" void solve(float* input, int N) {
    int N_half = (N - (N&1))>>1;
    const int blocks = (N_half + dataPerBlock - 1)/dataPerBlock;
    dim3 threads(THREAD_COLS , THREAD_ROWS); 
    reverse_array<<<blocks , threads>>>(input, N_half ,N);
    //cudaDeviceSynchronize();
}
