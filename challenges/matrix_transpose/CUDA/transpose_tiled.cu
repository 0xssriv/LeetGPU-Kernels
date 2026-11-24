#define TILE_DIM 32
#define THREAD_ROWS 8

#define FLAT_IDX_2D(y , x , cols) ((y) * (cols) + (x))

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{
    //Specific input data tile we are dealing with
    int tile_x_coord = blockIdx.x * TILE_DIM;  //tile x coordinate in input matrix plane
    int tile_y_coord = blockIdx.y * TILE_DIM;  // tile y coordinate in input matrix plane
    
    //Thread coordinates within the respective thread block pinpointed by (tile_x_coord, tile_y_coord).
    int thread_x_coord = tile_x_coord + threadIdx.x;  //thread x coordinate in input matrix plane
    int thread_y_coord = tile_y_coord + threadIdx.y;  //thread y coordinate in input matrix plane

    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    //A 32x8 block will load a 32x32 tile in memory, row wise. Thus the loads will be coalesced. each warp pulls in 128 Bytes or 4 contiguous HBM sectors
    // each iteration. 
    #pragma unroll
    for(int rowbase = 0; rowbase < TILE_DIM ; rowbase += THREAD_ROWS)
    {
	if(thread_x_coord < cols && (thread_y_coord + rowbase) < rows)
		tile[rowbase + threadIdx.y][threadIdx.x] = input[FLAT_IDX_2D(thread_y_coord + rowbase , thread_x_coord , cols)];
    }   
    __syncthreads(); 

   //After this barrier, the tile has fully loaded in the shared memory. Now we make the warp read the tile column wise
   //In the output buffer, the tile column will correspond to a row, because it is transposed and so our writes will be coalesced.
   //Reading tile column wise from shared memory, unlike global memory, has very less latency, but columnwise read cause bank conflicts which serialize the reads of column 
   //elements. To prevent that, we create the array to have TILE_DIM+1 columns so that the column elements are arranged jaggedly in shared memory
   //and columnwise reads hit different banks rather than one bank, which removes bank conflicts.

   /*
	Now we would like the warp to read out columns of the tile and then write it out to the destination, row wise, performing the transpose.
	Row wise write would mean that the writes are coalesced.
    */

   //Determine the output row index which this warp is targeting
   int dest_y = tile_x_coord + threadIdx.y; 
   
   //Determine specific data within that row which the particular warp thread must target 
   int dest_x = tile_y_coord + threadIdx.x;

   #pragma unroll
   for(int colbase = 0; colbase < TILE_DIM; colbase += THREAD_ROWS)
   {
	if((dest_y + colbase) < cols && dest_x < rows)
        {
		output[FLAT_IDX_2D(dest_y + colbase , dest_x , rows)] = tile[threadIdx.x][threadIdx.y + colbase];
   	}
   }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {

    //32x8 thread blocks(8 warps per block)
    dim3 threads_per_block(THREAD_COLS , THREAD_ROWS);
    //Blocks consists of the matrix split into 32x32 tiles
    dim3 blocks(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM

    );
    //Each 32x8 thread block processes a 32x32 tile of data.
    matrix_transpose_kernel<<<blocks , threads_per_block>>>(input, output, rows, cols);
    //cudaDeviceSynchronize();
}

