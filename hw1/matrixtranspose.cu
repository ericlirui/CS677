#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1
#define WIDTH 32
#define HEIGHT 32

__global__ void BlockTranspose(double* A, int A_width, int A_height)
{
    __shared__ double blockA[BLOCK_WIDTH][BLOCK_WIDTH];
    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;
    blockA[threadIdx.y][threadIdx.x] = A[baseIdx];
    __syncthreads();
    A[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

int main( int argc, char* argv[] )
{

        // Size of vectors
        int max_width = 32;
        int n = max_width * max_width;

        // Host input vectors
        double *h_a;

        // Device input vectors
        double *d_a;

        // Size, in bytes, of each vector
        size_t bytes = n * sizeof(double);

        // Allocate memory for each vector on host
        h_a = (double*)malloc(bytes);

        // Allocate memory for each vector on GPU
        cudaMalloc(&d_a, bytes);

        int i;
        // Initialize vectors on host
        for( i = 0; i < n; i++ ) {
            h_a[i] = sin(i)*sin(i);
        }

        // Copy host vectors to device
        cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);

        for ( int i = 0; i < WIDTH; i++){
            for(int j = 0; j < HEIGHT; j++){
                printf("%f ", h_a[i* WIDTH + j]);
            }
            printf("\n");

        }

        printf("After\n");

dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);

dim3 gridDim(WIDTH/BLOCK_WIDTH, HEIGHT/BLOCK_WIDTH);

        // Execute the kernel
        BlockTranspose<<<gridDim, blockDim>>>(d_a, WIDTH, HEIGHT);
        cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost );

          for ( int i = 0; i < WIDTH; i++){
              for(int j = 0; j < HEIGHT; j++){
                  printf("%f ", h_a[i* WIDTH + j]);
              }
              printf("\n");
          }

}