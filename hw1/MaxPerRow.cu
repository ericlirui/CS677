#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_HIGHT 1024

__device__ double golbal_max_per_row[MAX_HIGHT];

__global__ void max_per_row_matrix(double *a, int width, int height)
{
    double max_value = 0;
    double *ptr_max = &golbal_max_per_row[0];
    // Get our global thread ID
    int row = threadIdx.x;
    if (row < height){
        for(int j = 0; j < width; j++){
            if(a[row * width + j] > max_value){
                max_value = a[row * width + j];
            }
        }
        ptr_max[row] = max_value;
        printf("row %d , max is %f \n",row, ptr_max[row]);
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int max_width = 1024;
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

    //setup the execution configuration
    int  blockSize = 1024;
    int  gridSize = 1;
    // setup the width and height for the input matrix
    int width = 4;
    int height = 2;

    // Execute the kernel
    max_per_row_matrix<<<gridSize, blockSize>>>(d_a, width, height);

    // Release device memory
    cudaFree(d_a);

    // Release host memory
    free(h_a);

    return 0;
}


