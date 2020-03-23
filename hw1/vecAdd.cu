#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void matrix_output_element(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    c[id] = a[id] + b[id];
}

__global__ void matrix_output_row(double *a, double *b, double *c, int n)
{
    int i,tmp;
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        for (i = 0; i < n; i++) {
            tmp = id * n + i;
            c[tmp] = a[tmp] + b[tmp];
        }
    }
}
__global__ void matrix_output_column(double *a, double *b, double *c, int n)
{
    int i,tmp;
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n){
        for( i = 0 ; i < n; i++){
            tmp = id + i * n;
            c[tmp] = a[tmp] + b[tmp];
        }
    }

}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void print_report(double *h_a, double* h_b, double* h_c)
{
    int width = 1024;
    printf("the top-left 5*5 input a:\n");
    for ( int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", h_a[i* width + j]);
        }
        printf("\n");
    }
    printf("the top-left 5*5 input b:\n");
    for ( int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", h_b[i* width + j]);
        }
        printf("\n");
    }

    printf("the top-left 5*5 output c:\n");
    for ( int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", h_c[i* width + j]);
        }
        printf("\n");
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int width = 1024;
    int n = width * width;

    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);


    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //setup the execution configuration
    int blockSize, gridSize;

    // Execute the kernel
    //vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    blockSize = 1024;
    gridSize = (int)ceil((float)n/blockSize);
    matrix_output_element<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    cudaMemset( d_c, 0, bytes);
    print_report(h_a,h_b,h_c);

    blockSize = 1024;
    gridSize = 1;
    matrix_output_row<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    cudaMemset( d_c, 0, bytes);
    print_report(h_a,h_b,h_c);

    blockSize = 1024;
    gridSize = 1;
    matrix_output_column<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    //input report
    print_report(h_a,h_b,h_c);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

