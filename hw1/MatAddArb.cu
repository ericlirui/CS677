#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void matrix_output_element(double *a, double *b, double *c, int width, int height)
{
    // Get our global thread ID
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < height && col < width){
        int tmp = row * width + col;
        c[tmp] = a[tmp] + b[tmp];
    }
}

__global__ void matrix_output_row(double *a, double *b, double *c, int width, int height)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < height && col < width){
        for (int i = 0; i < width; i++) {
            int tmp = row + i * height;
            c[tmp] = a[tmp] + b[tmp];
        }
    }
}
__global__ void matrix_output_column(double *a, double *b, double *c, int width, int height)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < height && col < width){
        for(int i = 0 ; i < height; i++){
            int tmp = col + i * width;
            c[tmp] = a[tmp] + b[tmp];
        }
    }

}

void print_report(double *h_a, double* h_b, double* h_c, int width, int height)
{
    int max_width = 1024;
    printf("the top-left 5*5 input a:\n");
    for ( int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", h_a[i* max_width + j]);
        }
        printf("\n");
    }
    printf("the top-left 5*5 input b:\n");
    for ( int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", h_b[i* max_width + j]);
        }
        printf("\n");
    }

    printf("the output c:\n");
    for ( int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
            printf("%f ", h_c[i* width + j]);
        }
        printf("\n");
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int max_width = 1024;
    int n = max_width * max_width;

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
    // setup the width and height for the input matrix
    int width = 30;
    int height = 10;

    blockSize = 1024;
    gridSize = (int)ceil((float)n/blockSize);
    matrix_output_element<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    cudaMemset( d_c, 0, bytes);
    print_report(h_a,h_b,h_c,width,height);

    matrix_output_row<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    cudaMemset( d_c, 0, bytes);
    print_report(h_a,h_b,h_c,width,height);

    matrix_output_column<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    print_report(h_a,h_b,h_c,width,height);

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

