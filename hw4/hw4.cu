#ifdef _WIN32
#define NOMINMAX
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__constant__ float input_c[BLOCK_SIZE];
#define UNROLL_FACT  64
/*final optimized version*/
__global__ void
compute_kernel(float* result, float* input2)
{
    __shared__ float sdata[128 + UNROLL_FACT];
    __shared__ float sum;
    int tx = threadIdx.x;
    if (tx == 0){
        sum = 0;
        for (int k = 0; k < 128; k++) sum +=  input_c[k];
    }
    for (int i = 0; i < 128; i++) {
        int share_index = tx + UNROLL_FACT;
        int input_index = i * 128 + tx;
        sdata[share_index] = input2[input_index];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 1];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 2];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 4];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 8];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 16];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 32];
        __syncthreads();
        sdata[share_index] += sdata[share_index - 64];
        __syncthreads();
        result[input_index] = sum * input_c[tx] + sdata[share_index];
    }
}

__global__ void
compute_compare_3_kernel(float* result, float* input2)
{
//#define UNROLL_FACT  32
    __shared__ float sdata[128 + UNROLL_FACT];
    __shared__ float sum;
    int tx = threadIdx.x;
    if (tx == 0){
        sum = 0;
        for (int k = 0; k < 128; k++)
            sum +=  input_c[k];
    }
    for (int i = 0; i < 128; ++ i) {
        int index = tx + UNROLL_FACT;
        sdata[index] = input2[i * 128 + tx];
        __syncthreads();
        sdata[index] += sdata[index - 1];
        __syncthreads();
        sdata[index] += sdata[index - 2];
        __syncthreads();
        sdata[index] += sdata[index - 4];
        __syncthreads();
        sdata[index] += sdata[index - 8];
        __syncthreads();
        sdata[index] += sdata[index - 16];
        __syncthreads();
        sdata[index] += sdata[index - 32];
        for (int stride = 64; stride <= 128; stride *=2) {
            __syncthreads();
            if(index >= stride)
                sdata[index] += sdata[index - stride];
        }
        __syncthreads();
        result[i * 128 + tx] = sum * input_c[tx] + sdata[index];
    }
}

/*compare version 2*/
__global__ void
compute_compare_2_kernel(float* result, float* input2)
{
    __shared__ float sdata[128];
    __shared__ float sum;
    int tx = threadIdx.x;
    if (tx == 0){
        sum = 0;
        for (int k = 0; k < 128; k++)
            sum +=  input_c[k];
    }
    for (int i = 0; i < 128; ++ i) {
        sdata[tx] = input2[i * 128 + tx];
        for (int stride = 1; stride <= 128; stride *=2) {
            __syncthreads();
            if(tx >= stride)
                sdata[tx] += sdata[tx - stride];
        }
        __syncthreads();
        result[i * 128 + tx] = sum * input_c[tx] + sdata[tx];
    }
}

/* this function works as an GPU baseline */
__global__ void
compute_compare_1_kernel(float* result, float* input2, float * input_1)
{
    float sum;
    int tx = threadIdx.x;
    for (int k = 0; k < 128; k++) sum +=  input_1[k];

    for (int i = 0; i < 128; ++ i) {
        for (int stride = 1; stride <= 128; stride *=2) {
            __syncthreads();
            if (tx >= stride){
                input2[i * 128 + tx] += input2[i * 128 + tx - stride];
            }
        }
        __syncthreads();
        result[i * 128 + tx] = sum * input_1[tx] + input2[i * 128 + tx];
    }
}
/* kernel call kernel version , not efficient */
//__global__ void
//compute_col_kernel(float* result, float* input2, float sum)
//{
//    __shared__ float sdata[128];
//    int tx = threadIdx.x;
//    sdata[tx] = input2[tx];
//
//    for (int stride = 1; stride <= 128; stride *=2) {
//        __syncthreads();
//        if (tx >= stride)
//            sdata[tx] += sdata[tx- stride];
//    }
//    __syncthreads();
//    result[tx] = sdata[tx] + input_c[tx] * sum;
//    return;
//}
//
//__global__ void
//compute_row_kernel(float* result, float* input2)
//{
//    int tx = threadIdx.x;
//    float sum = 0.0f;
//    // call child kernel
//    if (tx < 128){
//        for (int i = 0; i < blockDim.x; i++) sum +=  input_c[i];
//        compute_col_kernel<<<1,BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(result + 128 * tx, input2 + 128 * tx, sum);
//    }
//    return;
//}

void
computeOnDevice(float* result,float* h_data1, float* h_data2, int num_elements)
{
    float milliseconds;
    float* d_data2, *d_data1;
    float* d_result;
    int array_memsize = num_elements * num_elements * sizeof(float);

    cudaMalloc(&d_data2, array_memsize);
    cudaMalloc(&d_result, array_memsize);
    cudaMalloc(&d_data1, num_elements* sizeof(float));

    cudaMemcpy(d_data2, h_data2, array_memsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_c, h_data1, num_elements * sizeof(float));
    cudaMemcpy(d_data1, h_data1, num_elements* sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(1, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //GPU baseline 470 us
    cudaEventRecord(start, 0);
    compute_compare_1_kernel<<<gridDim, blockDim>>>(d_result, d_data2, d_data1);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("For compare Baseline EXE TIME = %f us\n", milliseconds * 1000);
    cudaMemcpy(d_data2, h_data2, array_memsize, cudaMemcpyHostToDevice);

    //without unrolling 261.68us
//    cudaEventRecord(start, 0);
//    for (int i = 0; i < 100; ++i) {
//        compute_compare_2_kernel << < gridDim, blockDim >> > (d_result, d_data2);
//        cudaDeviceSynchronize();
//    }
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("For compare version 2 EXE TIME = %f us\n", milliseconds * 1000);

    /*unroll factor test */
    //factor 1 260 us
    //factor 2 252 us
    //factor 4 247 us
    //factor 8 246 us
    //facotr 16 248 us
    //facotr 32 245 us
    //facotr 64 231 us

//    cudaEventRecord(start, 0);
//    for (int i = 0; i < 1000; ++i) {
//        compute_compare_3_kernel<<<gridDim, blockDim>>>(d_result, d_data2);
//        cudaDeviceSynchronize();
//    }
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("For compare version 3 EXE TIME = %f us\n", milliseconds * 1000);

    //fully unroll
    cudaEventRecord(start, 0);
    compute_kernel<<<gridDim, blockDim>>>(d_result, d_data2);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Final Optimize EXE TIME = %f us \n", milliseconds * 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(result, d_result, array_memsize, cudaMemcpyDeviceToHost );
    cudaFree(d_data2);
    cudaFree(d_result);
    cudaFree(d_data1);

    return;
}
extern "C"
void computeGold(float* result, float* input1,  float* input2, int number)
{
    float temp[128];
    for (int i = 0; i < number; i++) {
        temp[i] = 0.0f;
        for (int j = 0; j < number; j++) {
            temp[i] += input2[i * number + j];
            result[i * number + j] = temp[i];
            for (int k = 0; k < number; k++) {
                result[i * number + j] += input1[j] * input1[k];
            }
        }
    }
    return;
}

void
verify_correct(float* result, float* reference, int num_elements)
{

    float epsilon = 50.0f;
    int flag = 0;
    unsigned int result_regtest;
    for (unsigned int i = 0; i < num_elements; ++i) {
        for (int j = 0; j < num_elements; ++j) {
            result_regtest = (abs(result[i * num_elements + j] - reference[i * num_elements + j]) <= epsilon);
            if (!result_regtest){
                flag = 1;
                printf( "i: %d, j: %d, device: %f  host: %f\n",i,j, result[i * num_elements+j], reference[i * num_elements+j]);
                printf( "Test FAILED\n");
            }
        }
    }
    if (!flag)
        printf( "Test PASSED\n");
}

void
runTest( int argc, char** argv) {

    int num_elements = 128;
    const unsigned int array_mem_size = sizeof(float) * num_elements * num_elements;

    // allocate host memory to store the input data
    float* h_data1 = (float*) malloc( num_elements  * sizeof(float));
    float* h_data2 = (float*) malloc(array_mem_size);
    float* result = (float*) malloc(array_mem_size);
    float* reference = (float*) malloc(array_mem_size);

    for (unsigned int i = 0; i < num_elements; ++i) {
        h_data1[i] = floorf(1000 * (rand() / (float) RAND_MAX));
        for (int j = 0; j < num_elements; ++j) {
            h_data2[i* num_elements + j] = floorf(1000 * (rand() / (float) RAND_MAX));
        }
    }
    // compute reference solution
    computeGold(reference, h_data1, h_data2, num_elements);

    computeOnDevice(result, h_data1, h_data2, num_elements);

    verify_correct(result,reference,num_elements);

    // cleanup memory
    free(h_data1);
    free(h_data2);
    free(result);
    free(reference);
}

int
main( int argc, char** argv)
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}
