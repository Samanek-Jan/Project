#include <cuda.h>
#include <stdio.h>

#include "chatGPT_matrix_mul.cu"

#define ROW_SIZE 3
#define COL_SIZE 3

void print_matrix(float* M, size_t row_size=ROW_SIZE, size_t col_size=COL_SIZE) {
    for (size_t i = 0; i < row_size; i++)
    {
        for (size_t j = 0; j < col_size-1; j++)
        {
            printf("%.2f, ", M[i*col_size+j]);
        }
        printf("%.2f\n", M[(i+1)*col_size-1]);
    }
}


int main(int argc, char **argv) {

    float *A_cpu, *B_cpu, *C_cpu;

    cudaMallocHost(&A_cpu, ROW_SIZE*COL_SIZE*sizeof(float));
    cudaMallocHost(&B_cpu, ROW_SIZE*COL_SIZE*sizeof(float));
    cudaMallocHost(&C_cpu, ROW_SIZE*COL_SIZE*sizeof(float));


    if (!A_cpu || !B_cpu || !C_cpu) {
        if (A_cpu) {
            free(A_cpu);
        }

        if (B_cpu) {
            free(B_cpu);
        }

        if (C_cpu) {
            free(C_cpu);
        }

        printf("Allocating memory failed\n");
        return -1;
    }


    float* A_gpu;
    float* B_gpu;
    float* C_gpu;

    if (cudaMalloc<float>(&A_gpu, sizeof(float)*ROW_SIZE*COL_SIZE) != cudaSuccess) {
        cudaFree(A_cpu);
        cudaFree(B_cpu);
        cudaFree(C_cpu);
        printf("cudaMalloc failed\n");
        return -1;
    } else if(cudaMalloc<float>(&B_gpu, sizeof(float)*ROW_SIZE*COL_SIZE) != cudaSuccess) {
        cudaFree(A_gpu);
        cudaFree(A_cpu);
        cudaFree(B_cpu);
        cudaFree(C_cpu);
        printf("cudaMalloc failed\n");
        return -1;
    } else if (cudaMalloc<float>(&C_gpu, sizeof(float)*ROW_SIZE*COL_SIZE) != cudaSuccess) {
        cudaFree(A_gpu);
        cudaFree(B_gpu);
        cudaFree(A_cpu);
        cudaFree(B_cpu);
        cudaFree(C_cpu);
        printf("cudaMalloc failed\n");
        return -1;
    }

    for (size_t i = 0; i < ROW_SIZE; i++)
    {
        for (size_t j = 0; j < COL_SIZE; j++)
        {
            A_cpu[i*COL_SIZE + j] = i*COL_SIZE + j + 1;
            B_cpu[i*COL_SIZE + j] = i*COL_SIZE + j + 1;
        }
    }

    cudaDeviceSynchronize();

    print_matrix(A_cpu);
    printf("\n");
    print_matrix(B_cpu);

    cudaMemcpy(A_gpu, A_cpu, ROW_SIZE*COL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, ROW_SIZE*COL_SIZE, cudaMemcpyHostToDevice);
    matrixMultiplication<<<64, 128>>>(A_gpu, B_gpu, C_gpu, ROW_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);
    cudaMemcpy(C_cpu, C_gpu, ROW_SIZE*COL_SIZE, cudaMemcpyDeviceToHost);
    
    printf("===============================\n");
    print_matrix(C_cpu);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(A_cpu);
    cudaFree(B_cpu);
    cudaFree(C_cpu);

    return 0;
}