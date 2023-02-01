#define blockSize 16

__global__ void matrixMultiplication(
    float *A, float *B, float *C, 
    int numARows, int numAColumns, 
    int numBRows, int numBColumns, 
    int numCRows, int numCColumns   )
{
    int blockRow = blockIdx.y * blockSize;
    int blockCol = blockIdx.x * blockSize;

    float subA[blockSize][blockSize];
    float subB[blockSize][blockSize];

    int row, col;
    for (row = 0; row < blockSize; ++row)
    {
        for (col = 0; col < blockSize; ++col)
        {
            if (blockRow + row < numARows && blockCol + col < numBColumns)
            {
                subA[row][col] = A[(blockRow + row) * numAColumns + (blockCol + col)];
                subB[row][col] = B[(blockCol + col) * numBColumns + (blockRow + row)];
            }
        }
    }

    float subC[blockSize][blockSize];
    for (row = 0; row < blockSize; ++row)
    {
        for (col = 0; col < blockSize; ++col)
        {
            float value = 0;
            for (int i = 0; i < blockSize; ++i)
                value += subA[row][i] * subB[i][col];
            subC[row][col] = value;
        }
    }

    for (row = 0; row < blockSize; ++row)
    {
        for (col = 0; col < blockSize; ++col)
        {
            if (blockRow + row < numCRows && blockCol + col < numCColumns)
                C[(blockRow + row) * numCColumns + (blockCol + col)] = subC[row][col];
        }
    }
}