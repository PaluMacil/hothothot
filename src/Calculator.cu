// Calculator.cu
// Dan Wolf

#include <cmath>
#include <iostream>
#include "Calculator.cuh"


// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void execGPU_d(int n, float sourceTemp, const float *currentArray, float *nextArray) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto idxBefore = i - 1;
        auto valBefore =
                idxBefore == -1 ?
                // if before the start of the array, use temperature of the heat source
                sourceTemp :
                currentArray[idxBefore];
        auto idxAfter = i + 1;
        auto valAfter =
                idxAfter == n ?
                // if the last element, use own temperature
                currentArray[i] :
                currentArray[idxAfter];
        nextArray[i] = (valBefore + valAfter) / 2;
    }
}

Calculator::Calculator(const config::Configuration &config) : config(config) {
    this->config = config;
    this->arraySize = this->config.slices * sizeof(float);
    array1 = (float *) malloc(this->arraySize);
    for (int i = 0; i < this->config.slices; i++) {
        array1[i] = this->config.ambientTemp;
    }
    if (this->config.device == config::DeviceType::CPU) {
        array2 = (float *) malloc(this->arraySize);
        array1_d = array2_d = {};
    } else {
        array2 = {};
        gpuErrchk(cudaMalloc(&this->array1_d, arraySize));
        gpuErrchk(cudaMalloc(&this->array2_d, arraySize));
        gpuErrchk(cudaMemcpy(
                array1_d,
                array1,
                this->arraySize,
                cudaMemcpyHostToDevice));
    }
    this->outputCSV = config.command == config::CommandType::Graph;
}

Calculator::~Calculator() {
    free(this->array1);
    if (this->config.device == config::DeviceType::CPU) {
        free(this->array2);
    } else {
        gpuErrchk(cudaFree(this->array1_d));
        gpuErrchk(cudaFree(this->array2_d));
    }
}

ObjectSnapshot Calculator::snapshotAt(int time) {
    bool currentArray1 = true;

    float *snapshotArray;

    for (int i = 0; i < this->config.time + 1; i++) {
        this->execCPU(
                currentArray1 ? this->array1 : this->array2,
                currentArray1 ? this->array2 : this->array1
        );
        currentArray1 = !currentArray1;
        if (i == time) {
            snapshotArray = currentArray1 ? this->array1 : this->array2;
        };
    }

    ObjectSnapshot snapshot(snapshotArray, this->config.slices);

    return snapshot;
}

float Calculator::exec() {
    bool currentArray1 = true;

    this->start = std::chrono::steady_clock::now();
    // for every time step through the requested time....
    for (int i = 0; i < this->config.time; i++) {
        if (this->config.device == config::DeviceType::CPU) {
            this->execCPU(
                    currentArray1 ? this->array1 : this->array2,
                    currentArray1 ? this->array2 : this->array1
            );

            // Every 1% of the way through the calculation, output the value at the point requested
            // output at 0
            if (this->outputCSV && (i == 0 || i % (this->config.time / 50) == 0)) {
                auto locationIndex = (int) (this->config.location * (float) this->config.slices);
                if (currentArray1) {
                    std::cout << i << ',' << array1[locationIndex] << std::endl;
                } else {
                    std::cout << i << ',' << array2[locationIndex] << std::endl;
                }
            }
        } else {
            this->execGPU(
                    currentArray1 ? this->array1_d : this->array2_d,
                    currentArray1 ? this->array2_d : this->array1_d
            );
        }
        currentArray1 = !currentArray1;
    }
    this->end = std::chrono::steady_clock::now();
    if (!this->outputCSV) {
        std::cout << "Calculation Time (sec): "
                  << (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000000.0
                  << std::endl;
    }

    float answer;
    auto i = (int) (this->config.location * (float) this->config.slices);
    if (this->config.device == config::DeviceType::CPU) {
        answer = currentArray1 ? this->array1[i] : this->array2[i];
    } else {
        gpuErrchk(cudaMemcpy(
                array1,
                currentArray1 ? array1_d : array2_d,
                this->arraySize,
                cudaMemcpyDeviceToHost));
        answer = this->array1[i];
    }
    return answer;
}

void Calculator::execCPU(const float *currentArray, float *nextArray) const {
    for (int i = 0; i < this->config.slices; i++) {
        auto idxBefore = i - 1;
        auto valBefore =
                idxBefore == -1 ?
                // if before the start of the array, use temperature of the heat source
                this->config.sourceTemp :
                currentArray[idxBefore];
        auto idxAfter = i + 1;
        auto valAfter =
                idxAfter == this->config.slices ?
                // if the last element, use own temperature
                currentArray[i] :
                currentArray[idxAfter];
        nextArray[i] = (valBefore + valAfter) / 2;
    }
}

void Calculator::execGPU(const float *currentArray, float *nextArray) const {
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid = (int) ceil((float) this->config.slices / BLOCK_SIZE);
    execGPU_d<<<dimGrid, dimBlock>>>(this->config.slices, this->config.sourceTemp, currentArray, nextArray);
    gpuErrchk(cudaPeekAtLastError());
}
