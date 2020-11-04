//
// Created by dan on 2020-10-27.
//

#ifndef HOTHOTHOT_CALCULATOR_CUH
#define HOTHOTHOT_CALCULATOR_CUH

#include <chrono>
#include "Configuration.h"

class Calculator {
public:
    explicit Calculator(const config::Configuration& config);

    ~Calculator();

    float exec();

private:
    config::Configuration config;
    unsigned long arraySize;
    float* array1;
    float* array2;
    float* array1_d;
    float* array2_d;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    bool outputCSV;

    void execCPU(const float* currentArray, float* nextArray) const;
    void execGPU(const float* currentArray, float* nextArray) const;
};

#endif //HOTHOTHOT_CALCULATOR_CUH
