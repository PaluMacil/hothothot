//
// Created by dan on 2020-11-01.
//

#include "gtest/gtest.h"
#include "Calculator.cuh"

TEST(Calculator, ExecCPU) {
    std::vector<std::string> args;
    args.emplace_back("TIME=10000000");
    args.emplace_back("LOCATION=.7");
    args.emplace_back("TIMEPOINT");
    config::Configuration conf(args);
    Calculator calc(conf);
    auto answer = calc.exec();

    EXPECT_FLOAT_EQ(answer, 85.5276794);
}

TEST(Calculator, ExecGPU) {
    std::vector<std::string> args;
    args.emplace_back("TIME=10000000");
    args.emplace_back("LOCATION=.7");
    args.emplace_back("TIMEPOINT");
    args.emplace_back("DEVICE=GPU");
    config::Configuration conf(args);
    Calculator calc(conf);
    auto answer = calc.exec();

    EXPECT_FLOAT_EQ(answer, 85.5276794);
}