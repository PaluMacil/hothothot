//
// Created by dan on 2020-10-27.
//

#include "gtest/gtest.h"
#include "Configuration.h"

TEST(Configuration, ArgumentCounts) {
    std::vector<std::string> args;
    args.emplace_back("SomeCommand");
    args.emplace_back("Setting=Yes");
    args.emplace_back("Thing=100");
    config::Configuration conf(args);

    auto count = conf.count.argc;

    EXPECT_EQ (count, 3);
    EXPECT_EQ (conf.count.standalone, 1);
    EXPECT_EQ (conf.count.pairs, 2);
}

TEST(Configuration, CommandType) {
    std::vector<std::string> argsInfo;
    argsInfo.emplace_back("INFO");
    argsInfo.emplace_back("Setting=Yes");
    argsInfo.emplace_back("Thing=100");
    config::Configuration conf1(argsInfo);
    EXPECT_EQ(conf1.command, config::CommandType::Info);

    std::vector<std::string> argsInvalid;
    argsInvalid.emplace_back("turnip");
    argsInvalid.emplace_back("Setting=Yes");
    argsInvalid.emplace_back("Thing=100");
    config::Configuration conf2(argsInvalid);
    EXPECT_EQ(conf2.command, config::CommandType::Help);

    std::vector<std::string> argsNoCmd;
    argsNoCmd.emplace_back("Setting=Yes");
    argsNoCmd.emplace_back("Thing=100");
    config::Configuration conf3(argsNoCmd);
    EXPECT_EQ(conf3.command, config::CommandType::Help);


    std::vector<std::string> argsTP;
    argsTP.emplace_back("Setting=Yes");
    argsTP.emplace_back("Thing=100");
    argsTP.emplace_back("TIMEPOINT");
    config::Configuration conf4(argsTP);
    EXPECT_EQ(conf4.command, config::CommandType::TimePoint);
}

TEST(Configuration, ArgumentValues) {
    std::vector<std::string> args;
    args.emplace_back("TIME=10000000");
    args.emplace_back("LOCATION=.7");
    args.emplace_back("TIMEPOINT");
    config::Configuration conf(args);

    EXPECT_EQ(conf.dimensions, config::Dimension::One);
    EXPECT_NEAR(conf.location, .7, .0000001);
    EXPECT_EQ(conf.device, config::DeviceType::CPU);
    EXPECT_EQ(conf.time, 10000000);
    EXPECT_EQ(conf.ambientTemp, 23);
    EXPECT_EQ(conf.sourceTemp, 100);
    EXPECT_EQ(conf.slices, 2500);
}

TEST(Configuration, ShortArgumentValues) {
    std::vector<std::string> args;
    args.emplace_back("DIM=1");
    args.emplace_back("L=.4");
    args.emplace_back("DEV=GPU");
    args.emplace_back("T=5000");
    args.emplace_back("AMBIENT=20");
    args.emplace_back("SOURCE=400");
    args.emplace_back("SLICES=6700");
    config::Configuration conf(args);

    EXPECT_EQ(conf.dimensions, config::Dimension::One);
    EXPECT_NEAR(conf.location, .4, .0000001);
    EXPECT_EQ(conf.device, config::DeviceType::CUDA_GPU);
    EXPECT_EQ(conf.time, 5000);
    EXPECT_EQ(conf.ambientTemp, 20);
    EXPECT_EQ(conf.sourceTemp, 400);
    EXPECT_EQ(conf.slices, 6700);
}